"""
LLM client for the validator.

Supports two backends (auto-selected):
  1. llama-server subprocess (bitnet.cpp or llama.cpp) — preferred when the binary
     is on PATH and BITNET_MODEL is set. Full OpenAI-compatible server, handles
     concurrent requests, can serve bitnet 1-bit kernels.
  2. llama-cpp-python in-process — fallback when llama-server is not available.
     Loads the GGUF model in the same process via the `llama-cpp-python` package.

Both backends expose the same `generate(prompt, max_tokens, temperature, stop)` call.

Env vars:
  BITNET_BIN     — path to the llama-server binary (default: llama-server in PATH)
  BITNET_MODEL   — path to the .gguf weights
                   (default: data/hf_cache/bitnet/ggml-model-i2_s.gguf)
  BITNET_PORT    — port for the llama-server backend (default: 8081)
  BITNET_THREADS — thread count (default: os.cpu_count())
  BITNET_CTX     — context size (default: 2048)
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = Path("data/hf_cache/bitnet/ggml-model-i2_s.gguf")


class _LlamaCppPythonBackend:
    """In-process backend using the llama-cpp-python package."""

    def __init__(self, model_path: Path, threads: int, ctx_size: int):
        from llama_cpp import Llama  # type: ignore[import]

        log.info("Loading GGUF model via llama-cpp-python: %s", model_path)
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=ctx_size,
            n_threads=threads,
            verbose=False,
        )
        log.info("Model loaded.")

    def generate(self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None) -> str:
        kwargs: dict = {"max_tokens": max_tokens, "temperature": temperature, "echo": False}
        if stop:
            kwargs["stop"] = stop
        output = self._llm(prompt, **kwargs)
        return output["choices"][0]["text"]


class _LlamaServerBackend:
    """Persistent llama-server subprocess backend."""

    def __init__(self, binary: str, model_path: Path, port: int, threads: int, ctx_size: int):
        self.binary = binary
        self.model_path = model_path
        self.port = port
        self.threads = threads
        self.ctx_size = ctx_size
        self.url = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        cmd = [
            self.binary,
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
            "-ngl", "0",     # no GPU offload (CPU-only bitnet inference)
        ]
        log.info("Starting llama-server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.stop)

        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    pass
                try:
                    r = requests.get(f"{self.url}/health", timeout=2)
                    if r.status_code == 200:
                        body = r.json() if r.content else {}
                        if body.get("status", "ok") == "ok":
                            log.info("llama-server ready on port %d", self.port)
                            return
                        # status="loading" — model still warming up
                except requests.RequestException:
                    pass
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("llama-server failed to become ready within 120s")

    def generate(self, prompt: str, max_tokens: int, temperature: float, stop: list[str] | None) -> str:
        for attempt in range(2):
            self._ensure_started()
            payload: dict = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "cache_prompt": True}
            if stop:
                payload["stop"] = stop
            try:
                # Retry 503 (slot busy) a few times before giving up
                for _ in range(5):
                    r = requests.post(f"{self.url}/completion", json=payload, timeout=90)
                    if r.status_code == 503:
                        log.debug("llama-server slot busy (503), retrying in 3s…")
                        time.sleep(3)
                        continue
                    r.raise_for_status()
                    return r.json().get("content", "")
                r.raise_for_status()
                return ""
            except requests.Timeout:
                # Kill stuck server and restart for the next attempt
                log.warning("llama-server timed out, restarting…")
                self.stop()
                self._proc = None
                time.sleep(2)
        raise RuntimeError("llama-server failed to respond after restart")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            log.info("Stopping llama-server")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


class BitnetClient:
    """
    Unified LLM client for the validator.

    Auto-selects backend:
      - llama-server subprocess if the binary is found on PATH
      - llama-cpp-python in-process otherwise
    """

    def __init__(
        self,
        model_path: Path | None = None,
        binary: str | None = None,
        port: int | None = None,
        threads: int | None = None,
        ctx_size: int | None = None,
    ):
        self._binary = binary or os.environ.get("BITNET_BIN", "llama-server")
        self._model_path = Path(
            model_path or os.environ.get("BITNET_MODEL", DEFAULT_MODEL_PATH)
        )
        self._port = int(port or os.environ.get("BITNET_PORT", "8081"))
        self._threads = int(threads or os.environ.get("BITNET_THREADS", os.cpu_count() or 4))
        self._ctx_size = int(ctx_size or os.environ.get("BITNET_CTX", "2048"))
        # Backend is initialised lazily on the first generate() call.
        self._backend: _LlamaServerBackend | _LlamaCppPythonBackend | None = None

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {self._model_path}. "
                "Download a model and set BITNET_MODEL, or rebuild the validator image."
            )

        has_binary = shutil.which(self._binary) is not None or Path(self._binary).exists()
        if has_binary:
            log.info("Using llama-server backend (%s)", self._binary)
            self._backend = _LlamaServerBackend(
                self._binary, self._model_path, self._port, self._threads, self._ctx_size
            )
        else:
            try:
                import llama_cpp  # noqa: F401  # type: ignore[import]
                log.info("llama-server not found, falling back to llama-cpp-python backend")
                self._backend = _LlamaCppPythonBackend(
                    self._model_path, self._threads, self._ctx_size
                )
            except ImportError:
                raise FileNotFoundError(
                    f"llama-server binary not found ({self._binary}) and llama-cpp-python "
                    "is not installed. Install one of them to run LLM evaluation."
                )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Run a single completion. Returns only the generated text."""
        self._ensure_backend()
        assert self._backend is not None
        return self._backend.generate(prompt, max_tokens, temperature, stop)

    def stop(self) -> None:
        if isinstance(self._backend, _LlamaServerBackend):
            self._backend.stop()

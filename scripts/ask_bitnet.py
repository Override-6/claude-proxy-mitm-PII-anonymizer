#!/usr/bin/env python3
"""
Prompt the BitNet LLM running inside the validator container.

The validator container runs llama-server (bitnet.cpp, TL2 kernels) on
127.0.0.1:8081 inside the container. This script:
  - auto-starts the server if it is not already running
  - supports one-shot mode (--prompt) and interactive REPL mode

Usage:
    # One-shot
    docker exec claude-validator poetry run python scripts/ask_bitnet.py \
        --prompt "What is the capital of France?"

    # Interactive REPL
    docker exec -it claude-validator poetry run python scripts/ask_bitnet.py

    # From the host (starts the container if needed):
    docker exec -it claude-validator poetry run python scripts/ask_bitnet.py
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import requests

MODEL_PATH = os.environ.get(
    "BITNET_MODEL",
    "data/hf_cache/bitnet/ggml-model-i2_s.gguf",
)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("BITNET_PORT", "8081"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
THREADS = int(os.environ.get("BITNET_THREADS", os.cpu_count() or 8))
CTX = int(os.environ.get("BITNET_CTX", "2048"))
BINARY = os.environ.get("BITNET_BIN", "llama-server")

_server_proc: subprocess.Popen | None = None
_llama_cpp_llm = None  # in-process fallback


def _find_binary() -> str | None:
    found = shutil.which(BINARY)
    if found:
        return found
    if os.path.isfile(BINARY):
        return BINARY
    return None


def _server_is_ready() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def ensure_server() -> None:
    global _server_proc, _llama_cpp_llm

    binary = _find_binary()

    if binary:
        # llama-server path
        if _server_is_ready():
            return

        if not os.path.exists(MODEL_PATH):
            sys.exit(
                f"Model not found: {MODEL_PATH}\n"
                "Set BITNET_MODEL or download the GGUF to data/hf_cache/bitnet/"
            )

        cmd = [
            binary,
            "-m", MODEL_PATH,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "-t", str(THREADS),
            "-c", str(CTX),
            "-ngl", "0",     # CPU-only
            "--log-disable",
        ]
        print(f"Starting llama-server on port {SERVER_PORT}…", flush=True)
        _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + 60
        while time.time() < deadline:
            if _server_proc.poll() is not None:
                sys.exit("llama-server exited unexpectedly during startup.")
            try:
                with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1):
                    pass
                if _server_is_ready():
                    print("Server ready.\n", flush=True)
                    return
            except OSError:
                time.sleep(0.5)

        sys.exit("llama-server did not become ready within 60s.")
    else:
        # in-process fallback via llama-cpp-python
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError:
            sys.exit(
                "llama-server binary not found and llama-cpp-python is not installed.\n"
                "Either rebuild the validator image (includes llama-server) or install llama-cpp-python."
            )

        if not os.path.exists(MODEL_PATH):
            sys.exit(
                f"Model not found: {MODEL_PATH}\n"
                "Set BITNET_MODEL or download the GGUF to data/hf_cache/bitnet/"
            )

        print(f"llama-server not found — loading model in-process via llama-cpp-python…", flush=True)
        _llama_cpp_llm = Llama(model_path=MODEL_PATH, n_ctx=CTX, n_threads=THREADS, verbose=False)
        print("Model loaded.\n", flush=True)


def stop_server() -> None:
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc.wait(timeout=5)


def complete(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    if _llama_cpp_llm is not None:
        output = _llama_cpp_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nHuman:", "\nUser:"],
            echo=False,
        )
        return output["choices"][0]["text"]

    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stop": ["\nHuman:", "\nUser:"],
        "cache_prompt": True,
    }
    r = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["content"]


def _build_prompt(history: list[tuple[str, str]], new_user: str) -> str:
    """Build a simple Human/Assistant prompt from conversation history."""
    parts = []
    for user_msg, assistant_msg in history:
        parts.append(f"Human: {user_msg}\n\nBITNETAssistant: {assistant_msg}")
    parts.append(f"Human: {new_user}\n\nBITNETAssistant:")
    return "\n\n".join(parts)


def run_repl() -> None:
    history: list[tuple[str, str]] = []
    print("BitNet-b1.58-2B-4T  |  type 'exit' or Ctrl-C to quit\n")

    def handle_sigint(sig, frame):
        print("\nBye.")
        stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "q"):
            break

        prompt = _build_prompt(history, user)
        try:
            answer = complete(prompt).strip()
        except Exception as e:
            print(f"[error] {e}\n")
            continue

        print(f"\nBitNet: {answer}\n")
        history.append((user, answer))

    stop_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt the BitNet LLM in the validator container")
    parser.add_argument("--prompt", "-p", help="One-shot prompt (non-interactive)")
    parser.add_argument("--max-tokens", "-n", type=int, default=512)
    parser.add_argument("--temperature", "-T", type=float, default=0.7)
    parser.add_argument("--no-server", action="store_true",
                        help="Skip server startup (assume it is already running)")
    args = parser.parse_args()

    if not args.no_server:
        ensure_server()

    if args.prompt:
        answer = complete(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        print(answer.strip())
        stop_server()
    else:
        run_repl()


if __name__ == "__main__":
    main()

"""
UDP control socket for the proxy.
Listens on 127.0.0.1:9999 and handles commands from console.py.
"""

import asyncio
import json


CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 9999


class _ControlProtocol(asyncio.DatagramProtocol):
    def __init__(self, state: dict):
        self._state = state
        self._transport = None

    def connection_made(self, transport):
        self._transport = transport

    def datagram_received(self, data: bytes, addr):
        try:
            cmd = data.decode("utf-8").strip()
            response = self._handle(cmd)
        except Exception as e:
            response = {"ok": False, "error": str(e)}

        self._transport.sendto(
            json.dumps(response, ensure_ascii=False).encode("utf-8"),
            addr,
        )

    def _handle(self, cmd: str) -> dict:
        cmd = cmd.lower()
        if cmd == "anon on":
            self._state["anon_enabled"] = True
            return {"ok": True, "anon_enabled": True}

        if cmd == "anon off":
            self._state["anon_enabled"] = False
            return {"ok": True, "anon_enabled": False}

        if cmd == "deanon on":
            self._state["deanon_enabled"] = True
            return {"ok": True, "deanon_enabled": True}

        if cmd == "deanon off":
            self._state["deanon_enabled"] = False
            return {"ok": True, "deanon_enabled": False}

        if cmd == "save images on":
            self._state["save_images"] = True
            return {"ok": True, "save_images": True}

        if cmd == "save images off":
            self._state["save_images"] = False
            return {"ok": True, "save_images": False}

        if cmd == "system prompt on":
            self._state["system_prompt_enabled"] = True
            return {"ok": True, "system_prompt_enabled": True}

        if cmd == "system prompt off":
            self._state["system_prompt_enabled"] = False
            return {"ok": True, "system_prompt_enabled": False}

        if cmd == "status":
            return {
                "ok": True,
                "anon_enabled": self._state["anon_enabled"],
                "deanon_enabled": self._state["deanon_enabled"],
                "save_images": self._state["save_images"],
                "system_prompt_enabled": self._state["system_prompt_enabled"],
            }

        if cmd == "dump":
            mappings = self._state["mappings"]
            entities = [
                {"sensitive": sensitive, "redacted": redacted}
                for sensitive, redacted in mappings._sensitive_to_redacted.items()
            ]
            return {"ok": True, "entities": entities}

        if cmd == "clear":
            mappings = self._state["mappings"]
            mappings._sensitive_to_redacted.clear()
            mappings._redacted_to_sensitive.clear()
            return {"ok": True, "cleared": True}

        return {"ok": False, "error": f"unknown command: {cmd!r}"}


async def init_control_socket(state: dict):
    loop = asyncio.get_event_loop()
    await loop.create_datagram_endpoint(
        lambda: _ControlProtocol(state),
        local_addr=(CONTROL_HOST, CONTROL_PORT),
    )
    print(f"[control] UDP socket listening on {CONTROL_HOST}:{CONTROL_PORT}")

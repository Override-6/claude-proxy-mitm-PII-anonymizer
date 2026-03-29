"""
TCP control socket for the proxy.
Listens on 0.0.0.0:9999 and handles commands from console.py.
Each connection sends one command and receives one JSON response, then closes.
"""

import asyncio
import json


CONTROL_HOST = "0.0.0.0"
CONTROL_PORT = 9999


def _handle(cmd: str, state: dict) -> dict:
    cmd = cmd.lower()
    if cmd == "anon on":
        state["anon_enabled"] = True
        return {"ok": True, "anon_enabled": True}

    if cmd == "anon off":
        state["anon_enabled"] = False
        return {"ok": True, "anon_enabled": False}

    if cmd == "deanon on":
        state["deanon_enabled"] = True
        return {"ok": True, "deanon_enabled": True}

    if cmd == "deanon off":
        state["deanon_enabled"] = False
        return {"ok": True, "deanon_enabled": False}

    if cmd == "save images on":
        state["save_images"] = True
        return {"ok": True, "save_images": True}

    if cmd == "save images off":
        state["save_images"] = False
        return {"ok": True, "save_images": False}

    if cmd == "system prompt on":
        state["system_prompt_enabled"] = True
        return {"ok": True, "system_prompt_enabled": True}

    if cmd == "system prompt off":
        state["system_prompt_enabled"] = False
        return {"ok": True, "system_prompt_enabled": False}

    if cmd == "status":
        return {
            "ok": True,
            "anon_enabled": state["anon_enabled"],
            "deanon_enabled": state["deanon_enabled"],
            "save_images": state["save_images"],
            "system_prompt_enabled": state["system_prompt_enabled"],
        }

    if cmd == "dump":
        mappings = state["mappings"]
        entities = [
            {"sensitive": sensitive, "redacted": redacted}
            for sensitive, redacted in mappings._sensitive_to_redacted.items()
        ]
        return {"ok": True, "entities": entities}

    if cmd == "clear":
        mappings = state["mappings"]
        mappings._sensitive_to_redacted.clear()
        mappings._redacted_to_sensitive.clear()
        return {"ok": True, "cleared": True}

    return {"ok": False, "error": f"unknown command: {cmd!r}"}


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: dict):
    try:
        data = await reader.read(1024)
        cmd = data.decode("utf-8").strip()
        response = _handle(cmd, state)
    except Exception as e:
        response = {"ok": False, "error": str(e)}
    finally:
        writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        await writer.drain()
        writer.close()


async def init_control_socket(state: dict):
    server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, state),
        CONTROL_HOST,
        CONTROL_PORT,
    )
    print(f"[control] TCP socket listening on {CONTROL_HOST}:{CONTROL_PORT}")
    asyncio.get_event_loop().create_task(server.serve_forever())

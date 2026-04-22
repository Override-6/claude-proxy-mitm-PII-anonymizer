"""
TCP control socket for the proxy.
Listens on 0.0.0.0:9999 and handles commands from console.py.
Each connection sends one command and receives one JSON response, then closes.
"""

import asyncio
import json
import logging

from proxy.engine import DLPProxy

log = logging.getLogger(__name__)

CONTROL_HOST = "0.0.0.0"
CONTROL_PORT = 9999


def _handle(cmd: str, proxy: DLPProxy) -> dict:
    cmd = cmd.lower()

    if cmd == "dump":
        mappings = proxy.mappings
        return {"ok": True, "entities": mappings.dump()}

    if cmd == "clear":
        mappings = proxy.mappings
        mappings.reset()
        return {"ok": True, "cleared": True}

    return {"ok": False, "error": f"unknown command: {cmd!r}"}


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, proxy: DLPProxy):
    try:
        data = await reader.read(1024)
        cmd = data.decode("utf-8").strip()
        response = _handle(cmd, proxy)
    except Exception as e:
        response = {"ok": False, "error": str(e)}
    finally:
        writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        await writer.drain()
        writer.close()


async def init_control_socket(proxy: DLPProxy):
    server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, proxy),
        CONTROL_HOST,
        CONTROL_PORT,
    )
    log.info("Control socket listening on %s:%d", CONTROL_HOST, CONTROL_PORT)
    asyncio.get_event_loop().create_task(server.serve_forever())

import asyncio
import json

import socket


class EventSocket(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport: asyncio.DatagramTransport = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport
        # Enable broadcast
        sock = transport.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def broadcast(self, payload: dict):
        if "body" in payload and isinstance(payload["body"], bytes):
            try:
                payload["body"] = json.loads(payload["body"])
            except json.JSONDecodeError:
                pass
        data = (json.dumps(payload) + "\n").encode("utf-8")
        # UDP max is ~65KB. For large payloads, split into chunks.
        MAX_CHUNK = 60000
        if len(data) <= MAX_CHUNK:
            self.transport.sendto(data, ("255.255.255.255", 8080))
        else:
            # Send a header with total size and chunk count, then chunks
            total = len(data)
            chunks = [data[i:i + MAX_CHUNK] for i in range(0, total, MAX_CHUNK)]
            chunk_id = id(payload) & 0xFFFFFFFF
            for idx, chunk in enumerate(chunks):
                header = json.dumps({
                    "type": "chunk",
                    "chunk_id": chunk_id,
                    "chunk_idx": idx,
                    "chunk_total": len(chunks),
                    "size": total,
                }).encode("utf-8") + b"\n"
                self.transport.sendto(header + chunk, ("255.255.255.255", 8080))


event_socket = None


def get_event_socket():
    return event_socket

async def init_event_socket():
    global event_socket

    loop = asyncio.get_event_loop()
    transport, event_socket = await loop.create_datagram_endpoint(
        EventSocket,
        local_addr=("127.0.0.1", 0)
    )
    return event_socket

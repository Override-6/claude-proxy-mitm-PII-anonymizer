import asyncio
import json

_pending_chunks: dict[int, dict] = {}  # chunk_id -> {total, parts: {idx: bytes}}

class JSONListenerProtocol(asyncio.DatagramProtocol):
    def datagram_received(self, data: bytes, addr: tuple):
        try:
            # Check if this is a chunked message (header + payload on same datagram)
            newline = data.find(b"\n")
            header_part = data[:newline] if newline != -1 else data
            header = json.loads(header_part.decode("utf-8"))

            line = None

            if header.get("type") == "chunk":
                # Reassemble chunked payload
                cid = header["chunk_id"]
                idx = header["chunk_idx"]
                total = header["chunk_total"]
                payload_data = data[newline + 1:] if newline != -1 else b""

                if cid not in _pending_chunks:
                    _pending_chunks[cid] = {"total": total, "parts": {}}
                _pending_chunks[cid]["parts"][idx] = payload_data

                if len(_pending_chunks[cid]["parts"]) == total:
                    # All chunks received — reassemble
                    full = b"".join(_pending_chunks[cid]["parts"][i] for i in range(total))
                    del _pending_chunks[cid]
                    payload = json.loads(full.decode("utf-8"))
                    line = json.dumps(payload, indent=2)
            else:
                 line = json.dumps(header, indent=2)

            if line:
                print(line)
                with open("ignore/events.jsonl", "a", buffering=1) as f:
                    f.write(line + "\n")

        except json.JSONDecodeError:
            print(f"[WARN] Non-JSON packet from {addr}: {data[:200]}")

async def main():
    loop = asyncio.get_event_loop()
    await loop.create_datagram_endpoint(
        JSONListenerProtocol,
        local_addr=("0.0.0.0", 8080)
    )
    await asyncio.sleep(9999)

asyncio.run(main())

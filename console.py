"""
Interactive console for the MITM proxy.
Communicates with the proxy via UDP on 127.0.0.1:9999.

Commands:
  anon on/off              Enable / disable request anonymization
  deanon on/off            Enable / disable response deanonymization
  save images on/off       Save redacted images to ignore/redacted_images/
  system prompt on/off     Enable / disable SYSTEM_PROMPT.md injection
  status                   Show current state
  dump                     Print all known entity mappings
  clear                    Wipe the entity mapping table
  help                     Show this message
  quit / exit              Leave the console
"""

import json
import socket
import sys

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9999
TIMEOUT    = 3.0  # seconds

HELP = __doc__


def send(cmd: str) -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(TIMEOUT)
        sock.sendto(cmd.encode("utf-8"), (PROXY_HOST, PROXY_PORT))
        data, _ = sock.recvfrom(65535)
    return json.loads(data.decode("utf-8"))


def fmt_bool(value: bool) -> str:
    return "\033[32mON\033[0m" if value else "\033[31mOFF\033[0m"


def handle(cmd: str):
    cmd = cmd.strip()

    if not cmd or cmd.startswith("#"):
        return

    if cmd in ("help", "?"):
        print(HELP)
        return

    if cmd in ("quit", "exit", "q"):
        print("bye")
        sys.exit(0)

    # Commands that map 1-to-1 to the socket protocol
    if cmd in ("anon on", "anon off", "deanon on", "deanon off",
               "save images on", "save images off",
               "system prompt on", "system prompt off",
               "status", "dump", "clear"):
        try:
            resp = send(cmd)
        except TimeoutError:
            print("[error] no response from proxy — is it running?")
            return
        except Exception as e:
            print(f"[error] {e}")
            return

        if not resp.get("ok"):
            print(f"[error] {resp.get('error', 'unknown error')}")
            return

        if cmd == "status":
            print(f"  anon           {fmt_bool(resp['anon_enabled'])}")
            print(f"  deanon         {fmt_bool(resp['deanon_enabled'])}")
            print(f"  save images    {fmt_bool(resp.get('save_images', False))}")
            print(f"  system prompt  {fmt_bool(resp.get('system_prompt_enabled', False))}")

        elif cmd == "dump":
            entities = resp.get("entities", [])
            if not entities:
                print("  (no entities recorded yet)")
            else:
                max_s = max(len(e["sensitive"]) for e in entities)
                for e in entities:
                    print(f"  {e['sensitive']:<{max_s}}  →  {e['redacted']}")
            print(f"\n  {len(entities)} entit{'y' if len(entities) == 1 else 'ies'} total")

        elif cmd == "clear":
            print("  entity table cleared")

        elif cmd.startswith("anon"):
            print(f"  anonymization {fmt_bool(resp['anon_enabled'])}")

        elif cmd.startswith("deanon"):
            print(f"  deanonymization {fmt_bool(resp['deanon_enabled'])}")

        elif cmd.startswith("save images"):
            print(f"  save images {fmt_bool(resp['save_images'])}")

        elif cmd.startswith("system prompt"):
            print(f"  system prompt {fmt_bool(resp['system_prompt_enabled'])}")

    else:
        print(f"  unknown command {cmd!r} — type 'help' for usage")


def main():
    print("Claude MITM proxy console  (type 'help' for commands)")
    print(f"Connecting to proxy at {PROXY_HOST}:{PROXY_PORT}\n")

    # Show initial status
    try:
        resp = send("status")
        print(f"  anon           {fmt_bool(resp['anon_enabled'])}")
        print(f"  deanon         {fmt_bool(resp['deanon_enabled'])}")
        print(f"  save images    {fmt_bool(resp.get('save_images', False))}")
        print(f"  system prompt  {fmt_bool(resp.get('system_prompt_enabled', False))}")
        print()
    except Exception:
        print("  [warn] proxy not reachable yet — commands will retry\n")

    while True:
        try:
            line = input("\033[36m> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        handle(line)


if __name__ == "__main__":
    main()

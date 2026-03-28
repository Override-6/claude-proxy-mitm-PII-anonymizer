"""
Transparent proxifier for Windows using WinDivert.

Forces ALL outgoing HTTPS (port 443) connections through a local mitmproxy
instance, regardless of whether the application respects proxy env vars.

How it works:
  1. WinDivert intercepts outgoing TCP packets destined for :443
  2. NAT rewrites redirect them to a local TCP relay (on machine's real IP)
  3. The relay opens an HTTP CONNECT tunnel through mitmproxy
  4. Data flows bidirectionally: app <-> relay <-> mitmproxy <-> real server

Requirements:
  pip install pydivert psutil
  Run as Administrator (WinDivert needs kernel-level access)
  mitmproxy running in regular mode on PROXY_PORT

Usage:
  python proxifier.py
"""

import ctypes
import logging
import os
import socket
import sys
import threading
import time

import psutil
import pydivert

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8080
RELAY_PORT = 44300
TARGET_PORT = 443

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxifier.log")

log = logging.getLogger("proxifier")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
log.addHandler(_ch)

_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
log.addHandler(_fh)

# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

LOCAL_IP = _get_local_ip()

# NAT table: (client_ip, client_port) -> (original_dst_ip, original_dst_port)
nat_table: dict[tuple[str, int], tuple[str, int]] = {}
nat_lock = threading.Lock()

passthrough: set[tuple[str, int]] = set()
passthrough_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Process exclusion — skip mitmproxy's own connections to avoid loops
# ---------------------------------------------------------------------------

_excluded_pids: set[int] = set()
_excluded_pids_lock = threading.Lock()


def _refresh_excluded_pids():
    pids: set[int] = set()
    try:
        for conn in psutil.net_connections("tcp4"):
            if (conn.status == "LISTEN"
                    and conn.laddr
                    and conn.laddr.port == PROXY_PORT
                    and conn.pid):
                pids.add(conn.pid)
                try:
                    for child in psutil.Process(conn.pid).children(recursive=True):
                        pids.add(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except (psutil.AccessDenied, OSError) as e:
        log.warning("Failed to enumerate excluded PIDs: %s", e)
    with _excluded_pids_lock:
        _excluded_pids.clear()
        _excluded_pids.update(pids)
    log.info("Excluded PIDs (mitmproxy): %s", pids or "NONE")


def _pid_refresh_loop():
    while True:
        time.sleep(10)
        _refresh_excluded_pids()


def _pid_of_connection(src_addr: str, src_port: int) -> int | None:
    try:
        for conn in psutil.net_connections("tcp4"):
            if (conn.laddr
                    and conn.laddr.port == src_port
                    and conn.laddr.ip in (src_addr, "0.0.0.0", "::")
                    and conn.pid):
                return conn.pid
    except (psutil.AccessDenied, OSError):
        pass
    return None


def _proc_name(pid: int | None) -> str:
    if pid is None:
        return "?"
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "?"


# ---------------------------------------------------------------------------
# TCP relay
# ---------------------------------------------------------------------------

def _pipe(src: socket.socket, dst: socket.socket):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle_client(client_sock: socket.socket, client_addr: tuple[str, int]):
    with nat_lock:
        orig = nat_table.get(client_addr)
    if not orig:
        time.sleep(0.1)
        with nat_lock:
            orig = nat_table.get(client_addr)
    if not orig:
        log.warning("[relay] no NAT entry for %s:%d — dropping", *client_addr)
        client_sock.close()
        return

    target = f"{orig[0]}:{orig[1]}"
    proxy = None
    try:
        proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy.settimeout(10)
        proxy.connect((PROXY_HOST, PROXY_PORT))

        proxy.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())

        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = proxy.recv(4096)
            if not chunk:
                raise ConnectionError("proxy closed during CONNECT")
            resp += chunk

        if b"200" not in resp.split(b"\r\n")[0]:
            log.error("[relay] CONNECT %s FAILED: %s", target,
                      resp.split(b"\r\n")[0].decode(errors="replace"))
            return

        proxy.settimeout(None)
        log.info("[relay] tunnel OPEN: %s:%d -> %s", *client_addr, target)

        t1 = threading.Thread(target=_pipe, args=(client_sock, proxy), daemon=True)
        t2 = threading.Thread(target=_pipe, args=(proxy, client_sock), daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
        log.debug("[relay] tunnel CLOSED: %s:%d -> %s", *client_addr, target)

    except Exception as e:
        log.error("[relay] error for %s: %s", target, e)
    finally:
        for s in (client_sock, proxy):
            if s:
                try: s.close()
                except OSError: pass
        with nat_lock:
            nat_table.pop(client_addr, None)


def _relay_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", RELAY_PORT))
    srv.listen(256)
    log.info("[relay] listening on 0.0.0.0:%d", RELAY_PORT)
    while True:
        client, addr = srv.accept()
        threading.Thread(target=_handle_client, args=(client, addr), daemon=True).start()


# ---------------------------------------------------------------------------
# WinDivert NAT — clear GetLastError before every w.send() to prevent
# pydivert's raise_on_error from picking up stale error codes from psutil.
# ---------------------------------------------------------------------------

def _clear_error():
    ctypes.windll.kernel32.SetLastError(0)


def _windivert_nat():
    filt = (
        f"(outbound and tcp.DstPort == {TARGET_PORT} "
        f"and ip.DstAddr != 127.0.0.1 and ip.DstAddr != {LOCAL_IP}) or "
        f"(inbound and tcp.SrcPort == {RELAY_PORT})"
    )

    log.info("[nat] local IP: %s", LOCAL_IP)
    log.info("[nat] filter: %s", filt)

    with pydivert.WinDivert(filt) as w:
        for packet in w:
            # --- outbound: app -> real server  ->  rewrite to relay ---
            if packet.is_outbound and packet.dst_port == TARGET_PORT:
                key = (packet.src_addr, packet.src_port)

                if packet.tcp.syn and not packet.tcp.ack:
                    pid = _pid_of_connection(packet.src_addr, packet.src_port)
                    with _excluded_pids_lock:
                        is_excluded = pid is not None and pid in _excluded_pids

                    if is_excluded:
                        with passthrough_lock:
                            passthrough.add(key)
                        log.info("[nat] PASS %s:%d -> %s:%d (pid %s = mitmproxy)",
                                 key[0], key[1], packet.dst_addr, packet.dst_port, pid)
                        _clear_error(); w.send(packet)
                        continue

                    with nat_lock:
                        nat_table[key] = (packet.dst_addr, packet.dst_port)
                    log.info("[nat] NAT  %s:%d -> %s:%d (pid %s, %s)",
                             key[0], key[1], packet.dst_addr, packet.dst_port,
                             pid, _proc_name(pid))

                with passthrough_lock:
                    if key in passthrough:
                        if packet.tcp.fin or packet.tcp.rst:
                            passthrough.discard(key)
                        _clear_error(); w.send(packet)
                        continue

                packet.dst_addr = LOCAL_IP
                packet.dst_port = RELAY_PORT

            # --- inbound: relay -> app  ->  restore original src ---
            elif packet.is_inbound and packet.src_port == RELAY_PORT:
                key = (packet.dst_addr, packet.dst_port)
                with nat_lock:
                    orig = nat_table.get(key)
                if orig:
                    packet.src_addr = orig[0]
                    packet.src_port = orig[1]
                else:
                    _clear_error(); w.send(packet)
                    continue

            _clear_error(); w.send(packet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 50)
    log.info("  Proxifier  (WinDivert + TCP relay)")
    log.info("=" * 50)
    log.info("  local IP:  %s", LOCAL_IP)
    log.info("  proxy:     %s:%d", PROXY_HOST, PROXY_PORT)
    log.info("  intercept: *:%d", TARGET_PORT)
    log.info("  relay:     0.0.0.0:%d (redirect via %s)", RELAY_PORT, LOCAL_IP)
    log.info("  log file:  %s", LOG_FILE)
    log.info("")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((PROXY_HOST, PROXY_PORT))
        s.close()
        log.info("mitmproxy reachable on %s:%d", PROXY_HOST, PROXY_PORT)
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        log.error("mitmproxy NOT reachable on %s:%d — %s", PROXY_HOST, PROXY_PORT, e)
        sys.exit(1)

    _refresh_excluded_pids()

    threading.Thread(target=_relay_server, daemon=True).start()
    threading.Thread(target=_pid_refresh_loop, daemon=True).start()

    try:
        _windivert_nat()
    except KeyboardInterrupt:
        log.info("Stopped by user")
    except PermissionError:
        log.error("Must run as Administrator")
        sys.exit(1)
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

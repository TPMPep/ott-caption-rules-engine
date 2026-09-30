"""Fail-closed engine authentication and outbound URL policy (no dependencies)."""
import hmac
import http.client
import ipaddress
import os
import re
import socket
import ssl
import threading
import time
from collections import deque
from contextlib import contextmanager
from urllib.parse import urlsplit
from fastapi import HTTPException

APP_ID = "69cf27eee8db99cfbe0e6245"
CALLBACK_TARGETS = {
    ("transperfectmediacreator.base44.app", "/functions/ccEngineCallback"),
    ("app.base44.app", f"/api/apps/{APP_ID}/functions/ccEngineCallback"),
}
S3_HOST = re.compile(r"^[a-z0-9][a-z0-9.-]*\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com$")
_ADMISSIONS = deque()
_ADMISSION_LOCK = threading.Lock()


def require_engine_secret(supplied):
    expected = os.getenv("ENGINE_SHARED_SECRET", "").strip()
    if not expected or not isinstance(supplied, str) or not hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid X-Engine-Secret")


def validate_url(url, kind="media"):
    parsed = urlsplit(str(url))
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("Only credential-free HTTPS URLs on port 443 are permitted")
    host = parsed.hostname.lower()
    if kind == "callback":
        if (host, parsed.path) not in CALLBACK_TARGETS or parsed.query:
            raise ValueError("Callback must target this app's ccEngineCallback endpoint")
    else:
        # Signed AWS storage URLs are the normal intake. Custom storage hosts need
        # an explicit deployment allowlist; caller-supplied job env cannot alter it.
        extra_hosts = {h.strip().lower() for h in os.getenv("ENGINE_MEDIA_HOSTS", "").split(",") if h.strip()}
        if not S3_HOST.fullmatch(host) and host not in extra_hosts:
            raise ValueError("Media/baseline host is not an approved storage host")
    resolved = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    addresses = []
    for entry in resolved:
        address = entry[4][0]
        ip = ipaddress.ip_address(address)
        effective = getattr(ip, "ipv4_mapped", None) or ip
        if (not effective.is_global or effective.is_multicast or effective.is_reserved
                or getattr(ip, "sixtofour", None) or getattr(ip, "teredo", None)):
            raise ValueError("Non-public network destinations are forbidden")
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ValueError("Destination did not resolve to a public address")
    return parsed, addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, address, timeout):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._public_address = address

    def connect(self):
        # Connect to the address we validated, not a second DNS lookup. Keep the
        # original hostname for TLS certificate verification and SNI.
        raw = socket.create_connection((self._public_address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


@contextmanager
def safe_urlopen(request, timeout, kind="media"):
    parsed, addresses = validate_url(request.full_url, kind)
    connection = _PinnedHTTPSConnection(parsed.hostname, addresses[0], timeout)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        headers = {k: v for k, v in request.header_items() if k.lower() != "host"}
        connection.request(request.get_method(), path, body=request.data, headers=headers)
        response = connection.getresponse()
        # Never follow redirects: they can move a validated request (or callback
        # secret) to an internal network or attacker-controlled destination.
        if not 200 <= response.status < 300:
            response.close()
            raise ValueError(f"Approved destination returned HTTP {response.status}; redirects are forbidden")
        try:
            yield response
        finally:
            response.close()
    finally:
        connection.close()


def admit_job():
    # Lock makes admission atomic across FastAPI's threads. The single-process
    # engine bounds dispatch bursts without imposing a per-user bottleneck.
    limit = max(1, min(10000, int(os.getenv("ENGINE_JOB_RATE_LIMIT_PER_MINUTE", "300"))))
    now = time.monotonic()
    with _ADMISSION_LOCK:
        while _ADMISSIONS and _ADMISSIONS[0] <= now - 60:
            _ADMISSIONS.popleft()
        if len(_ADMISSIONS) >= limit:
            raise HTTPException(status_code=429, detail="Engine job admission limit reached", headers={"Retry-After": "60"})
        _ADMISSIONS.append(now)

"""Offline adversarial tests: no provider, storage, or callback requests."""
import importlib.util
import pathlib
import socket
import sys
import types
import unittest
from unittest.mock import patch, MagicMock
import urllib.request

# The sandbox may lack engine-only dependencies. The production Docker image
# installs FastAPI; its exception contract is all these pure boundary tests need.
try:
    from fastapi import HTTPException
except ModuleNotFoundError:
    class HTTPException(Exception):
        def __init__(self, status_code, detail, headers=None):
            super().__init__(detail)
            self.status_code, self.detail, self.headers = status_code, detail, headers
    sys.modules["fastapi"] = types.SimpleNamespace(HTTPException=HTTPException)

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("request_security", ROOT / "services/request_security.py")
security = importlib.util.module_from_spec(spec)
spec.loader.exec_module(security)
PUBLIC = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("54.231.1.1", 443))]
MEDIA = "https://bucket.s3.us-east-1.amazonaws.com/source.wav?X-Amz-Signature=example"
CALLBACK = "https://transperfectmediacreator.base44.app/functions/ccEngineCallback"


class RequestSecurityTests(unittest.TestCase):
    def test_secret_required_even_when_unconfigured(self):
        for env, supplied in [({}, None), ({"ENGINE_SHARED_SECRET": "known"}, None),
                              ({"ENGINE_SHARED_SECRET": "known"}, "wrong"),
                              ({"ENGINE_SHARED_SECRET": "known"}, "ü")]:
            with patch.dict(security.os.environ, env, clear=True):
                with self.assertRaises(HTTPException) as caught:
                    security.require_engine_secret(supplied)
                self.assertEqual(caught.exception.status_code, 401)
        with patch.dict(security.os.environ, {"ENGINE_SHARED_SECRET": "known"}, clear=True):
            security.require_engine_secret("known")

    @patch.object(security.socket, "getaddrinfo", return_value=PUBLIC)
    def test_https_and_storage_allowlist(self, resolve):
        security.validate_url(MEDIA)
        for url in ["http://bucket.s3.amazonaws.com/file", "https://attacker.example/file",
                    "https://bucket.s3.amazonaws.com.attacker.example/file",
                    "https://user:pass@bucket.s3.amazonaws.com/file",
                    "https://bucket.s3.amazonaws.com:8443/file", "https://127.0.0.1/file"]:
            with self.assertRaises(ValueError):
                security.validate_url(url)

    def test_dns_private_mixed_and_ipv6_addresses_rejected(self):
        for address in ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1", "::ffff:127.0.0.1"]:
            private = [(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]
            with patch.object(security.socket, "getaddrinfo", return_value=PUBLIC + private):
                with self.assertRaises(ValueError):
                    security.validate_url(MEDIA)

    @patch.object(security.socket, "getaddrinfo", return_value=PUBLIC)
    def test_callback_pinned_to_exact_app_and_function(self, resolve):
        security.validate_url(CALLBACK, "callback")
        security.validate_url(f"https://app.base44.app/api/apps/{security.APP_ID}/functions/ccEngineCallback", "callback")
        for url in ["https://attacker.example/", CALLBACK + "?redirect=https://attacker.example",
                    "https://transperfectmediacreator.base44.app/functions/anotherFunction",
                    "https://app.base44.app/api/apps/another-app/functions/ccEngineCallback"]:
            with self.assertRaises(ValueError):
                security.validate_url(url, "callback")

    @patch.object(security.socket, "getaddrinfo", return_value=PUBLIC)
    def test_redirect_never_followed_and_connection_closed(self, resolve):
        conn = MagicMock()
        conn.getresponse.return_value.status = 302
        with patch.object(security, "_PinnedHTTPSConnection", return_value=conn) as constructor:
            with self.assertRaises(ValueError):
                with security.safe_urlopen(urllib.request.Request(CALLBACK, data=b"{}"), 10, "callback"):
                    self.fail("redirect must not be delivered")
            constructor.assert_called_once_with("transperfectmediacreator.base44.app", "54.231.1.1", 10)
            self.assertEqual(conn.request.call_count, 1)
            conn.close.assert_called_once()

    def test_dns_is_not_resolved_again_at_connect(self):
        raw, wrapped = MagicMock(), MagicMock()
        with patch.object(security.ssl, "create_default_context") as context:
            context.return_value.wrap_socket.return_value = wrapped
            conn = security._PinnedHTTPSConnection("bucket.s3.amazonaws.com", "54.231.1.1", 20)
            with patch.object(security.socket, "create_connection", return_value=raw) as connect:
                conn.connect()
                connect.assert_called_once_with(("54.231.1.1", 443), 20)
                context.return_value.wrap_socket.assert_called_once_with(raw, server_hostname="bucket.s3.amazonaws.com")

    def test_rate_limiter_returns_429_then_recovers(self):
        security._ADMISSIONS.clear()
        with patch.dict(security.os.environ, {"ENGINE_JOB_RATE_LIMIT_PER_MINUTE": "2"}), patch.object(security.time, "monotonic", return_value=100):
            security.admit_job()
            security.admit_job()
            with self.assertRaises(HTTPException) as caught:
                security.admit_job()
            self.assertEqual(caught.exception.status_code, 429)
        with patch.object(security.time, "monotonic", return_value=161):
            security.admit_job()
        security._ADMISSIONS.clear()

    def test_actual_sinks_and_admission_use_shared_boundary(self):
        source = (ROOT / "main.py").read_text()
        self.assertNotIn("urllib.request.urlopen(", source)
        self.assertIn("with safe_urlopen(req, timeout=BASELINE_FETCH_TIMEOUT_SECONDS)", source)
        self.assertIn('with safe_urlopen(req, timeout=CALLBACK_TIMEOUT_SECONDS, kind="callback")', source)
        admission = source[source.index("def create_job("):source.index('@app.get("/v1/jobs/{job_id}")')]
        self.assertLess(admission.index("_check_secret("), admission.index("validate_url("))
        self.assertLess(admission.index("validate_url("), admission.index("JOBS[job_id] ="))
        self.assertLess(admission.index("admit_job()"), admission.index("submit_transcription_job("))


if __name__ == "__main__":
    unittest.main()

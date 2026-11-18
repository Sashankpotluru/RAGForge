"""Opt-in HTTPS loader with DNS pinning, TLS hostname validation and bounded bodies."""

import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urljoin, urlsplit

from ragforge.config import Settings
from ragforge.ingestion.loaders import LoadedDocument, load_bytes

MAX_BYTES = 20 * 1024 * 1024


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, ip: str) -> None:
        self.tls_context = ssl.create_default_context()
        super().__init__(host, timeout=5, context=self.tls_context)
        self.ip = ip

    def connect(self) -> None:
        # Numeric-address socket bypasses a second hostname DNS lookup entirely.
        address = ipaddress.ip_address(self.ip)
        raw = socket.socket(
            socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_STREAM
        )
        raw.settimeout(5)
        try:
            raw.connect((self.ip, 443))
            self.sock = self.tls_context.wrap_socket(raw, server_hostname=self.host)
            self.sock.settimeout(20)
        except BaseException:
            raw.close()
            raise


class SafeUrlLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _resolve(self, host: str) -> list[str]:
        return list(
            {str(result[4][0]) for result in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
        )

    def _fetch(self, host: str, ip: str, path: str) -> tuple[int, dict[str, str], bytes]:
        connection = PinnedHTTPSConnection(host, ip)
        try:
            connection.request(
                "GET",
                path,
                headers={"Accept-Encoding": "identity", "User-Agent": "RAGForge/0.1", "Host": host},
            )
            response = connection.getresponse()
            headers = {k.lower(): v for k, v in response.getheaders()}
            if int(headers.get("content-length", "0")) > MAX_BYTES:
                raise ValueError("URL body exceeds 20 MiB")
            body = response.read(MAX_BYTES + 1)
            return response.status, headers, body
        finally:
            connection.close()

    def load(self, url: str) -> LoadedDocument:
        if not self.settings.url_ingestion_enabled:
            raise ValueError("URL ingestion is disabled")
        for hop in range(4):
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            if (
                parsed.scheme != "https"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
                or host not in self.settings.url_allowed_hosts
                or any(ord(c) < 32 for c in url)
            ):
                raise ValueError("URL must be allowlisted HTTPS without credentials on port 443")
            try:
                ips = self._resolve(host)
            except OSError as error:
                raise ValueError("URL hostname resolution failed") from error
            addresses = [ipaddress.ip_address(ip) for ip in ips]
            if not addresses or any(
                not ip.is_global
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
                or ip.is_loopback
                or ip.is_link_local
                for ip in addresses
            ):
                raise ValueError("URL must resolve exclusively to public addresses")
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            try:
                status, headers, body = self._fetch(host, sorted(ips)[0], path)
            except (OSError, http.client.HTTPException) as error:
                raise ValueError("URL fetch failed or timed out") from error
            if len(body) > MAX_BYTES:
                raise ValueError("URL body exceeds 20 MiB")
            if status in (301, 302, 303, 307, 308):
                if hop == 3 or not headers.get("location"):
                    raise ValueError("URL redirect limit exceeded or missing location")
                url = urljoin(url, headers["location"])
                continue
            if status != 200 or headers.get("content-encoding", "identity") != "identity":
                raise ValueError("unsupported URL response")
            mime = headers.get("content-type", "").split(";")[0].strip().lower()
            if mime not in {"text/plain", "text/html", "application/pdf"}:
                raise ValueError("unsupported URL content type")
            return load_bytes(body, mime, url)
        raise ValueError("redirect limit exceeded")

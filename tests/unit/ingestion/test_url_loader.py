import pytest

from ragforge.config import Settings


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.example/a",
        "https://other.example/",
        "https://user:pass@docs.example/",
        "https://docs.example:8443/",
    ],
)
def test_url_policy_rejects_unsafe_targets(url) -> None:
    from ragforge.ingestion.url_loader import SafeUrlLoader

    loader = SafeUrlLoader(
        Settings(url_ingestion_enabled=True, url_allowed_hosts=("docs.example",))
    )
    with pytest.raises(ValueError):
        loader.load(url)


def test_disabled_private_dns_redirect_and_pinning(monkeypatch) -> None:
    from ragforge.ingestion.url_loader import SafeUrlLoader

    with pytest.raises(ValueError, match="disabled"):
        SafeUrlLoader(Settings()).load("https://docs.example/a")
    loader = SafeUrlLoader(
        Settings(url_ingestion_enabled=True, url_allowed_hosts=("docs.example",))
    )
    monkeypatch.setattr(loader, "_resolve", lambda host: ["127.0.0.1"])
    with pytest.raises(ValueError, match="public"):
        loader.load("https://docs.example/a")
    monkeypatch.setattr(loader, "_resolve", lambda host: ["93.184.216.34"])

    def fetch(host, ip, path):
        assert host == "docs.example" and ip == "93.184.216.34"
        return 200, {"content-type": "text/plain"}, b"safe text"

    monkeypatch.setattr(loader, "_fetch", fetch)
    assert loader.load("https://docs.example/a").text == "safe text"
    monkeypatch.setattr(loader, "_fetch", lambda *a: (302, {"location": "https://127.0.0.1/"}, b""))
    with pytest.raises(ValueError):
        loader.load("https://docs.example/a")


def test_url_oversize_type_and_redirect_limits(monkeypatch) -> None:
    from ragforge.ingestion.url_loader import SafeUrlLoader

    loader = SafeUrlLoader(
        Settings(url_ingestion_enabled=True, url_allowed_hosts=("docs.example",))
    )
    monkeypatch.setattr(loader, "_resolve", lambda host: ["93.184.216.34"])
    for response in [
        (200, {"content-type": "application/zip"}, b"zip"),
        (200, {"content-type": "text/plain"}, b"x" * (20 * 1024 * 1024 + 1)),
        (302, {"location": "/again"}, b""),
    ]:
        monkeypatch.setattr(loader, "_fetch", lambda *a, response=response: response)
        with pytest.raises(ValueError):
            loader.load("https://docs.example/a")

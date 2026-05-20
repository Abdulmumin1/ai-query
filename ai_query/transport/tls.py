"""TLS verification helpers for HTTP transports."""

from __future__ import annotations

import ssl


def certifi_ca_bundle() -> str | bool:
    """Return certifi's CA bundle path for httpx, or True for system defaults."""
    try:
        import certifi
    except ImportError:
        return True
    return certifi.where()


def certifi_ssl_context() -> ssl.SSLContext:
    """Return an SSL context using certifi when available."""
    ca_bundle = certifi_ca_bundle()
    if isinstance(ca_bundle, str):
        return ssl.create_default_context(cafile=ca_bundle)
    return ssl.create_default_context()

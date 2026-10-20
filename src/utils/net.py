"""Network helpers."""

from __future__ import annotations

import socket
from typing import Any

import urllib3.util.connection as urllib3_cn

_orig_getaddrinfo = socket.getaddrinfo
_patched = False


def force_ipv4() -> None:
    """Make every in-process client (urllib3, httplib2, http.client) skip IPv6.

    The dev machine has a black-holed IPv6 route: each new connection stalls for the full
    connect timeout before falling back to IPv4, which made Earth Engine calls crawl.
    urllib3 is switched off for IPv6 and ``socket.getaddrinfo`` is restricted to AF_INET,
    which also covers httplib2 (used by earthengine-api). See ADR-023. Idempotent.
    """
    global _patched
    urllib3_cn.HAS_IPV6 = False
    if _patched:
        return

    def _ipv4_only(
        host: Any, port: Any, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
    ) -> Any:
        return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only
    _patched = True

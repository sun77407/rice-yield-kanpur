"""Network helpers."""

from __future__ import annotations

import urllib3.util.connection as urllib3_cn


def force_ipv4() -> None:
    """Make urllib3 (hence requests, google-auth, google-cloud-storage) skip IPv6.

    The dev machine has a black-holed IPv6 route: each new connection stalls for the full
    connect timeout before falling back to IPv4, which made Earth Engine init hang.
    """
    urllib3_cn.HAS_IPV6 = False

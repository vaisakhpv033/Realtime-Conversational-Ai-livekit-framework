"""SSRF security controls and pre-dispatch DNS validation.

Provides robust validation of webhook targets to block:
- Non-HTTP(S) schemes (file://, gopher://, ftp://, etc.)
- Cloud metadata service endpoints (169.254.169.254, fd00:ec2::254, 100.100.100.200, metadata.google.internal, etc.)
- Loopback (127.0.0.0/8, ::1) and local hostnames
- RFC 1918 private subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Link-local (169.254.0.0/16, fe80::/10), reserved, multicast, and carrier-grade NAT (100.64.0.0/10) ranges
- Alternative IP representations (decimal, hex, octal notations)
- Pre-dispatch DNS rebinding attacks via socket.getaddrinfo inspection
- Optional allowed domain whitelisting
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("agent_framework.reporting.security")

# Known cloud provider metadata hostnames
CLOUD_METADATA_HOSTNAMES = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.internal",
    "instance-data",
    "100.100.100.200",  # Alibaba Cloud metadata IP
    "fd00:ec2::254",    # AWS IPv6 IMDS
})

# Known cloud provider metadata IP literals
CLOUD_METADATA_IPS = frozenset({
    "169.254.169.254",
    "100.100.100.200",
    "fd00:ec2::254",    # AWS IPv6 IMDS
})

# Shared / carrier-grade NAT (RFC 6598) and benchmark network (RFC 2544)
CARRIER_GRADE_NAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
BENCHMARK_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def validate_ip_address(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    allow_local: bool = False,
) -> tuple[bool, str | None]:
    """Evaluates an IP address against blocked private, loopback, and metadata ranges.

    Args:
        ip: IPv4 or IPv6 address object.
        allow_local: If True, permits localhost/private RFC 1918 IPs for local testing.
                     Cloud metadata IPs, link-local, multicast, and unspecified IPs remain
                     strictly blocked regardless of allow_local.

    Returns:
        tuple[bool, str | None]: (True, None) if allowed; (False, reason) if blocked.
    """
    # Normalize IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1 -> 127.0.0.1)
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    ip_str = str(ip).lower()

    # Cloud metadata IPs are strictly forbidden under all circumstances
    if ip_str in CLOUD_METADATA_IPS:
        return False, f"Cloud metadata IP address '{ip_str}' is forbidden"

    # Multicast, unspecified, and reserved IPs are never valid webhook targets
    if ip.is_multicast:
        return False, f"Multicast IP address '{ip_str}' is blocked"

    if ip.is_unspecified:
        return False, f"Unspecified IP address '{ip_str}' is blocked"

    if ip.is_reserved:
        return False, f"Reserved IP address '{ip_str}' is blocked"

    # Link-local addresses (169.254.0.0/16, fe80::/10) are used for auto-config and cloud metadata,
    # and must never be permitted even under allow_local=True
    if ip.is_link_local:
        return False, f"Link-local IP address '{ip_str}' is blocked"

    # Local testing permits loopback and private RFC 1918 ranges
    if allow_local:
        return True, None

    if ip.is_loopback:
        return False, f"Loopback IP address '{ip_str}' is blocked"

    if ip.is_private:
        return False, f"Private network IP address '{ip_str}' is blocked"

    if isinstance(ip, ipaddress.IPv4Address):
        if ip in CARRIER_GRADE_NAT_NETWORK:
            return False, f"Carrier-grade NAT IP address '{ip_str}' is blocked"
        if ip in BENCHMARK_NETWORK:
            return False, f"Benchmark test IP address '{ip_str}' is blocked"

    return True, None


def is_safe_webhook_url(
    url: str,
    allow_local: bool = False,
    allowed_domains: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Validates that a webhook URL is safe against SSRF and DNS rebinding attacks.

    Rejection criteria:
    - Non-string or empty URL
    - Non-HTTP/HTTPS schemes (e.g. file://, gopher://, ftp://)
    - Invalid or out-of-range port numbers (< 1 or > 65535)
    - Cloud metadata IPs and endpoints (169.254.169.254, fd00:ec2::254, metadata.google.internal, etc.)
    - Localhost and private RFC 1918 subnets (unless allow_local=True)
    - Alternative IP encodings (decimal, octal, hex)
    - Hostnames failing the optional allowed_domains whitelist
    - Domains whose DNS pre-resolution points to private, loopback, or metadata IPs (DNS rebinding)

    Args:
        url: Webhook target URL string.
        allow_local: Whether local/private addresses are allowed (testing only).
        allowed_domains: Optional list of whitelisted domain names / suffixes.

    Returns:
        tuple[bool, str | None]: (True, None) if safe; (False, reason) if unsafe.
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    url_clean = url.strip()
    if not url_clean:
        return False, "URL cannot be empty or whitespace"

    try:
        parsed = urlparse(url_clean)
        scheme = (parsed.scheme or "").lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as e:
        return False, f"Invalid URL format: {e}"
    except Exception as e:
        return False, f"Invalid URL format: {e}"

    if scheme not in ("http", "https"):
        return False, f"Invalid URL scheme '{parsed.scheme}': only http and https are allowed"

    if not hostname:
        return False, "URL is missing a valid hostname"

    if port is not None and not (1 <= port <= 65535):
        return False, f"Port '{port}' is out of valid range (1-65535)"

    hostname_lower = hostname.lower().strip().rstrip(".")

    # Cloud metadata service endpoints
    if (
        hostname_lower in CLOUD_METADATA_HOSTNAMES
        or hostname_lower.endswith(".metadata.google.internal")
        or hostname_lower.endswith(".metadata.azure.internal")
        or hostname_lower.endswith(".instance-data")
    ):
        return False, f"Cloud metadata host '{hostname}' is forbidden"

    # Allowed domains whitelist check
    if allowed_domains:
        normalized_allowed = [
            d.strip().lower().strip(".")
            for d in allowed_domains
            if d and isinstance(d, str) and d.strip()
        ]
        if normalized_allowed:
            matched = False
            for allowed in normalized_allowed:
                if hostname_lower == allowed or hostname_lower.endswith(f".{allowed}"):
                    matched = True
                    break
            if not matched:
                return False, f"Domain '{hostname}' is not in allowed_domains whitelist: {allowed_domains}"

    # Literal IP address check (standard dot/colon notation or alternative integer/hex/octal encodings)
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip = ipaddress.ip_address(hostname_lower)
    except ValueError:
        # Check alternative IPv4 representations (e.g. integer 2130706433, hex 0x7f000001)
        if all(c in "0123456789.xXabcdefABCDEF" for c in hostname_lower) and not hostname_lower.endswith(".com"):
            try:
                packed = socket.inet_aton(hostname_lower)
                ip = ipaddress.ip_address(socket.inet_ntoa(packed))
            except (OSError, ValueError):
                ip = None

    if ip is not None:
        return validate_ip_address(ip, allow_local=allow_local)

    # Localhost domains blocked when allow_local is False
    if not allow_local:
        if (
            hostname_lower == "localhost"
            or hostname_lower.endswith(".localhost")
            or hostname_lower.endswith(".local")
        ):
            return False, f"Localhost domain '{hostname}' is blocked when allow_local=False"

    # Pre-dispatch DNS resolution using socket.getaddrinfo to prevent DNS rebinding
    resolved_port = port or (443 if scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname_lower, resolved_port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, socket.herror, OSError) as e:
        return False, f"DNS resolution failed for hostname '{hostname}': {e}"
    except Exception as e:
        return False, f"DNS resolution error for hostname '{hostname}': {e}"

    if not addr_info:
        return False, f"No DNS records found for hostname '{hostname}'"

    for entry in addr_info:
        sockaddr = entry[4]
        resolved_ip_str = sockaddr[0]
        try:
            resolved_ip = ipaddress.ip_address(resolved_ip_str)
        except ValueError:
            return False, f"Invalid IP returned by DNS for '{hostname}': {resolved_ip_str}"

        is_safe, reason = validate_ip_address(resolved_ip, allow_local=allow_local)
        if not is_safe:
            return (
                False,
                f"Domain '{hostname}' resolved to blocked IP '{resolved_ip_str}': {reason} (DNS rebinding risk)",
            )

    return True, None


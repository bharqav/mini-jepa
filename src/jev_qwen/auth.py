"""API key registry and client-address resolution.

The service accepts multiple bearer tokens so keys can be rotated without
downtime and scoped to individual endpoints. Tokens are 256-bit random values,
so they are stored as keyed HMAC-SHA-256 digests and compared with
:func:`hmac.compare_digest`; the plaintext token is never kept in the registry
after startup.

Configuration (environment):

``JEV_API_KEYS``
    JSON object mapping key id to a token string or an object::

        {"ci": {"token": "...", "scopes": ["decide"], "rps": 5, "burst": 10}}

``JEV_API_KEYS_FILE``
    Path to a JSON file with the same shape. Mutually exclusive with
    ``JEV_API_KEYS``.

``JEV_API_TOKEN``
    Legacy single-key configuration. Equivalent to one key with id
    ``default`` and both scopes.

``JEV_TRUSTED_PROXIES``
    Comma-separated CIDRs of reverse proxies whose ``X-Forwarded-For`` header
    may be trusted. Empty (the default) means the header is ignored.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

SCOPES = frozenset({"decide", "answer"})
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_TOKEN_HASH_KEY = b"jev-qwen/api-key-digest/v1"


def _token_digest(token: str) -> bytes:
    return hmac.new(_TOKEN_HASH_KEY, token.encode("utf-8"), hashlib.sha256).digest()


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    token_digest: bytes
    scopes: frozenset[str]
    rps: float | None = None
    burst: float | None = None


@dataclass(frozen=True)
class AuthContext:
    key_id: str
    scopes: frozenset[str]
    rps: float | None = None
    burst: float | None = None
    dev: bool = False


def _parse_key(key_id: str, value: object, source: str) -> ApiKey:
    if not _KEY_ID_RE.match(key_id):
        raise ValueError(f"{source}: invalid key id {key_id!r} (use [A-Za-z0-9_.:-], 1-64 chars)")
    if isinstance(value, str):
        token = value.strip()
        scopes = SCOPES
        rps = burst = None
    elif isinstance(value, Mapping):
        raw_token = value.get("token")
        if not isinstance(raw_token, str) or not raw_token.strip():
            raise ValueError(f"{source}: key {key_id!r} needs a non-empty 'token' string")
        token = raw_token.strip()
        raw_scopes = value.get("scopes", sorted(SCOPES))
        if not isinstance(raw_scopes, list) or not all(isinstance(s, str) for s in raw_scopes):
            raise ValueError(f"{source}: key {key_id!r} 'scopes' must be a list of strings")
        unknown = set(raw_scopes) - SCOPES
        if unknown:
            raise ValueError(
                f"{source}: key {key_id!r} has unknown scopes {sorted(unknown)}; "
                f"valid scopes are {sorted(SCOPES)}"
            )
        if not raw_scopes:
            raise ValueError(f"{source}: key {key_id!r} has no scopes")
        scopes = frozenset(raw_scopes)
        rps = value.get("rps")
        burst = value.get("burst")
        if rps is not None:
            if not isinstance(rps, (int, float)) or isinstance(rps, bool) or rps < 0:
                raise ValueError(f"{source}: key {key_id!r} 'rps' must be a number >= 0")
            rps = float(rps)
        if burst is not None:
            if not isinstance(burst, (int, float)) or isinstance(burst, bool) or burst < 1:
                raise ValueError(f"{source}: key {key_id!r} 'burst' must be a number >= 1")
            burst = float(burst)
        if rps is not None and burst is None:
            burst = max(1.0, rps)
    else:
        raise ValueError(f"{source}: key {key_id!r} must be a token string or an object")
    if not token:
        raise ValueError(f"{source}: key {key_id!r} has an empty token")
    return ApiKey(
        key_id=key_id,
        token_digest=_token_digest(token),
        scopes=scopes,
        rps=rps,
        burst=burst,
    )


def parse_api_keys(raw: str, source: str = "JEV_API_KEYS") -> list[ApiKey]:
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source}: invalid JSON: {e}") from e
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{source}: expected a non-empty JSON object")
    keys = [_parse_key(str(key_id), value, source) for key_id, value in config.items()]
    _reject_duplicates(keys, source)
    return keys


def _reject_duplicates(keys: list[ApiKey], source: str) -> None:
    seen_ids: set[str] = set()
    seen_tokens: set[bytes] = set()
    for key in keys:
        if key.key_id in seen_ids:
            raise ValueError(f"{source}: duplicate key id {key.key_id!r}")
        if key.token_digest in seen_tokens:
            raise ValueError(f"{source}: duplicate token for key {key.key_id!r}")
        seen_ids.add(key.key_id)
        seen_tokens.add(key.token_digest)


def single_key(token: str, key_id: str = "default") -> ApiKey:
    token = token.strip()
    if not token:
        raise ValueError("API token must not be empty")
    return ApiKey(key_id=key_id, token_digest=_token_digest(token), scopes=SCOPES)


def load_keys_from_env(env: Mapping[str, str] | None = None) -> list[ApiKey]:
    """Load keys from the environment, preserving the legacy single-token path."""
    env = os.environ if env is None else env
    raw_keys = env.get("JEV_API_KEYS", "").strip()
    keys_file = env.get("JEV_API_KEYS_FILE", "").strip()
    legacy = env.get("JEV_API_TOKEN", "").strip()
    if raw_keys and keys_file:
        raise ValueError("set only one of JEV_API_KEYS and JEV_API_KEYS_FILE")
    if (raw_keys or keys_file) and legacy:
        raise ValueError(
            "set JEV_API_TOKEN only for single-key deployments; "
            "use JEV_API_KEYS/JEV_API_KEYS_FILE for multiple keys"
        )
    if keys_file:
        try:
            raw_keys = Path(keys_file).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"cannot read JEV_API_KEYS_FILE {keys_file!r}: {e}") from e
        return parse_api_keys(raw_keys, source="JEV_API_KEYS_FILE")
    if raw_keys:
        return parse_api_keys(raw_keys)
    if legacy:
        return [single_key(legacy)]
    return []


class ApiKeyRegistry:
    """Immutable set of accepted keys with constant-time token comparison."""

    def __init__(self, keys: Iterable[ApiKey]):
        self._keys = tuple(keys)
        _reject_duplicates(list(self._keys), "api keys")

    def __len__(self) -> int:
        return len(self._keys)

    def authenticate(self, token: str) -> ApiKey | None:
        digest = _token_digest(token)
        found: ApiKey | None = None
        for key in self._keys:
            if hmac.compare_digest(digest, key.token_digest):
                found = key
        return found

    def by_id(self, key_id: str) -> ApiKey | None:
        for key in self._keys:
            if key.key_id == key_id:
                return key
        return None


def extract_bearer(header: str | None) -> str | None:
    scheme, _, supplied = (header or "").partition(" ")
    supplied = supplied.strip()
    if scheme.lower() != "bearer" or not supplied:
        return None
    return supplied


class TrustedProxies:
    """Resolve the client address without blindly trusting X-Forwarded-For."""

    def __init__(self, networks: Iterable[str] = ()):
        parsed: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for raw in networks:
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed.append(ipaddress.ip_network(raw, strict=False))
            except ValueError as e:
                raise ValueError(f"invalid trusted proxy CIDR {raw!r}: {e}") from e
        self.networks = tuple(parsed)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TrustedProxies:
        env = os.environ if env is None else env
        return cls(env.get("JEV_TRUSTED_PROXIES", "").split(","))

    def contains(self, address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(ip in network for network in self.networks)

    def client_ip(self, peer: str | None, forwarded_for: str | None) -> str:
        if not peer:
            return "unknown"
        if not self.networks or not forwarded_for or not self.contains(peer):
            return peer
        candidates: list[str] = []
        for raw in forwarded_for.split(","):
            candidate = raw.strip()
            if not candidate:
                continue
            try:
                candidates.append(str(ipaddress.ip_address(candidate)))
            except ValueError:
                continue
        for candidate in reversed(candidates):
            if not self.contains(candidate):
                return candidate
        return candidates[0] if candidates else peer

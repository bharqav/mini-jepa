"""API key registry, scopes, proxy resolution and environment loading."""

import json

import pytest

from jev_qwen.auth import (
    ApiKeyRegistry,
    TrustedProxies,
    extract_bearer,
    load_keys_from_env,
    parse_api_keys,
    single_key,
)


def test_parse_api_keys_accepts_shorthand_and_objects():
    keys = parse_api_keys(
        json.dumps(
            {
                "ci": "token-ci",
                "ops": {"token": "token-ops", "scopes": ["decide"], "rps": 2, "burst": 4},
            }
        )
    )
    assert {k.key_id for k in keys} == {"ci", "ops"}
    ops = next(k for k in keys if k.key_id == "ops")
    assert ops.scopes == frozenset({"decide"})
    assert (ops.rps, ops.burst) == (2.0, 4.0)


@pytest.mark.parametrize(
    "raw,why",
    [
        ("not json", "invalid json"),
        ("[]", "not an object"),
        ("{}", "empty"),
        (json.dumps({"k": {"token": ""}}), "empty token"),
        (json.dumps({"k": {"token": "t", "scopes": ["admin"]}}), "unknown scope"),
        (json.dumps({"k": {"token": "t", "scopes": []}}), "no scopes"),
        (json.dumps({"k": {"token": "t", "rps": -1}}), "negative rps"),
        (json.dumps({"k": {"token": "t", "burst": 0}}), "zero burst"),
        (json.dumps({"bad key": "t"}), "bad key id"),
        (
            json.dumps({"a": {"token": "same"}, "b": {"token": "same"}}),
            "duplicate token",
        ),
    ],
)
def test_parse_api_keys_rejects_invalid_config(raw, why):
    with pytest.raises(ValueError):
        parse_api_keys(raw)


def test_registry_authenticates_and_rejects_unknown_tokens():
    reg = ApiKeyRegistry([single_key("one", "a"), single_key("two", "b")])
    assert reg.authenticate("one").key_id == "a"
    assert reg.authenticate("two").key_id == "b"
    assert reg.authenticate("three") is None
    assert reg.authenticate("") is None
    assert len(reg) == 2


def test_extract_bearer_is_case_insensitive_and_strict():
    assert extract_bearer("Bearer abc") == "abc"
    assert extract_bearer("bearer abc") == "abc"
    assert extract_bearer("Basic abc") is None
    assert extract_bearer("Bearer") is None
    assert extract_bearer(None) is None


def test_load_keys_prefers_environment_shapes(monkeypatch, tmp_path):
    monkeypatch.delenv("JEV_API_TOKEN", raising=False)
    monkeypatch.delenv("JEV_API_KEYS_FILE", raising=False)
    monkeypatch.setenv("JEV_API_KEYS", json.dumps({"ci": "token"}))
    keys = load_keys_from_env()
    assert [k.key_id for k in keys] == ["ci"]

    path = tmp_path / "keys.json"
    path.write_text(json.dumps({"file": "token-file"}), encoding="utf-8")
    monkeypatch.delenv("JEV_API_KEYS")
    monkeypatch.setenv("JEV_API_KEYS_FILE", str(path))
    assert load_keys_from_env()[0].key_id == "file"

    monkeypatch.setenv("JEV_API_TOKEN", "legacy")
    with pytest.raises(ValueError, match="single-key"):
        load_keys_from_env()

    monkeypatch.delenv("JEV_API_KEYS_FILE")
    assert load_keys_from_env()[0].key_id == "default"


def test_trusted_proxies_only_use_forwarded_for_from_a_trusted_peer():
    proxies = TrustedProxies(["10.0.0.0/8"])
    assert proxies.client_ip("198.51.100.9", "203.0.113.5") == "198.51.100.9"
    assert proxies.client_ip("10.1.2.3", "203.0.113.5") == "203.0.113.5"
    assert proxies.client_ip("10.1.2.3", "192.0.2.7, 203.0.113.5, 10.9.9.9") == "203.0.113.5"
    assert proxies.client_ip("10.1.2.3", "10.9.9.9, 10.8.8.8") == "10.9.9.9"
    assert proxies.client_ip("10.1.2.3", "garbage") == "10.1.2.3"
    assert TrustedProxies().client_ip("10.1.2.3", "203.0.113.5") == "10.1.2.3"
    with pytest.raises(ValueError, match="invalid trusted proxy"):
        TrustedProxies(["not-a-cidr"])

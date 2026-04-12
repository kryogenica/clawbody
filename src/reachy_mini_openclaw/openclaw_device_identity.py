"""OpenClaw WebSocket device identity (Ed25519 + v3 signed payload).

Matches OpenClaw's ``buildDeviceAuthPayloadV3`` / ``signDevicePayload`` in
``src/gateway/device-auth.ts`` and ``src/infra/device-identity.ts``. Current
gateways require this for ``operator.write`` and ``chat.send`` over LAN.

See: https://docs.openclaw.ai/gateway/protocol (Device identity + pairing).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _ascii_lower_metadata(value: str | None) -> str:
    """Lowercase ASCII A–Z only; same intent as OpenClaw ``normalizeDeviceMetadataForAuth``."""
    if not value or not isinstance(value, str):
        return ""
    trimmed = value.strip()
    if not trimmed:
        return ""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in trimmed)


def build_device_auth_payload_v3(
    *,
    device_id: str,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    signed_at_ms: int,
    token: str,
    nonce: str,
    platform: str | None,
    device_family: str | None,
) -> str:
    scopes_csv = ",".join(scopes)
    tok = token or ""
    return "|".join(
        [
            "v3",
            device_id,
            client_id,
            client_mode,
            role,
            scopes_csv,
            str(signed_at_ms),
            tok,
            nonce,
            _ascii_lower_metadata(platform),
            _ascii_lower_metadata(device_family),
        ]
    )


def device_id_from_private_key(priv: ed25519.Ed25519PrivateKey) -> str:
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def public_key_raw_b64url(priv: ed25519.Ed25519PrivateKey) -> str:
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _b64url(raw)


def sign_device_payload(priv: ed25519.Ed25519PrivateKey, payload: str) -> str:
    sig = priv.sign(payload.encode("utf-8"))
    return _b64url(sig)


def default_identity_path() -> Path:
    override = os.environ.get("OPENCLAW_DEVICE_IDENTITY_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "clawbody" / "openclaw_device.json"


def load_or_create_identity(path: Path) -> tuple[ed25519.Ed25519PrivateKey, str]:
    """Load or create a stable Ed25519 identity; returns (private_key, device_id_hex)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
            if raw.get("version") == 1 and raw.get("privateKeyPem"):
                key = serialization.load_pem_private_key(
                    raw["privateKeyPem"].encode(),
                    password=None,
                )
                if not isinstance(key, ed25519.Ed25519PrivateKey):
                    raise TypeError("stored key is not Ed25519")
                derived_id = device_id_from_private_key(key)
                if raw.get("deviceId") != derived_id:
                    raw["deviceId"] = derived_id
                    path.write_text(json.dumps(raw, indent=2) + "\n")
                    try:
                        path.chmod(0o600)
                    except OSError:
                        pass
                return key, derived_id
        except Exception as e:
            logger.warning("Could not load OpenClaw device identity from %s: %s", path, e)

    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    did = device_id_from_private_key(priv)
    store: dict[str, Any] = {
        "version": 1,
        "deviceId": did,
        "publicKeyPem": pub_pem,
        "privateKeyPem": priv_pem,
        "createdAtMs": int(time.time() * 1000),
    }
    path.write_text(json.dumps(store, indent=2) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    logger.info(
        "Created new OpenClaw device identity at %s (deviceId=%s). "
        "If the gateway rejects pairing, approve this device in OpenClaw.",
        path,
        did[:16] + "…",
    )
    return priv, did


def load_stored_device_token(path: Path) -> str | None:
    """Return ``lastDeviceToken`` written after a successful connect, if any."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        tok = data.get("lastDeviceToken")
        if isinstance(tok, str) and tok.strip():
            return tok.strip()
    except Exception:
        pass
    return None


def maybe_store_device_token(path: Path, device_token: str | None) -> None:
    """Persist gateway-issued device token for debugging / future auth combinations."""
    if not device_token or not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            return
        data["lastDeviceToken"] = device_token
        path.write_text(json.dumps(data, indent=2) + "\n")
    except Exception as e:
        logger.debug("Could not store device token: %s", e)


def device_disabled() -> bool:
    v = os.environ.get("OPENCLAW_DISABLE_DEVICE_IDENTITY", "").strip().lower()
    return v in ("1", "true", "yes", "on")

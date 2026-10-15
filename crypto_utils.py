"""
crypto_utils.py
─────────────────────────────────────────────────────────────────────────────
Cryptographic core for the CEPL AI Provenance & Audit Trail system.

Responsibilities
────────────────
• RSA-2048 key-pair generation / loading (private_key.pem, public_key.pem)
• Chunked SHA-256 hashing of arbitrarily-large model weight files
• Canonical JSON hashing of metadata / configuration dictionaries
• RSA-PSS signing and verification of combined provenance hashes
"""

from __future__ import annotations

import hashlib
import json
import os
import base64
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

# ─── Key file paths ──────────────────────────────────────────────────────────
PRIVATE_KEY_PATH = Path("private_key.pem")
PUBLIC_KEY_PATH  = Path("public_key.pem")


# ─── Key Management ──────────────────────────────────────────────────────────

def generate_rsa_key_pair() -> tuple[str, str]:
    """Generate a new RSA-2048 key pair and persist them to disk.

    Returns
    -------
    (private_key_pem, public_key_pem) as PEM-encoded strings.
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    PRIVATE_KEY_PATH.write_bytes(private_pem)
    PUBLIC_KEY_PATH.write_bytes(public_pem)

    return private_pem.decode(), public_pem.decode()


def load_or_create_keys() -> tuple[str, str]:
    """Return (private_pem, public_pem), generating the pair if absent."""
    if PRIVATE_KEY_PATH.exists() and PUBLIC_KEY_PATH.exists():
        return (
            PRIVATE_KEY_PATH.read_text(),
            PUBLIC_KEY_PATH.read_text(),
        )
    return generate_rsa_key_pair()


def get_public_key_pem() -> str:
    """Return the public key PEM, generating keys if necessary."""
    _, public_pem = load_or_create_keys()
    return public_pem


def get_private_key_pem() -> str:
    """Return the private key PEM, generating keys if necessary."""
    private_pem, _ = load_or_create_keys()
    return private_pem


# ─── Hashing ─────────────────────────────────────────────────────────────────

def hash_file_sha256(filepath: str | Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 digest of a (potentially large) file in chunks.

    Uses streaming reads so that multi-GB model weights never exhaust RAM.

    Parameters
    ----------
    filepath   : Path to the model artifact (.pt, .bin, .safetensors, .pkl …)
    chunk_size : Read buffer size in bytes (default 64 KiB).

    Returns
    -------
    Lowercase hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_metadata(dict_data: dict) -> str:
    """Deterministically SHA-256-hash a metadata / config dictionary.

    Uses canonical JSON (sorted keys, no whitespace) to guarantee that
    identical dicts always produce identical digests regardless of insertion
    order.

    Parameters
    ----------
    dict_data : Arbitrary JSON-serialisable dictionary.

    Returns
    -------
    Lowercase hexadecimal SHA-256 digest string.
    """
    canonical = json.dumps(dict_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_block_fields(fields: dict) -> str:
    """Combine multiple block fields into a single deterministic SHA-256.

    Used to compute `current_block_hash` from all ledger row values.
    """
    return hash_metadata(fields)


# ─── Signing & Verification ──────────────────────────────────────────────────

def sign_hash(data_hash_hex: str, private_key_pem: str) -> str:
    """Sign a hex-digest string using RSA-PSS (SHA-256).

    Parameters
    ----------
    data_hash_hex  : Hex string to sign (typically artifact_hash + config_hash
                     + previous_block_hash, concatenated before hashing).
    private_key_pem: PEM-encoded RSA private key.

    Returns
    -------
    Base64-encoded signature string.
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None,
        backend=default_backend(),
    )

    message = data_hash_hex.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    return base64.b64encode(signature).decode("utf-8")


def verify_signature(
    data_hash_hex: str,
    signature_b64: str,
    public_key_pem: str,
) -> bool:
    """Verify an RSA-PSS signature against a public key.

    Parameters
    ----------
    data_hash_hex : Original hex string that was signed.
    signature_b64 : Base64-encoded signature (from `sign_hash`).
    public_key_pem: PEM-encoded RSA public key.

    Returns
    -------
    True if the signature is valid; False otherwise.
    """
    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode(),
            backend=default_backend(),
        )
        signature = base64.b64decode(signature_b64)
        message = data_hash_hex.encode("utf-8")

        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True

    except (InvalidSignature, Exception):
        return False


def build_signing_payload(
    artifact_hash: str,
    config_hash: str,
    previous_block_hash: str,
) -> str:
    """Construct the canonical string to sign for a ledger block.

    Concatenates hashes in a deterministic order and returns the
    hex-digest of the combined SHA-256 — this is what `sign_hash` receives.
    """
    combined = artifact_hash + config_hash + previous_block_hash
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ─── Initialise keys on import ───────────────────────────────────────────────
load_or_create_keys()

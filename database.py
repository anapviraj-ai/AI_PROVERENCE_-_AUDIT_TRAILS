"""
database.py
─────────────────────────────────────────────────────────────────────────────
Blockchain-like SQLite audit ledger for the CEPL AI Provenance system.

Each row in ModelLedger is an immutable "block" whose integrity is
guaranteed by:
  1. `artifact_hash`      — SHA-256 of the model weights file
  2. `config_hash`        — SHA-256 of the hyper-parameter / config dict
  3. `digital_signature`  — RSA-PSS signature of (artifact_hash + config_hash
                            + previous_block_hash)
  4. `previous_block_hash`— Hash of the preceding block (genesis = "0")
  5. `current_block_hash` — SHA-256 of every field in this row → chain link

`verify_chain_integrity()` re-derives hashes and signatures from stored
data to detect any post-commit tampering.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import crypto_utils as cu

# ─── Engine / Session ─────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./cepl_provenance.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# Enable WAL mode for better concurrent read performance
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─── ORM Model ───────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class ModelLedger(Base):
    """Immutable provenance block for a single AI model artifact event."""

    __tablename__ = "model_ledger"

    id                  = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp           = Column(String(40),  nullable=False)
    model_name          = Column(String(256), nullable=False)
    event_type          = Column(String(64),  nullable=False)   # GENESIS_INGESTION | FINE_TUNING | PRUNING_QUANTIZATION
    parent_block_id     = Column(Integer,     nullable=True)    # NULL for genesis
    artifact_hash       = Column(String(64),  nullable=False)   # SHA-256 of weights file
    config_hash         = Column(String(64),  nullable=False)   # SHA-256 of config/hparams dict
    metadata_json       = Column(Text,        nullable=False)   # JSON blob (author, source, etc.)
    digital_signature   = Column(Text,        nullable=False)   # Base64 RSA-PSS signature
    previous_block_hash = Column(String(64),  nullable=False)   # "0" for genesis
    current_block_hash  = Column(String(64),  nullable=False)   # SHA-256 of all fields above

    def to_dict(self) -> dict:
        return {
            "id":                   self.id,
            "timestamp":            self.timestamp,
            "model_name":           self.model_name,
            "event_type":           self.event_type,
            "parent_block_id":      self.parent_block_id,
            "artifact_hash":        self.artifact_hash,
            "config_hash":          self.config_hash,
            "metadata_json":        self.metadata_json,
            "digital_signature":    self.digital_signature,
            "previous_block_hash":  self.previous_block_hash,
            "current_block_hash":   self.current_block_hash,
        }


Base.metadata.create_all(bind=engine)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_current_block_hash(record: dict) -> str:
    """Re-derive the current_block_hash from all other fields (excludes itself and auto-increment id)."""
    fields = {k: v for k, v in record.items() if k not in ("current_block_hash", "id")}
    return cu.hash_metadata(fields)


# ─── Public API ──────────────────────────────────────────────────────────────

def append_record(
    *,
    model_name:       str,
    event_type:       str,
    artifact_hash:    str,
    config_hash:      str,
    metadata:         dict,
    parent_block_id:  Optional[int] = None,
) -> ModelLedger:
    """Create and persist a new provenance block.

    Parameters
    ----------
    model_name      : Human-readable model identifier.
    event_type      : One of GENESIS_INGESTION / FINE_TUNING / PRUNING_QUANTIZATION.
    artifact_hash   : SHA-256 hex digest of the model weights file.
    config_hash     : SHA-256 hex digest of the config / hyper-param dictionary.
    metadata        : Arbitrary dict stored as JSON (author, source URL, etc.).
    parent_block_id : ID of the parent block (None for genesis events).

    Returns
    -------
    The newly created and committed ModelLedger ORM instance.
    """
    with SessionLocal() as session:
        # Resolve previous block hash
        if parent_block_id is not None:
            parent = session.get(ModelLedger, parent_block_id)
            previous_block_hash = parent.current_block_hash if parent else "0"
        else:
            # For non-genesis records without explicit parent, chain to latest
            last = (
                session.query(ModelLedger)
                .order_by(ModelLedger.id.desc())
                .first()
            )
            previous_block_hash = last.current_block_hash if last else "0"

        # Sign the block
        private_pem = cu.get_private_key_pem()
        signing_payload = cu.build_signing_payload(
            artifact_hash, config_hash, previous_block_hash
        )
        digital_signature = cu.sign_hash(signing_payload, private_pem)

        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata, sort_keys=True)

        # Compute current block hash over all fields (excluding itself)
        pre_image = {
            "timestamp":            timestamp,
            "model_name":           model_name,
            "event_type":           event_type,
            "parent_block_id":      parent_block_id,
            "artifact_hash":        artifact_hash,
            "config_hash":          config_hash,
            "metadata_json":        metadata_json,
            "digital_signature":    digital_signature,
            "previous_block_hash":  previous_block_hash,
        }
        current_block_hash = cu.hash_metadata(pre_image)

        record = ModelLedger(
            timestamp           = timestamp,
            model_name          = model_name,
            event_type          = event_type,
            parent_block_id     = parent_block_id,
            artifact_hash       = artifact_hash,
            config_hash         = config_hash,
            metadata_json       = metadata_json,
            digital_signature   = digital_signature,
            previous_block_hash = previous_block_hash,
            current_block_hash  = current_block_hash,
        )

        session.add(record)
        session.commit()
        session.refresh(record)
        return record


def get_all_records() -> list[dict]:
    """Return every ledger block as a list of dicts, ordered by id."""
    with SessionLocal() as session:
        rows = session.query(ModelLedger).order_by(ModelLedger.id.asc()).all()
        return [r.to_dict() for r in rows]


def get_record_by_id(block_id: int) -> Optional[dict]:
    """Return a single ledger block dict, or None if not found."""
    with SessionLocal() as session:
        row = session.get(ModelLedger, block_id)
        return row.to_dict() if row else None


def verify_chain_integrity() -> dict:
    """Re-derive and cross-check every hash and signature in the ledger.

    Checks performed per block
    ─────────────────────────
    1. current_block_hash matches re-derived hash of all other fields.
    2. previous_block_hash equals the current_block_hash of the prior block
       (or "0" for block #1).
    3. digital_signature is valid against the stored public key.

    Returns
    -------
    {
        "is_valid":     bool,
        "total_blocks": int,
        "results":      [{ "block_id": int, "checks": {...}, "ok": bool }],
    }
    """
    public_pem = cu.get_public_key_pem()
    records = get_all_records()
    results = []
    all_ok  = True

    prev_hash = "0"

    records_by_id = {r["id"]: r for r in records}

    for i, rec in enumerate(records):
        checks: dict[str, bool] = {}

        # 1 ─ current_block_hash integrity
        # id is excluded because it was not known when the hash was first computed
        # (auto-increment is assigned by the DB after the INSERT)
        pre_image = {k: v for k, v in rec.items() if k not in ("current_block_hash", "id")}
        expected_block_hash = cu.hash_metadata(pre_image)
        checks["block_hash_valid"] = (rec["current_block_hash"] == expected_block_hash)

        # 2 ─ chain linkage (checks parent block if specified, else prior sequential block)
        if rec.get("parent_block_id") is not None:
            parent_rec = records_by_id.get(rec["parent_block_id"])
            expected_prev = parent_rec["current_block_hash"] if parent_rec else "0"
        else:
            expected_prev = "0" if i == 0 else records[i - 1]["current_block_hash"]
        checks["chain_link_valid"] = (rec["previous_block_hash"] == expected_prev)

        # 3 ─ digital signature
        signing_payload = cu.build_signing_payload(
            rec["artifact_hash"],
            rec["config_hash"],
            rec["previous_block_hash"],
        )
        checks["signature_valid"] = cu.verify_signature(
            signing_payload,
            rec["digital_signature"],
            public_pem,
        )

        block_ok = all(checks.values())
        all_ok   = all_ok and block_ok
        prev_hash = rec["current_block_hash"]

        results.append({
            "block_id": rec["id"],
            "model_name": rec["model_name"],
            "event_type": rec["event_type"],
            "timestamp":  rec["timestamp"],
            "checks":     checks,
            "ok":         block_ok,
        })

    return {
        "is_valid":     all_ok,
        "total_blocks": len(records),
        "results":      results,
    }

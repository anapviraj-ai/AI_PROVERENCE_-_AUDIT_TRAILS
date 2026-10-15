"""
ingest_service.py
─────────────────────────────────────────────────────────────────────────────
Ingestion & evolution handler for the CEPL AI Provenance system.

Two public pipelines
────────────────────
• ingest_model_artifact()  — Genesis ingestion of an existing pre-trained model.
• log_model_update()       — Record a fine-tuning / quantization / pruning step
                             as a child block chained to its parent.

Both pipelines:
  1. Compute SHA-256 of the artifact file (chunked, memory-safe).
  2. Hash the supplied config / hyper-parameter dictionary.
  3. Sign the combined payload with the local RSA private key.
  4. Commit the block to the SQLite audit ledger.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import crypto_utils as cu
import database as db

# ─── Allowed event types ──────────────────────────────────────────────────────
EVENT_GENESIS          = "GENESIS_INGESTION"
EVENT_FINE_TUNING      = "FINE_TUNING"
EVENT_PRUNING_QUANT    = "PRUNING_QUANTIZATION"

VALID_EVENT_TYPES = {EVENT_GENESIS, EVENT_FINE_TUNING, EVENT_PRUNING_QUANT}


# ─── Ingestion Pipeline ───────────────────────────────────────────────────────

def ingest_model_artifact(
    artifact_path: str | Path,
    model_name: str,
    config: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Ingest an existing pre-trained model artifact as a Genesis block.

    Parameters
    ----------
    artifact_path : Absolute or relative path to the model weights file.
                    Supports .pt, .bin, .safetensors, .pkl, .onnx, etc.
    model_name    : Human-readable name (e.g. "BERT-Base-Pretrained").
    config        : Optional dict of model config / hyper-parameters
                    (architecture, num_layers, vocab_size, …).
    metadata      : Optional dict of provenance metadata
                    (author, source_url, dataset_lineage, framework, …).

    Returns
    -------
    dict with { block_id, model_name, artifact_hash, config_hash,
                digital_signature, current_block_hash, event_type }.
    """
    artifact_path = Path(artifact_path)
    config   = config   or {}
    metadata = metadata or {}

    # 1. Hash the weights file
    artifact_hash = cu.hash_file_sha256(artifact_path)

    # 2. Hash the config dict
    config_hash = cu.hash_metadata(config)

    # 3. Enrich metadata with framework hints from file extension
    ext = artifact_path.suffix.lower()
    framework_hints = {
        ".pt":           "PyTorch",
        ".pth":          "PyTorch",
        ".bin":          "PyTorch / HuggingFace",
        ".safetensors":  "SafeTensors / HuggingFace",
        ".pkl":          "Scikit-learn / Pickle",
        ".onnx":         "ONNX",
        ".h5":           "Keras / TensorFlow",
        ".pb":           "TensorFlow",
        ".tflite":       "TensorFlow Lite",
    }
    metadata.setdefault("file_extension",     ext)
    metadata.setdefault("detected_framework", framework_hints.get(ext, "Unknown"))
    metadata.setdefault("file_size_bytes",    os.path.getsize(artifact_path))
    metadata.setdefault("original_filename",  artifact_path.name)

    # 4. Write to the ledger
    record = db.append_record(
        model_name    = model_name,
        event_type    = EVENT_GENESIS,
        artifact_hash = artifact_hash,
        config_hash   = config_hash,
        metadata      = metadata,
        parent_block_id = None,
    )

    return _record_to_response(record)


# ─── Evolution Pipeline ───────────────────────────────────────────────────────

def log_model_update(
    artifact_path:   str | Path,
    parent_block_id: int,
    event_type:      str,
    model_name:      Optional[str] = None,
    config:          Optional[dict] = None,
    metadata:        Optional[dict] = None,
) -> dict:
    """Record a fine-tuning / quantization / pruning step for an existing model.

    Parameters
    ----------
    artifact_path    : Path to the updated / fine-tuned model weights.
    parent_block_id  : ID of the parent Genesis (or prior update) block.
    event_type       : "FINE_TUNING" or "PRUNING_QUANTIZATION".
    model_name       : Optional override; inherits from parent if omitted.
    config           : New hyper-parameters / quantization config dict.
    metadata         : Update-specific metadata (dataset, epochs, lr, …).

    Returns
    -------
    Same dict shape as `ingest_model_artifact`.
    """
    artifact_path = Path(artifact_path)
    config   = config   or {}
    metadata = metadata or {}

    if event_type not in (EVENT_FINE_TUNING, EVENT_PRUNING_QUANT):
        raise ValueError(
            f"event_type must be one of {EVENT_FINE_TUNING!r} or "
            f"{EVENT_PRUNING_QUANT!r}, got {event_type!r}"
        )

    # Resolve parent record
    parent = db.get_record_by_id(parent_block_id)
    if parent is None:
        raise ValueError(f"Parent block ID {parent_block_id} not found in ledger.")

    resolved_name = model_name or parent["model_name"]

    # 1. Hash the updated artifact
    artifact_hash = cu.hash_file_sha256(artifact_path)

    # 2. Hash the new config
    config_hash = cu.hash_metadata(config)

    # 3. Enrich metadata
    metadata.setdefault("file_size_bytes",   os.path.getsize(artifact_path))
    metadata.setdefault("original_filename", artifact_path.name)
    metadata.setdefault("parent_model_name", parent["model_name"])
    metadata.setdefault("parent_event_type", parent["event_type"])

    # 4. Write to ledger
    record = db.append_record(
        model_name      = resolved_name,
        event_type      = event_type,
        artifact_hash   = artifact_hash,
        config_hash     = config_hash,
        metadata        = metadata,
        parent_block_id = parent_block_id,
    )

    return _record_to_response(record)


# ─── Verification Helper ──────────────────────────────────────────────────────

def verify_artifact_against_block(
    artifact_path: str | Path,
    block_id:      int,
) -> dict:
    """Check whether an uploaded file matches the stored provenance record.

    Computes the file's SHA-256 and compares it to the stored artifact_hash.
    Also re-validates the block's digital signature.

    Returns
    -------
    {
        "block_id":             int,
        "hash_matches":         bool,
        "signature_valid":      bool,
        "computed_hash":        str,
        "stored_hash":          str,
        "is_authentic":         bool,    # both checks pass
        "model_name":           str,
        "event_type":           str,
        "timestamp":            str,
    }
    """
    artifact_path = Path(artifact_path)
    record = db.get_record_by_id(block_id)

    if record is None:
        raise ValueError(f"Block ID {block_id} not found in ledger.")

    computed_hash = cu.hash_file_sha256(artifact_path)
    hash_matches  = (computed_hash == record["artifact_hash"])

    # Re-verify the stored signature
    public_pem = cu.get_public_key_pem()
    signing_payload = cu.build_signing_payload(
        record["artifact_hash"],
        record["config_hash"],
        record["previous_block_hash"],
    )
    signature_valid = cu.verify_signature(
        signing_payload,
        record["digital_signature"],
        public_pem,
    )

    return {
        "block_id":        block_id,
        "hash_matches":    hash_matches,
        "signature_valid": signature_valid,
        "computed_hash":   computed_hash,
        "stored_hash":     record["artifact_hash"],
        "is_authentic":    hash_matches and signature_valid,
        "model_name":      record["model_name"],
        "event_type":      record["event_type"],
        "timestamp":       record["timestamp"],
    }


# ─── Internal ─────────────────────────────────────────────────────────────────

def _record_to_response(record) -> dict:
    """Convert an ORM ModelLedger instance to a clean API response dict."""
    return {
        "block_id":            record.id,
        "model_name":          record.model_name,
        "event_type":          record.event_type,
        "artifact_hash":       record.artifact_hash,
        "config_hash":         record.config_hash,
        "digital_signature":   record.digital_signature,
        "current_block_hash":  record.current_block_hash,
        "previous_block_hash": record.previous_block_hash,
        "timestamp":           record.timestamp,
    }

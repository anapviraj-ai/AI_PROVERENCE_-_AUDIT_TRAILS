"""
main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI REST backend for the CEPL AI Provenance & Audit Trail system.

Endpoints
─────────
  POST /api/ingest       — Upload a model file → write Genesis provenance record.
  POST /api/log-update   — Upload updated model + parent block ID → record evolution.
  GET  /api/ledger       — Full provenance chain + chain integrity status.
  POST /api/verify       — Upload a file + block_id → tamper detection result.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database as db
import ingest_service as svc

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AI Provenance & Audit Trail",
    description = (
        "Verifiable, tamper-evident provenance records for existing AI model "
        "artifacts using cryptographic hashing and digital signatures."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ─── Temp upload directory ───────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _save_upload(upload_file: UploadFile) -> Path:
    """Persist an uploaded file to a temporary path and return it."""
    suffix = Path(upload_file.filename or "model").suffix or ".bin"
    tmp    = tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=suffix, delete=False
    )
    try:
        shutil.copyfileobj(upload_file.file, tmp)
    finally:
        tmp.close()
    return Path(tmp.name)


# ─── Response Schemas ─────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    block_id:            int
    model_name:          str
    event_type:          str
    artifact_hash:       str
    config_hash:         str
    digital_signature:   str
    current_block_hash:  str
    previous_block_hash: str
    timestamp:           str


class LedgerResponse(BaseModel):
    is_chain_valid: bool
    total_blocks:   int
    records:        list[dict]
    integrity:      dict


class VerifyResponse(BaseModel):
    block_id:        int
    hash_matches:    bool
    signature_valid: bool
    computed_hash:   str
    stored_hash:     str
    is_authentic:    bool
    model_name:      str
    event_type:      str
    timestamp:       str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"service": "AI Provenance & Audit API", "status": "running", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/api/ingest", response_model=IngestResponse, tags=["Provenance"])
async def ingest_model(
    file:        UploadFile = File(..., description="Model artifact file (.pt, .bin, .safetensors, .pkl, .onnx, …)"),
    model_name:  str        = Form(..., description="Human-readable model name"),
    config_json: str        = Form("{}",  description="JSON string: model config / architecture hyper-parameters"),
    metadata_json: str      = Form("{}", description="JSON string: provenance metadata (author, source_url, …)"),
):
    """
    ## Genesis Ingestion

    Upload an existing pre-trained model artifact and register it as the
    first (genesis) block in its provenance chain.

    **Form fields**
    - `file`          : The model weights file.
    - `model_name`    : e.g. `"BERT-Base-Uncased"`.
    - `config_json`   : JSON-serialised config dict (architecture, vocab_size, …).
    - `metadata_json` : JSON-serialised metadata (author, huggingface_url, …).
    """
    try:
        config   = json.loads(config_json)
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}")

    tmp_path = _save_upload(file)
    try:
        result = svc.ingest_model_artifact(
            artifact_path = tmp_path,
            model_name    = model_name,
            config        = config,
            metadata      = metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


@app.post("/api/log-update", response_model=IngestResponse, tags=["Provenance"])
async def log_update(
    file:            UploadFile = File(..., description="Updated model artifact file"),
    parent_block_id: int        = Form(..., description="Block ID of the parent / predecessor model"),
    event_type:      str        = Form("FINE_TUNING", description="FINE_TUNING or PRUNING_QUANTIZATION"),
    model_name:      str        = Form("",  description="Override model name (leave blank to inherit from parent)"),
    config_json:     str        = Form("{}", description="JSON: new hyper-parameters / quantization config"),
    metadata_json:   str        = Form("{}", description="JSON: update metadata (dataset, epochs, lr, …)"),
):
    """
    ## Log Model Update / Evolution

    Record a fine-tuning, quantization, or pruning step as a child block
    chained to the specified parent provenance record.
    """
    try:
        config   = json.loads(config_json)
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}")

    if event_type not in (svc.EVENT_FINE_TUNING, svc.EVENT_PRUNING_QUANT):
        raise HTTPException(
            status_code=422,
            detail=f"event_type must be FINE_TUNING or PRUNING_QUANTIZATION, got {event_type!r}",
        )

    tmp_path = _save_upload(file)
    try:
        result = svc.log_model_update(
            artifact_path   = tmp_path,
            parent_block_id = parent_block_id,
            event_type      = event_type,
            model_name      = model_name or None,
            config          = config,
            metadata        = metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


@app.get("/api/ledger", response_model=LedgerResponse, tags=["Audit"])
def get_ledger():
    """
    ## Full Provenance Ledger

    Returns every block in the audit chain together with a live
    `is_chain_valid` integrity check (re-derives all hashes and signatures).
    """
    records   = db.get_all_records()
    integrity = db.verify_chain_integrity()

    return LedgerResponse(
        is_chain_valid = integrity["is_valid"],
        total_blocks   = integrity["total_blocks"],
        records        = records,
        integrity      = integrity,
    )


@app.post("/api/verify", response_model=VerifyResponse, tags=["Audit"])
async def verify_artifact(
    file:     UploadFile = File(..., description="Model file to verify"),
    block_id: int        = Form(..., description="Block ID to verify against"),
):
    """
    ## Independent Tamper Verification

    Upload any model file and a Block ID.  The API computes the file's
    SHA-256 and compares it to the stored provenance record.

    Returns a green (authentic) or red (tampered) verdict including:
    - `hash_matches`    — file bytes unchanged since ingestion.
    - `signature_valid` — ledger block's RSA signature still holds.
    - `is_authentic`    — both checks passed.
    """
    tmp_path = _save_upload(file)
    try:
        result = svc.verify_artifact_against_block(tmp_path, block_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return VerifyResponse(**result)


@app.get("/api/block/{block_id}", tags=["Audit"])
def get_block(block_id: int):
    """Return a single ledger block by ID."""
    record = db.get_record_by_id(block_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Block {block_id} not found.")
    return record


@app.post("/api/verify-chain", tags=["Audit"])
def verify_chain():
    """Re-run full chain integrity verification and return detailed results."""
    return db.verify_chain_integrity()

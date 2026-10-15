"""
generate_rich_demo_data.py
─────────────────────────────────────────────────────────────────────────────
Reset the database and generate 4 realistic, independent blockchain trees
representing different AI frameworks:
  1. Hugging Face : BERT-Base ➔ BERT-Finance-Adapter (Fine-Tuning)
  2. PyTorch      : ResNet-50 ➔ ResNet-50-Pruned (Pruning & Sparsity)
  3. ONNX         : YOLOv8 ➔ YOLOv8-INT8-Quantized (Quantization)
  4. Scikit-Learn : RandomForest-FraudDetection (Single-node Standalone)

Run:
    python generate_rich_demo_data.py
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import struct
import sys
import time
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crypto_utils as cu
import database as db
import ingest_service as svc

SAMPLE_DIR = Path("sample_models")


# ─── 1. Environment Reset ──────────────────────────────────────────────────────

def reset_environment():
    """Safely clear existing SQLite database and sample models directory."""
    print("=" * 70)
    print("[*] STEP 1: Resetting Environment & Initializing Fresh Database")
    print("=" * 70)

    # 1. Dispose active SQLAlchemy connections
    try:
        db.engine.dispose()
    except Exception as e:
        print(f"  Note: Engine dispose: {e}")

    # 2. Try removing SQLite database files if not locked
    db_files = ["cepl_provenance.db", "cepl_provenance.db-wal", "cepl_provenance.db-shm"]
    for fname in db_files:
        fpath = Path(fname)
        if fpath.exists():
            try:
                fpath.unlink()
                print(f"  [OK] Deleted old database file: {fname}")
            except PermissionError:
                print(f"  [WARN] {fname} locked by active process - will truncate tables via SQL.")

    # 3. Always drop & re-create tables cleanly using SQLAlchemy schema
    db.Base.metadata.drop_all(bind=db.engine)
    db.Base.metadata.create_all(bind=db.engine)
    print("  [OK] Initialized fresh tables from database.py schema (model_ledger).")

    # 4. Clear and recreate sample_models/ directory
    if SAMPLE_DIR.exists():
        for item in SAMPLE_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"  Note: clearing {item.name}: {e}")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Cleared and created fresh directory: {SAMPLE_DIR}/")


# ─── 2. Artifact Creators ─────────────────────────────────────────────────────

def create_safetensors_file(filename: str, tensor_names: list[str], metadata: dict, size_kb: int = 180) -> Path:
    """Generate a realistic SafeTensors binary file format."""
    out_path = SAMPLE_DIR / filename
    
    # Formulate SafeTensors header dictionary
    header_dict = {}
    current_offset = 0
    tensor_bytes_each = (size_kb * 1024) // max(len(tensor_names), 1)

    for name in tensor_names:
        header_dict[name] = {
            "dtype": "F32",
            "shape": [64, 128],
            "data_offsets": [current_offset, current_offset + tensor_bytes_each],
        }
        current_offset += tensor_bytes_each

    header_dict["__metadata__"] = metadata
    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    
    # SafeTensors begins with an 8-byte uint64 header size
    header_size = len(header_json)
    size_prefix = struct.pack("<Q", header_size)

    # Tensor payload (pseudo-random floats)
    rng = random.Random(hash(filename))
    raw_tensor_data = bytes([rng.randint(0, 255) for _ in range(current_offset)])

    out_path.write_bytes(size_prefix + header_json + raw_tensor_data)
    print(f"  [OK] Created SafeTensors artifact: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def create_pytorch_state_dict_file(filename: str, layer_names: list[str], size_kb: int = 220) -> Path:
    """Generate a realistic PyTorch state_dict .pt artifact."""
    out_path = SAMPLE_DIR / filename
    rng = random.Random(hash(filename))

    state_dict = {}
    for layer in layer_names:
        # Realistic weight array simulation
        weights = [rng.uniform(-0.1, 0.1) for _ in range(32 * 32)]
        state_dict[layer] = {
            "type": "torch.FloatTensor",
            "shape": [32, 32],
            "data": weights[:128], # sample
            "norm": sum(abs(w) for w in weights[:64])
        }

    with out_path.open("wb") as f:
        pickle.dump(state_dict, f, protocol=4)

    # Append filler to match expected model weight size
    filler = bytes([rng.randint(0, 255) for _ in range(size_kb * 1024)])
    with out_path.open("ab") as f:
        f.write(filler)

    print(f"  [OK] Created PyTorch artifact:     {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def create_onnx_model_file(filename: str, graph_name: str, opset: int = 17, size_kb: int = 160) -> Path:
    """Generate a realistic ONNX Protobuf binary .onnx artifact."""
    out_path = SAMPLE_DIR / filename
    rng = random.Random(hash(filename))

    # ONNX Magic and Protobuf simulated structure
    onnx_header = b"\x08\x07\x12\x04onnx\x1a\x08" + graph_name.encode("utf-8") + b"\x20" + bytes([opset])
    raw_graph_data = bytes([rng.randint(0, 255) for _ in range(size_kb * 1024)])

    out_path.write_bytes(onnx_header + raw_graph_data)
    print(f"  [OK] Created ONNX artifact:        {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def create_scikit_learn_file(filename: str) -> Path:
    """Train and serialize a genuine Scikit-Learn RandomForestClassifier."""
    out_path = SAMPLE_DIR / filename
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=800,
            n_features=15,
            n_informative=10,
            n_classes=2,
            random_state=42,
        )
        clf = RandomForestClassifier(n_estimators=75, max_depth=10, random_state=42)
        clf.fit(X, y)

        with out_path.open("wb") as f:
            pickle.dump(clf, f)
    except ImportError:
        # Fallback if scikit-learn not available
        dummy_obj = {"model": "RandomForestClassifier", "n_estimators": 75, "params": {"max_depth": 10}}
        with out_path.open("wb") as f:
            pickle.dump(dummy_obj, f)

    print(f"  [OK] Created Scikit-Learn artifact: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


# ─── 3. Ingestion & Ledger Construction ────────────────────────────────────────

def build_rich_ledger():
    """Build the 4 distinct framework lineages."""
    print("\n" + "=" * 70)
    print("[+] STEP 2: Generating Framework-Specific Model Artifacts")
    print("=" * 70)

    # 1. Hugging Face
    hf_genesis_file = create_safetensors_file(
        "bert_base.safetensors",
        tensor_names=[
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.key.weight",
            "bert.pooler.dense.weight",
        ],
        metadata={"format": "pt", "model_type": "bert", "framework": "Hugging Face"},
        size_kb=180,
    )

    hf_child_file = create_safetensors_file(
        "bert_finance_adapter.safetensors",
        tensor_names=[
            "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight",
            "base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.weight",
            "classifier.weight",
            "classifier.bias",
        ],
        metadata={"peft_type": "LORA", "target": "bert-base", "task": "sentiment"},
        size_kb=95,
    )

    # 2. PyTorch
    pt_genesis_file = create_pytorch_state_dict_file(
        "resnet50.pt",
        layer_names=["conv1.weight", "bn1.weight", "layer1.0.conv1.weight", "layer2.0.conv1.weight", "fc.weight"],
        size_kb=240,
    )

    pt_child_file = create_pytorch_state_dict_file(
        "resnet50_pruned.pt",
        layer_names=["conv1.weight", "bn1.weight", "layer1.0.conv1.weight_mask", "fc.weight_mask"],
        size_kb=150,
    )

    # 3. ONNX
    onnx_genesis_file = create_onnx_model_file(
        "yolov8.onnx",
        graph_name="yolov8_detector_graph",
        opset=17,
        size_kb=200,
    )

    onnx_child_file = create_onnx_model_file(
        "yolov8_int8_quantized.onnx",
        graph_name="yolov8_quantized_graph",
        opset=17,
        size_kb=85,
    )

    # 4. Scikit-Learn
    sklearn_genesis_file = create_scikit_learn_file("random_forest.pkl")

    # Compatibility copies for any legacy lookups
    shutil.copyfile(SAMPLE_DIR / "random_forest.pkl", SAMPLE_DIR / "dummy_sklearn_model.pkl")
    shutil.copyfile(SAMPLE_DIR / "resnet50.pt", SAMPLE_DIR / "dummy_pytorch_model.pt")
    shutil.copyfile(SAMPLE_DIR / "resnet50_pruned.pt", SAMPLE_DIR / "dummy_finetuned_model.pt")

    print("\n" + "=" * 70)
    print("[+] STEP 3: Ledger Construction (Building 4 Independent Lineages)")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # TREE 1: Hugging Face (BERT-Base -> BERT-Finance-Adapter)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Tree 1: Hugging Face NLP]")
    b1 = svc.ingest_model_artifact(
        artifact_path = hf_genesis_file,
        model_name    = "BERT-Base",
        config        = {
            "architecture": "BertForMaskedLM",
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 512,
            "hidden_act": "gelu",
        },
        metadata      = {
            "author": "Hugging Face Model Hub",
            "framework": "Hugging Face / SafeTensors",
            "source_url": "https://huggingface.co/google-bert/bert-base-uncased",
            "license": "Apache-2.0",
            "description": "Pre-trained BERT encoder base checkpoint trained on BookCorpus & English Wikipedia.",
        },
    )
    print(f"  [Genesis] Block #{b1['block_id']}: '{b1['model_name']}' [GENESIS_INGESTION]")
    print(f"     Artifact Hash: {b1['artifact_hash'][:24]}...")
    print(f"     Block Hash:    {b1['current_block_hash'][:24]}...")

    b2 = svc.log_model_update(
        artifact_path   = hf_child_file,
        parent_block_id = b1["block_id"],
        event_type      = "FINE_TUNING",
        model_name      = "BERT-Finance-Adapter",
        config          = {
            "peft_type": "LORA",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["query", "value"],
            "learning_rate": "3e-4",
            "batch_size": 32,
            "epochs": 5,
        },
        metadata        = {
            "author": "Financial NLP Research Lab",
            "framework": "Hugging Face PEFT",
            "dataset": "Financial-PhraseBank-v2",
            "metrics": {"macro_f1": 0.942, "accuracy": 0.951},
            "description": "LoRA adapter fine-tuned on financial sentiment and earnings call transcripts.",
        },
    )
    print(f"  [Child]   Block #{b2['block_id']}: '{b2['model_name']}' [FINE_TUNING] -> Parent #{b1['block_id']}")
    print(f"     Artifact Hash: {b2['artifact_hash'][:24]}...")

    # ─────────────────────────────────────────────────────────────────────────
    # TREE 2: PyTorch (ResNet-50 -> ResNet-50-Pruned)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Tree 2: PyTorch Computer Vision]")
    b3 = svc.ingest_model_artifact(
        artifact_path = pt_genesis_file,
        model_name    = "ResNet-50",
        config        = {
            "architecture": "ResNet50",
            "num_classes": 1000,
            "layers": [3, 4, 6, 3],
            "input_shape": [3, 224, 224],
            "normalization": "ImageNet",
        },
        metadata      = {
            "author": "TorchVision Model Zoo",
            "framework": "PyTorch",
            "source_url": "https://download.pytorch.org/models/resnet50.pth",
            "license": "BSD-3-Clause",
            "description": "Deep residual network with 50 layers pre-trained on ImageNet-1K benchmark.",
        },
    )
    print(f"  [Genesis] Block #{b3['block_id']}: '{b3['model_name']}' [GENESIS_INGESTION]")
    print(f"     Artifact Hash: {b3['artifact_hash'][:24]}...")

    b4 = svc.log_model_update(
        artifact_path   = pt_child_file,
        parent_block_id = b3["block_id"],
        event_type      = "PRUNING_QUANTIZATION",
        model_name      = "ResNet-50-Pruned",
        config          = {
            "pruning_method": "L1Unstructured",
            "sparsity_ratio": 0.50,
            "target_layers": ["layer2", "layer3", "layer4"],
            "speedup_factor": "1.7x",
            "retained_accuracy": 0.748,
        },
        metadata        = {
            "author": "Edge Inference Optimization Team",
            "framework": "PyTorch Pruning",
            "dataset": "ImageNet-Val-50k",
            "description": "50% weight sparsity applied via L1 norm magnitude pruning for low-latency inference.",
        },
    )
    print(f"  [Child]   Block #{b4['block_id']}: '{b4['model_name']}' [PRUNING_QUANTIZATION] -> Parent #{b3['block_id']}")

    # ─────────────────────────────────────────────────────────────────────────
    # TREE 3: ONNX (YOLOv8 -> YOLOv8-INT8-Quantized)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Tree 3: ONNX Object Detection]")
    b5 = svc.ingest_model_artifact(
        artifact_path = onnx_genesis_file,
        model_name    = "YOLOv8",
        config        = {
            "model_type": "YOLOv8-Small",
            "task": "object-detection",
            "input_resolution": [640, 640],
            "ir_version": 8,
            "opset_version": 17,
            "num_classes": 80,
        },
        metadata      = {
            "author": "Ultralytics",
            "framework": "ONNX Runtime",
            "source_url": "https://github.com/ultralytics/assets/releases/yolov8s.onnx",
            "license": "AGPL-3.0",
            "description": "High-efficiency real-time object detection model exported to ONNX format.",
        },
    )
    print(f"  [Genesis] Block #{b5['block_id']}: '{b5['model_name']}' [GENESIS_INGESTION]")
    print(f"     Artifact Hash: {b5['artifact_hash'][:24]}...")

    b6 = svc.log_model_update(
        artifact_path   = onnx_child_file,
        parent_block_id = b5["block_id"],
        event_type      = "PRUNING_QUANTIZATION",
        model_name      = "YOLOv8-INT8-Quantized",
        config          = {
            "quantization_type": "Post-Training Quantization (PTQ)",
            "precision": "INT8",
            "calibration_dataset": "COCO-val2017-sample500",
            "quantized_nodes": 142,
            "compression_ratio": "4.1x",
            "latency_reduction_percent": 68,
        },
        metadata        = {
            "author": "Mobile Embedded Systems Team",
            "framework": "ONNX Runtime Quantization",
            "description": "Static INT8 quantization calibrating activations and weights for edge NPU hardware.",
        },
    )
    print(f"  [Child]   Block #{b6['block_id']}: '{b6['model_name']}' [PRUNING_QUANTIZATION] -> Parent #{b5['block_id']}")

    # ─────────────────────────────────────────────────────────────────────────
    # TREE 4: Scikit-Learn (RandomForest-FraudDetection Single-Node)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Tree 4: Scikit-Learn Tabular Classifier]")
    b7 = svc.ingest_model_artifact(
        artifact_path = sklearn_genesis_file,
        model_name    = "RandomForest-FraudDetection",
        config        = {
            "n_estimators": 75,
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
        },
        metadata      = {
            "author": "FinTech Fraud Prevention Group",
            "framework": "Scikit-Learn",
            "dataset": "Synthetic-Financial-Transactions-v4",
            "metrics": {"auc_roc": 0.987, "precision": 0.963, "recall": 0.941},
            "description": "Ensemble decision forest trained to classify fraudulent e-commerce credit card transactions in real-time.",
        },
    )
    print(f"  [Genesis] Block #{b7['block_id']}: '{b7['model_name']}' [GENESIS_INGESTION]")
    print(f"     Artifact Hash: {b7['artifact_hash'][:24]}...")

    # ─── Verification ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[+] STEP 4: Mathematical Blockchain Integrity Verification")
    print("=" * 70)
    report = db.verify_chain_integrity()
    is_valid = report.get("is_valid", False)
    total = report.get("total_blocks", 0)

    print(f"  Total Ledger Blocks Committed: {total}")
    print(f"  Cryptographic Chain Status:    {'[VALID] 100% Intact' if is_valid else '[FAILED] Compromised'}")

    for item in report.get("results", []):
        bid = item["block_id"]
        c = item["checks"]
        status = "PASSED" if item["ok"] else "FAILED"
        print(f"    * Block #{bid}: Hash={c.get('block_hash_valid')} | Link={c.get('chain_link_valid')} | RSA-Sig={c.get('signature_valid')} -> {status}")

    if not is_valid:
        print("\n[ERROR] Blockchain mathematical validation failed!")
        sys.exit(1)

    print("\n[SUCCESS] Rich multi-framework demo data generated successfully!")
    print(f"   4 Framework Trees | 7 Immutable Blocks | Ready for live demonstration.\n")


if __name__ == "__main__":
    reset_environment()
    build_rich_ledger()

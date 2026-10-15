# 🔐 CEPL — AI Model Provenance & Cryptographic Audit Trail System

> **A verifiable, tamper-evident lineage and provenance ledger for enterprise AI models using cryptographic hashing, Merkle-linked blockchain records, and RSA-2048 digital signatures.**

---

## 📌 Overview

Modern enterprise AI systems lack verifiable provenance: when a model is deployed, updated, quantized, or fine-tuned, there is rarely an immutable record proving its origin, training dataset hashes, hyperparameters, or integrity. 

**CEPL AI Provenance & Cryptographic Audit Trail** solves this problem by providing:
- **Streaming SHA-256 Model Weight Hashing**: Efficient chunked hashing supporting multi-gigabyte files without memory exhaustion (`.safetensors`, `.pt`, `.onnx`, `.pkl`, etc.).
- **Canonical Metadata Hashes**: Deterministic JSON hashing of datasets, training regimes, hyperparameters, and licenses.
- **RSA-2048 PSS Digital Signatures**: Every provenance record is signed using RSA-PSS with SHA-256 for mathematical non-repudiation.
- **Merkle-Linked Blockchain Ledger**: Every child model (fine-tuned, quantized, pruned) links cryptographically to its parent block hash (`prev_hash`).
- **Interactive Tamper & Bit-Flip Attack Simulator**: Prove cryptographic tamper detection in real-time by mutating weight bytes or metadata.
- **Full-Featured Dashboard & REST API**: High-contrast, interactive Streamlit cockpit backed by a fast, asynchronous FastAPI REST engine.

---

## 🏗️ Architecture

```
                                  ┌──────────────────────────────┐
                                  │   AI Model Artifact (.pt,    │
                                  │   .safetensors, .onnx, .pkl) │
                                  └──────────────┬───────────────┘
                                                 │
                                       Chunked SHA-256 Hash
                                                 │
                                                 ▼
┌──────────────────────────────┐       ┌──────────────────┐
│ Canonical Metadata & Config  │──────▶│ Combined Hash    │
│ (JSON Normalized SHA-256)    │       │ (Weight + Meta)  │
└──────────────────────────────┘       └─────────┬────────┘
                                                 │
┌──────────────────────────────┐                 ▼
│ Previous Block Hash          │──────▶ ┌─────────────────┐
│ (Blockchain / Merkle Link)   │        │ RSA-2048 Sign   │
└──────────────────────────────┘        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ Provenance Record│
                                        │ (SQLite Ledger) │
                                        └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone <YOUR-GITHUB-REPO-URL>
cd CEPL

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Sample Models & Demo Ledger (Optional)

Generate realistic pre-trained model weights (`BERT`, `ResNet`, `YOLO`, `RandomForest`) and an initial cryptographic audit ledger:

```bash
# Create sample model weight files
python create_sample_model.py

# Seed rich demonstration ledger with lineage chains
python generate_rich_demo_data.py
```

### 4. Run the Backend API

Start the FastAPI REST backend:

```bash
uvicorn main:app --reload --port 8000
```
Interactive Swagger API documentation will be available at:  
👉 **`http://localhost:8000/docs`**

### 5. Run the Interactive Dashboard

In a separate terminal (with the virtual environment activated):

```bash
streamlit run app.py
```
The Streamlit UI will open at:  
👉 **`http://localhost:8501`**

---

## 🖥️ Dashboard Capabilities

| Tab | Feature | Description |
|---|---|---|
| **1. Ingest Existing AI** | **Genesis Block Registration** | Upload any pre-trained model file or quick-load sample models. Computes streaming SHA-256, generates metadata hash, signs with RSA-2048, and creates Block 1. |
| **2. Log Model Update** | **Evolution & Child Blocks** | Register fine-tuned, quantized, pruned, or distilled variants. Cryptographically links to the parent block (`prev_hash`). |
| **3. Audit Ledger & Lineage** | **Ledger Explorer & Tree** | Searchable audit trail table, full JSON record inspection, whole-chain integrity verification, and interactive parent-child lineage tree. |
| **4. Independent Verifier** | **Tamper Attack Simulator** | Upload any candidate file against a block ID, or simulate bit-flip attacks on weights and metadata to demonstrate instant cryptographic detection. |

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest` | Upload model file + metadata → creates Genesis provenance block. |
| `POST` | `/api/log-update` | Upload modified model + parent block ID → logs evolutionary child record. |
| `GET` | `/api/ledger` | Retrieve full provenance ledger and verify end-to-end chain integrity. |
| `POST` | `/api/verify` | Upload file + block ID to verify cryptographic validity and detect tampering. |

---

## 🧪 Testing

Execute the automated end-to-end test suite:

```bash
pytest test_e2e.py -v
```

---

## 🔒 Security & Privacy

- **Private Keys**: `private_key.pem` is automatically generated on first run if not present, and is excluded via `.gitignore` to prevent secret leakage.
- **Large File Handling**: Files are processed using streaming chunks (64 KiB buffer), ensuring consistent low-memory execution regardless of model parameter size.
- **Deterministic Hashing**: Canonical JSON serialization (sorted keys, compact delimiters) ensures reproducible cryptographic signatures across environments.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

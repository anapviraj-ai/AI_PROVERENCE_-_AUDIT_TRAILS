"""Quick end-to-end test for the CEPL provenance API."""
import requests
import json

BASE = "http://localhost:8000"

# ── 1. Genesis ingestion ──────────────────────────────────────────────────────
print("\n=== INGEST GENESIS ===")
with open("sample_models/dummy_sklearn_model.pkl", "rb") as f:
    r = requests.post(f"{BASE}/api/ingest",
        data={
            "model_name":    "Dummy-RandomForest-v1",
            "config_json":   json.dumps({"n_estimators": 50, "max_depth": None}),
            "metadata_json": json.dumps({"author": "AI Security Test", "framework": "scikit-learn"}),
        },
        files={"file": ("dummy_sklearn_model.pkl", f, "application/octet-stream")},
    )
b1 = r.json()
print(f"  Status : {r.status_code}")
print(f"  Block  : #{b1['block_id']}  event={b1['event_type']}")
print(f"  Hash   : {b1['artifact_hash'][:32]}...")

# ── 2. Fine-tuning update ─────────────────────────────────────────────────────
print("\n=== LOG UPDATE (FINE_TUNING) ===")
with open("sample_models/dummy_finetuned_model.pt", "rb") as f:
    r2 = requests.post(f"{BASE}/api/log-update",
        data={
            "parent_block_id": b1["block_id"],
            "event_type":      "FINE_TUNING",
            "model_name":      "Dummy-RandomForest-v1-finetuned",
            "config_json":     json.dumps({"epochs": 3, "lr": "2e-5"}),
            "metadata_json":   json.dumps({"notes": "Simulated fine-tune run"}),
        },
        files={"file": ("dummy_finetuned_model.pt", f, "application/octet-stream")},
    )
b2 = r2.json()
print(f"  Status : {r2.status_code}")
print(f"  Block  : #{b2['block_id']}  parent=#{b1['block_id']}")
print(f"  Hash   : {b2['artifact_hash'][:32]}...")

# ── 3. Chain integrity ────────────────────────────────────────────────────────
print("\n=== CHAIN INTEGRITY ===")
chain = requests.post(f"{BASE}/api/verify-chain").json()
print(f"  Total blocks : {chain['total_blocks']}")
print(f"  Chain valid  : {chain['is_valid']}")
for res in chain["results"]:
    tag = "OK  " if res["ok"] else "FAIL"
    c = res["checks"]
    print(f"  Block #{res['block_id']} [{res['event_type'][:16]:16}] {tag}"
          f"  block_hash={c['block_hash_valid']}  "
          f"chain_link={c['chain_link_valid']}  "
          f"sig={c['signature_valid']}")

# ── 4. Verify authentic file ──────────────────────────────────────────────────
print("\n=== VERIFY (original file -> should be AUTHENTIC) ===")
with open("sample_models/dummy_sklearn_model.pkl", "rb") as f:
    v = requests.post(f"{BASE}/api/verify",
        data={"block_id": b1["block_id"]},
        files={"file": ("dummy_sklearn_model.pkl", f, "application/octet-stream")},
    ).json()
print(f"  Hash match  : {v['hash_matches']}")
print(f"  Sig valid   : {v['signature_valid']}")
print(f"  Authentic   : {v['is_authentic']}")

# ── 5. Tamper test: wrong file vs block 1 ────────────────────────────────────
print("\n=== VERIFY (WRONG file vs block #1 -> should FAIL) ===")
with open("sample_models/dummy_finetuned_model.pt", "rb") as f:
    v2 = requests.post(f"{BASE}/api/verify",
        data={"block_id": b1["block_id"]},
        files={"file": ("dummy_finetuned_model.pt", f, "application/octet-stream")},
    ).json()
print(f"  Hash match  : {v2['hash_matches']}")
print(f"  Sig valid   : {v2['signature_valid']}")
print(f"  Authentic   : {v2['is_authentic']}")

print("\nAll tests done.")

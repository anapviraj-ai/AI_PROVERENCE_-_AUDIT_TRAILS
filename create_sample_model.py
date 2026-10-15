"""
create_sample_model.py
─────────────────────────────────────────────────────────────────────────────
Utility to generate ready-to-use sample model artifacts for testing the
CEPL AI Provenance & Audit Trail system out of the box.

Generates
─────────
  sample_models/dummy_sklearn_model.pkl       — Scikit-learn RandomForest
  sample_models/dummy_pytorch_model.pt        — PyTorch MLP (state_dict)
  sample_models/dummy_finetuned_model.pt      — "Fine-tuned" variant (slightly different weights)

Run:
    python create_sample_model.py
"""

from __future__ import annotations

import pickle
import struct
import os
from pathlib import Path

OUTPUT_DIR = Path("sample_models")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_sklearn_model() -> Path:
    """Create a dummy Scikit-learn RandomForestClassifier .pkl artifact."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import numpy as np

        print("  [sklearn] Generating RandomForestClassifier...")
        X, y = make_classification(
            n_samples    = 500,
            n_features   = 20,
            n_informative= 10,
            random_state = 42,
        )
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        out_path = OUTPUT_DIR / "dummy_sklearn_model.pkl"
        with out_path.open("wb") as f:
            pickle.dump(model, f)

        print(f"  [sklearn] Saved -> {out_path}  ({out_path.stat().st_size:,} bytes)")
        return out_path

    except ImportError:
        print("  [sklearn] scikit-learn not installed - writing raw binary fallback.")
        return _write_raw_binary("dummy_sklearn_model.pkl", seed=1)


def create_pytorch_model() -> Path:
    """Create a dummy PyTorch MLP state_dict .pt artifact."""
    try:
        import torch
        import torch.nn as nn

        print("  [pytorch] Generating MLP state_dict...")

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                )

            def forward(self, x):
                return self.net(x)

        model = SimpleMLP()
        out_path = OUTPUT_DIR / "dummy_pytorch_model.pt"
        torch.save(model.state_dict(), str(out_path))

        print(f"  [pytorch] Saved -> {out_path}  ({out_path.stat().st_size:,} bytes)")
        return out_path

    except ImportError:
        print("  [pytorch] torch not installed - writing raw binary fallback.")
        return _write_raw_binary("dummy_pytorch_model.pt", seed=2)


def create_finetuned_pytorch_model() -> Path:
    """Create a slightly modified PyTorch model simulating a fine-tuned checkpoint."""
    try:
        import torch
        import torch.nn as nn

        print("  [pytorch] Generating fine-tuned MLP variant...")

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                )

            def forward(self, x):
                return self.net(x)

        torch.manual_seed(99)  # Different seed -> different weights
        model = SimpleMLP()

        # Simulate fine-tuning: small perturbations to all weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        out_path = OUTPUT_DIR / "dummy_finetuned_model.pt"
        torch.save(model.state_dict(), str(out_path))

        print(f"  [pytorch] Saved -> {out_path}  ({out_path.stat().st_size:,} bytes)")
        return out_path

    except ImportError:
        print("  [pytorch] torch not installed - writing raw binary fallback.")
        return _write_raw_binary("dummy_finetuned_model.pt", seed=3)


def _write_raw_binary(filename: str, seed: int = 0) -> Path:
    """Fallback: write a deterministic pseudo-random binary blob."""
    import random
    rng  = random.Random(seed)
    data = bytes([rng.randint(0, 255) for _ in range(128 * 1024)])  # 128 KiB

    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(data)
    print(f"  [binary]  Saved -> {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


def main():
    print("\n" + "=" * 60)
    print("  AI Provenance - Sample Model Generator")
    print("=" * 60)

    paths = []
    paths.append(create_sklearn_model())
    paths.append(create_pytorch_model())
    paths.append(create_finetuned_pytorch_model())

    print("\n" + "=" * 60)
    print("  Generated artifacts:")
    for p in paths:
        print(f"    * {p}")
    print("\n  Usage:")
    print("  1. Start the API:        uvicorn main:app --reload --port 8000")
    print("  2. Start the dashboard:  streamlit run app.py")
    print("  3. In Tab 1, upload  sample_models/dummy_sklearn_model.pkl")
    print("  4. In Tab 2, upload  sample_models/dummy_finetuned_model.pt")
    print("  5. In Tab 3, explore the provenance chain.")
    print("  6. In Tab 4, verify any file against its block ID.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

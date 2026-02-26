#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "huggingface-hub",
#     "hf-transfer",
# ]
# ///

"""
download_phi3.py — Phi-3 ONNX Model Downloader for Project David

Downloads the DirectML INT4-AWQ quantized Phi-3 Mini model from Hugging Face
into the Xbox_AI_Server/Assets/Model directory.

Usage:
    uv run download_phi3.py
"""

import os
import sys

# Enable blazing fast downloads via hf_transfer BEFORE importing huggingface_hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

REPO_ID = "microsoft/Phi-3-mini-4k-instruct-onnx"

# The DirectML INT4-AWQ quantized variant — optimized for AMD/NVIDIA GPUs
ALLOW_PATTERNS = "directml/directml-int4-awq-block-128/*"

# Download into Assets/Model relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(SCRIPT_DIR, "Assets", "Model")


# ──────────────────────────────────────────────
#  Download
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Project David — Phi-3 ONNX Model Downloader")
    print("=" * 60)
    print()
    print(f"  Repository:  {REPO_ID}")
    print(f"  Variant:     directml-int4-awq-block-128")
    print(f"  Target dir:  {LOCAL_DIR}")
    print()

    os.makedirs(LOCAL_DIR, exist_ok=True)

    print("Downloading model files (this may take several minutes)...")
    print()

    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,  # Xbox needs real files, not symlinks
    )

    print()
    print(f"✅ Download complete!")
    print(f"   Files saved to: {downloaded_path}")
    print()

    # List what was downloaded
    model_subdir = os.path.join(LOCAL_DIR, "directml", "directml-int4-awq-block-128")
    if os.path.isdir(model_subdir):
        print("Downloaded files:")
        for f in sorted(os.listdir(model_subdir)):
            fpath = os.path.join(model_subdir, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f:40s}  {size_mb:8.1f} MB")

        print()
        print("IMPORTANT: Update the model path in InferenceEngine.cs to point to:")
        print(f"  Assets/Model/directml/directml-int4-awq-block-128")
        print()
        print("Or flatten the files directly into Assets/Model/ with:")
        print(f"  mv {model_subdir}/* {LOCAL_DIR}/")
    else:
        print("Downloaded files:")
        for root, dirs, files in os.walk(LOCAL_DIR):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, LOCAL_DIR)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {rel:40s}  {size_mb:8.1f} MB")

if __name__ == "__main__":
    main()
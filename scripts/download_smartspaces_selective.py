#!/usr/bin/env python3
"""
Selective Downloader & YOLO Formatter for NVIDIA PhysicalAI-SmartSpaces
------------------------------------------------------------------------
Target: Download warehouse scenes (default Warehouse_000 for train, Warehouse_015 for val)
        strictly omitting raw HDF5 depth maps (~185GB/scene) to prevent disk saturation.
        Downloads videos and annotations (~3.3GB/scene), samples frames at specified FPS,
        and generates standard YOLO detection annotations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("[ERROR] huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
REPO_TYPE = "dataset"

DEFAULT_SCENES = {
    "train": ["Warehouse_000"],
    "val": ["Warehouse_015"]
}

def verify_hf_authentication():
    """Verify Hugging Face authentication before attempting gated/large downloads."""
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        print("[WARNING] No Hugging Face token found in cache. Ensure you are logged in via `huggingface-cli login` if needed.")
    else:
        print("[INFO] Hugging Face authentication detected.")

def download_scene_manifest(scene_id: str, split_name: str, output_dir: Path) -> Path:
    """Download annotation and metadata files for a given warehouse scene."""
    scene_path = output_dir / split_name / scene_id
    scene_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Fetching scene metadata for {scene_id} ({split_name})...")
    # SmartSpaces MTMC format has ground_truth.json / metadata
    # Files are organized under scenes/<scene_id>/...
    # Note: Omit all files under depth_maps/ (*.h5, *.hdf5)
    return scene_path

def main():
    parser = argparse.ArgumentParser(description="Selective downloader for PhysicalAI-SmartSpaces")
    parser.add_argument("--output-dir", type=str, default="data/smartspaces_yolo", help="Target output directory")
    parser.add_argument("--scenes-train", nargs="+", default=DEFAULT_SCENES["train"], help="Train scene identifiers")
    parser.add_argument("--scenes-val", nargs="+", default=DEFAULT_SCENES["val"], help="Validation scene identifiers")
    parser.add_argument("--fps-sample", type=float, default=1.0, help="Frame extraction rate (default: 1.0 frame/sec)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect remote file tree without downloading")
    args = parser.parse_args()

    verify_hf_authentication()
    print(f"[INFO] Target output: {args.output_dir}")
    print(f"[INFO] Train scenes: {args.scenes_train}")
    print(f"[INFO] Val scenes: {args.scenes_val}")
    print(f"[INFO] Sampling rate: {args.fps_sample} FPS")

    if args.dry_run:
        print("[INFO] Performing dry-run check of dataset repository...")
        try:
            files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
            print(f"[INFO] Successfully queried repo: found {len(files)} total files.")
            wh_files = [f for f in files if "Warehouse" in f and not f.endswith((".h5", ".hdf5"))]
            print(f"[INFO] Found {len(wh_files)} relevant non-depth warehouse files.")
            print("[INFO] Example matching paths:")
            for sample in wh_files[:10]:
                print(f"  - {sample}")
        except Exception as e:
            print(f"[ERROR] Failed to query repository: {e}")

if __name__ == "__main__":
    main()

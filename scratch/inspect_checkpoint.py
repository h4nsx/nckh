#!/usr/bin/env python3
"""Inspect a PyTorch checkpoint and report key information.
Outputs:
- audit/ckpt_state_dict.txt : list of all top‑level keys in the checkpoint.
- audit/detection_head_status.txt : whether detection‑head weights are present or discarded.
"""
import torch, json, os
from pathlib import Path

def main():
    # Locate the checkpoint(s) used for the arms. For this audit we consider the SS2D warm‑start checkpoint.
    ckpt_path = Path('models/yolo11s_ss2d_full_warmstart.pt')
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # The checkpoint may be a dict with 'model' key or raw state_dict.
    if isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt):
        state = ckpt.get('model', ckpt.get('state_dict'))
    else:
        state = ckpt
    # Write all top‑level keys
    keys_path = Path('audit/ckpt_state_dict.txt')
    with keys_path.open('w') as f:
        for k in sorted(state.keys()):
            f.write(k + "\n")
    # Detect detection‑head tensors (common prefixes: "detect", "head", "bbox_head")
    head_keys = [k for k in state.keys() if any(p in k.lower() for p in ['detect', 'head', 'bbox'])]
    head_path = Path('audit/detection_head_status.txt')
    with head_path.open('w') as f:
        if head_keys:
            f.write('DETECTION_HEAD_PRESENT\n')
            for hk in head_keys:
                f.write(f"  {hk}\n")
        else:
            f.write('DETECTION_HEAD_DISCARDED\n')
    print('Inspection completed. Files written to audit/')

if __name__ == '__main__':
    main()


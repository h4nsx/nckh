#!/usr/bin/env python3
"""Safely inspect a YOLO/Ultralytics checkpoint.
It loads with `weights_only=False` (trusted source) and registers Ultralytics globals.
Outputs:
- audit/ckpt_state_dict.txt (list of keys)
- audit/detection_head_status.txt (presence of detection head tensors)
"""
import torch, os
from pathlib import Path

def safe_load(path):
    # Allow Ultralytics classes as safe globals
    try:
        import ultralytics
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    except Exception:
        pass
    # Load with weights_only=False (trusted checkpoint)
    return torch.load(path, map_location='cpu', weights_only=False)

def main():
    ckpt_path = Path('models/yolo11s_ss2d_full_warmstart.pt')
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = safe_load(ckpt_path)
    # Determine if dict has a nested state_dict
    if isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt):
        state = ckpt.get('model', ckpt.get('state_dict'))
    else:
        state = ckpt
    # Write all keys
    keys_path = Path('audit/ckpt_state_dict.txt')
    with keys_path.open('w') as f:
        for k in sorted(state.keys()):
            f.write(k + '\n')
    # Detect detection‑head tensors (common patterns)
    head_keys = [k for k in state.keys() if any(p in k.lower() for p in ['detect', 'head', 'bbox'])]
    head_path = Path('audit/detection_head_status.txt')
    with head_path.open('w') as f:
        if head_keys:
            f.write('DETECTION_HEAD_PRESENT\n')
            for hk in head_keys:
                f.write(f"  {hk}\n")
        else:
            f.write('DETECTION_HEAD_DISCARDED\n')
    print('Checkpoint inspection completed.')

if __name__ == '__main__':
    main()


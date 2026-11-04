#!/usr/bin/env python3
"""Inspect a YOLO/Ultralytics checkpoint (dict format).

Outputs:
- audit/ckpt_state_dict.txt          – all parameter keys (sorted)
- audit/detection_head_status.txt    – presence of detection‑head tensors
"""
import sys
from pathlib import Path
import torch

# Make the repository importable (some checkpoints reference `models` inside)
repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / 'models'
for p in (repo_root, models_dir):
    if str(p) not in sys.path:
        sys.path.append(str(p))

def main():
    ckpt_path = Path('models/yolo11s_ss2d_full_warmstart.pt')
    if not ckpt_path.is_file():
        print(f'Checkpoint not found: {ckpt_path}')
        return

    # Load as a plain dict – this works for the provided checkpoint.
    try:
        import models  # noqa: F401
    except Exception as e:
        print(f"Warning: could not import models package: {e}")
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

    # 1️⃣ Write all sorted keys
    keys_path = Path('audit/ckpt_state_dict.txt')
    with keys_path.open('w') as f:
        for k in sorted(state.keys()):
            f.write(f'{k}\n')

    # 2️⃣ Detect detection‑head tensors
    head_keys = [k for k in state.keys()
                if any(p in k.lower() for p in ('detect', 'head', 'bbox'))]

    status_path = Path('audit/detection_head_status.txt')
    with status_path.open('w') as f:
        if head_keys:
            f.write('DETECTION_HEAD_PRESENT\n')
            for hk in head_keys:
                f.write(f'  {hk}\n')
        else:
            f.write('DETECTION_HEAD_DISCARDED\n')

    print('Checkpoint inspection completed.')

if __name__ == '__main__':
    main()
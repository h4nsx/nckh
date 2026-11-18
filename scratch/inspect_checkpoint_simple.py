#!/usr/bin/env python3
"""Simple safe checkpoint inspection using weights_only=True.
Outputs:
- audit/ckpt_state_dict.txt : list of all parameter keys (sorted).
- audit/detection_head_status.txt : indicates presence of detection head tensors.
"""
import torch
from pathlib import Path

def main():
    ckpt_path = Path('models/yolo11s_ss2d_full_warmstart.pt')
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    # Load only tensor weights (no class objects) – safe for large files.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # ckpt should now be a dict of tensors.
    if isinstance(ckpt, dict):
        state = ckpt
    elif hasattr(ckpt, 'state_dict'):
        state = ckpt.state_dict()
    else:
        print('Unsupported checkpoint format')
        return
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
    print('Simple checkpoint inspection completed.')

if __name__ == '__main__':
    main()


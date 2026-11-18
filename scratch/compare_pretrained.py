#!/usr/bin/env python3
"""Compare pretrained YOLO11s checkpoint with a SS2D checkpoint.
Outputs:
- audit/ckpt_transfer.csv : columns `key,status` where status can be
  MATCHED, MISSING_IN_SS2D, MISSING_IN_PRETRAINED, SHAPE_MISMATCH.
"""
import sys
import torch, csv, os
from pathlib import Path

# Ensure repository importable for any custom modules (e.g., `models`).
repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / "models"
for p in (repo_root, models_dir):
    if str(p) not in sys.path:
        sys.path.append(str(p))


def safe_load(path):
    """Load a checkpoint safely.
    Tries a weights‑only load first; if that fails, registers Ultralytics
    classes and imports the `models` package before loading with full globals.
    """
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except Exception:
        # Register Ultralytics safe globals.
        try:
            import ultralytics
            torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
        except Exception:
            pass
        # Import models to satisfy pickle globals.
        try:
            import models  # noqa: F401
        except Exception as e:
            print(f"Warning: cannot import models package: {e}")
        return torch.load(str(path), map_location='cpu', weights_only=False)

def load_state(path):
    ckpt = safe_load(path)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt):
        return ckpt.get('model', ckpt.get('state_dict'))
    return ckpt

def main():
    pretrained_path = Path('yolo11s.pt')
    ss2d_path = Path('models/yolo11s_ss2d_full_warmstart.pt')
    if not pretrained_path.is_file() or not ss2d_path.is_file():
        print('Checkpoint files missing.')
        return
    pre_state = load_state(pretrained_path)
    ss_state = load_state(ss2d_path)
    # Ensure we have dicts for key iteration
    if isinstance(pre_state, torch.nn.Module):
        pre_state = pre_state.state_dict()
    if isinstance(ss_state, torch.nn.Module):
        ss_state = ss_state.state_dict()
    out_path = Path('audit/ckpt_transfer.csv')
    with out_path.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key','status'])
        all_keys = set(pre_state.keys()) | set(ss_state.keys())
        for k in sorted(all_keys):
            if k in pre_state and k in ss_state:
                pre_val, ss_val = pre_state[k], ss_state[k]
                if isinstance(pre_val, torch.Tensor) and isinstance(ss_val, torch.Tensor):
                    status = 'MATCHED' if pre_val.shape == ss_val.shape else 'SHAPE_MISMATCH'
                else:
                    status = 'MATCHED'
            elif k in pre_state:
                status = 'MISSING_IN_SS2D'
            else:
                status = 'MISSING_IN_PRETRAINED'
            writer.writerow([k, status])
    print('Comparison written to', out_path)

if __name__ == '__main__':
    main()


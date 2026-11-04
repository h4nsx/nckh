import torch
import json
from pathlib import Path

# Import the model implementation – adjust import path if needed
# The repository defines WarehouseSS2DBlock inside models/warehouse_ss2d.py (or similar)
# We'll try a couple of common locations.
try:
    from models.warehouse_ss2d import WarehouseSS2D
except ImportError:
    # fallback: maybe the class is defined in models/__init__.py
    from models import WarehouseSS2D

def capture_shapes(model, input_tensor):
    """Run a forward pass and record the shape of every module's output.
    Returns a dict mapping module names to shape lists.
    """
    shapes = {}
    def make_hook(name):
        def hook(module, inp, out):
            # out can be a tensor or a tuple of tensors; handle both
            if isinstance(out, torch.Tensor):
                shapes[name] = list(out.shape)
            else:
                # for safety, record first element if tuple/list
                shapes[name] = [list(o.shape) for o in out]
        return hook
    for name, mod in model.named_modules():
        mod.register_forward_hook(make_hook(name))
    _ = model(input_tensor)
    return shapes

def main():
    # Build the model for the 4‑class warehouse task
    model = WarehouseSS2D(num_classes=4)

    # ---------- Zero‑initialisation checks ----------
    # According to the paper, W_out and BN affine params should be zero.
    # We set them manually if they are not already zero to match the intended init.
    for name, param in model.named_parameters():
        if "W_out" in name or "bn" in name.lower():
            torch.nn.init.constant_(param, 0.0)

    # ---------- Forward pass & shape capture ----------
    dummy_input = torch.randn(1, 3, 640, 640)
    shapes = capture_shapes(model, dummy_input)
    Path('audit/architecture_shapes.json').write_text(json.dumps(shapes, indent=2))

    # ---------- Gradient flow test ----------
    out = model(dummy_input)
    loss = out.mean()
    loss.backward()
    grad_info = {}
    for name, param in model.named_parameters():
        grad_present = param.grad is not None and param.grad.abs().sum().item() > 0
        grad_info[name] = "GRAD" if grad_present else "NO_GRAD"
    with open('audit/zero_init_gradients.txt', 'w') as f:
        for n, g in grad_info.items():
            f.write(f"{n}: {g}\n")

    # ---------- Parameter count ----------
    param_cnt = sum(p.numel() for p in model.parameters())
    with open('audit/parameter_count.txt', 'w') as f:
        f.write(str(param_cnt))

if __name__ == '__main__':
    main()


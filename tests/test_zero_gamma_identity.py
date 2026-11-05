import torch
import torch.nn as nn
from models.warehouse_ss2d import WarehouseSS2DBlock

def test_zero_gamma_identity_and_gradients():
    """
    Phase 5 / Task 5.2 Test Suite:
    Comprehensive verification of Zero-Gamma initialization and gradient dynamics:
    1. Exact step-0 identity mapping (||Y - X|| == 0.0) in both eval and train modes.
    2. Forward pass shape and numerical finiteness (no NaN / Inf).
    3. Initial backward pass: verifies that norm3.weight (gamma) receives a finite non-zero gradient.
    4. Post-step gradient propagation: after moving gamma away from 0.0 with an optimizer step,
       upstream SSM branch parameters (A_logs, conv2d, in_proj) receive active, non-zero gradients.
    5. Multi-step optimization stability: perturbation grows smoothly without exploding.
    """
    print("\n" + "=" * 70)
    print("TEST: Zero-Gamma Residual Identity, Gradient Flow & Multi-Step Training")
    print("=" * 70)

    torch.manual_seed(42)
    block = WarehouseSS2DBlock(c1=512, c2=512, d_state=16, d_conv=3, expand=0.5)

    # 1. Step 0 exact identity (eval mode and train mode)
    block.eval()
    x = torch.randn(2, 512, 10, 10)
    with torch.no_grad():
        y0_eval = block(x)
        diff_eval = (y0_eval - x).abs().max().item()
        assert diff_eval == 0.0, f"Step 0 identity failed in eval mode: diff = {diff_eval}"
    print(f"  [PASS] 1. Step 0 Identity (eval mode): max abs diff = {diff_eval:.2e}")

    block.train()
    with torch.no_grad():
        y0_train = block(x)
        diff_train = (y0_train - x).abs().max().item()
        assert diff_train == 0.0, f"Step 0 identity failed in train mode: diff = {diff_train}"
    print(f"  [PASS] 2. Step 0 Identity (train mode): max abs diff = {diff_train:.2e}")

    # 2. Forward validity: shape and finiteness
    assert y0_train.shape == x.shape, f"Shape mismatch: {y0_train.shape} vs {x.shape}"
    assert torch.isfinite(y0_train).all(), "Forward output contains NaN/Inf values"
    print(f"  [PASS] 3. Forward output shape and finiteness verified: shape={list(y0_train.shape)}")

    # 3. Initial backward pass: verify norm3.weight (gamma) receives non-zero gradient
    block.zero_grad()
    x_in = torch.randn(2, 512, 10, 10, requires_grad=True)
    y_out = block(x_in)
    target = torch.randn_like(y_out)
    loss = (y_out - target).pow(2).mean()
    loss.backward()

    assert block.norm3.weight.grad is not None, "norm3.weight.grad is None"
    gamma_grad_norm = block.norm3.weight.grad.abs().sum().item()
    assert torch.isfinite(block.norm3.weight.grad).all(), "norm3.weight gradient has NaN/Inf"
    assert gamma_grad_norm > 0.0, "norm3.weight (gamma) gradient is zero"
    print(f"  [PASS] 4. Initial backward pass: gamma gradient norm = {gamma_grad_norm:.4f} (finite & non-zero)")

    # 4. Post-step gradient propagation: update gamma and verify upstream SSM gradients
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01)
    optimizer.step()  # moves gamma away from 0.0

    # Second backward pass with gamma != 0
    optimizer.zero_grad()
    x_step2 = torch.randn(2, 512, 10, 10)
    y_step2 = block(x_step2)
    target2 = torch.randn_like(y_step2)
    loss2 = (y_step2 - target2).pow(2).mean()
    loss2.backward()

    # Upstream SSM parameters must now receive non-zero gradients
    conv_grad_norm = block.dw_conv.weight.grad.abs().sum().item()
    proj_grad_norm = block.in_proj.weight.grad.abs().sum().item()
    a_grad_norm = block.ss2d.A_logs.grad.abs().sum().item()

    assert torch.isfinite(block.dw_conv.weight.grad).all(), "dw_conv gradient contains NaN/Inf"
    assert torch.isfinite(block.in_proj.weight.grad).all(), "in_proj gradient contains NaN/Inf"
    assert torch.isfinite(block.ss2d.A_logs.grad).all(), "A_logs gradient contains NaN/Inf"
    assert conv_grad_norm > 0.0, "Upstream dw_conv gradient is zero after gamma update"
    assert proj_grad_norm > 0.0, "Upstream in_proj gradient is zero after gamma update"
    assert a_grad_norm > 0.0, "Upstream A_logs gradient is zero after gamma update"

    print(f"  [PASS] 5. Post-step gradient flow to SSM branch: dw_conv={conv_grad_norm:.4f}, in_proj={proj_grad_norm:.4f}, A_logs={a_grad_norm:.4f}")

    # 5. Multi-step optimization stability
    for step in range(2, 11):
        optimizer.zero_grad()
        x_m = torch.randn(2, 512, 10, 10)
        y_m = block(x_m)
        target_m = torch.randn_like(y_m)
        loss_m = (y_m - target_m).pow(2).mean()
        loss_m.backward()
        optimizer.step()

        with torch.no_grad():
            x_test = torch.randn(2, 512, 10, 10)
            y_test = block(x_test)
            rel_diff = ((y_test - x_test).abs() / (x_test.abs() + 1e-6)).mean().item()
            max_diff = (y_test - x_test).abs().max().item()

        assert rel_diff < 1.0, f"Step {step} exploded: rel_diff = {rel_diff}"
        if step in [2, 5, 10]:
            print(f"  [PASS] Multi-step {step:2d}: max diff = {max_diff:.4f}, mean relative diff = {rel_diff:.4e}")

    print("  [SUCCESS] All zero-gamma identity, gradient flow, and stability tests passed cleanly.\n")

if __name__ == "__main__":
    test_zero_gamma_identity_and_gradients()

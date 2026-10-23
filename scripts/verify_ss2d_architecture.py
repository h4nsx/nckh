#!/usr/bin/env python3
"""
Unit Test & Shape Verification Script for WarehouseSS2DBlock & YOLO11s-SS2D
--------------------------------------------------------------------------
Verifies the following requirements from yolo11s_ss2d_architecture_implementation.md:
1. Shape contract: Input [B, C, H, W] -> Output [B, C, H, W].
2. Finite numbers: No NaN or Inf in outputs.
3. Backward pass: Gradients propagate to all trainable parameters.
4. Batch size 1 compatibility.
5. Integration into YOLO11s model replacing Layer 10 (C2PSA) and full forward pass through Detect head.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.warehouse_ss2d import WarehouseSS2DBlock, PurePyTorch2DSelectiveScan
from ultralytics import YOLO

def test_isolated_ss2d_block():
    print("=" * 60)
    print("TEST 1: Isolated WarehouseSS2DBlock Shape & Gradient Contract")
    print("=" * 60)

    B, C, H, W = 2, 512, 20, 20
    block = WarehouseSS2DBlock(c1=C, c2=C, d_state=16, expand=0.5)
    
    # 1. Forward shape test
    x = torch.randn(B, C, H, W, requires_grad=True)
    y = block(x)
    
    assert y.shape == (B, C, H, W), f"Shape mismatch: expected {(B, C, H, W)}, got {y.shape}"
    print(f"  [PASS] Forward shape contract: {list(x.shape)} -> {list(y.shape)}")

    # 2. Finite numbers test
    assert not torch.isnan(y).any(), "NaN found in output"
    assert not torch.isinf(y).any(), "Inf found in output"
    print("  [PASS] Output tensor contains only finite numbers (zero NaN / Inf).")

    # 3. Backward gradient propagation test
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not propagate back to input x"
    print("  [PASS] Backward pass: gradient propagated cleanly to input tensor.")

    # 4. Check block parameters have gradients
    param_grads = [p.grad is not None for p in block.parameters() if p.requires_grad]
    assert all(param_grads), "Some block parameters did not receive gradients"
    print(f"  [PASS] All {len(param_grads)} trainable parameters received valid gradients.")

    # 5. Batch size 1 test
    x_b1 = torch.randn(1, C, H, W)
    y_b1 = block(x_b1)
    assert y_b1.shape == (1, C, H, W), f"Batch size 1 failed: {y_b1.shape}"
    print("  [PASS] Batch size 1 inference works as expected.")

def test_yolo11s_case_b_graph_replacement():
    print("\n" + "=" * 60)
    print("TEST 2: YOLO11s Case B (P5 Layer 10 Replacement) Graph Test")
    print("=" * 60)

    # Load baseline YOLO11s architecture
    yolo_model = YOLO("yolo11s.yaml")
    torch_model = yolo_model.model

    print(f"  Loaded baseline YOLO11s with {len(torch_model.model)} layers.")
    original_l10 = torch_model.model[10]
    print(f"  Original Layer 10: {original_l10.__class__.__name__}")

    # Verify original layer 10 is C2PSA
    assert original_l10.__class__.__name__ == "C2PSA", f"Expected Layer 10 to be C2PSA, got {original_l10.__class__.__name__}"

    # Instantiate Shape-Compatible SS2D Block
    # YOLO11s P5 channels = 512
    ss2d_block = WarehouseSS2DBlock(c1=512, c2=512, d_state=16, expand=0.5)

    # Ultralytics attaches model graph metadata: i (layer index), f (from index), type, np (num params)
    ss2d_block.i = original_l10.i
    ss2d_block.f = original_l10.f
    ss2d_block.type = "WarehouseSS2DBlock"
    ss2d_block.np = sum(p.numel() for p in ss2d_block.parameters())

    # Perform surgical replacement for Case B
    torch_model.model[10] = ss2d_block
    print(f"  Replaced Layer 10 with: {torch_model.model[10].__class__.__name__}")

    # Test full end-to-end forward pass with 640x640 input
    dummy_img = torch.randn(1, 3, 640, 640)
    torch_model.eval()
    with torch.no_grad():
        preds = torch_model(dummy_img)

    print("  [PASS] Full end-to-end forward pass through modified backbone, neck, and Detect head succeeded.")
    if isinstance(preds, (list, tuple)):
        for idx, p in enumerate(preds):
            if isinstance(p, torch.Tensor):
                print(f"    - Output[{idx}] tensor shape: {list(p.shape)}")
            elif isinstance(p, (list, tuple)):
                print(f"    - Output[{idx}] pyramid shapes: {[list(t.shape) for t in p if isinstance(t, torch.Tensor)]}")
    else:
        print(f"    - Single output tensor shape: {list(preds.shape)}")

if __name__ == "__main__":
    test_isolated_ss2d_block()
    test_yolo11s_case_b_graph_replacement()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED: Architecture contract & shape compatibility verified.")
    print("=" * 60)

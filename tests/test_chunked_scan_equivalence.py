import torch
from models.warehouse_ss2d import (
    PurePyTorch2DSelectiveScan,
    WarehouseSS2DBlock,
    WarehouseConvBlock,
    WarehouseSS2DBlock2Dir,
    set_scan_chunk_size,
)


def test_chunked_matches_sequential():
    """
    The chunked scan must reproduce the sequential reference recurrence (forward and backward),
    including when the sequence length is not a multiple of the chunk size.
    """
    print("\n" + "=" * 65)
    print("TEST: Chunked Scan Numerical Equivalence")
    print("=" * 65)

    torch.manual_seed(0)
    for (H, W), chunk in [((20, 20), 20), ((20, 20), 16), ((7, 9), 10), ((40, 40), 32)]:
        scan = PurePyTorch2DSelectiveScan(d_model=32, d_state=16).double()
        x = torch.randn(2, 32, H, W, dtype=torch.double, requires_grad=True)

        scan.chunk_size = 0
        y_seq = scan(x)
        g_seq = torch.autograd.grad(y_seq.square().sum(), [x] + list(scan.parameters()))

        scan.chunk_size = chunk
        y_chk = scan(x)
        g_chk = torch.autograd.grad(y_chk.square().sum(), [x] + list(scan.parameters()))

        max_diff = (y_seq - y_chk).abs().max().item()
        assert max_diff < 1e-8, f"Forward mismatch at {H}x{W}, chunk={chunk}: {max_diff}"
        for a, b in zip(g_seq, g_chk):
            assert torch.allclose(a, b, atol=1e-7), f"Gradient mismatch at {H}x{W}, chunk={chunk}"
        print(f"  [PASS] {H}x{W}, chunk={chunk}: max forward diff = {max_diff:.2e}")


def test_ablation_variants():
    """Shape contract and zero-gamma identity for the ablation blocks."""
    print("\n" + "=" * 65)
    print("TEST: Ablation Variants (Conv-only, 2-direction)")
    print("=" * 65)

    x = torch.randn(2, 512, 20, 20)
    for cls in (WarehouseSS2DBlock, WarehouseConvBlock, WarehouseSS2DBlock2Dir):
        block = cls(c1=512, c2=512, d_state=16, d_conv=3, expand=0.5).eval()
        with torch.no_grad():
            y = block(x)
        assert y.shape == x.shape, f"{cls.__name__} changed shape: {tuple(y.shape)}"
        assert (y - x).abs().max().item() == 0.0, f"{cls.__name__} is not identity at init"
        print(f"  [PASS] {cls.__name__}: shape preserved, identity at init")

    block = WarehouseSS2DBlock(c1=512, c2=512, d_state=16, d_conv=3, expand=0.5)
    assert set_scan_chunk_size(block, 20) == 1
    assert block.ss2d.chunk_size == 20
    print("  [PASS] set_scan_chunk_size switches the scan mode")


if __name__ == "__main__":
    test_chunked_matches_sequential()
    test_ablation_variants()

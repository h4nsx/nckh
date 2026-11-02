import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PurePyTorch2DSelectiveScan(nn.Module):
    """
    Vectorized pure PyTorch 4-direction 2D Selective Scan (v2 - Spatially Aligned).
    Runs on CPU or GPU without requiring custom CUDA compilation.

    Directional Scan Mapping:
    1. Direction 1: Row-major, forward (Top-down, left-to-right)
    2. Direction 2: Row-major, reversed (Bottom-up, right-to-left)
    3. Direction 3: Column-major, forward (Left-to-right, top-down)
    4. Direction 4: Column-major, reversed (Right-to-left, bottom-up)
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Continuous state matrix parameters A and skip parameter D per direction
        self.A_logs = nn.Parameter(torch.randn(4, d_model, d_state))
        self.D = nn.Parameter(torch.ones(4, d_model))

        # Direction-specific parameter projections for (B, C, Delta)
        # Each direction projects from its own spatially aligned tokens:
        # Projection output size: 2 * d_state + d_model
        self.proj_params = nn.Linear(d_model, 2 * self.d_state + self.d_model, bias=False)

    def _selective_scan_1d(self, u, delta, A, B, C, D):
        """
        Scan 1D sequence of shape [B, L, D]
        u: [B, L, D]
        delta: [B, L, D] (strictly positive via softplus)
        A: [D, N]
        B: [B, L, N]
        C: [B, L, N]
        D: [D]
        """
        batch_size, seq_len, d_model = u.shape
        d_state = A.shape[-1]

        # Simplified ZOH discretization:
        # delta_A = exp(delta * A) where A has negative diagonal
        delta_A = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_model, d_state))
        # delta_B_u = (delta * u) * B
        delta_B_u = (delta.unsqueeze(-1) * u.unsqueeze(-1)) * B.unsqueeze(2)

        # Sequential scan along sequence length L
        states = []
        h = torch.zeros(batch_size, d_model, d_state, device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            states.append(y_t)

        y = torch.stack(states, dim=1)  # [B, L, D]
        # Skip connection D * u
        y = y + u * D.view(1, 1, d_model)
        return y

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        L = H * W

        # 1. Spatially aligned 4 directional sequences:
        # Direction 1: Row-major, forward [B, H*W, C]
        x_1 = x.permute(0, 2, 3, 1).reshape(B, L, C)
        # Direction 2: Row-major, reversed [B, H*W, C]
        x_2 = torch.flip(x_1, dims=[1])
        # Direction 3: Column-major, forward [B, W*H, C]
        x_3 = x.permute(0, 3, 2, 1).reshape(B, L, C)
        # Direction 4: Column-major, reversed [B, W*H, C]
        x_4 = torch.flip(x_3, dims=[1])

        # 2. Extract direction-specific continuous A matrices:
        A = -torch.exp(self.A_logs)

        # 3. Process each direction with strictly matched, self-projected parameters:
        # Direction 1:
        p_1 = self.proj_params(x_1)
        B_1, C_1, d_1 = torch.split(p_1, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_1 = self._selective_scan_1d(x_1, F.softplus(d_1), A[0], B_1, C_1, self.D[0])

        # Direction 2:
        p_2 = self.proj_params(x_2)
        B_2, C_2, d_2 = torch.split(p_2, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_2 = self._selective_scan_1d(x_2, F.softplus(d_2), A[1], B_2, C_2, self.D[1])
        # Restore row-major orientation
        y_2 = torch.flip(y_2, dims=[1])

        # Direction 3:
        p_3 = self.proj_params(x_3)
        B_3, C_3, d_3 = torch.split(p_3, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_3 = self._selective_scan_1d(x_3, F.softplus(d_3), A[2], B_3, C_3, self.D[2])
        # Restore from column-major [B, W, H, C] to row-major [B, H, W, C]
        y_3 = y_3.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        # Direction 4:
        p_4 = self.proj_params(x_4)
        B_4, C_4, d_4 = torch.split(p_4, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_4 = self._selective_scan_1d(x_4, F.softplus(d_4), A[3], B_4, C_4, self.D[3])
        # Reverse back and restore from column-major to row-major
        y_4 = torch.flip(y_4, dims=[1]).reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        # 4. Merge 4 directions and restore to [B, C, H, W]
        y = y_1 + y_2 + y_3 + y_4
        out = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class WarehouseSS2DBlock(nn.Module):
    """
    Drop-in replacement block for C2PSA at Layer 10 (P5 stage).
    Maintains [B, C, H, W] -> [B, C, H, W] exact contract.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Robust constructor argument resolution:
        # Case A: Explicit channels passed: (c1, c2, d_state=16, d_conv=3, expand=0.5)
        # Case B: Legacy 3-arg YAML tuple: (d_state=16, d_conv=3, expand=0.5)
        if len(args) == 5:
            c1, c2, d_state, d_conv, expand = args
        elif len(args) == 4:
            c1, c2, d_state, d_conv = args
            expand = kwargs.get("expand", 0.5)
        elif len(args) == 3:
            # Check if args are [c1, c2, d_state] or [d_state, d_conv, expand]
            if args[0] > 64 and args[1] > 64:
                c1, c2, d_state = args
                d_conv = kwargs.get("d_conv", 3)
                expand = kwargs.get("expand", 0.5)
            else:
                d_state, d_conv, expand = args
                c1 = kwargs.get("c1", 512)
                c2 = kwargs.get("c2", 512)
        elif len(args) == 2:
            c1, c2 = args
            d_state = kwargs.get("d_state", 16)
            d_conv = kwargs.get("d_conv", 3)
            expand = kwargs.get("expand", 0.5)
        elif len(args) == 1:
            c1 = c2 = args[0]
            d_state = kwargs.get("d_state", 16)
            d_conv = kwargs.get("d_conv", 3)
            expand = kwargs.get("expand", 0.5)
        else:
            c1 = kwargs.get("c1", 512)
            c2 = kwargs.get("c2", 512)
            d_state = kwargs.get("d_state", 16)
            d_conv = kwargs.get("d_conv", 3)
            expand = kwargs.get("expand", 0.5)

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = float(expand)
        self.d_inner = max(1, int(self.expand * self.c2))

        # Input projection & Depthwise conv for local context
        self.in_proj = nn.Conv2d(c1, self.d_inner, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.d_inner)
        self.act1 = nn.SiLU()

        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=False
        )
        self.norm2 = nn.BatchNorm2d(self.d_inner)
        self.act2 = nn.SiLU()

        # 2D Selective Scan core (Directionally Aligned)
        self.ss2d = PurePyTorch2DSelectiveScan(d_model=self.d_inner, d_state=d_state)

        # Output projection
        self.out_proj = nn.Conv2d(self.d_inner, c2, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(c2)

        # Zero-gamma initialization (Goyal et al., 2017):
        # Initializing norm3 gamma (weight) and beta (bias) to 0 ensures Y = X at step 0,
        # and crucially prevents variance amplification across subsequent training steps.
        nn.init.zeros_(self.norm3.weight)
        nn.init.zeros_(self.norm3.bias)

        # Residual connection
        self.has_residual = (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.in_proj(x)
        out = self.act1(self.norm1(out))

        out = self.dw_conv(out)
        out = self.act2(self.norm2(out))

        out = self.ss2d(out)

        out = self.norm3(self.out_proj(out))

        if self.has_residual:
            out = out + res

        return out

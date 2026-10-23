import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PurePyTorch2DSelectiveScan(nn.Module):
    """
    Vectorized pure PyTorch 4-direction 2D Selective Scan.
    Runs on CPU or GPU without requiring custom CUDA compilation.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Continuous state matrix parameters A and delta projection
        # Log parameterization ensures stability
        self.A_logs = nn.Parameter(torch.randn(4, d_model, d_state))
        self.D = nn.Parameter(torch.ones(4, d_model))

        # Input projections for B, C, and Delta across the 4 scan directions
        # Total channels = 4 * (2 * d_state + d_model)
        self.proj_params = nn.Linear(d_model, 4 * (2 * d_state + d_model), bias=False)

    def _selective_scan_1d(self, u, delta, A, B, C, D):
        """
        Scan 1D sequence of shape [B, L, D]
        u: [B, L, D]
        delta: [B, L, D]
        A: [D, N]
        B: [B, L, N]
        C: [B, L, N]
        D: [D]
        """
        batch_size, seq_len, d_model = u.shape
        d_state = A.shape[-1]

        # Discretize: deltaA = exp(delta * A)
        # delta: [B, L, D, 1], A: [1, 1, D, N] -> deltaA: [B, L, D, N]
        delta_A = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_model, d_state))
        # deltaB_u: delta * B * u -> [B, L, D, N]
        delta_B_u = (delta.unsqueeze(-1) * u.unsqueeze(-1)) * B.unsqueeze(2)

        # Sequential or associative scan across sequence length L
        # In P5 feature map, L = H * W = 20 * 20 = 400
        states = []
        h = torch.zeros(batch_size, d_model, d_state, device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            # y_t = (h_t * C_t).sum(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            states.append(y_t)

        y = torch.stack(states, dim=1) # [B, L, D]
        y = y + u * D.view(1, 1, d_model)
        return y

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        L = H * W

        # 4 scan sequences:
        # 1. Top-down, left-to-right (standard row-major)
        x_1 = x.permute(0, 2, 3, 1).reshape(B, L, C)
        # 2. Bottom-up, right-to-left (reversed row-major)
        x_2 = torch.flip(x_1, dims=[1])
        # 3. Left-to-right, top-down (column-major)
        x_3 = x.permute(0, 3, 2, 1).reshape(B, L, C)
        # 4. Right-to-left, bottom-up (reversed column-major)
        x_4 = torch.flip(x_3, dims=[1])

        # Project parameters
        # Shared or direction-specific parameterization
        # Use average feature map across directions to condition params
        p1 = self.proj_params(x_1) # [B, L, 4*(2N + C)]
        splits = torch.split(p1, [self.d_state, self.d_state, self.d_model] * 4, dim=-1)

        # Compute A from logs
        A = -torch.exp(self.A_logs)

        # Direction 1
        B_1, C_1, delta_1 = splits[0], splits[1], F.softplus(splits[2])
        y_1 = self._selective_scan_1d(x_1, delta_1, A[0], B_1, C_1, self.D[0])

        # Direction 2
        B_2, C_2, delta_2 = splits[3], splits[4], F.softplus(splits[5])
        y_2 = self._selective_scan_1d(x_2, delta_2, A[1], B_2, C_2, self.D[1])
        y_2 = torch.flip(y_2, dims=[1])

        # Direction 3
        B_3, C_3, delta_3 = splits[6], splits[7], F.softplus(splits[8])
        y_3 = self._selective_scan_1d(x_3, delta_3, A[2], B_3, C_3, self.D[2])
        y_3 = y_3.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        # Direction 4
        B_4, C_4, delta_4 = splits[9], splits[10], F.softplus(splits[11])
        y_4 = self._selective_scan_1d(x_4, delta_4, A[3], B_4, C_4, self.D[3])
        y_4 = torch.flip(y_4, dims=[1])
        y_4 = y_4.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        # Combine 4 directions
        y = y_1 + y_2 + y_3 + y_4
        # Restore [B, C, H, W]
        out = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class WarehouseSS2DBlock(nn.Module):
    """
    Drop-in replacement block for C2PSA at Layer 10 (P5 stage).
    Maintains [B, C, H, W] -> [B, C, H, W] exact contract.
    """
    def __init__(self, d_state: int = 16, d_conv: int = 3, expand: float = 0.5, c1: int = 512, c2: int = 512):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.d_inner = int(expand * c2)

        # Input projection & Depthwise conv for local context
        self.in_proj = nn.Conv2d(c1, self.d_inner, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.d_inner)
        self.act1 = nn.SiLU()

        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=False
        )
        self.norm2 = nn.BatchNorm2d(self.d_inner)
        self.act2 = nn.SiLU()

        # 2D Selective Scan core
        self.ss2d = PurePyTorch2DSelectiveScan(d_model=self.d_inner, d_state=d_state)

        # Output projection with identity-preserving zero initialization
        self.out_proj = nn.Conv2d(self.d_inner, c2, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(c2)

        # Zero-initialize the final projection so the block acts as identity pass-through at start
        nn.init.zeros_(self.out_proj.weight)
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def _scan_loop(delta_A: torch.Tensor, delta_B_u: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Sequential recurrence h_t = dA_t * h_{t-1} + dBu_t, y_t = <h_t, C_t>, as a scripted loop.
    delta_A, delta_B_u: [B, L, D, N]; C: [B, L, N]; returns [B, L, D]."""
    h = torch.zeros_like(delta_B_u[:, 0])
    ys = torch.zeros_like(delta_B_u[:, :, :, 0])
    for t in range(delta_A.size(1)):
        h = delta_A[:, t] * h + delta_B_u[:, t]
        ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
    return ys


class PurePyTorch2DSelectiveScan(nn.Module):
    """
    Pure PyTorch 4-direction 2D Selective Scan (v2 - Spatially Aligned).
    Runs on CPU or GPU without requiring custom CUDA compilation.

    Directional Scan Mapping:
    1. Direction 1: Row-major, forward (Top-down, left-to-right)
    2. Direction 2: Row-major, reversed (Bottom-up, right-to-left)
    3. Direction 3: Column-major, forward (Left-to-right, top-down)
    4. Direction 4: Column-major, reversed (Right-to-left, bottom-up)

    chunk_size: -2 runs the same recurrence as a parallel associative scan
    (ceil(log2 L) whole-sequence steps); -1 runs it as one scripted loop (one ONNX Loop);
    0 runs the reference sequential recurrence (one step per token,
    which unrolls into ~L graph blocks on ONNX export). A positive value runs the
    mathematically equivalent chunked scan, which loops over ceil(L / chunk_size)
    chunks and resolves each chunk in closed form. Parameters are identical, so
    a trained checkpoint can be evaluated with either mode.
    n_directions: 4 (row and column, both ways) or 2 (row-major only, both ways).

    Ultralytics rebuilds models from YAML inside the trainer, so per-instance settings
    are lost there; set the class attribute `default_chunk_size` before training to
    choose the scan mode for every instance created afterwards.
    """
    default_chunk_size = 0

    def __init__(self, d_model: int, d_state: int = 16, chunk_size: int = None, n_directions: int = 4):
        super().__init__()
        assert n_directions in (2, 4), "n_directions must be 2 or 4"
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = self.default_chunk_size if chunk_size is None else chunk_size
        self.n_directions = n_directions

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

        # Continuous-to-discrete parameter mapping:
        # Exact matrix exponential for diagonal transition: delta_A = exp(delta * A)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_model, d_state))
        # First-order Euler input modulation approximation: delta_B_u = (delta * u) * B
        delta_B_u = (delta.unsqueeze(-1) * u.unsqueeze(-1)) * B.unsqueeze(2)

        # Checkpoints pickled before chunk_size existed fall back to the class default
        chunk_size = getattr(self, "chunk_size", self.default_chunk_size)
        if chunk_size == -2:
            # Same recurrence evaluated as a parallel (Hillis-Steele) associative scan:
            # ceil(log2 L) whole-sequence steps instead of L dependent steps (runtime-only
            # intervention on the graph form; see _parallel_scan).
            h = self._parallel_scan(delta_A, delta_B_u)
            y = (h * C.unsqueeze(2)).sum(dim=-1)
            return y + u * D.view(1, 1, d_model)
        if chunk_size == -1:
            # Same sequential recurrence and arithmetic, but as one scripted loop that the
            # TorchScript-based ONNX exporter emits as a single ONNX Loop node instead of
            # unrolling it (runtime-only intervention on the graph form; see _scan_loop).
            y = _scan_loop(delta_A, delta_B_u, C)
            return y + u * D.view(1, 1, d_model)
        if chunk_size and chunk_size > 0:
            log_A = delta.unsqueeze(-1) * A.view(1, 1, d_model, d_state)  # log(delta_A), <= 0
            y = self._chunked_scan(log_A, delta_B_u, C, chunk_size)
            return y + u * D.view(1, 1, d_model)

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

    @staticmethod
    def _parallel_scan(a, b):
        """
        Inclusive parallel scan of h_t = a_t * h_{t-1} + b_t (h_0 = 0) over dim 1.
        Uses the associative operator (a1, b1) o (a2, b2) = (a1 * a2, a2 * b1 + b2) with the
        Hillis-Steele schedule: at offset k each position combines with position t - k, so
        after ceil(log2 L) steps b_t holds h_t. Positions t < k combine with the identity
        element (a = 1, b = 0). Only products of decays in (0, 1] are formed, so the scan is
        as stable as the sequential recurrence. a, b: [B, L, D, N]; returns h: [B, L, D, N].
        Work is O(L log L) instead of O(L), but there is no dependency chain of length L.
        """
        seq_len = b.shape[1]
        k = 1
        while k < seq_len:
            b_prev = F.pad(b[:, : seq_len - k], (0, 0, 0, 0, k, 0), value=0.0)
            b = a * b_prev + b
            if 2 * k < seq_len:  # the decay products are not needed after the last step
                a_prev = F.pad(a[:, : seq_len - k], (0, 0, 0, 0, k, 0), value=1.0)
                a = a * a_prev
            k *= 2
        return b

    @staticmethod
    def _chunked_scan(log_A, bu, C, chunk_size):
        """
        Chunked closed-form evaluation of h_t = exp(log_A_t) * h_{t-1} + bu_t.
        Within a chunk, with S_t = cumsum(log_A)_t:
            h_t = exp(S_t) * h_prev + sum_{s<=t} exp(S_t - S_s) * bu_s
        log_A, bu: [B, L, D, N]; C: [B, L, N]; returns y: [B, L, D]
        """
        batch_size, seq_len, d_model, d_state = bu.shape
        h_prev = torch.zeros(batch_size, d_model, d_state, device=bu.device, dtype=bu.dtype)
        outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            T = end - start
            cs = torch.cumsum(log_A[:, start:end], dim=1)               # [B, T, D, N]
            seg = cs.unsqueeze(2) - cs.unsqueeze(1)                     # [B, T(t), T(s), D, N]
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=bu.device))
            seg = seg.masked_fill(~causal.view(1, T, T, 1, 1), float("-inf"))
            h_intra = (torch.exp(seg) * bu[:, start:end].unsqueeze(1)).sum(dim=2)
            h = h_intra + torch.exp(cs) * h_prev.unsqueeze(1)           # [B, T, D, N]
            outputs.append((h * C[:, start:end].unsqueeze(2)).sum(dim=-1))
            h_prev = h[:, -1]
        return torch.cat(outputs, dim=1)

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

        # 3. Process each direction with strictly matched, self-projected parameters.
        # proj_params acts per token and the four sequences are permutations of the same tokens, so
        # with hoist_proj=True the projection is computed once and permuted like the inputs; this is
        # mathematically identical and costs 1/4 of the MACs (runtime-only option, same weights).
        hoist = getattr(self, "hoist_proj", False)
        if hoist:
            p_row = self.proj_params(x_1)                                         # [B, L, P], row-major
            p_col = p_row.reshape(B, H, W, -1).permute(0, 2, 1, 3).reshape(B, L, -1)  # column-major
        # Direction 1:
        p_1 = p_row if hoist else self.proj_params(x_1)
        B_1, C_1, d_1 = torch.split(p_1, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_1 = self._selective_scan_1d(x_1, F.softplus(d_1), A[0], B_1, C_1, self.D[0])

        # Direction 2:
        p_2 = torch.flip(p_row, dims=[1]) if hoist else self.proj_params(x_2)
        B_2, C_2, d_2 = torch.split(p_2, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_2 = self._selective_scan_1d(x_2, F.softplus(d_2), A[1], B_2, C_2, self.D[1])
        # Restore row-major orientation
        y_2 = torch.flip(y_2, dims=[1])

        if getattr(self, "n_directions", 4) == 2:
            y = y_1 + y_2
            return y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Direction 3:
        p_3 = p_col if hoist else self.proj_params(x_3)
        B_3, C_3, d_3 = torch.split(p_3, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_3 = self._selective_scan_1d(x_3, F.softplus(d_3), A[2], B_3, C_3, self.D[2])
        # Restore from column-major [B, W, H, C] to row-major [B, H, W, C]
        y_3 = y_3.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        # Direction 4:
        p_4 = torch.flip(p_col, dims=[1]) if hoist else self.proj_params(x_4)
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


class WarehouseConvBlock(WarehouseSS2DBlock):
    """
    Ablation control: identical to WarehouseSS2DBlock (1x1 conv, DWConv, out_proj,
    zero-gamma BN, residual) but with the selective scan replaced by identity.
    Isolates the contribution of the SSM from that of the surrounding convolutions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ss2d = nn.Identity()


class WarehouseSS2DBlock2Dir(WarehouseSS2DBlock):
    """
    Ablation: row-major scan only (forward and reversed). A_logs and D keep their
    4-direction shape for checkpoint compatibility; rows 2-3 are unused, so report
    the parameter count of this variant from its trainable, gradient-receiving tensors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ss2d.n_directions = 2


def set_scan_chunk_size(model: nn.Module, chunk_size: int) -> int:
    """Switch every selective scan in `model` to sequential (0) or chunked mode. Returns count."""
    count = 0
    for m in model.modules():
        if isinstance(m, PurePyTorch2DSelectiveScan):
            m.chunk_size = chunk_size
            count += 1
    return count

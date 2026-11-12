"""Rotary position embedding without complex numbers.

FlashHead applies RoPE by viewing each (even, odd) feature pair as a float64
complex number and multiplying by a unit complex. That is two float64 complex
multiplies per transformer layer, which consumer GPUs do slowly and which
torch.compile cannot generate code for at all (it leaves the op, and the graph
break around it, in eager mode).

The same rotation in real float32 arithmetic:

    even' = even * cos - odd * sin
    odd'  = even * sin + odd * cos

The result is cast back to the model's bfloat16 either way, so the two agree
to within bfloat16 rounding.
"""
from __future__ import annotations

import torch


def install(model, grid: tuple[int, int, int] | None = None):
    """Swap the vendor `rope_apply` for the real-valued one. The token grid
    (latent frames, height, width) is fixed for a given output size, so the
    cos/sin tables are built once; pass `grid` to build them up front."""
    import flash_head.src.modules.flash_head_model as fm

    tables: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def build(freqs, f, h, w, c):
        parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        grid = torch.cat([
            parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1)
        return grid.real.float().contiguous(), grid.imag.float().contiguous()

    def rope_apply(x, freqs, grid_sizes, *_unused, **_kw):
        # x: [1, L, heads, C]; rotate each (even, odd) pair by its position angle
        f, h, w = int(grid_sizes[0]), int(grid_sizes[1]), int(grid_sizes[2])
        c = x.size(3) // 2
        key = (f, h, w, c)
        if key not in tables:
            tables[key] = build(freqs.to(x.device), f, h, w, c)
        cos, sin = tables[key]
        pairs = x.float().unflatten(-1, (c, 2))
        even, odd = pairs[..., 0], pairs[..., 1]
        out = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return out.flatten(-2).to(x.dtype)

    if grid is not None:
        # build ahead of time so the compiled graph never sees the dict change
        c = model.dim // model.blocks[0].self_attn.num_heads // 2
        tables[(*grid, c)] = build(model.freqs.to(next(model.parameters()).device), *grid, c)

    fm.rope_apply = rope_apply
    return model

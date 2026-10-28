"""HK-RoPE: the rotary encoding of the joint axis.

An MLP maps the heat-kernel signature of each joint (src/data/skeleton_spectral.py) to head_dim / 2 angles. Queries and
keys of the spatial attention are rotated by these angles after the per-head RMS normalisation, so the attention logit
between two joints depends on the difference of their angles. Channel i is paired with channel i + head_dim / 2.
"""
import torch
import torch.nn as nn

from src.data.skeleton_spectral import hks_scales


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """(a, b) -> (-b, a) for the two halves of the last dimension."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat((-second, first), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate q and k by the angles behind (cos, sin); the arithmetic is float32, the result has q's dtype."""
    out_dtype = q.dtype
    q32, k32 = q.float(), k.float()
    q_rot = (q32 * cos) + (rotate_pairs(q32) * sin)
    k_rot = (k32 * cos) + (rotate_pairs(k32) * sin)
    return q_rot.to(out_dtype), k_rot.to(out_dtype)


class HeatKernelAngles(nn.Module):
    """MLP from the signature at `num_scales` diffusion times to `out_dim` angles. The diffusion times are stored as a
    buffer, so weights trained with other times are refused when they are loaded."""

    def __init__(self, num_scales: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.num_scales = num_scales
        self.net = nn.Sequential(nn.Linear(num_scales, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim))
        self.register_buffer("scales", torch.tensor(hks_scales(num_scales), dtype=torch.float32))
        self._register_load_state_dict_pre_hook(self._check_scales)

    def _check_scales(self, state_dict, prefix, *args):
        key = prefix + "scales"
        if key in state_dict:
            loaded = state_dict[key].detach().cpu().to(self.scales.dtype)
            if not torch.equal(loaded, self.scales.detach().cpu()):
                raise RuntimeError(f"{key}: weights use diffusion times {loaded.tolist()}, "
                                   f"this model {self.scales.tolist()}")

    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        """[B, J, S] -> [B, J, out_dim]"""
        return self.net(signature)


def _xavier_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class HKRoPE(nn.Module):
    """Heat-kernel signature -> (cos, sin) of one angle per channel pair, each [B, J, head_dim]."""

    def __init__(self, head_dim: int, num_scales: int = 8, hidden_dim: int = 64):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.num_scales = num_scales
        self.spectral_encoder = HeatKernelAngles(num_scales=num_scales, hidden_dim=hidden_dim, out_dim=head_dim // 2)
        self.apply(_xavier_linear)

    def cos_sin(self, signature: torch.Tensor):
        half = self.spectral_encoder(signature)            # [B, J, head_dim / 2]
        angles = torch.cat([half, half], dim=-1)           # [B, J, head_dim]
        return angles.cos(), angles.sin()

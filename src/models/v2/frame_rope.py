"""Rotary encoding of the frame axis: frame f is rotated by f * omega_m, omega_m = base^(-2m / head_dim)."""
import torch


def frame_rope_cos_sin(n_pos: int, head_dim: int, base: float, *, device=None, dtype=torch.float32):
    """(cos, sin), each [n_pos, head_dim], for frames 0 .. n_pos-1; channel i is paired with i + head_dim / 2."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if not base > 1.0:
        raise ValueError(f"base must be > 1, got {base}")
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    omega = 1.0 / (base ** exponents)                      # [head_dim / 2]
    frames = torch.arange(n_pos, device=device, dtype=torch.float32)
    angles = torch.outer(frames, omega)                    # [n_pos, head_dim / 2]
    angles = torch.cat([angles, angles], dim=-1)           # [n_pos, head_dim]
    return angles.cos().to(dtype), angles.sin().to(dtype)

"""The TopX denoiser, a factorised joint-by-frame transformer trained by flow matching, and its sampler.

Every (frame, joint) cell is one token. Each block attends over the frames of every joint (frame RoPE), then over the
joints of every frame (HK-RoPE and additive skeleton-graph biases), then applies an MLP; every sub-layer is
AdaLN-modulated by the diffusion time and the caption embedding. The sequence is [rest demo | target]: the rest frame
is given clean and the target frames are generated from noise.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from src.models.v2.hk_rope import HKRoPE, apply_rope
from src.models.v2.frame_rope import frame_rope_cos_sin


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding [B, dim] of the times t [B]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1])) if dim % 2 else emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention over the second-to-last axis, with optional per-head RMS normalisation of q
    and k, a rotary encoding, an additive bias and a key-padding mask.
    """

    def __init__(self, dim: int, n_heads: int, qk_norm: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.dh, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.dh, eps=1e-6)

    def forward(self, x, attn_bias=None, key_pad=None, rope=None):
        """x [..., N, D]; attn_bias broadcastable to [..., H, N, N]; key_pad [..., N] (True = valid);
        rope = (cos, sin), each broadcastable to [..., H, N, D / H].
        """
        N, D = x.shape[-2], x.shape[-1]
        lead = x.shape[:-2]
        qkv = self.qkv(x).reshape(*lead, N, 3, self.h, self.dh)
        q, k, v = (t.transpose(-3, -2) for t in qkv.movedim(-3, 0))
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, k, rope[0], rope[1])
        bias = attn_bias
        if key_pad is not None:
            m = torch.zeros(*key_pad.shape[:-1], 1, 1, N, device=x.device, dtype=q.dtype)
            m = m.masked_fill(~key_pad[..., None, None, :], torch.finfo(q.dtype).min)
            bias = m if bias is None else bias + m
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(o.transpose(-3, -2).reshape(*lead, N, D))


class Block(nn.Module):
    """One factorised block: frame attention -> joint attention -> MLP, every sub-layer AdaLN-modulated by the
    conditioning vector c.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False):
        """The AdaLN weights start small and non-zero and the three gates start at 1, so a block starts as an
        ordinary residual block that the conditioning already reaches.
        """
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.t_attn = Attention(dim, n_heads, qk_norm=qk_norm)
        self.s_attn = Attention(dim, n_heads, qk_norm=qk_norm)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.normal_(self.ada[-1].weight, std=0.5); nn.init.zeros_(self.ada[-1].bias)
        with torch.no_grad():
            for k in (2, 5, 8):
                self.ada[-1].bias[k * dim:(k + 1) * dim].fill_(1.0)

    def forward(self, x, c, joint_bias=None, frame_valid=None, joint_valid=None, rope_cos=None, rope_sin=None,
                trope_cos=None, trope_sin=None):
        """x [B, T, J, D], c [B, D]; rope_cos / rope_sin reach the joint attention, trope_cos / trope_sin the
        frame attention.
        """
        B, T, J, D = x.shape
        p = self.ada(c)
        (st, sc, sg, st2, sc2, sg2, mt, mc, mg) = p.chunk(9, dim=-1)
        e = lambda v: v[:, None, None, :]

        y = modulate(self.n1(x), e(st), e(sc)).permute(0, 2, 1, 3).reshape(B * J, T, D)
        fv = frame_valid.repeat_interleave(J, 0) if frame_valid is not None else None
        y = self.t_attn(y, key_pad=fv, rope=(trope_cos, trope_sin) if trope_cos is not None else None
                        ).reshape(B, J, T, D).permute(0, 2, 1, 3)
        x = x + e(sg) * y

        y = modulate(self.n2(x), e(st2), e(sc2))
        jb = joint_bias
        if jb is not None:
            jb = jb[:, None, None] if jb.dim() == 3 else jb[:, None]
        jv = None if jb is not None else \
            (joint_valid[:, None] if joint_valid is not None else None)
        y = self.s_attn(y, attn_bias=jb, key_pad=jv, rope=(rope_cos, rope_sin) if rope_cos is not None else None)
        x = x + e(sg2) * y

        x = x + e(mg) * self.mlp(modulate(self.n3(x), e(mt), e(mc)))
        return x


class InContextMotionDiT(nn.Module):
    """Motion tokens [B, T, J, C] -> predicted clean motion [B, T, J, C].

    The diffusion time and the caption embedding modulate every block (AdaLN). Per joint, the token receives the
    joint-description embedding and the structural and world-frame rest features (`use_struct_feats`,
    `struct_world_rest`); the joint attention receives the skeleton graph as additive biases (`joint_bias` from the
    collated batch, plus learnt up/down hop embeddings with `use_dir_bias`) and HK-RoPE (`use_spec_rope`,
    `spec_rope_hks`). With `use_temporal_rope` the frame axis is encoded by RoPE instead of a learnt table.
    """

    def __init__(self, in_ch=13, dim=256, depth=6, n_heads=8, d_text=4096, d_blueprint=16,
                 d_joint_sem=4096, mlp_ratio=4.0, use_struct_feats=False, use_dir_bias=False,
                 local_root_dim=0, use_ref_text=False, grad_ckpt=False, qk_norm=False, use_geo_bias=True,
                 use_spec_rope=False, spec_rope_k=8, spec_rope_hks=False, use_temporal_rope=False, trope_base=700.0,
                 struct_world_rest=False):
        super().__init__()
        self.grad_ckpt = bool(grad_ckpt)
        self.dim = dim
        self.n_heads = n_heads
        self.x_in = nn.Linear(in_ch, dim)
        self.use_struct_feats = bool(use_struct_feats)
        self.struct_world_rest = bool(struct_world_rest)
        if self.struct_world_rest and not self.use_struct_feats:
            raise ValueError("struct_world_rest extends the struct_feats table: it needs use_struct_feats=True")
        self.struct_in = 8 + (6 if self.struct_world_rest else 0)
        self.use_dir_bias = bool(use_dir_bias)
        self.use_geo_bias = bool(use_geo_bias)
        self.use_spec_rope = bool(use_spec_rope)
        self.spec_rope_k = int(spec_rope_k)
        self.spec_rope_hks = bool(spec_rope_hks)
        if self.spec_rope_hks and not self.use_spec_rope:
            raise ValueError("spec_rope_hks needs use_spec_rope=True: the heat-kernel signature is the spectral RoPE's "
                             "coordinate, not a module of its own")
        self.use_temporal_rope = bool(use_temporal_rope)
        if self.use_temporal_rope:
            self.trope_base_value = float(trope_base)
            self._register_load_state_dict_pre_hook(self._refuse_trope_base_drift)
        if not self.use_geo_bias:
            self.register_buffer("geo_bias_off", torch.ones(()))
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.max_T, self.max_J = 4096, 160
        t_pos = torch.zeros(1, self.max_T, 1, dim)
        j_pos = torch.zeros(1, 1, self.max_J, dim)
        nn.init.normal_(t_pos, std=0.02); nn.init.normal_(j_pos, std=0.02)
        if not self.use_temporal_rope:
            self.t_pos = nn.Parameter(t_pos)
        else:
            self.register_buffer("trope_base", torch.tensor(float(trope_base)))
        if not self.use_spec_rope:
            self.j_pos = nn.Parameter(j_pos)
        self.joint_sem = nn.Linear(d_joint_sem, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.text_mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.bp_mlp = nn.Sequential(nn.Linear(d_blueprint, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_ratio, qk_norm=qk_norm)
                                     for _ in range(depth)])
        self.n_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch)
        nn.init.normal_(self.ada_out[-1].weight, std=0.5); nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.normal_(self.out.weight, std=0.02); nn.init.zeros_(self.out.bias)

        if self.use_struct_feats:
            self.struct_mlp = nn.Sequential(nn.Linear(8, dim), nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.struct_mlp[-1].weight)
            nn.init.zeros_(self.struct_mlp[-1].bias)
        if self.use_dir_bias:
            self.e_up = nn.Embedding(16, n_heads)
            self.e_down = nn.Embedding(16, n_heads)
            nn.init.zeros_(self.e_up.weight)
            nn.init.zeros_(self.e_down.weight)
        self.use_ref_text = bool(use_ref_text)
        if self.use_ref_text:
            self.ref_text_mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, dim),
                                              nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.ref_text_mlp[-1].weight)
            nn.init.zeros_(self.ref_text_mlp[-1].bias)
        self.local_root_mlp = None
        if local_root_dim > 0:
            self.local_root_mlp = nn.Linear(local_root_dim, dim)
            nn.init.zeros_(self.local_root_mlp.weight)
            nn.init.zeros_(self.local_root_mlp.bias)
        self.spec_rope = None
        if self.use_spec_rope:
            if not self.spec_rope_hks:
                raise ValueError("this release encodes the joint axis with the heat-kernel signature only "
                                 "(spec_rope_hks=True)")
            self.spec_rope = HKRoPE(head_dim=dim // n_heads, num_scales=self.spec_rope_k)
        self.struct_rest_in = None
        if self.struct_world_rest:
            self.struct_rest_in = nn.Linear(6, dim, bias=False)
    def _refuse_trope_base_drift(self, state_dict, prefix, *args):
        """Refuse weights trained with a different frame-RoPE base."""
        key = prefix + "trope_base"
        if key in state_dict and float(state_dict[key]) != self.trope_base_value:
            raise RuntimeError(
                f"[refuse] this checkpoint was trained with trope_base={float(state_dict[key])} but the model was built "
                f"with {self.trope_base_value}")

    def forward(self, x, t, *, is_target, joint_sem=None, text=None, blueprint=None,
                joint_bias=None, frame_valid=None, joint_valid=None,
                struct_feats=None, updown=None, local_root=None, demo_text=None, spectral_feats=None):
        """x [B, T, J, C]; t [B] in [0, 1]; is_target [B, T] marks the frames to generate;
        joint_sem [B, J, d_joint_sem]; text [B, d_text]; struct_feats [B, J, 8] (14 with struct_world_rest),
        updown [B, J, J, 2] and spectral_feats [B, J, K] (the heat-kernel signature) as the flags require.
        """
        B, T, J, _ = x.shape
        h = self.x_in(x)
        trope_cos = trope_sin = None
        if self.use_temporal_rope:
            cs, sn = frame_rope_cos_sin(T, self.dim // self.n_heads, self.trope_base_value,
                                        device=h.device, dtype=torch.float32)
            trope_cos, trope_sin = cs[None, None], sn[None, None]
        else:
            h = h + self.t_pos[:, :T]
        rope_cos = rope_sin = None
        if self.use_spec_rope:
            if spectral_feats is None:
                raise ValueError("model built with use_spec_rope=True but batch has no spectral_feats -- dataset must "
                                 "carry the heat-kernel signature (spectral_feats)")
            if spectral_feats.shape[-1] != self.spec_rope_k:
                raise ValueError(f"spectral_feats has K={spectral_feats.shape[-1]}, model built with spec_rope_k={self.spec_rope_k}")
            cos, sin = self.spec_rope.cos_sin(spectral_feats)
            rope_cos, rope_sin = cos[:, None, None], sin[:, None, None]
        else:
            if spectral_feats is not None:
                raise ValueError("spectral_feats passed but the model was built with use_spec_rope=False")
            h = h + self.j_pos[:, :, :J]
        h = h + is_target[..., None, None].to(h.dtype) * self.mask_token
        if self.use_ref_text and demo_text is not None:
            tgt_v = self.ref_text_mlp(text if text is not None else torch.zeros_like(demo_text))
            dem_v = self.ref_text_mlp(demo_text)
            per_frame = torch.where(is_target[..., None], tgt_v[:, None], dem_v[:, None])
            h = h + per_frame[:, :, None, :].to(h.dtype)
        if local_root is not None:
            if self.local_root_mlp is None:
                raise ValueError("local_root passed but the model was built with "
                                 "local_root_dim=0")
            h = h + self.local_root_mlp(local_root.to(h.dtype))[:, :, None, :]
        if joint_sem is not None:
            h = h + self.joint_sem(joint_sem)[:, None]
        if self.use_struct_feats:
            if struct_feats is None:
                raise ValueError("model built with use_struct_feats=True but batch has no "
                                 "struct_feats")
            if struct_feats.shape[-1] != self.struct_in:
                raise ValueError(f"struct_feats has {struct_feats.shape[-1]} columns but the model was built for "
                                 f"{self.struct_in} (struct_world_rest={self.struct_world_rest}): dataset and model disagree")
            z = self.struct_mlp[0](struct_feats[..., :8])
            if self.struct_world_rest:
                z = z + self.struct_rest_in(struct_feats[..., 8:])
            h = h + self.struct_mlp[2](self.struct_mlp[1](z))[:, None]
        if not self.use_geo_bias and joint_bias is not None:
            joint_bias = torch.where(joint_bias <= -1e3, joint_bias, torch.zeros_like(joint_bias))
        if self.use_dir_bias:
            if updown is None:
                raise ValueError("model built with use_dir_bias=True but batch has no updown")
            dir_b = (self.e_up(updown[..., 0]) + self.e_down(updown[..., 1])).permute(0, 3, 1, 2)
            base = joint_bias.unsqueeze(1) if (joint_bias is not None and joint_bias.dim() == 3) \
                else joint_bias
            joint_bias = dir_b.to(h.dtype) if base is None else base.to(h.dtype) + dir_b.to(h.dtype)

        c = self.t_mlp(timestep_embedding(t * 1000.0, self.dim))
        if text is not None:
            c = c + self.text_mlp(text)
        if blueprint is not None:
            c = c + self.bp_mlp(blueprint)

        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, c, joint_bias, frame_valid, joint_valid, rope_cos, rope_sin,
                    trope_cos, trope_sin, use_reentrant=False)
            else:
                h = blk(h, c, joint_bias=joint_bias, frame_valid=frame_valid,
                        joint_valid=joint_valid, rope_cos=rope_cos, rope_sin=rope_sin,
                        trope_cos=trope_cos, trope_sin=trope_sin)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.n_out(h), shift[:, None, None], scale[:, None, None])
        return self.out(h)


@torch.no_grad()
def sample(model, x_ref, is_target, steps, cfg_text=1.0, cfg_demo=1.0, demo_frames=None,
           channel_valid=None, anchor=None, heading_valid=None, **cond):
    """Euler integration of the flow from noise to motion in `steps` steps, driven by the predicted clean motion.

    Each step moves the target frames by (x1_pred - x) / (steps - i), the remaining distance over the remaining steps;
    the other frames stay x_ref. cfg_text != 1 applies classifier-free guidance on the caption (the unconditional pass
    zeroes the text embedding), cfg_demo != 1 also on the demo frames. channel_valid [B, J, C] zeroes the channels a
    joint does not have, heading_valid [B, T] the root heading of frames without one.
    """
    cv = channel_valid[:, None].to(x_ref.dtype) if channel_valid is not None else None
    if cv is not None:
        x_ref = x_ref * cv
        if heading_valid is not None and x_ref.shape[-1] >= 17:
            hv = heading_valid.to(x_ref.dtype).clone()
            hv[is_target] = 1.0
            x_ref[:, :, 0, 15:17] *= hv[:, :, None]
    noise = torch.randn_like(x_ref)
    if anchor is not None:
        noise = noise + anchor.to(noise.dtype)
    if cv is not None:
        noise = noise * cv
    x = torch.where(is_target[..., None, None], noise, x_ref)

    guided = (cfg_text != 1.0) or (cfg_demo != 1.0)
    if guided:
        cond_dt = cond
        cond_du = ({**cond, "text": torch.zeros_like(cond["text"])}
                   if cond.get("text") is not None else cond)
        if cfg_demo != 1.0:
            assert demo_frames is not None, \
                "guided sampling needs demo_frames to build the demo-drop"
            fv = cond.get("frame_valid")
            fv_u = fv.clone() if fv is not None else None
            if fv_u is not None:
                fv_u[:, :demo_frames] = False
            cond_uu = {**cond_du, "frame_valid": fv_u}
            if cond.get("demo_text") is not None:
                cond_uu = {**cond_uu, "demo_text": torch.zeros_like(cond["demo_text"])}

    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        if not guided:
            x1_pred = model(x, t, is_target=is_target, **cond)
        elif cfg_demo == 1.0:
            x_dt = model(x, t, is_target=is_target, **cond_dt)
            x_du = model(x, t, is_target=is_target, **cond_du)
            x1_pred = x_du + cfg_text * (x_dt - x_du)
        else:
            x_dt = model(x, t, is_target=is_target, **cond_dt)
            x_du = model(x, t, is_target=is_target, **cond_du)
            x_u = x.clone(); x_u[:, :demo_frames] = 0.0
            x_uu = model(x_u, t, is_target=is_target, **cond_uu)
            x1_pred = x_uu + cfg_demo * (x_du - x_uu) + cfg_text * (x_dt - x_du)
        step = (x1_pred - x) / (steps - i)
        x = torch.where(is_target[..., None, None], x + step, x_ref)
        if cv is not None:
            x = x * cv
    return x

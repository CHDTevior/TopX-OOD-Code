"""Per-joint skeleton features served with every item (tree hops, structural and world-frame rest features),
and the batch collation that pads the joint axis."""
from __future__ import annotations

import numpy as np
import torch

from src.data.constants import _STD_FLOOR  # noqa: F401


REST_DEMO_CLAMP = 5.0


GEODESIC_CLIP = 8.0


PAD_BIAS = -1e4


UPDOWN_CLIP = 15


def _world_rest_feats(parents, offsets, P_rest):
    """[J, 6] float32 world-frame rest descriptor, appended to the structural features when struct_world_rest is on:
      0:3  u_j = (P_rest[j] - P_rest[parent(j)]) normalised: the world direction of bone j at rest (root and zero-length
           bones: zeros);
      3:6  P_rest[j] with the root's XZ subtracted, divided by the rig's mean bone length |offset| (height kept: the rest
           pose is grounded at the data level, so this is height above ground in bone lengths), then radially
           log-compressed, c -> c * log1p(|c|) / |c| (direction kept, invertible).
    Read off the rest positions themselves rather than R_rest @ offset: the skeleton files' parent-frame offsets are not
    FK-consistent with P_rest on every rig, and an augmented item's transform rebuilds P_rest by FK of its (possibly
    rotated) rest rotations, so on both paths P_rest IS the served rest pose and u follows it exactly."""
    par = np.asarray(parents, dtype=np.int64); J = len(par)
    off = np.asarray(offsets, dtype=np.float64)[:J]
    P = np.asarray(P_rest, dtype=np.float64)[:J]
    blen = np.linalg.norm(off, axis=-1)
    mean_bone = max(float(blen[1:].mean()) if J > 1 else 1.0, 1e-6)
    u = np.zeros((J, 3))
    if J > 1:
        d = P[1:] - P[par[1:]]
        n = np.linalg.norm(d, axis=1)
        ok = n > 1e-8
        u[1:][ok] = d[ok] / n[ok, None]
    c = (P - np.array([P[0, 0], 0.0, P[0, 2]])[None]) / mean_bone
    r = np.linalg.norm(c, axis=1)
    c = c * np.where(r > 0, np.log1p(r) / np.maximum(r, 1e-12), 1.0)[:, None]
    return np.concatenate([u, c], axis=1).astype(np.float32)


def _graph_v2_tables(parents, offsets, geodesic_raw):
    """Per-rig structural joint features [J,8] and the directional hop matrix [J,J,2] (up-steps to the lowest
    common ancestor, then down-steps). up + down must equal the tree hop distance exactly; a mismatch means the joint
    order or the parent table is corrupt and raises.

    Feature layout (scale-free; the rig's mean bone length is the unit):
      0:3 rest offset direction (unit; zeros for the root and zero-length bones)
      3   log1p(bone length / mean bone)
      4   depth from the root in hops / 16
      5   log1p(path length to the root / mean bone) / 4
      6   child count, clipped at 4, / 4
      7   leaf flag
    """
    J = len(parents)
    par = [int(p) for p in parents]
    depth = np.zeros(J, np.int64)
    for j in range(1, J):
        depth[j] = depth[par[j]] + 1
    anc = []
    for j in range(J):
        c, k = {}, j
        while k >= 0:
            c[k] = depth[j] - depth[k]      # up-steps from j to ancestor k
            k = par[k]
        anc.append(c)
    ud = np.zeros((J, J, 2), np.int64)
    for i in range(J):
        ai = anc[i]
        for j in range(J):
            aj = anc[j]
            best, up = -1, 0
            for k, u in ai.items():         # ancestors of i, nearest first is not guaranteed --
                if k in aj and depth[k] > best:   # pick the DEEPEST common ancestor (the LCA)
                    best, up = depth[k], u
            ud[i, j, 0] = up
            ud[i, j, 1] = depth[j] - best         # down-steps = depth(j) - depth(LCA)
    hops = ud.sum(-1)
    geo = np.asarray(geodesic_raw, dtype=np.int64)[:J, :J]
    if not np.array_equal(hops, geo):
        bad = int(np.abs(hops - geo).max())
        raise AssertionError(f"LCA hop decomposition disagrees with the tree hop distance "
                             f"(max |up+down - geo| = {bad}); joint order or parents corrupted")
    off = np.asarray(offsets, dtype=np.float64)[:J]
    blen = np.linalg.norm(off, axis=-1)
    mean_bone = max(float(blen[1:].mean()) if J > 1 else 1.0, 1e-6)
    unit = np.zeros((J, 3))
    nz = blen > 1e-8
    unit[nz] = off[nz] / blen[nz, None]
    phys = np.zeros(J)
    for j in range(1, J):
        phys[j] = phys[par[j]] + blen[j]
    n_child = np.zeros(J)
    for j in range(1, J):
        n_child[par[j]] += 1
    feats = np.zeros((J, 8), np.float32)
    feats[:, 0:3] = unit
    feats[:, 3] = np.log1p(blen / mean_bone)
    feats[:, 4] = depth / 16.0
    feats[:, 5] = np.log1p(phys / mean_bone) / 4.0
    feats[:, 6] = np.clip(n_child, 0, 4) / 4.0
    feats[:, 7] = (n_child == 0).astype(np.float32)
    return feats, np.clip(ud, 0, UPDOWN_CLIP)


def collate(batch):
    """Pad the joint axis to the batch max; frames are already fixed length.

    joint_valid MUST reach the loss. Padded joints are numerically zero, and counting them as valid
    would let padding dominate the per-group means: a J=9 rig in a batch whose max is 142 would be
    94% padding.
    """
    B, T, C = len(batch), batch[0]["x"].shape[0], batch[0]["x"].shape[2]
    Jm = max(b["n_joints"] for b in batch)

    x = torch.zeros(B, T, Jm, C)
    joint_valid = torch.zeros(B, Jm, dtype=torch.bool)
    bias = torch.full((B, Jm, Jm), PAD_BIAS)
    has_sem = "joint_sem" in batch[0]
    sem = torch.zeros(B, Jm, batch[0]["joint_sem"].shape[1]) if has_sem else None

    for k, b in enumerate(batch):
        J = b["n_joints"]
        x[k, :, :J] = b["x"]
        joint_valid[k, :J] = True
        bias[k, :J, :J] = -b["geodesic"]          # nearer joints attend more; padding stays at -1e4
        if has_sem:
            sem[k, :J] = b["joint_sem"]

    out = {
        "x": x,
        "joint_valid": joint_valid,
        "joint_bias": bias,
        "is_target": torch.stack([b["is_target"] for b in batch]),
        "frame_valid": torch.stack([b["frame_valid"] for b in batch]),
        "object_type": [b["object_type"] for b in batch],
        "motion_id": [b["motion_id"] for b in batch],
        "demo_id": [b["demo_id"] for b in batch],
        "lock_denominator": torch.tensor([float(b.get("lock_denominator", 0.0)) for b in batch]),
    }
    if has_sem:
        out["joint_sem"] = sem
    if "text" in batch[0]:
        out["text"] = torch.stack([b["text"] for b in batch])
    if "demo_text" in batch[0]:
        out["demo_text"] = torch.stack([b["demo_text"] for b in batch])
    if "norm_mean" in batch[0]:
        Cs = batch[0]["norm_mean"].shape[1]
        am = torch.zeros(B, Jm, Cs); asd = torch.zeros(B, Jm, Cs)
        par = torch.full((B, Jm), -1, dtype=torch.long); ro = torch.zeros(B, Jm, 3)
        for k, b in enumerate(batch):
            J = b["n_joints"]
            am[k, :J] = b["norm_mean"]; asd[k, :J] = b["norm_std"]
            par[k, :J] = b["parents"]; ro[k, :J] = b["rest_offsets"]
        out.update(norm_mean=am, norm_std=asd, parents=par, rest_offsets=ro,
                   n_joints=torch.tensor([b["n_joints"] for b in batch], dtype=torch.long))
        if "R_rest_global" in batch[0]:
            rr = torch.zeros(B, Jm, 3, 3)
            for k, b in enumerate(batch):
                rr[k, :b["n_joints"]] = b["R_rest_global"]
            out["R_rest_global"] = rr
    if "channel_valid" in batch[0]:
        cvb = torch.zeros(B, Jm, batch[0]["channel_valid"].shape[1], dtype=torch.bool)
        for k, b in enumerate(batch):
            cvb[k, :b["n_joints"]] = b["channel_valid"]
        out["channel_valid"] = cvb
    if "struct_feats" in batch[0]:
        # structural fields, padded to Jm. Padded rows are zeros; padded PAIRS are irrelevant
        # because the -1e4 PAD_BIAS already excludes them from attention, and zero-index lookups
        # into the zero-initialised direction tables contribute exactly 0 at start anyway.
        sf = torch.zeros(B, Jm, batch[0]["struct_feats"].shape[1])
        ud = torch.zeros(B, Jm, Jm, 2, dtype=torch.long)
        for k, b in enumerate(batch):
            J = b["n_joints"]
            sf[k, :J] = b["struct_feats"]
            ud[k, :J, :J] = b["updown"]
        out.update(struct_feats=sf, updown=ud)
    if "spectral_feats" in batch[0]:
        # heat-kernel signatures padded to Jm with zeros; PAD_BIAS removes padded joints from every softmax
        sp = torch.zeros(B, Jm, batch[0]["spectral_feats"].shape[1])
        for k, b in enumerate(batch):
            sp[k, :b["n_joints"]] = b["spectral_feats"]
        out["spectral_feats"] = sp
    out["valid"] = out["frame_valid"][:, :, None] & joint_valid[:, None, :]      # [B,T,Jm]
    return out


def hop_matrix(parents) -> np.ndarray:
    """[J,J] float32 tree hop distances."""
    par = [int(p) for p in parents]
    J = len(par)
    depth = np.zeros(J, np.int64)
    anc = []
    for j in range(J):
        if j:
            depth[j] = depth[par[j]] + 1
        c, k = {}, j
        while k >= 0:
            c[k] = depth[j] - depth[k]
            k = par[k]
        anc.append(c)
    g = np.zeros((J, J), np.float32)
    for i in range(J):
        for j in range(J):
            common = anc[i].keys() & anc[j].keys()
            l = max(common, key=lambda k: depth[k])
            g[i, j] = (depth[i] - depth[l]) + (depth[j] - depth[l])
    return g

#!/usr/bin/env python
"""Text-to-motion on one skeleton with TopX -- only the skeleton file is needed.

    python scripts/generate.py --ckpt topx.pt --skeleton my_rig.bvh --up +Y --forward +Z \
        --text "An object walks forward and sits down." --frames 120 --out out/

Skeleton
  --skeleton  a BVH file, or a skeleton .npz in the corpus format (P_rest_global / R_rest_global / offset_parent_local
              / ...). For a BVH the rest pose is the FK of frame --rest_frame (default 0: export the rig with its rest
              pose as the first frame; -1 = the OFFSETs with zero rotations). End Sites are not joints. --up / --forward
              name the BVH axis pointing up and the axis the character faces; the model's frame is +Y up, +Z forward,
              left = +X.
  Each joint's rest frame is re-expressed in the corpus convention: local +Y along the bone to its primary child
  (shortest-arc swing). The generated rotations are rest deltas (world rotation of each joint from its rest pose),
  which do not depend on that convention, so the BVH written back uses the input's own joint frames.

Text
  --text      the prompt (training captions read "An object <does something>."; on a new character also try naming
              it), encoded with LLM2Vec (McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp + -supervised on
              meta-llama/Meta-Llama-3-8B-Instruct, gated on HuggingFace; about 16 GB in bf16) as the training captions
              were: the mean of the sentence-span token states. --text_emb <.npy [4096]> skips the encoder.
  Joints whose names carry a side (Left / Right / .L / _R ...) are checked against the model's frame (+X = left).
  Joint descriptions (the model's per-joint semantics) are sentences such as "Left Thigh joint.", derived from the
  joint names through a lexicon learnt from the training corpus (configs/joint_name_lexicon.json) and, for names that
  carry no anatomy, from the skeleton's geometry (src/data/joint_descriptions.py). --descriptions <json
  {joint name: sentence}> overrides any of them; --joint_sem <.npy [J,4096]> skips their encoding; --describe_only
  prints them (and the rig) and stops.

Outputs  <out>/<name>.{npz,gif,bvh}
  .npz  world positions of both decode paths [T,J,3] (direct = the position channels; fk = the rotation channels
        through the skeleton), the rest-delta rotations [T,J,3,3], the normalised model output, joint names / parents,
        fps 30, the frame (+Y up, +Z forward, rest pose grounded).
  .gif  rest pose | generated (position channels) | generated (rotation FK), at 30 fps.
  .bvh  the input hierarchy (BVH input) or a hierarchy built from the rest pose (npz input), animated with the
        generated rotations and root translation, in the input's own axes and units.

A new skeleton needs no motion statistics: under rest normalisation the mean is the rig's rest frame and the scales
are s_rig / (per-block gain), s_rig being the diagonal of the rest pose's bounding box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.constants import _STD_FLOOR                                              # noqa: E402
from src.data.rig_features import (GEODESIC_CLIP, REST_DEMO_CLAMP, _graph_v2_tables,    # noqa: E402
                                      _world_rest_feats, collate)
from src.data.rig_features import hop_matrix                                               # noqa: E402
from src.data.ktjd17.decoder import decode_ktjd17                                            # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                              # noqa: E402

FPS = 30.0
# Per-block gains of the rest normalisation: the scales are s_rig / gain for the position (0:3), velocity (9:12) and
# root-track (13:15) blocks. The weights file records the gains it was trained with; they must match.
GAINS = np.array([3.867547101351066, 2.943516881261983, 3.3212471860907744], dtype=np.float64)
CAPTION_MAX_LENGTH = 128        # tokenizer cap of the training captions
CAPTION_STORE_MAX_TOKENS = 113  # the caption build refused longer sentence spans
DESC_MAX_LENGTH = 64            # LLM2VecEncoder default, used for the joint-description table
DEFAULT_LEXICON = REPO / "configs/joint_name_lexicon.json"
AXES = {"+X": np.array([1.0, 0, 0]), "-X": np.array([-1.0, 0, 0]), "+Y": np.array([0, 1.0, 0]),
        "-Y": np.array([0, -1.0, 0]), "+Z": np.array([0, 0, 1.0]), "-Z": np.array([0, 0, -1.0])}


# ------------------------------------------------------------------------------------------------ BVH
class BvhNode:
    __slots__ = ("name", "parent", "offset", "channels", "is_end")

    def __init__(self, name, parent, is_end):
        self.name, self.parent, self.is_end = name, parent, is_end
        self.offset = np.zeros(3)
        self.channels: list[str] = []


def read_bvh(path):
    """-> (nodes [BvhNode] incl. End Sites in file order, frames [F, n_channels] float64, frame_time)."""
    lines = Path(path).read_text().splitlines()
    nodes, stack, i = [], [], 0
    while i < len(lines) and lines[i].strip().upper() != "HIERARCHY":
        i += 1
    if i == len(lines):
        raise SystemExit(f"[refuse] {path}: no HIERARCHY section")
    i += 1
    pending = None
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            continue
        head = s.split()[0].upper()
        if head in ("ROOT", "JOINT"):
            name = s.split(None, 1)[1].strip() if len(s.split(None, 1)) > 1 else f"joint{len(nodes)}"
            pending = BvhNode(name, stack[-1] if stack else -1, False)
        elif head == "END":
            pending = BvhNode(f"{nodes[stack[-1]].name}_End", stack[-1], True)
        elif s == "{":
            if pending is None:
                raise SystemExit(f"[refuse] {path}:{i}: '{{' without a node")
            nodes.append(pending)
            stack.append(len(nodes) - 1)
            pending = None
        elif s == "}":
            stack.pop()
        elif head == "OFFSET":
            nodes[stack[-1]].offset = np.array([float(v) for v in s.split()[1:4]], dtype=np.float64)
        elif head == "CHANNELS":
            p = s.split()
            n = int(p[1])
            nodes[stack[-1]].channels = [c.lower() for c in p[2:2 + n]]
        elif head == "MOTION":
            break
        else:
            raise SystemExit(f"[refuse] {path}:{i}: unexpected hierarchy line {s!r}")
    if stack:
        raise SystemExit(f"[refuse] {path}: unbalanced braces in HIERARCHY")
    frames, frame_time = 0, 1.0 / FPS
    rest_vals = []
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            continue
        if s.lower().startswith("frames:"):
            frames = int(s.split(":")[1])
        elif s.lower().startswith("frame time:"):
            frame_time = float(s.split(":")[1])
            rest_vals = " ".join(lines[i:]).split()
            break
    nch = sum(len(n.channels) for n in nodes)
    vals = np.array([float(v) for v in rest_vals], dtype=np.float64)
    if frames and vals.size != frames * nch:
        raise SystemExit(f"[refuse] {path}: {vals.size} motion values, expected {frames} frames x {nch} channels")
    return nodes, (vals.reshape(frames, nch) if frames else np.zeros((0, nch))), frame_time


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def bvh_frame_fk(nodes, values):
    """FK of one BVH frame (position channels are the local translation, the root's are absolute;
    rotations compose in channel order). values None = the OFFSETs with zero rotations.
    -> (P [N,3], G [N,3,3], T_local [N,3]) for every node (End Sites included)."""
    N = len(nodes)
    P, G, T = np.zeros((N, 3)), np.zeros((N, 3, 3)), np.zeros((N, 3))
    k = 0
    for j, n in enumerate(nodes):
        t, R = n.offset.copy(), np.eye(3)
        if values is not None:
            pos = [None, None, None]
            for c in n.channels:
                v = values[k]
                k += 1
                if c.endswith("position"):
                    pos["xyz".index(c[0])] = v
                else:
                    R = R @ _rot(c[0], v)
            if all(p is not None for p in pos):
                t = np.array(pos, dtype=np.float64)
            elif any(p is not None for p in pos):
                raise SystemExit(f"[refuse] joint {n.name!r}: a partial set of position channels {n.channels}")
        T[j] = t
        if n.parent < 0:
            P[j], G[j] = t, R
        else:
            P[j] = P[n.parent] + G[n.parent] @ t
            G[j] = G[n.parent] @ R
    return P, G, T


def euler_from_matrix(R, order):
    """R [...,3,3] -> angles (degrees) such that R = R_order[0] @ R_order[1] @ R_order[2] (BVH channel order)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_euler(order.upper(), degrees=True).reshape(R.shape[:-2] + (3,))


# ------------------------------------------------------------------------------------------------ rig
def canonical_basis(up: str, forward: str) -> np.ndarray:
    """C with C @ v_source = v_canonical (+Y up, +Z forward, +X = up x forward = the creature's left)."""
    if up not in AXES or forward not in AXES:
        raise SystemExit(f"[refuse] --up / --forward must be one of {sorted(AXES)}")
    u, f = AXES[up], AXES[forward]
    if abs(float(u @ f)) > 0.5:
        raise SystemExit(f"[refuse] --up {up} and --forward {forward} are not perpendicular")
    return np.stack([np.cross(u, f), u, f])


def _unit(v, eps=1e-9):
    n = float(np.linalg.norm(v))
    return v / n if n > eps else None


def _swing_from_y(d):
    """Shortest-arc rotation taking +Y to the unit vector d."""
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.clip(y @ d, -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(y, d)
    ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    s = np.sqrt(max(0.0, 1.0 - c * c))
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def continuation_frames(parents, P):
    """[J,3,3] rest frames in the corpus convention: local +Y along the bone to the primary child -- the child that
    best continues the incoming bone (root: world up); a leaf (or a joint whose children all coincide with it) continues
    its incoming bone.
    """
    par = np.asarray(parents, dtype=np.int64)
    J = len(par)
    children = [[] for _ in range(J)]
    for j in range(1, J):
        children[par[j]].append(j)
    R = np.zeros((J, 3, 3))
    up = np.array([0.0, 1.0, 0.0])
    for j in range(J):
        inc = _unit(P[j] - P[par[j]]) if j > 0 else up
        if inc is None:
            inc = R[par[j]][:, 1]
        cands = [(c, _unit(P[c] - P[j])) for c in children[j]]
        cands = [(c, d) for c, d in cands if d is not None]
        d = max(cands, key=lambda cd: float(cd[1] @ inc))[1] if cands else inc
        R[j] = _swing_from_y(d)
    return R


def _finish_rig(names, parents, P, R, kinds, source):
    """Common fields of a served rig from canonical, grounded rest positions and rest frames."""
    par = np.asarray(parents, dtype=np.int64)
    J = len(par)
    off = np.zeros((J, 3))
    Rl = np.zeros((J, 3, 3))
    Rl[0] = R[0]
    for j in range(1, J):
        off[j] = R[par[j]].T @ (P[j] - P[par[j]])
        Rl[j] = R[par[j]].T @ R[j]
    s_rig = float(np.linalg.norm(P.max(0) - P.min(0)))
    if not s_rig > 0:
        raise SystemExit("[refuse] the rest pose has zero extent")
    return {"joint_names": [str(n) for n in names], "parents": par.astype(np.int32), "P_rest_global": P,
            "R_rest_global": R, "R_rest_local": Rl, "offset_parent_local": off,
            "rotation_source_kind": np.array(kinds), "serve_kinds": np.array(kinds), "s_rig": s_rig, "source": source}


def rig_from_bvh(path, up, forward, rest_frame):
    nodes, frames, frame_time = read_bvh(path)
    if rest_frame >= 0 and rest_frame >= len(frames):
        raise SystemExit(f"[refuse] --rest_frame {rest_frame} but the BVH has {len(frames)} frames")
    for n in nodes:
        if n.is_end:
            continue
        rot = sorted(c[0] for c in n.channels if c.endswith("rotation"))
        if rot not in ([], ["x", "y", "z"]):
            raise SystemExit(f"[refuse] joint {n.name!r} has rotation channels {n.channels}: only none or all three of "
                             f"X/Y/Z are supported")
    if sorted(c for c in nodes[0].channels if c.endswith("position")) != ["xposition", "yposition", "zposition"]:
        raise SystemExit(f"[refuse] the root {nodes[0].name!r} has no X/Y/Z position channels: the generated root path "
                         f"could not be written")
    Pn, Gn, Tn = bvh_frame_fk(nodes, frames[rest_frame] if rest_frame >= 0 else None)
    jidx = [k for k, n in enumerate(nodes) if not n.is_end]
    remap = {k: i for i, k in enumerate(jidx)}
    parents = [(-1 if nodes[k].parent < 0 else remap[nodes[k].parent]) for k in jidx]
    if parents[0] != -1 or any(p < 0 for p in parents[1:]) or any(p >= i for i, p in enumerate(parents) if i):
        raise SystemExit("[refuse] the BVH hierarchy is not a single tree in depth-first order")
    C = canonical_basis(up, forward)
    P = Pn[jidx] @ C.T
    shift = np.array([P[0, 0], P[:, 1].min(), P[0, 2]])
    P = P - shift
    kinds = ["animated_dof" if any(c.endswith("rotation") for c in nodes[k].channels) else "fixed_dof" for k in jidx]
    rig = _finish_rig([nodes[k].name for k in jidx], parents, P, continuation_frames(parents, P), kinds,
                      f"bvh:{path} rest_frame={rest_frame} up={up} forward={forward}")
    # a joint without rotation channels is decoded / exported as fixed (it keeps its rest local rotation), but the MODEL is
    # served every row as animated: no rig of the training corpus has a fixed joint (the UniML3D builder writes
    # animated_dof everywhere), so a masked rotation row would be a pattern it never saw
    rig["serve_kinds"] = np.array(["animated_dof"] * len(jidx))
    rig["bvh"] = {"nodes": nodes, "jidx": jidx, "C": C, "shift": shift, "G0": Gn, "T0": Tn, "frame_time": frame_time}
    return rig


def side_check(rig):
    """(agree, n, forward_canonical or None): of the joints whose NAME carries a side and that sit off the mid-line
    (|x| > 0.02 s_rig, the corpus's SIDE_TOL), how many lie on that side of the model's frame (+X = left; the corpus puts
    99.75% of its "Left" joints at x > 0); and the forward direction the named sides imply (left x up)."""
    P = np.asarray(rig["P_rest_global"], dtype=np.float64)
    s = float(rig["s_rig"])
    left, right = [], []
    agree = n = 0
    for j, nm in enumerate(rig["joint_names"]):
        side, _ = name_side_key(nm)
        if side is None:
            continue
        (left if side == "left" else right).append(j)
        if abs(P[j, 0]) > 0.02 * s:
            n += 1
            agree += int((P[j, 0] > 0) == (side == "left"))
    fwd = None
    if left and right:
        lv = P[left].mean(0) - P[right].mean(0)
        lv[1] = 0.0
        if np.linalg.norm(lv) > 1e-9:
            fwd = np.cross(lv / np.linalg.norm(lv), np.array([0.0, 1.0, 0.0]))
    return agree, n, fwd


def rig_from_npz(path, reconvention):
    z = np.load(path, allow_pickle=False)
    names = [str(x) for x in z["joint_names"]]
    rig = {"joint_names": names, "parents": np.asarray(z["parents"]), "P_rest_global": np.asarray(z["P_rest_global"], dtype=np.float64),
           "R_rest_global": np.asarray(z["R_rest_global"], dtype=np.float64),
           "R_rest_local": np.asarray(z["R_rest_local"], dtype=np.float64),
           "offset_parent_local": np.asarray(z["offset_parent_local"], dtype=np.float64),
           "rotation_source_kind": np.asarray(z["rotation_source_kind"]), "s_rig": float(z["s_rig"]),
           "source": f"npz:{path}"}
    rig["serve_kinds"] = rig["rotation_source_kind"]
    for k in ("R_rest_global", "R_rest_local"):
        if z[k].dtype != np.float64:     # stored in lower precision: snap to the nearest rotation
            u, _, vt = np.linalg.svd(rig[k])
            u[..., -1] *= np.sign(np.linalg.det(u @ vt))[..., None]
            rig[k] = u @ vt
    if "joint_descriptions" in z.files:
        rig["joint_descriptions"] = [str(x) for x in z["joint_descriptions"]]
    if reconvention:
        # the corpus rig re-expressed in the rule above (what a BVH import of the same rest pose would serve)
        kinds = [str(k) for k in rig["rotation_source_kind"]]
        keep = {k: rig[k] for k in ("joint_descriptions",) if k in rig}
        rig = _finish_rig(names, rig["parents"], np.asarray(rig["P_rest_global"], dtype=np.float64),
                          continuation_frames(rig["parents"], np.asarray(rig["P_rest_global"], dtype=np.float64)),
                          kinds, f"npz:{path} reconvention=continuation")
        rig.update(keep)
    return rig


# ------------------------------------------------------------------------------------------------ descriptions
_SIDE_PREFIX = re.compile(r"^(left|right|l|r|lft|rgt)(?=[_\-. ]|[A-Z]|$)", re.I)
_SIDE_SUFFIX = re.compile(r"(?:[_\-. ])(l|r|left|right)$", re.I)
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def name_side_key(raw: str):
    """A joint name -> (side 'left'/'right'/None, normalized key). Namespaces ('mixamorig:'), 'Armature|' paths, the
    corpus's per-rig '_<index>' suffix and Blender '.001' duplicates are dropped; side markers (Left/Right words,
    L/R tokens, '.L' / '_R' suffixes) become the side; the remaining word tokens, lower-cased and without digits, are
    the key ('mixamorig:LeftUpLeg_056' -> ('left', 'up leg'), 'UpLeg.L' -> ('left', 'up leg'))."""
    s = raw.split(":")[-1].split("|")[-1].strip()
    s = re.sub(r"_\d+$", "", s)
    s = re.sub(r"\.\d{3}$", "", s)
    side = None
    m = _SIDE_SUFFIX.search(s)
    if m:
        side = "left" if m.group(1).lower().startswith("l") else "right"
        s = s[:m.start()]
    toks = []
    for part in re.split(r"[^A-Za-z0-9]+", s):
        toks += [t for t in _CAMEL.split(part) if t]
    out = []
    for t in toks:
        tl = t.lower()
        if tl in ("left", "right", "lft", "rgt") or (tl in ("l", "r") and len(toks) > 1):
            side = side or ("left" if tl.startswith("l") else "right")
            continue
        tl = re.sub(r"\d+", "", tl)
        if tl:
            out.append(tl)
    return side, " ".join(out)


def load_lexicon(path):
    """{key: (clean name, sided)}; `sided` marks a key whose corpus descriptions carry a side although the key function
    sees no side marker in the names (foreign or glued side words: 'mao esquerda', 'lhand'). The side itself is then read
    off the geometry (the corpus's own rule, +X = left beyond 0.02 s_rig), never learnt: a few corpus assets have names
    mirrored against their geometry, so a learnt side can be inverted."""
    lx = json.loads(Path(path).read_text())
    return {k: (v["clean"], bool(v.get("sided", False))) for k, v in lx["entries"].items() if v["share"] >= 0.5}


SIDE_TOL = 0.02   # |x| / s_rig below this is the mid-line


def describe_joints(rig, lexicon, overrides=None, body_plan=None):
    """-> (descriptions [J], source per joint: 'override' / 'lexicon' / 'structure')."""
    from src.data.joint_descriptions import describe_rig, is_low_info
    names = rig["joint_names"]
    xs = np.asarray(rig["P_rest_global"], dtype=np.float64)[:, 0] / float(rig["s_rig"])
    desc, src = [], []
    for j, nm in enumerate(names):
        if overrides and nm in overrides:
            desc.append(str(overrides[nm]))
            src.append("override")
            continue
        side, key = name_side_key(nm)
        clean, sided = lexicon.get(key, (None, False))
        geo = "left" if xs[j] > SIDE_TOL else ("right" if xs[j] < -SIDE_TOL else None)
        if side is None and sided:
            side = geo
        elif side is not None and geo is not None and geo != side:
            side = geo      # a name mirrored against the geometry: the corpus's descriptions follow the geometry
        if clean:
            desc.append(f"{side.capitalize() + ' ' if side else ''}{clean} joint.")
            src.append("lexicon")
        else:
            desc.append("Bone joint.")                   # a placeholder: the structural builder replaces it
            src.append("structure")
    low = [is_low_info(d) for d in desc]
    src = [("structure" if lo and s != "override" else s) for s, lo in zip(src, low)]
    desc, _, _ = describe_rig(names, desc, np.asarray(rig["parents"], dtype=np.int64),    # -> (descriptions, low, stats)
                              np.asarray(rig["P_rest_global"], dtype=np.float64), float(rig["s_rig"]), body_plan)
    desc = [str(overrides[nm]) if s == "override" else str(d) for nm, d, s in zip(names, desc, src)]
    return desc, src


# ------------------------------------------------------------------------------------------------ text
def encode_texts(caption, descriptions, device):
    """LLM2Vec embeddings computed as for training: the caption = mean of its sentence-span token states
    (tokenizer cap 128), each joint description = LLM2Vec's pooled embedding (tokenizer cap 64), one string per
    forward pass.
    """
    from src.data.text_encoders import LLM2VecEncoder
    from llm2vec.llm2vec import batch_to_device
    enc = LLM2VecEncoder(device=device, max_length=DESC_MAX_LENGTH)
    dev = next(enc.l2v.model.parameters()).device
    sem = None
    if descriptions is not None:
        # one string per forward pass: in a batch a string's embedding depends on its left-padded neighbours, and the
        # library's encode() starts a process pool (and a second copy of the model) on a multi-GPU machine
        uniq = sorted(set(descriptions))
        E = np.zeros((len(uniq), enc.dim), dtype=np.float32)
        for i, s in enumerate(uniq):
            feat = enc.l2v.tokenize([enc.l2v.prepare_for_tokenization(enc.l2v._convert_to_str("", s))])
            with torch.no_grad():
                E[i] = enc.l2v.forward(batch_to_device(feat, dev)).float().cpu().numpy()[0]
        if not np.isfinite(E).all():
            raise SystemExit("[refuse] non-finite joint-description embedding")
        row = {t: i for i, t in enumerate(uniq)}
        sem = np.stack([E[row[d]] for d in descriptions]).astype(np.float32)
    cap = None
    if caption is not None:
        caption = " ".join(caption.split())
        enc.l2v.max_length = CAPTION_MAX_LENGTH
        enc.max_length = CAPTION_MAX_LENGTH
        n = int(enc.encoded_token_lengths([caption])[0])
        if n > CAPTION_STORE_MAX_TOKENS:
            raise SystemExit(f"[refuse] the prompt spans {n} tokens; training captions were at most {CAPTION_STORE_MAX_TOKENS}")
        H, M = enc.encode_tokens([caption], bs=1)
        h = H[0][M[0]]
        if h.shape[0] == 0 or not np.isfinite(h).all():
            raise SystemExit("[refuse] the prompt encoded to no / non-finite token states")
        cap = h.mean(0).astype(np.float32)
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cap, sem


# ------------------------------------------------------------------------------------------------ model
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ck or "model" not in ck:
        raise SystemExit(f"[refuse] {ckpt_path} is not a TopX weights file (expected the keys 'config' and 'model')")
    ca = dict(ck["config"])
    if [float(g) for g in ca.get("gains", [])] != [float(g) for g in GAINS]:
        raise SystemExit(f"[refuse] the weights' normalisation gains {ca.get('gains')} differ from this script's {GAINS}")
    model = InContextMotionDiT(
        in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"], d_text=4096, d_joint_sem=4096,
        use_struct_feats=bool(ca.get("struct_feats", False)),
        struct_world_rest=bool(ca.get("struct_world_rest", False)),
        use_dir_bias=bool(ca.get("dir_bias", False)), qk_norm=bool(ca.get("qk_norm", False)),
        use_ref_text=False, use_geo_bias=bool(ca.get("geo_bias", True)),
        use_spec_rope=bool(ca.get("spec_rope", False)), spec_rope_k=int(ca.get("spec_rope_k", 8)),
        spec_rope_hks=bool(ca.get("spec_rope_hks", False)),
        use_temporal_rope=bool(ca.get("temporal_rope", False)), trope_base=float(ca.get("trope_base", 700.0))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ca, ck


# ------------------------------------------------------------------------------------------------ serving
def rig_stats(rig):
    """(mean [J,17] float32, std [J,17] float32, channel_valid [J,17] bool) of the rest normalisation: the mean is the
    rest pose's own channels, the scale is the skeleton size over the gains for positions, velocities and root motion
    and 1 elsewhere.
    """
    P = np.asarray(rig["P_rest_global"], dtype=np.float64)
    J = len(rig["parents"])
    raw = np.zeros((J, 17), dtype=np.float64)
    raw[:, 0:3] = P
    raw[:, 0] -= P[0, 0]
    raw[:, 2] -= P[0, 2]
    raw[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)[None]
    raw[0, 15] = 1.0
    mu = raw.astype(np.float32)
    s = float(rig["s_rig"])
    scale = np.ones((J, 18), dtype=np.float32)
    scale[:, 0:3] = s / GAINS[0]
    scale[:, 9:12] = s / GAINS[1]
    scale[0, 13:15] = s / GAINS[2]
    sd = np.ascontiguousarray((scale - _STD_FLOOR).astype(np.float32)[:, :17]).astype(np.float32)
    cv = np.zeros((J, 17), dtype=bool)
    cv[:, :13] = True
    cv[0, 13:17] = True
    cv[np.asarray(rig.get("serve_kinds", rig["rotation_source_kind"])).astype(str) == "fixed_dof", 3:9] = False
    return mu, sd, cv


def rest_demo(rig, mu, sd, cv):
    """[J,18] the one-frame rest demo, normalised and clamped to +-REST_DEMO_CLAMP."""
    P = np.asarray(rig["P_rest_global"], dtype=np.float64)
    J = len(rig["parents"])
    raw = np.zeros((J, 17), dtype=np.float64)
    q = P.copy()
    q[:, 0] -= P[0, 0]
    q[:, 2] -= P[0, 2]
    raw[:, 0:3] = q
    raw[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)[None]
    raw[0, 15] = 1.0
    raw[~cv] = 0.0
    raw[~cv] = np.asarray(mu, dtype=np.float64)[~cv]
    n = (raw.astype(np.float32) - mu) / (sd + _STD_FLOOR)
    d = np.concatenate([n, np.ones((J, 1), np.float32)], axis=1).astype(np.float32)
    return np.clip(d, -REST_DEMO_CLAMP, REST_DEMO_CLAMP)


def build_item(rig, ca, text_emb, joint_sem, frames):
    """The model input for (this rig, its rest demo, a target of `frames` frames); the sampler replaces the target
    frames with noise, so their content is never read.
    """
    Tt = int(ca["target_frames"])
    if not 1 <= frames <= Tt:
        raise SystemExit(f"[refuse] --frames must be in [1, {Tt}] (the checkpoint's target window)")
    par = np.asarray(rig["parents"], dtype=np.int64)
    J = len(par)
    mu, sd, cv = rig_stats(rig)
    x = np.zeros((1 + Tt, J, 18), dtype=np.float32)
    x[0] = rest_demo(rig, mu, sd, cv)
    t_valid = np.zeros(Tt, dtype=bool)
    t_valid[:frames] = True
    geo_raw = hop_matrix(par)
    item = {"x": torch.from_numpy(x),
            "is_target": torch.from_numpy(np.concatenate([np.zeros(1, bool), t_valid])),
            "frame_valid": torch.from_numpy(np.concatenate([np.ones(1, bool), t_valid])),
            "geodesic": torch.from_numpy(np.clip(geo_raw, 0.0, GEODESIC_CLIP)),
            "object_type": "deploy", "motion_id": "deploy", "demo_id": "rest", "n_joints": J, "lock_denominator": 0.0,
            "text": torch.as_tensor(np.asarray(text_emb, dtype=np.float32)).float(), "is_identity": False,
            "joint_sem": torch.as_tensor(np.asarray(joint_sem, dtype=np.float32)).float()}
    if tuple(item["joint_sem"].shape) != (J, 4096) or tuple(item["text"].shape) != (4096,):
        raise SystemExit(f"[refuse] joint_sem {tuple(item['joint_sem'].shape)} / text {tuple(item['text'].shape)}: "
                         f"expected ({J}, 4096) / (4096,)")
    off32 = np.asarray(rig["offset_parent_local"], dtype=np.float32)
    if bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)):
        feats, ud = _graph_v2_tables(par, off32, geo_raw)
        if bool(ca.get("struct_world_rest", False)):
            feats = np.concatenate([feats, _world_rest_feats(par, off32, np.asarray(rig["P_rest_global"], np.float32))], 1)
        item["struct_feats"] = torch.from_numpy(feats)
        item["updown"] = torch.from_numpy(ud)
    if bool(ca.get("spec_rope", False)):
        from src.data.skeleton_spectral import heat_kernel_signature, hks_scales
        K = int(ca.get("spec_rope_k", 8))
        spec = heat_kernel_signature(par, hks_scales(K))
        item["spectral_feats"] = torch.from_numpy(spec)
    return item, (mu, sd, cv)


@torch.no_grad()
def generate(model, ca, rig, text_emb, joint_sem, frames, seed, steps, cfg_text, device):
    """-> (normalised output [frames, J, 17] float32, decoded dict)."""
    item, (mu, sd, cv) = build_item(rig, ca, text_emb, joint_sem, frames)
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
    J = int(item["n_joints"])
    torch.manual_seed(seed)
    g2kw = {k: b[k] for k in ("struct_feats", "updown", "spectral_feats") if k in b}
    cvt = torch.zeros(1, b["x"].shape[2], 17, dtype=torch.bool, device=device)
    cvt[0, :J] = torch.from_numpy(cv).to(device)
    g2kw["channel_valid"] = cvt
    g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
    gen = sample(model, b["x"][..., :17].contiguous(), b["is_target"], steps, cfg_text=cfg_text, demo_frames=1,
                 joint_bias=b["joint_bias"], frame_valid=b["frame_valid"], joint_valid=b["joint_valid"],
                 text=b["text"], joint_sem=b["joint_sem"], **g2kw)
    seg = gen[0].float().cpu().numpy()[1:1 + frames, :J]
    return seg, decode(rig, seg, mu, sd)


def decode(rig, seg, mu, sd):
    """De-normalise, then decode the position channels directly and the rotations by forward kinematics."""
    J = seg.shape[1]
    raw = (seg[..., :17] * (sd[:J][None] + _STD_FLOOR) + mu[:J][None]).astype(np.float64)
    dec = decode_ktjd17(raw, parents=rig["parents"], R_rest_global=rig["R_rest_global"],
                        R_rest_local=rig["R_rest_local"], offset_parent_local=rig["offset_parent_local"],
                        rotation_source_kind=rig["rotation_source_kind"], strict_gt=False)
    W = np.matmul(dec.global_rotations, np.swapaxes(np.asarray(rig["R_rest_global"], dtype=np.float64), -1, -2)[None])
    return {"positions_direct": dec.positions_direct, "positions_fk": dec.positions_fk, "rest_delta": W,
            "degenerate_6d": np.asarray(dec.model_d6_degenerate)}


# ------------------------------------------------------------------------------------------------ BVH out
def write_bvh(path, rig, dec, root_from="direct"):
    """Animate the rig as a BVH in its input axes and units. BVH input: its own hierarchy (offsets, End Sites, channel
    orders; non-root position channels hold their rest-frame translation); npz input: a hierarchy of the rest bone
    vectors with identity rest frames, root 6 channels, others ZXY. Joint j's global frame at t is W_j(t) G0_j (W = the
    world rotation from rest, G0 = the joint's rest-frame global rotation in the output hierarchy)."""
    W = dec["rest_delta"]
    root = dec["positions_direct" if root_from == "direct" else "positions_fk"][:, 0]
    T, J = W.shape[:2]
    bv = rig.get("bvh")
    if bv is not None:
        C, shift, nodes, jidx = bv["C"], bv["shift"], bv["nodes"], bv["jidx"]
        Wsrc = np.einsum("ab,tjbc,cd->tjad", C.T, W, C)
        G0 = bv["G0"][jidx]
        Tloc = bv["T0"]
        root_src = (root + shift) @ C                    # C^T applied to row vectors
        chans = [n.channels for n in nodes]
    else:
        nodes, jidx = [], list(range(J))
        par = np.asarray(rig["parents"], dtype=np.int64)
        P = np.asarray(rig["P_rest_global"], dtype=np.float64)
        for j in range(J):
            n = BvhNode(re.sub(r"\s+", "_", rig["joint_names"][j]), -1 if j == 0 else int(par[j]), False)
            n.offset = np.zeros(3) if j == 0 else P[j] - P[par[j]]      # root channels are absolute positions
            n.channels = (["xposition", "yposition", "zposition"] if j == 0 else []) + ["zrotation", "xrotation", "yrotation"]
            nodes.append(n)
        kids = {int(p) for p in par[1:]}
        for j in range(J):
            if j not in kids:
                e = BvhNode(f"{nodes[j].name}_End", j, True)
                inc = P[j] - P[par[j]] if j > 0 else np.array([0.0, 0.1, 0.0])
                e.offset = 0.25 * inc
                nodes.append(e)
        order = sorted(range(len(nodes)), key=lambda k: _dfs_key(nodes, k))
        pos_of = {k: i for i, k in enumerate(order)}
        nodes = [nodes[k] for k in order]
        for n in nodes:
            n.parent = -1 if n.parent < 0 else pos_of[n.parent]
        jidx = [pos_of[j] for j in range(J)]
        Wsrc, G0 = W, np.tile(np.eye(3), (J, 1, 1))
        Tloc = np.array([n.offset for n in nodes])
        root_src = root.copy()
        chans = [n.channels for n in nodes]
    G = np.einsum("tjab,jbc->tjac", Wsrc, G0)
    j_of_node = {k: i for i, k in enumerate(jidx)}
    rows = np.zeros((T, sum(len(c) for c in chans)))
    col = 0
    for k, n in enumerate(nodes):
        if n.is_end or not n.channels:
            continue
        j = j_of_node[k]
        pj = n.parent
        L = G[:, j] if pj < 0 else np.einsum("tba,tbc->tac", G[:, j_of_node[pj]], G[:, j])
        rot_axes = "".join(c[0] for c in n.channels if c.endswith("rotation"))
        ang = euler_from_matrix(L, rot_axes) if rot_axes else np.zeros((T, 0))
        ang = np.degrees(np.unwrap(np.radians(ang), axis=0))       # continuous curves (no +-360 jumps between frames)
        ri = 0
        for c in n.channels:
            if c.endswith("position"):
                rows[:, col] = (root_src if pj < 0 else np.broadcast_to(Tloc[k], (T, 3)))[:, "xyz".index(c[0])]
            else:
                rows[:, col] = ang[:, ri]
                ri += 1
            col += 1
    out = ["HIERARCHY"]
    depth_of = {}
    for k, n in enumerate(nodes):
        d = 0 if n.parent < 0 else depth_of[n.parent] + 1
        depth_of[k] = d
    nd = max(6, 5 + int(np.ceil(-np.log10(float(rig["s_rig"])))))   # 6 decimals unless the rig is under ~0.1 units
    stack = []
    for k, n in enumerate(nodes):
        while stack and stack[-1] != n.parent:
            stack.pop()
            out.append("  " * len(stack) + "}")
        ind = "  " * len(stack)
        if n.is_end:
            out.append(f"{ind}End Site")
        else:
            out.append(f"{ind}{'ROOT' if n.parent < 0 else 'JOINT'} {n.name}")
        out.append(f"{ind}{{")
        out.append(f"{ind}  OFFSET {n.offset[0]:.{nd}f} {n.offset[1]:.{nd}f} {n.offset[2]:.{nd}f}")
        if not n.is_end:
            out.append(f"{ind}  CHANNELS {len(n.channels)}" + "".join(" " + c[0].upper() + c[1:] for c in n.channels))
        stack.append(k)
    while stack:
        stack.pop()
        out.append("  " * len(stack) + "}")
    out += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / FPS:.8f}"]
    out += [" ".join(f"{v:.{nd}f}" for v in r) for r in rows]
    tmp = Path(str(path) + ".tmp")
    if Path(path).exists():
        Path(path).unlink()                              # never leave a previous run's BVH beside this run's npz / gif
    tmp.write_text("\n".join(out) + "\n")
    # read it back: FK of the written file must reproduce the generated FK positions (in the output axes); the file is
    # moved into place only after that
    nodes2, fr2, _ = read_bvh(tmp)
    if len(nodes2) != len(nodes) or len(fr2) != T:
        raise SystemExit(f"[refuse] the written BVH reads back as {len(nodes2)} nodes / {len(fr2)} frames")
    # the file holds the nodes in the written (depth-first) order; jidx maps rig joint j -> its node there
    got = np.stack([bvh_frame_fk(nodes2, fr2[t])[0][jidx] for t in range(T)])
    want = dec["positions_fk"].copy()                    # its root row IS the direct root (decode_ktjd17 FK)
    if bv is not None:
        want = (want + bv["shift"]) @ bv["C"]
    scale = float(rig["s_rig"])
    err = float(np.abs(got - want).max()) / scale
    if err > 1e-4:
        raise SystemExit(f"[refuse] the written BVH does not reproduce the generated FK positions: {err:.2e} x s_rig "
                         f"(left at {tmp})")
    tmp.replace(path)
    return err


def _dfs_key(nodes, k):
    path = []
    while k >= 0:
        path.append(k)
        k = nodes[k].parent
    return tuple(reversed(path))


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--skeleton", required=True, help="BVH, or a KTJD-17 skeleton .npz")
    ap.add_argument("--up", default="+Y", help="BVH axis pointing up (+X -X +Y -Y +Z -Z)")
    ap.add_argument("--forward", default="+Z", help="BVH axis the creature faces")
    ap.add_argument("--rest_frame", type=int, default=0, help="BVH frame holding the rest pose; -1 = OFFSETs only")
    ap.add_argument("--keep_rest_rotations", action="store_true",
                    help="npz input: serve the file's own rest frames instead of re-deriving them (corpus rigs)")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text_emb", default=None, help=".npy [4096] precomputed caption embedding")
    ap.add_argument("--descriptions", default=None, help="JSON {joint name: sentence}")
    ap.add_argument("--joint_sem", default=None, help=".npy [J,4096] precomputed joint-description embeddings")
    ap.add_argument("--lexicon", default=str(DEFAULT_LEXICON))
    ap.add_argument("--body_plan", default=None,
                    choices=[None, "bipedal", "quadrupedal", "insectoid", "avian", "marine", "serpentine",
                             "articulated_rigid"], help="only words the structural descriptions")
    ap.add_argument("--describe_only", action="store_true", help="print the rig and its joint descriptions, then stop")
    ap.add_argument("--frames", type=int, default=120, help="output length at 30 fps (<= 240)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tf32", type=int, choices=(0, 1), default=1,
                    help="TF32 matrix products on CUDA (1, as in the paper's evaluation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="output file stem (default: <skeleton stem>_s<seed>)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--allow_axis_mismatch", action="store_true",
                    help="run although the joint-name sides contradict --up / --forward")
    a = ap.parse_args()
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if (a.text is None) == (a.text_emb is None) and not a.describe_only:
        raise SystemExit("[refuse] give exactly one of --text / --text_emb")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sk = Path(a.skeleton)
    name = a.name or f"{sk.stem}_s{a.seed}"
    outs = [out / f"{name}{ext}" for ext in (".npz", ".gif", ".bvh", ".descriptions.json", ".joint_sem.npy",
                                              ".text_emb.npy", ".rest.gif")]
    if any(o.resolve() == sk.resolve() for o in outs):
        raise SystemExit(f"[refuse] an output would overwrite the input skeleton {sk}: choose another --out / --name")
    if not a.force and not a.describe_only and any(o.exists() for o in outs[:3]):
        raise SystemExit(f"[refuse] {out / name}.* exists: pass --force or another --name")
    if a.text is not None and not a.text.strip().lower().startswith("an object"):
        print('[topx] note: every training caption reads "An object <does something>." -- this prompt is phrased '
              'otherwise; on an unseen rig compare it with the "An object ..." phrasing (see the README)', flush=True)

    if sk.suffix.lower() == ".bvh":
        rig = rig_from_bvh(sk, a.up, a.forward, a.rest_frame)
        print(f"[topx] rest pose = {'frame ' + str(a.rest_frame) if a.rest_frame >= 0 else 'the OFFSETs with zero rotations'} "
              f"of the BVH -- check {name}.rest.gif (--describe_only) before trusting it", flush=True)
    else:
        rig = rig_from_npz(sk, reconvention=not a.keep_rest_rotations)
    J = len(rig["parents"])
    if J > 142:
        print(f"[topx] WARNING: {J} joints -- the training rigs have at most 142", flush=True)
    agree, n_sided, fwd = side_check(rig)
    if n_sided >= 4:
        hint = ""
        if fwd is not None:
            if "bvh" in rig:
                f_src = rig["bvh"]["C"].T @ fwd
                cand = [k for k in AXES if abs(float(AXES[k] @ AXES[a.up])) < 0.5]
                hint = f" (the names imply --forward={max(cand, key=lambda k: float(AXES[k] @ f_src))})"
        msg = f"{agree}/{n_sided} side-named joints lie on their named side of the model's frame (+X = left)"
        if agree < 0.8 * n_sided:
            why = (f"either --forward is wrong{hint} or the rig's Left/Right names are mirrored against its geometry -- "
                   f"look at {name}.rest.gif (--describe_only), then fix --up / --forward or pass --allow_axis_mismatch")
            if "bvh" in rig and not (a.allow_axis_mismatch or a.describe_only):
                raise SystemExit(f"[refuse] axes: {msg}: {why}")
            print(f"[topx] WARNING axes: {msg}: {why}", flush=True)
        else:
            print(f"[topx] axes: {msg}", flush=True)
    else:
        print(f"[topx] axes: only {n_sided} off-mid-line joints carry a side in their name -- --up / --forward are "
              f"unchecked; look at the rest pose (--describe_only)", flush=True)
    overrides = json.loads(Path(a.descriptions).read_text()) if a.descriptions else None
    if overrides:
        unknown = sorted(set(overrides) - set(rig["joint_names"]))
        if unknown:
            print(f"[topx] WARNING: --descriptions names {len(unknown)} joints the rig does not have: {unknown[:5]}",
                  flush=True)
    if rig.get("joint_descriptions") is not None and overrides is None and a.keep_rest_rotations:
        desc, dsrc = rig["joint_descriptions"], ["file"] * J
    else:
        desc, dsrc = describe_joints(rig, load_lexicon(a.lexicon), overrides, a.body_plan)
    P = rig["P_rest_global"]
    print(f"[topx] rig {name}: J={J}, s_rig={rig['s_rig']:.4g}, rest height {P[:, 1].max():.4g} "
          f"(extent x {np.ptp(P[:, 0]):.3g} y {np.ptp(P[:, 1]):.3g} z {np.ptp(P[:, 2]):.3g}; +Y up, +Z forward, +X left)",
          flush=True)
    counts = {s: dsrc.count(s) for s in sorted(set(dsrc))}
    print(f"[topx] joint descriptions: {counts}", flush=True)
    for nm, d, s in zip(rig["joint_names"], desc, dsrc):
        print(f"    {nm:40s} [{s:9s}] {d}", flush=True)
    (out / f"{name}.descriptions.json").write_text(json.dumps(dict(zip(rig["joint_names"], desc)), indent=1))
    if a.describe_only:
        _rest_png(out / f"{name}.rest.gif", rig, name)
        print(f"[topx] rest pose preview -> {out / (name + '.rest.gif')}", flush=True)
        return

    text_emb = np.load(a.text_emb).astype(np.float32) if a.text_emb else None
    joint_sem = np.load(a.joint_sem).astype(np.float32) if a.joint_sem else None
    if text_emb is None or joint_sem is None:
        cap, sem = encode_texts(a.text if text_emb is None else None, desc if joint_sem is None else None, a.device)
        text_emb = cap if text_emb is None else text_emb
        joint_sem = sem if joint_sem is None else joint_sem
        np.save(out / f"{name}.joint_sem.npy", joint_sem)
        if a.text is not None:
            np.save(out / f"{name}.text_emb.npy", text_emb)
    model, ca, ck = load_model(a.ckpt, a.device)
    seg, dec = generate(model, ca, rig, text_emb, joint_sem, a.frames, a.seed, a.steps, a.cfg_text, a.device)
    gap = float(np.abs(dec['positions_direct'] - dec['positions_fk']).max() / rig['s_rig'])
    print(f"[topx] generated {a.frames} frames; |direct - fk| max {gap:.3f} x s_rig; degenerate 6D cells "
          f"{int(np.asarray(dec['degenerate_6d']).sum())}", flush=True)
    if gap > 0.3:
        print("[topx] WARNING: the position and rotation decodes disagree by more than 0.3 x the rig size -- the prompt "
              "or the rig is probably out of the training distribution (the BVH plays the rotation decode)", flush=True)
    sha = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()
    np.savez(out / f"{name}.npz", positions_direct=dec["positions_direct"], positions_fk=dec["positions_fk"],
             rest_delta_rotations=dec["rest_delta"], model_output=seg, joint_names=np.array(rig["joint_names"]),
             parents=np.asarray(rig["parents"], dtype=np.int64), P_rest=np.asarray(P, dtype=np.float64), fps=np.array(FPS),
             text=np.array(a.text or f"<{a.text_emb}>"), seed=np.array(a.seed), steps=np.array(a.steps),
             cfg_text=np.array(a.cfg_text), ckpt=np.array(str(a.ckpt)), ckpt_sha256=np.array(sha),
             skeleton=np.array(rig["source"]), joint_descriptions=np.array(desc),
             frame=np.array("canonical: +Y up, +Z forward, +X left; rest pose grounded (min y = 0), root at x = z = 0"))
    from src.render import render_gif
    render_gif(out / f"{name}.gif", [("demo", "REST", np.asarray(P)[None]), ("gen_ric", "GEN position channels",
                                                                               dec["positions_direct"]),
                                     ("gen_fk", "GEN rotation FK", dec["positions_fk"])],
               [int(p) for p in rig["parents"]], a.text or "", name, fps=int(FPS))
    err = write_bvh(out / f"{name}.bvh", rig, dec)
    print(f"[topx] wrote {out / name}.npz / .gif / .bvh (BVH read-back FK error {err:.1e} x s_rig)", flush=True)


def _rest_png(path, rig, name):
    from src.render import render_gif
    P = np.asarray(rig["P_rest_global"])
    render_gif(path, [("demo", "REST (+Y up, +Z forward)", P[None])], [int(p) for p in rig["parents"]], "", name, fps=30)


if __name__ == "__main__":
    main()

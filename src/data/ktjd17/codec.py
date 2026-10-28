"""Numerical primitives of the KTJD-17 per-joint motion representation.

All conversion-facing functions operate in float64 and fail closed on non-finite or geometrically invalid inputs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp


KTJD17_D = 17
GT_D6_DEGENERACY_EPS = 1e-6
MODEL_D6_DEGENERACY_EPS = 1e-8


class Ktjd17CodecError(RuntimeError):
    """An input cannot satisfy the numerical KTJD-17 contract."""


@dataclasses.dataclass(frozen=True)
class SmootherConfig:
    """Full-clip root-XZ smoother configuration."""

    method: str = "butterworth_filtfilt"
    order: int = 4
    cutoff_hz: float = 1.0
    padtype: str = "odd"
    padlen: int = 15
    short_clip_cycles: float = 3.0
    short_clip_fallback: str = "ols_line"

    def validate(self, fps: float) -> None:
        _positive_scalar("fps", fps)
        if self.method != "butterworth_filtfilt":
            raise Ktjd17CodecError(f"unsupported smoother method {self.method!r}")
        if self.order != 4:
            raise Ktjd17CodecError(f"the root smoother order must be 4, got {self.order}")
        cutoff = _positive_scalar("cutoff_hz", self.cutoff_hz)
        if cutoff >= 0.5 * float(fps):
            raise Ktjd17CodecError(
                f"cutoff_hz must be below Nyquist: {cutoff} >= {0.5 * float(fps)}"
            )
        if self.padtype != "odd" or self.padlen != 15:
            raise Ktjd17CodecError(
                "the root smoother requires padtype='odd', padlen=15"
            )
        if self.short_clip_cycles <= 0.0 or self.short_clip_fallback != "ols_line":
            raise Ktjd17CodecError("invalid short-clip smoother rule")

    def as_schema_params(self) -> dict[str, object]:
        return {
            "order": self.order,
            "cutoff_hz": self.cutoff_hz,
            "padtype": self.padtype,
            "padlen": self.padlen,
            "short_clip_cycles": self.short_clip_cycles,
        }

    @property
    def schema_short_clip_rule(self) -> str:
        return (
            "if T < short_clip_cycles*round(fps_target/cutoff_hz), "
            "fit independent OLS lines to root x/z; T=1 identity"
        )


@dataclasses.dataclass(frozen=True)
class ResampledMotion:
    root_positions: np.ndarray
    local_rotations: np.ndarray
    source_times: np.ndarray
    target_times: np.ndarray
    mode: str


@dataclasses.dataclass(frozen=True)
class EncodedChannels:
    """Float64 semantic channels before storage quantization."""

    motion: np.ndarray
    heading_valid: np.ndarray
    origin_xz: np.ndarray
    positions_clip: np.ndarray
    positions_absolute: np.ndarray
    global_rotations: np.ndarray
    local_rotations: np.ndarray
    smooth_root_absolute: np.ndarray
    ground_shift_y: float


def _positive_scalar(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise Ktjd17CodecError(f"{name} must be finite and >0, got {value!r}")
    return number


def require_float64_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise Ktjd17CodecError(f"{name} must be float64, got {array.dtype}")
    if not np.isfinite(array).all():
        location = np.argwhere(~np.isfinite(array))[0].tolist()
        raise Ktjd17CodecError(f"{name} has non-finite data at {location}")
    return array


def validate_parent_tree(parents: Sequence[int] | np.ndarray, joint_count: int) -> np.ndarray:
    values = np.asarray(parents, dtype=np.int64)
    if values.shape != (joint_count,):
        raise Ktjd17CodecError(
            f"parents must have shape ({joint_count},), got {values.shape}"
        )
    if joint_count <= 0 or int(values[0]) != -1:
        raise Ktjd17CodecError("physical joint 0 must be the single root")
    for child in range(1, joint_count):
        parent = int(values[child])
        if not 0 <= parent < child:
            raise Ktjd17CodecError(
                f"parent-before-child violated at child={child}, parent={parent}"
            )
    return values


def rotation_diagnostics(rotations: np.ndarray) -> dict[str, float]:
    matrices = require_float64_finite("rotations", np.asarray(rotations))
    if matrices.shape[-2:] != (3, 3):
        raise Ktjd17CodecError(f"rotations must end in [3,3], got {matrices.shape}")
    gram = np.matmul(np.swapaxes(matrices, -1, -2), matrices)
    determinants = np.linalg.det(matrices)
    return {
        "orthogonality_max_abs": float(
            np.max(np.abs(gram - np.eye(3, dtype=np.float64)))
        ),
        "determinant_min": float(np.min(determinants)),
        "determinant_max": float(np.max(determinants)),
    }


def require_so3(name: str, rotations: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    matrices = require_float64_finite(name, np.asarray(rotations))
    diagnostics = rotation_diagnostics(matrices)
    if diagnostics["orthogonality_max_abs"] > tolerance:
        raise Ktjd17CodecError(
            f"{name} is not orthogonal: {diagnostics['orthogonality_max_abs']}"
        )
    if (
        abs(diagnostics["determinant_min"] - 1.0) > tolerance
        or abs(diagnostics["determinant_max"] - 1.0) > tolerance
    ):
        raise Ktjd17CodecError(
            f"{name} is not right-handed SO(3): "
            f"det=[{diagnostics['determinant_min']},{diagnostics['determinant_max']}]"
        )
    return matrices


def encode_column_cont6d(rotations: np.ndarray) -> np.ndarray:
    """Encode first two *columns* as R00,R10,R20,R01,R11,R21."""
    matrices = require_so3("rotations_to_encode", np.asarray(rotations))
    encoded = np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)
    return require_float64_finite("encoded_column_cont6d", encoded)


def decode_column_cont6d(
    d6: np.ndarray,
    *,
    degeneracy_eps: float = GT_D6_DEGENERACY_EPS,
    strict: bool = True,
) -> np.ndarray:
    """Decode the exact KTJD-17 Gram-Schmidt column-cont6d contract."""
    values = require_float64_finite("column_cont6d", np.asarray(d6))
    if values.shape[-1] != 6:
        raise Ktjd17CodecError(f"d6 must end in 6, got {values.shape}")
    epsilon = _positive_scalar("degeneracy_eps", degeneracy_eps)
    a1 = values[..., :3]
    a2 = values[..., 3:]
    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    bad1 = norm1[..., 0] < epsilon
    if strict and np.any(bad1):
        location = np.argwhere(bad1)[0].tolist()
        raise Ktjd17CodecError(
            f"d6 first column is degenerate at {location}: "
            f"norm={float(norm1[tuple(location) + (0,)])}"
        )
    safe_norm1 = np.maximum(norm1, MODEL_D6_DEGENERACY_EPS)
    b1 = a1 / safe_norm1
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    bad2 = norm2[..., 0] < epsilon
    if strict and np.any(bad2):
        location = np.argwhere(bad2)[0].tolist()
        raise Ktjd17CodecError(
            f"d6 second column is degenerate at {location}: "
            f"norm={float(norm2[tuple(location) + (0,)])}"
        )
    safe_norm2 = np.maximum(norm2, MODEL_D6_DEGENERACY_EPS)
    b2 = u2 / safe_norm2
    b3 = np.cross(b1, b2)
    matrices = np.stack((b1, b2, b3), axis=-1)
    return require_float64_finite("decoded_column_cont6d", matrices)


def global_to_local_rotations(parents: np.ndarray, global_rotations: np.ndarray) -> np.ndarray:
    matrices = require_so3("global_rotations", np.asarray(global_rotations))
    if matrices.ndim != 4:
        raise Ktjd17CodecError(
            f"global_rotations must be [T,J,3,3], got {matrices.shape}"
        )
    parent_array = validate_parent_tree(parents, matrices.shape[1])
    local = np.empty_like(matrices)
    local[:, 0] = matrices[:, 0]
    for child in range(1, matrices.shape[1]):
        parent = int(parent_array[child])
        local[:, child] = np.matmul(
            np.swapaxes(matrices[:, parent], -1, -2), matrices[:, child]
        )
    return require_so3("local_rotations", local)


def local_to_global_rotations(parents: np.ndarray, local_rotations: np.ndarray) -> np.ndarray:
    matrices = require_so3("local_rotations", np.asarray(local_rotations))
    if matrices.ndim != 4:
        raise Ktjd17CodecError(
            f"local_rotations must be [T,J,3,3], got {matrices.shape}"
        )
    parent_array = validate_parent_tree(parents, matrices.shape[1])
    global_rotations = np.empty_like(matrices)
    global_rotations[:, 0] = matrices[:, 0]
    for child in range(1, matrices.shape[1]):
        parent = int(parent_array[child])
        global_rotations[:, child] = np.matmul(
            global_rotations[:, parent], matrices[:, child]
        )
    return require_so3("global_rotations", global_rotations)


def fk_from_global_rotations(
    parents: np.ndarray,
    root_positions: np.ndarray,
    global_rotations: np.ndarray,
    offset_parent_local: np.ndarray,
) -> np.ndarray:
    roots = require_float64_finite("root_positions", np.asarray(root_positions))
    rotations = require_so3("global_rotations", np.asarray(global_rotations))
    offsets = require_float64_finite(
        "offset_parent_local", np.asarray(offset_parent_local)
    )
    if roots.ndim != 2 or roots.shape[-1] != 3:
        raise Ktjd17CodecError(f"root_positions must be [T,3], got {roots.shape}")
    if rotations.ndim != 4 or rotations.shape[0] != roots.shape[0]:
        raise Ktjd17CodecError(
            f"rotation/root shape mismatch: {rotations.shape} vs {roots.shape}"
        )
    joint_count = rotations.shape[1]
    parents_array = validate_parent_tree(parents, joint_count)
    if offsets.shape != (joint_count, 3):
        raise Ktjd17CodecError(
            f"offset_parent_local must be [{joint_count},3], got {offsets.shape}"
        )
    positions = np.empty((roots.shape[0], joint_count, 3), dtype=np.float64)
    positions[:, 0] = roots
    for child in range(1, joint_count):
        parent = int(parents_array[child])
        positions[:, child] = positions[:, parent] + np.einsum(
            "tij,j->ti", rotations[:, parent], offsets[child]
        )
    return require_float64_finite("fk_positions", positions)


def timestamp_grid(frame_count: int, fps: float) -> np.ndarray:
    if frame_count <= 0:
        raise Ktjd17CodecError(f"frame_count must be positive, got {frame_count}")
    rate = _positive_scalar("fps", fps)
    return np.arange(frame_count, dtype=np.float64) / rate


def resample_root_and_local_rotations(
    root_positions: np.ndarray,
    local_rotations: np.ndarray,
    *,
    fps_src: float,
    fps_target: float,
) -> ResampledMotion:
    """Timestamp resample root translation + per-joint local SO(3) SLERP."""
    roots = require_float64_finite("root_positions", np.asarray(root_positions))
    rotations = require_so3("local_rotations", np.asarray(local_rotations))
    if roots.ndim != 2 or roots.shape[-1] != 3:
        raise Ktjd17CodecError(f"root_positions must be [T,3], got {roots.shape}")
    if rotations.ndim != 4 or rotations.shape[0] != roots.shape[0]:
        raise Ktjd17CodecError(
            f"root/local rotation shape mismatch: {roots.shape} vs {rotations.shape}"
        )
    source_fps = _positive_scalar("fps_src", fps_src)
    target_fps = _positive_scalar("fps_target", fps_target)
    source_times = timestamp_grid(roots.shape[0], source_fps)
    if source_fps == target_fps:
        return ResampledMotion(
            root_positions=roots.copy(),
            local_rotations=rotations.copy(),
            source_times=source_times,
            target_times=source_times.copy(),
            mode="exact_fps_identity_bypass",
        )
    if roots.shape[0] == 1:
        return ResampledMotion(
            root_positions=roots.copy(),
            local_rotations=rotations.copy(),
            source_times=source_times,
            target_times=np.asarray([0.0], dtype=np.float64),
            mode="single_frame_identity",
        )
    duration = float(source_times[-1])
    target_count = math.floor(duration * target_fps) + 1
    target_times = timestamp_grid(target_count, target_fps)
    if target_times[-1] > source_times[-1]:
        raise Ktjd17CodecError(
            "timestamp grid attempted extrapolation: "
            f"{target_times[-1]} > {source_times[-1]}"
        )
    target_roots = np.stack(
        [np.interp(target_times, source_times, roots[:, axis]) for axis in range(3)],
        axis=-1,
    )
    target_local = np.empty(
        (target_count, rotations.shape[1], 3, 3), dtype=np.float64
    )
    for joint in range(rotations.shape[1]):
        source_rotation = Rotation.from_matrix(rotations[:, joint])
        target_local[:, joint] = Slerp(source_times, source_rotation)(
            target_times
        ).as_matrix()
    require_so3("resampled_local_rotations", target_local)
    return ResampledMotion(
        root_positions=require_float64_finite("resampled_root_positions", target_roots),
        local_rotations=target_local,
        source_times=source_times,
        target_times=target_times,
        mode="timestamp_linear_root_local_so3_slerp_then_fk",
    )


def _ols_line(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return values.copy()
    x = np.arange(values.shape[0], dtype=np.float64)
    design = np.stack((x, np.ones_like(x)), axis=-1)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return design @ coefficients


def smooth_root_xz(
    root_xz: np.ndarray,
    *,
    fps: float,
    config: SmootherConfig,
) -> tuple[np.ndarray, str]:
    values = require_float64_finite("root_xz", np.asarray(root_xz))
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] <= 0:
        raise Ktjd17CodecError(f"root_xz must be nonempty [T,2], got {values.shape}")
    rate = _positive_scalar("fps", fps)
    config.validate(rate)
    if values.shape[0] == 1:
        return values.copy(), "single_frame_identity"
    length_floor = round(rate / config.cutoff_hz)
    short_limit = config.short_clip_cycles * length_floor
    if values.shape[0] < short_limit or values.shape[0] <= config.padlen:
        return require_float64_finite("smooth_root_xz", _ols_line(values)), "ols_line"
    normalized_cutoff = config.cutoff_hz / (0.5 * rate)
    b, a = butter(config.order, normalized_cutoff, btype="low", analog=False)
    smoothed = filtfilt(
        b,
        a,
        values,
        axis=0,
        padtype=config.padtype,
        padlen=config.padlen,
    )
    return require_float64_finite("smooth_root_xz", smoothed), "butterworth_filtfilt"


def world_velocity(positions: np.ndarray, *, fps: float) -> np.ndarray:
    values = require_float64_finite("positions", np.asarray(positions))
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[0] <= 0:
        raise Ktjd17CodecError(f"positions must be nonempty [T,J,3], got {values.shape}")
    rate = _positive_scalar("fps", fps)
    velocity = np.zeros_like(values)
    if values.shape[0] >= 2:
        velocity[:-1] = (values[1:] - values[:-1]) * rate
        velocity[-1] = velocity[-2]
    return require_float64_finite("world_velocity", velocity)


def joint_proxy_contact(
    positions: np.ndarray,
    velocity: np.ndarray,
    *,
    s_rig: float,
    tau_h: float,
    tau_v: float,
) -> np.ndarray:
    points = require_float64_finite("positions", np.asarray(positions))
    speeds = require_float64_finite("velocity", np.asarray(velocity))
    if points.shape != speeds.shape or points.ndim != 3 or points.shape[-1] != 3:
        raise Ktjd17CodecError(
            f"contact position/velocity shape mismatch: {points.shape} vs {speeds.shape}"
        )
    scale = _positive_scalar("s_rig", s_rig)
    height_threshold = _positive_scalar("tau_h", tau_h)
    speed_threshold = _positive_scalar("tau_v", tau_v)
    height_norm = points[..., 1] / scale
    speed_norm = np.linalg.norm(speeds, axis=-1) / scale
    contact = (height_norm <= height_threshold) & (speed_norm <= speed_threshold)
    if points.shape[0] >= 2:
        contact[-1] = contact[-2]
    return contact.astype(np.float64)


def heading_from_global_rotation(
    global_rotations: np.ndarray,
    *,
    carrier_joint: int,
    u_forward_local: np.ndarray,
    eps_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = require_so3("global_rotations", np.asarray(global_rotations))
    if rotations.ndim != 4:
        raise Ktjd17CodecError(
            f"global_rotations must be [T,J,3,3], got {rotations.shape}"
        )
    carrier = int(carrier_joint)
    if not 0 <= carrier < rotations.shape[1]:
        raise Ktjd17CodecError(f"heading carrier {carrier} is outside J={rotations.shape[1]}")
    forward_local = require_float64_finite(
        "u_forward_local", np.asarray(u_forward_local)
    )
    if forward_local.shape != (3,):
        raise Ktjd17CodecError(
            f"u_forward_local must be [3], got {forward_local.shape}"
        )
    forward_norm = float(np.linalg.norm(forward_local))
    if abs(forward_norm - 1.0) > 1e-10:
        raise Ktjd17CodecError(
            f"u_forward_local must be unit length, got {forward_norm}"
        )
    epsilon = _positive_scalar("eps_h", eps_h)
    forward_world = np.einsum(
        "tij,j->ti", rotations[:, carrier], forward_local
    )
    horizontal_norm = np.hypot(forward_world[:, 0], forward_world[:, 2])
    valid = horizontal_norm >= epsilon
    heading = np.zeros((rotations.shape[0], 2), dtype=np.float64)
    heading[valid, 0] = forward_world[valid, 2] / horizontal_norm[valid]
    heading[valid, 1] = forward_world[valid, 0] / horizontal_norm[valid]
    return (
        require_float64_finite("heading", heading),
        valid,
        require_float64_finite("heading_horizontal_norm", horizontal_norm),
    )


def encode_ktjd17_channels(
    *,
    parents: np.ndarray,
    root_positions: np.ndarray,
    local_rotations: np.ndarray,
    offset_parent_local: np.ndarray,
    R_rest_global: np.ndarray,
    s_rig: float,
    fps_target: float,
    smoother: SmootherConfig,
    contact_tau_h: float,
    contact_tau_v: float,
    heading_carrier_joint: int,
    u_forward_local: np.ndarray,
    heading_eps_h: float,
) -> EncodedChannels:
    """Encode already-resampled canonical root/local rotations to KTJD-17."""
    roots = require_float64_finite("root_positions", np.asarray(root_positions))
    local = require_so3("local_rotations", np.asarray(local_rotations))
    if local.shape[:2] != (roots.shape[0], len(parents)):
        raise Ktjd17CodecError(
            f"root/local/topology mismatch: {roots.shape}, {local.shape}, J={len(parents)}"
        )
    global_rotations = local_to_global_rotations(parents, local)
    positions = fk_from_global_rotations(
        parents, roots, global_rotations, offset_parent_local
    )
    ground_shift_y = -float(np.min(positions[..., 1]))
    positions_absolute = positions.copy()
    positions_absolute[..., 1] += ground_shift_y

    smooth_absolute, _ = smooth_root_xz(
        positions_absolute[:, 0][:, [0, 2]], fps=fps_target, config=smoother
    )
    q_position = positions_absolute.copy()
    q_position[..., 0] -= smooth_absolute[:, None, 0]
    q_position[..., 2] -= smooth_absolute[:, None, 1]

    delta = np.matmul(
        global_rotations,
        np.swapaxes(
            require_so3("R_rest_global", np.asarray(R_rest_global)), -1, -2
        )[None],
    )
    d6 = encode_column_cont6d(delta)
    velocity = world_velocity(positions_absolute, fps=fps_target)
    contact = joint_proxy_contact(
        positions_absolute,
        velocity,
        s_rig=s_rig,
        tau_h=contact_tau_h,
        tau_v=contact_tau_v,
    )
    heading, heading_valid, _ = heading_from_global_rotation(
        global_rotations,
        carrier_joint=heading_carrier_joint,
        u_forward_local=u_forward_local,
        eps_h=heading_eps_h,
    )

    origin_xz = smooth_absolute[0].copy()
    smooth_stored = smooth_absolute - origin_xz[None]
    positions_clip = positions_absolute.copy()
    positions_clip[..., 0] -= origin_xz[0]
    positions_clip[..., 2] -= origin_xz[1]
    motion = np.zeros(
        (positions.shape[0], positions.shape[1], KTJD17_D), dtype=np.float64
    )
    motion[..., 0:3] = q_position
    motion[..., 3:9] = d6
    motion[..., 9:12] = velocity
    motion[..., 12] = contact
    motion[:, 0, 13:15] = smooth_stored
    motion[:, 0, 15:17] = heading
    if np.any(motion[:, 1:, 13:17] != 0.0):
        raise Ktjd17CodecError("non-root root-global channels must be exact zero")
    return EncodedChannels(
        motion=require_float64_finite("ktjd17_motion", motion),
        heading_valid=heading_valid,
        origin_xz=require_float64_finite("origin_xz", origin_xz),
        positions_clip=require_float64_finite("positions_clip", positions_clip),
        positions_absolute=require_float64_finite(
            "positions_absolute", positions_absolute
        ),
        global_rotations=global_rotations,
        local_rotations=local,
        smooth_root_absolute=smooth_absolute,
        ground_shift_y=ground_shift_y,
    )


def direct_decode_positions(motion: np.ndarray) -> np.ndarray:
    values = require_float64_finite("motion", np.asarray(motion))
    if values.ndim != 3 or values.shape[-1] != KTJD17_D:
        raise Ktjd17CodecError(f"motion must be [T,J,17], got {values.shape}")
    result = values[..., 0:3].copy()
    result[..., 0] += values[:, 0, 13][:, None]
    result[..., 2] += values[:, 0, 14][:, None]
    return require_float64_finite("direct_positions", result)


def restore_origin_xz(positions_clip: np.ndarray, origin_xz: np.ndarray) -> np.ndarray:
    positions = require_float64_finite("positions_clip", np.asarray(positions_clip))
    origin = require_float64_finite("origin_xz", np.asarray(origin_xz))
    if positions.ndim != 3 or positions.shape[-1] != 3 or origin.shape != (2,):
        raise Ktjd17CodecError(
            f"origin restore expects [T,J,3] + [2], got {positions.shape} + {origin.shape}"
        )
    result = positions.copy()
    result[..., 0] += origin[0]
    result[..., 2] += origin[1]
    return require_float64_finite("positions_absolute", result)

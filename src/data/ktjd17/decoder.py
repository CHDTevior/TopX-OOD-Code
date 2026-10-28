"""Independent frame-local decode paths for KTJD-17 v1."""

from __future__ import annotations

import dataclasses

import numpy as np

from .codec import (
    Ktjd17CodecError,
    decode_column_cont6d,
    direct_decode_positions,
    fk_from_global_rotations,
    require_float64_finite,
    require_so3,
    validate_parent_tree,
)


@dataclasses.dataclass(frozen=True)
class DecodedMotion:
    positions_direct: np.ndarray
    positions_fk: np.ndarray
    positions_direct_minus_fk: np.ndarray
    global_rotations: np.ndarray
    local_rotations: np.ndarray
    fixed_dof_overridden: np.ndarray
    model_d6_degenerate: np.ndarray


def _decode_global_rotations(
    motion: np.ndarray,
    *,
    parents: np.ndarray,
    R_rest_global: np.ndarray,
    R_rest_local: np.ndarray,
    rotation_source_kind: np.ndarray,
    strict_gt: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = require_float64_finite("motion", np.asarray(motion))
    joint_count = values.shape[1]
    parent_array = validate_parent_tree(parents, joint_count)
    rest_global = require_so3("R_rest_global", np.asarray(R_rest_global))
    rest_local = require_so3("R_rest_local", np.asarray(R_rest_local))
    if rest_global.shape != (joint_count, 3, 3) or rest_local.shape != (
        joint_count,
        3,
        3,
    ):
        raise Ktjd17CodecError("rest rotation shape does not match motion J")
    kinds = np.asarray(rotation_source_kind).astype(str)
    if kinds.shape != (joint_count,) or not set(kinds).issubset(
        {"animated_dof", "fixed_dof"}
    ):
        raise Ktjd17CodecError("rotation_source_kind has invalid values or shape")

    d6 = values[..., 3:9]
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    norm1 = np.linalg.norm(a1, axis=-1)
    b1 = a1 / np.maximum(norm1[..., None], 1e-8)
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(u2, axis=-1)
    degenerate = (norm1 < 1e-6) | (norm2 < 1e-6)
    delta = decode_column_cont6d(
        d6,
        degeneracy_eps=1e-6 if strict_gt else 1e-8,
        strict=strict_gt,
    )
    proposed_global = np.matmul(delta, rest_global[None])
    global_rotations = np.empty_like(proposed_global)
    local_rotations = np.empty_like(proposed_global)
    overridden = np.zeros((values.shape[0], joint_count), dtype=bool)
    for joint in range(joint_count):
        parent = int(parent_array[joint])
        if kinds[joint] == "fixed_dof":
            overridden[:, joint] = True
            if joint == 0:
                local_rotations[:, joint] = rest_local[joint]
                global_rotations[:, joint] = rest_local[joint]
            else:
                local_rotations[:, joint] = rest_local[joint]
                global_rotations[:, joint] = np.matmul(
                    global_rotations[:, parent], rest_local[joint]
                )
        else:
            global_rotations[:, joint] = proposed_global[:, joint]
            if joint == 0:
                local_rotations[:, joint] = global_rotations[:, joint]
            else:
                local_rotations[:, joint] = np.matmul(
                    np.swapaxes(global_rotations[:, parent], -1, -2),
                    global_rotations[:, joint],
                )
    return (
        require_float64_finite("decoded_global_rotations", global_rotations),
        require_float64_finite("decoded_local_rotations", local_rotations),
        overridden,
        degenerate,
    )


def decode_ktjd17(
    motion: np.ndarray,
    *,
    parents: np.ndarray,
    R_rest_global: np.ndarray,
    R_rest_local: np.ndarray,
    offset_parent_local: np.ndarray,
    rotation_source_kind: np.ndarray,
    strict_gt: bool = True,
) -> DecodedMotion:
    """Decode direct and FK paths without temporal integration."""
    values = require_float64_finite("motion", np.asarray(motion))
    if values.ndim != 3 or values.shape[-1] != 17:
        raise Ktjd17CodecError(f"motion must be [T,J,17], got {values.shape}")
    direct = direct_decode_positions(values)
    global_rotations, local_rotations, overridden, degenerate = (
        _decode_global_rotations(
            values,
            parents=parents,
            R_rest_global=R_rest_global,
            R_rest_local=R_rest_local,
            rotation_source_kind=rotation_source_kind,
            strict_gt=strict_gt,
        )
    )
    fk = fk_from_global_rotations(
        parents,
        direct[:, 0],
        global_rotations,
        offset_parent_local,
    )
    return DecodedMotion(
        positions_direct=direct,
        positions_fk=fk,
        positions_direct_minus_fk=require_float64_finite(
            "positions_direct_minus_fk", direct - fk
        ),
        global_rotations=global_rotations,
        local_rotations=local_rotations,
        fixed_dof_overridden=overridden,
        model_d6_degenerate=degenerate,
    )

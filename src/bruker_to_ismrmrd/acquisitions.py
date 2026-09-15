"""Build ISMRMRD acquisitions from decoded k-space data.

This module takes the already-prepared k-space array, dimension labels,
orientation data, and trajectory, and produces a list of
``ismrmrd.Acquisition`` objects with proper encoding indices, flags,
and metadata.
"""

from __future__ import annotations

import ismrmrd
import numpy as np

from bruker_to_ismrmrd.constants import DIMTYPE_TO_IDX
from bruker_to_ismrmrd.geometry import compute_orientation_and_position
from bruker_to_ismrmrd.parameters import get_float


def build_acquisitions(
    dataset: object,
    data: np.ndarray,
    dim_type: list[str],
    traj_data: np.ndarray | None = None,
) -> list[ismrmrd.Acquisition]:
    """Build a list of ISMRMRD acquisitions from k-space data.

    Parameters
    ----------
    dataset
        A ``brukerapi.dataset.Dataset`` instance (used for orientation
        and dwell-time parameters).
    data
        Decoded k-space array whose axes correspond to ``dim_type``.
    dim_type
        Dimension labels (e.g. ``['k_space_encode_step_0', 'k_space_encode_step_1',
        'object', 'repetition', 'channel']``).
    traj_data
        Optional trajectory array of shape ``(traj_dims, n_samples, n_projections)``,
        already scaled to encoding-step units.

    Returns
    -------
        List of ISMRMRD acquisitions, one per readout line.
    """
    if 'k_space_encode_step_0' not in dim_type:
        raise ValueError(f'No k_space_encode_step_0 in dim_type {dim_type}. Cannot build acquisitions.')

    k0_axis = dim_type.index('k_space_encode_step_0')
    ch_axis = dim_type.index('channel') if 'channel' in dim_type else None
    n_k0 = data.shape[k0_axis]
    n_coils = data.shape[ch_axis] if ch_axis is not None else 1
    center_sample = n_k0 // 2
    traj_dims = traj_data.shape[0] if traj_data is not None else 0

    # Identify loop dimensions (everything except k0 and channel)
    loop_axes = _get_loop_axes(dim_type, k0_axis, ch_axis, data.shape)
    loop_shape = tuple(size for _, _, size in loop_axes)
    n_total = int(np.prod(loop_shape)) if loop_shape else 1

    # Compute orientation and position
    n_objects = next((s for _, label, s in loop_axes if label == 'object'), 1)
    read_dir, phase_dir, slice_dir, position = compute_orientation_and_position(dataset, n_total, n_objects)

    # Get dwell time
    dwell_time = get_float(dataset, 'PVM_DigDw', 'DWELL')

    # Pre-compute loop sizes for flag determination
    n_k1_loop = next((s for _, label, s in loop_axes if label == 'k_space_encode_step_1'), 1)
    n_k2_loop = next((s for _, label, s in loop_axes if label == 'k_space_encode_step_2'), 1)
    n_obj_loop = next((s for _, label, s in loop_axes if label == 'object'), 1)

    acquisitions: list[ismrmrd.Acquisition] = []
    loop_iter = np.ndindex(*loop_shape) if loop_shape else [()]

    for acq_idx, multi_idx in enumerate(loop_iter):
        readout_data, idx_vals = _extract_readout(data, k0_axis, ch_axis, n_coils, n_k0, loop_axes, multi_idx)

        acq = ismrmrd.Acquisition()
        acq.resize(number_of_samples=n_k0, active_channels=n_coils, trajectory_dimensions=traj_dims)
        acq.available_channels = n_coils
        acq.center_sample = center_sample
        acq.data[:] = readout_data

        # Encoding indices
        for label, field_name in DIMTYPE_TO_IDX.items():
            if label in idx_vals:
                setattr(acq.idx, field_name, idx_vals[label])

        # Flags
        acq.flags = _compute_flags(idx_vals, n_k1_loop, n_k2_loop, n_obj_loop)

        # Orientation
        _set_orientation(acq, acq_idx, read_dir, phase_dir, slice_dir, position)

        # Trajectory
        if traj_data is not None:
            proj_idx = idx_vals.get('k_space_encode_step_1', 0)
            acq.traj[:] = traj_data[:, :n_k0, proj_idx].T

        # Dwell time
        if dwell_time is not None:
            acq.sample_time_us = float(dwell_time)

        acquisitions.append(acq)

    return acquisitions


def _get_loop_axes(
    dim_type: list[str],
    k0_axis: int,
    ch_axis: int | None,
    shape: tuple[int, ...],
) -> list[tuple[int, str, int]]:
    """Identify loop dimensions (all axes except k0 and channel).

    Returns
    -------
        List of ``(axis_index, label, size)`` tuples.
    """
    return [(axis, label, shape[axis]) for axis, label in enumerate(dim_type) if axis != k0_axis and axis != ch_axis]


def _extract_readout(
    data: np.ndarray,
    k0_axis: int,
    ch_axis: int | None,
    n_coils: int,
    n_k0: int,
    loop_axes: list[tuple[int, str, int]],
    multi_idx: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract a single readout line and return it with its index values.

    Returns
    -------
        Tuple of ``(readout_data, idx_vals)`` where ``readout_data`` has
        shape ``(n_coils, n_k0)`` and ``idx_vals`` maps dimension labels
        to their index values.
    """
    idx: list[int | slice] = [slice(None)] * data.ndim
    idx_vals: dict[str, int] = {}
    for (axis, label, _), dim_val in zip(loop_axes, multi_idx, strict=True):
        idx[axis] = dim_val
        idx_vals[label] = dim_val

    readout_line = data[tuple(idx)]

    # Reshape to (n_coils, n_k0)
    if ch_axis is not None and readout_line.ndim == 2:
        readout_data = readout_line.T if k0_axis < ch_axis else readout_line
    elif readout_line.ndim == 1:
        readout_data = readout_line.reshape(1, -1)
    else:
        readout_data = readout_line.reshape(n_coils, n_k0)

    return readout_data, idx_vals


def _compute_flags(idx_vals: dict[str, int], n_k1: int, n_k2: int, n_obj: int) -> int:
    """Compute ISMRMRD acquisition flags from encoding indices."""
    k1_val = idx_vals.get('k_space_encode_step_1', 0)
    k2_val = idx_vals.get('k_space_encode_step_2', 0)
    obj_val = idx_vals.get('object', 0)

    is_first_encode = k1_val == 0 and k2_val == 0
    is_last_encode = k1_val == n_k1 - 1 and k2_val == n_k2 - 1

    flags = 0
    if is_first_encode:
        flags |= ismrmrd.ACQ_FIRST_IN_SLICE
    if is_last_encode:
        flags |= ismrmrd.ACQ_LAST_IN_SLICE
    if is_first_encode and obj_val == 0:
        flags |= ismrmrd.ACQ_FIRST_IN_REPETITION
    if is_last_encode and obj_val == n_obj - 1:
        flags |= ismrmrd.ACQ_LAST_IN_REPETITION
    return flags


def _set_orientation(
    acq: ismrmrd.Acquisition,
    acq_idx: int,
    read_dir: np.ndarray | None,
    phase_dir: np.ndarray | None,
    slice_dir: np.ndarray | None,
    position: np.ndarray | None,
) -> None:
    """Set orientation vectors on an acquisition, defaulting to identity."""
    acq.read_dir[:] = read_dir[acq_idx] if read_dir is not None else [1, 0, 0]
    acq.phase_dir[:] = phase_dir[acq_idx] if phase_dir is not None else [0, 1, 0]
    acq.slice_dir[:] = slice_dir[acq_idx] if slice_dir is not None else [0, 0, 1]
    if position is not None:
        acq.position[:] = position[acq_idx]

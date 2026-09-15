"""Compute per-acquisition orientation and position vectors.

Handles the coordinate transform from Bruker's gradient / patient
coordinate system to ISMRMRD's read / phase / slice directions.
"""

from __future__ import annotations

import numpy as np

from bruker_to_ismrmrd.constants import POSITION_MATRICES
from bruker_to_ismrmrd.parameters import get_array, get_str


def compute_orientation_and_position(
    dataset: object,
    n_acquisitions: int,
    n_objects: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute per-acquisition read/phase/slice directions and position.

    Parameters
    ----------
    dataset
        A ``brukerapi.dataset.Dataset`` instance.
    n_acquisitions
        Total number of acquisitions to create.
    n_objects
        Number of objects/slices (NI).

    Returns
    -------
        Tuple of ``(read_direction, phase_direction, slice_direction, position)``,
        each of shape ``(n_acquisitions, 3)`` or ``None`` if not available.
    """
    subject_position = get_str(dataset, 'SUBJECT_position')
    subject_rotation = POSITION_MATRICES.get(subject_position or '', np.eye(3, dtype=np.float32))

    read_direction, phase_direction, slice_direction = _compute_directions(
        dataset, subject_rotation, n_acquisitions, n_objects
    )
    position = _compute_position(dataset, n_acquisitions, n_objects)

    return read_direction, phase_direction, slice_direction, position


def _select_frames(values: np.ndarray, width: int, n_acquisitions: int, n_objects: int) -> np.ndarray | None:
    """Select and tile orientation/position frames to match the acquisition count.

    Parameters
    ----------
    values
        Raw array from the Bruker parameter.
    width
        Number of elements per frame (9 for orientation matrices, 3 for positions).
    n_acquisitions
        Total number of acquisitions.
    n_objects
        Number of objects/slices.

    Returns
    -------
        Array of shape ``(n_acquisitions, width)`` or ``None`` if the frame
        count does not match any expected pattern.
    """
    reshaped = values.reshape(-1, width)
    n_frames = reshaped.shape[0]
    if n_frames == 1:
        return np.repeat(reshaped, n_acquisitions, axis=0)
    if n_frames == n_acquisitions:
        return reshaped
    if n_frames == n_objects and n_objects > 1 and n_acquisitions % n_objects == 0:
        return np.tile(reshaped, (n_acquisitions // n_objects, 1))
    return None


def _compute_directions(
    dataset: object,
    subject_rotation: np.ndarray,
    n_acquisitions: int,
    n_objects: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute read/phase/slice direction vectors from orientation parameters."""
    # Try VisuCoreOrientation first, then fall back to PVM_SPackArrGradOrient
    visu_orientation = get_array(dataset, 'VisuCoreOrientation', dtype=np.float32)
    if visu_orientation is not None and visu_orientation.size:
        selected = _select_frames(visu_orientation, 9, n_acquisitions, n_objects)
        if selected is not None:
            matrices = subject_rotation @ np.transpose(selected.reshape(-1, 3, 3), (0, 2, 1))
            return matrices[:, :, 0], matrices[:, :, 1], matrices[:, :, 2]

    grad_orientation = get_array(dataset, 'PVM_SPackArrGradOrient', dtype=np.float32)
    if grad_orientation is not None and grad_orientation.size:
        selected = _select_frames(grad_orientation, 9, n_acquisitions, n_objects)
        if selected is not None:
            matrices = subject_rotation @ selected.reshape(-1, 3, 3)
            return matrices[:, :, 0], matrices[:, :, 1], matrices[:, :, 2]

    return None, None, None


def _compute_position(
    dataset: object,
    n_acquisitions: int,
    n_objects: int,
) -> np.ndarray | None:
    """Compute per-acquisition position vectors."""
    visu_position = get_array(dataset, 'VisuCorePosition', dtype=np.float32)
    if visu_position is not None and visu_position.size:
        return _select_frames(visu_position, 3, n_acquisitions, n_objects)
    return None

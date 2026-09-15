"""Static lookup tables mapping Bruker concepts to ISMRMRD equivalents.

This module also contains the scheme-ID resolution logic that determines
the acquisition type (Cartesian 2D/3D, radial, spiral, EPI, ...).
"""

from __future__ import annotations

import contextlib

import numpy as np
from brukerapi.dataset import Dataset
from ismrmrd import xsd

# Subject position -> rotation matrix (Bruker patient coordinate system to MRD).
POSITION_MATRICES: dict[str, np.ndarray] = {
    # PV5/6 format
    'Head_Supine': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    'Head_Prone': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
    'Head_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    'Head_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
    'Foot_Supine': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
    'Tail_Supine': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
    'Foot_Prone': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    'Tail_Prone': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    'Foot_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    'Tail_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    'Foot_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
    'Tail_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
    # PV7+ format (assume head-first orientation)
    'SUBJ_POS_Supine': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
    'SUBJ_POS_Prone': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
    'SUBJ_POS_Left': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    'SUBJ_POS_Right': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
}

# Bruker subject position -> ISMRMRD patient position enum.
BRUKER_TO_PATIENT_POSITION: dict[str, xsd.patientPositionType] = {
    'Head_Supine': xsd.patientPositionType.HFS,
    'Head_Prone': xsd.patientPositionType.HFP,
    'Head_Left': xsd.patientPositionType.HFDL,
    'Head_Right': xsd.patientPositionType.HFDR,
    'Foot_Supine': xsd.patientPositionType.FFS,
    'Foot_Prone': xsd.patientPositionType.FFP,
    'Foot_Left': xsd.patientPositionType.FFDL,
    'Foot_Right': xsd.patientPositionType.FFDR,
    'Tail_Supine': xsd.patientPositionType.FFS,
    'Tail_Prone': xsd.patientPositionType.FFP,
    'Tail_Left': xsd.patientPositionType.FFDL,
    'Tail_Right': xsd.patientPositionType.FFDR,
    'SUBJ_POS_Supine': xsd.patientPositionType.HFS,
    'SUBJ_POS_Prone': xsd.patientPositionType.HFP,
    'SUBJ_POS_Left': xsd.patientPositionType.HFDL,
    'SUBJ_POS_Right': xsd.patientPositionType.HFDR,
}

# brukerapi scheme_id -> ISMRMRD trajectory type.
SCHEME_TO_TRAJECTORY: dict[str, xsd.trajectoryType] = {
    'CART_2D': xsd.trajectoryType.CARTESIAN,
    'CART_3D': xsd.trajectoryType.CARTESIAN,
    'RADIAL': xsd.trajectoryType.RADIAL,
    'SPIRAL': xsd.trajectoryType.SPIRAL,
    'EPI': xsd.trajectoryType.EPI,
    'dEPI': xsd.trajectoryType.EPI,
}

# Dimension-type labels -> ISMRMRD acquisition index field names.
DIMTYPE_TO_IDX: dict[str, str] = {
    'k_space_encode_step_1': 'kspace_encode_step_1',
    'k_space_encode_step_2': 'kspace_encode_step_2',
    'object': 'slice',
    'repetition': 'repetition',
    'echo': 'contrast',
}


def resolve_scheme_id(dataset: Dataset) -> str:
    """Determine the acquisition scheme for a brukerapi Dataset.

    For ``fid`` datasets, ``dataset.scheme_id`` is set by brukerapi.
    For ``rawdata`` (PV-360) datasets, ``scheme_id`` is not available as an
    attribute; this function falls back to ``_infer_scheme_id()`` which uses
    ``PULPROG``, ``Method``, and other heuristics.

    Returns ``'CART_2D'`` when no non-Cartesian scheme can be identified.
    """
    scheme = getattr(dataset, 'scheme_id', None)
    if scheme is None:
        with contextlib.suppress(AttributeError):
            scheme = dataset._infer_scheme_id()
    return scheme or 'CART_2D'

"""Convert Bruker raw data to ISMRMRD format.

This module provides the :func:`convert` function, which orchestrates the
full conversion pipeline: loading the Bruker dataset, building the ISMRMRD
header and acquisitions, and returning an :class:`~bruker_to_ismrmrd.MrdData`
instance.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
from xsdata.models.datatype import XmlDate, XmlTime

from bruker_to_ismrmrd.acquisitions import build_acquisitions
from bruker_to_ismrmrd.constants import resolve_scheme_id
from bruker_to_ismrmrd.header import build_ismrmrd_header
from bruker_to_ismrmrd.mrd_data import MrdData
from bruker_to_ismrmrd.parameters import get_array, get_int, get_str, parse_bruker_datetime


def convert(raw_path: str | Path) -> MrdData:
    """Convert Bruker raw data to ISMRMRD format.

    Uses ``brukerapi.dataset.Dataset`` to read the binary data and parameters,
    then creates an ISMRMRD header and list of acquisitions with proper
    encoding indices, orientation, and trajectory data.

    Parameters
    ----------
    raw_path
        Path to the Bruker experiment directory (containing acqp, method, fid)
        or directly to the ``fid`` / ``rawdata.job0`` file.

    Returns
    -------
        An :class:`~bruker_to_ismrmrd.MrdData` instance containing the
        ISMRMRD header and acquisitions.
    """
    from brukerapi.dataset import Dataset
    from brukerapi.jcampdx import JCAMPDX

    dataset = Dataset(str(raw_path))

    # Load auxiliary parameter files
    experiment_dir = Path(dataset.path).parent
    _load_auxiliary_parameters(dataset, experiment_dir, JCAMPDX)

    scheme_id = resolve_scheme_id(dataset)

    # Extract k-space data
    data, dim_type = _extract_kspace(dataset, scheme_id)

    # Load trajectory
    traj_data = _load_trajectory(dataset)

    # Truncate readout to trajectory-valid samples for non-Cartesian
    if traj_data is not None:
        k0_axis = dim_type.index('k_space_encode_step_0')
        n_k0 = data.shape[k0_axis]
        n_traj_samples = traj_data.shape[1]

        if n_traj_samples < n_k0:
            pre_size = get_int(dataset, 'PVM_SpiralPreSize') or 0
            slc = [slice(None)] * data.ndim
            slc[k0_axis] = slice(pre_size, pre_size + n_traj_samples)
            data = data[tuple(slc)]

    # Build ISMRMRD header and acquisitions
    ismrmrd_header = build_ismrmrd_header(dataset, data_shape=data.shape, dim_type=dim_type)

    # Store acquisition datetime in the header
    _set_acquisition_time(dataset, ismrmrd_header)

    acquisitions = build_acquisitions(dataset, data, dim_type, traj_data=traj_data)

    return MrdData(header=ismrmrd_header, acquisitions=acquisitions)


# ---------------------------------------------------------------------------
# Bruker dataset loading helpers
# ---------------------------------------------------------------------------


def _load_auxiliary_parameters(dataset: object, experiment_dir: Path, jcampdx_cls: type) -> None:
    """Load subject and visu_pars parameter files into the dataset."""
    subject_path = experiment_dir.parent / 'subject'
    if subject_path.is_file():
        dataset._parameters['subject'] = jcampdx_cls(str(subject_path))  # type: ignore[attr-defined]

    visu_path = experiment_dir / 'pdata' / '1' / 'visu_pars'
    if visu_path.is_file() and 'visu_pars' not in dataset._parameters:  # type: ignore[attr-defined]
        dataset._parameters['visu_pars'] = jcampdx_cls(str(visu_path))  # type: ignore[attr-defined]


def _extract_kspace(dataset: object, scheme_id: str) -> tuple[np.ndarray, list[str]]:
    """Extract k-space data and determine dimension labels.

    Handles the special cases for PV360 rawdata (non-Cartesian and Cartesian).
    """
    if dataset.type == 'rawdata' and scheme_id in ('RADIAL', 'SPIRAL', 'ZTE'):  # type: ignore[attr-defined]
        return _extract_noncartesian_rawdata(dataset)

    data = np.asarray(dataset.kspace, dtype=np.complex64)  # type: ignore[attr-defined]
    dim_type = dataset.dim_type  # type: ignore[attr-defined]

    # For PV360 rawdata Cartesian, dim_type describes the raw storage layout
    # (e.g. ['sample', 'channel', 'scan']), not the kspace axes.
    if dataset.type == 'rawdata' and 'k_space_encode_step_0' not in dim_type:  # type: ignore[attr-defined]
        acq_dim = get_int(dataset, 'ACQ_dim') or 2
        if acq_dim == 3:
            dim_type = [
                'k_space_encode_step_0',
                'k_space_encode_step_1',
                'k_space_encode_step_2',
                'repetition',
                'channel',
            ]
        else:
            dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'repetition', 'channel']

    return data, dim_type


def _extract_noncartesian_rawdata(dataset: object) -> tuple[np.ndarray, list[str]]:
    """Extract non-Cartesian rawdata (PV360) and reshape to a standard layout.

    brukerapi's ``.kspace`` intentionally refuses to reshape non-Cartesian
    rawdata, so we use ``.raw`` and arrange the axes ourselves.
    """
    raw = np.asarray(dataset.raw, dtype=np.complex64)  # type: ignore[attr-defined]
    n_samples, n_scans, n_receivers = raw.shape
    ni = get_int(dataset, 'NI') or 1
    nr = get_int(dataset, 'NR') or 1
    n_projections = n_scans // (ni * nr)
    # Bruker nesting (inner -> outer): projections -> NI -> NR
    data = raw.reshape(n_samples, n_projections, ni, nr, n_receivers)
    dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'repetition', 'channel']
    return data, dim_type


def _load_trajectory(dataset: object) -> np.ndarray | None:
    """Load and scale trajectory data from the dataset.

    The brukerapi trajectory is normalized to ``[-0.5, 0.5]``; this function
    scales it to encoding-step units (``[-N/2, N/2]``) using ``PVM_EncMatrix``.

    Returns
    -------
        Trajectory array of shape ``(traj_dims, n_samples, n_projections)``
        or ``None`` if no trajectory is available.
    """
    traj_data = None
    with contextlib.suppress(AttributeError):
        if dataset._traj is not None:  # type: ignore[attr-defined]
            traj_data = np.asarray(dataset.traj, dtype=np.float32)  # type: ignore[attr-defined]
            traj_dims = traj_data.shape[0]

            enc_matrix = get_array(dataset, 'PVM_EncMatrix', 'PVM_Matrix', dtype=np.float32)
            if enc_matrix is not None:
                enc_matrix = enc_matrix.reshape(-1)
                for dim_i in range(min(traj_dims, len(enc_matrix))):
                    traj_data[dim_i] *= enc_matrix[dim_i]

    return traj_data


def _set_acquisition_time(dataset: object, header: object) -> None:
    """Store the acquisition datetime in the ISMRMRD header's measurementInformation."""
    acq_time_str = get_str(dataset, 'ACQ_time')
    acquisition_time = parse_bruker_datetime(acq_time_str)
    if acquisition_time is None:
        with contextlib.suppress(AttributeError):
            acquisition_time = dataset.date  # type: ignore[attr-defined]

    if acquisition_time is not None and header.measurementInformation is not None:  # type: ignore[attr-defined]
        header.measurementInformation.seriesDate = XmlDate(  # type: ignore[attr-defined]
            acquisition_time.year, acquisition_time.month, acquisition_time.day
        )
        header.measurementInformation.seriesTime = XmlTime(  # type: ignore[attr-defined]
            acquisition_time.hour, acquisition_time.minute, acquisition_time.second, 0
        )

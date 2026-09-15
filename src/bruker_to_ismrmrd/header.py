"""Build an ISMRMRD header from Bruker parameters."""

from __future__ import annotations

import numpy as np
from ismrmrd import xsd

from bruker_to_ismrmrd.constants import BRUKER_TO_PATIENT_POSITION, SCHEME_TO_TRAJECTORY, resolve_scheme_id
from bruker_to_ismrmrd.parameters import get_array, get_float, get_int, get_str


def build_ismrmrd_header(
    dataset: object,
    data_shape: tuple[int, ...] | None = None,
    dim_type: list[str] | None = None,
) -> xsd.ismrmrdschema.ismrmrdHeader:
    """Build an ISMRMRD header from a brukerapi Dataset.

    Parameters
    ----------
    dataset
        A ``brukerapi.dataset.Dataset`` instance.
    data_shape
        Shape of the decoded k-space data (from ``dataset.data``).
        Used as fallback for encoding dimensions when ``PVM_EncMatrix`` is missing.
    dim_type
        Dimension labels corresponding to ``data_shape``.

    Returns
    -------
        Populated ISMRMRD header.
    """
    encoding = _build_encoding(dataset, data_shape, dim_type)
    experimental_conditions = _build_experimental_conditions(dataset)
    acquisition_system_information = _build_acquisition_system_information(dataset)
    measurement_information = _build_measurement_information(dataset)
    subject_information = _build_subject_information(dataset)
    sequence_parameters = _build_sequence_parameters(dataset)

    return xsd.ismrmrdschema.ismrmrdHeader(
        encoding=[encoding],
        experimentalConditions=experimental_conditions,
        acquisitionSystemInformation=acquisition_system_information,
        measurementInformation=measurement_information,
        subjectInformation=subject_information,
        sequenceParameters=sequence_parameters,
    )


def _build_encoding(
    dataset: object,
    data_shape: tuple[int, ...] | None,
    dim_type: list[str] | None,
) -> xsd.encodingType:
    """Build the encoding section of the ISMRMRD header."""
    enc_matrix = get_array(dataset, 'PVM_EncMatrix', 'PVM_Matrix', dtype=np.int64)
    if enc_matrix is not None:
        enc_matrix = enc_matrix.reshape(-1)
    elif data_shape is not None and dim_type is not None:
        dims = {label: data_shape[i] for i, label in enumerate(dim_type)}
        enc_matrix = np.array(
            [
                dims.get('k_space_encode_step_0', 1),
                dims.get('k_space_encode_step_1', 1),
                dims.get('k_space_encode_step_2', 1),
            ],
            dtype=np.int64,
        )
    else:
        raise ValueError('Missing Bruker parameter PVM_EncMatrix or PVM_Matrix.')

    n_k0 = int(enc_matrix[0])
    n_k1 = int(enc_matrix[1]) if enc_matrix.size >= 2 else 1
    n_k2 = int(enc_matrix[2]) if enc_matrix.size >= 3 else 1

    # FOV in mm (default to 1mm if missing, e.g. for spectroscopy)
    fov = get_array(dataset, 'PVM_Fov', dtype=np.float64)
    if fov is None:
        fov = np.ones(enc_matrix.size, dtype=np.float64)
    fov = fov.reshape(-1)
    fov_x = float(fov[0])
    fov_y = float(fov[1]) if fov.size >= 2 else fov_x
    fov_z = float(fov[2]) if fov.size >= 3 else 0.0

    # Slice thickness (for 2D acquisitions)
    slice_thickness = get_float(dataset, 'PVM_SliceThick')
    if n_k2 <= 1 and slice_thickness is not None:
        fov_z = slice_thickness

    # Encoding / recon spaces
    encoded_space = xsd.encodingSpaceType(
        matrixSize=xsd.matrixSizeType(x=n_k0, y=n_k1, z=max(n_k2, 1)),
        fieldOfView_mm=xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z if fov_z > 0 else 1.0),
    )
    recon_space = xsd.encodingSpaceType(
        matrixSize=xsd.matrixSizeType(x=n_k0, y=n_k1, z=max(n_k2, 1)),
        fieldOfView_mm=xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z if fov_z > 0 else 1.0),
    )

    # Encoding limits
    encoding_limits = _build_encoding_limits(dataset, n_k1, n_k2, data_shape, dim_type)

    scheme_id = resolve_scheme_id(dataset)
    trajectory = SCHEME_TO_TRAJECTORY.get(scheme_id, xsd.trajectoryType.OTHER)
    echo_train_length = get_int(dataset, 'PVM_RareFactor', 'ACQ_phase_factor') or 1

    return xsd.encodingType(
        trajectory=trajectory,
        encodedSpace=encoded_space,
        reconSpace=recon_space,
        encodingLimits=encoding_limits,
        echoTrainLength=echo_train_length,
    )


def _build_encoding_limits(
    dataset: object,
    n_k1: int,
    n_k2: int,
    data_shape: tuple[int, ...] | None,
    dim_type: list[str] | None,
) -> xsd.encodingLimitsType:
    """Build encoding limits from dataset parameters and data shape."""
    encoding_limits = xsd.encodingLimitsType(
        kspace_encoding_step_1=xsd.limitType(minimum=0, maximum=n_k1 - 1, center=n_k1 // 2),
        kspace_encoding_step_2=xsd.limitType(minimum=0, maximum=max(n_k2 - 1, 0), center=max(n_k2 // 2, 0)),
    )

    n_slices = get_int(dataset, 'PVM_SPackArrNSlices', 'NI') or 1
    if dim_type is not None and 'object' in dim_type:
        n_slices = data_shape[dim_type.index('object')]  # type: ignore[index]
    if n_slices > 1:
        encoding_limits.slice = xsd.limitType(minimum=0, maximum=n_slices - 1, center=0)

    n_repetitions = get_int(dataset, 'NR') or 1
    if dim_type is not None and 'repetition' in dim_type:
        n_repetitions = data_shape[dim_type.index('repetition')]  # type: ignore[index]
    if n_repetitions > 1:
        encoding_limits.repetition = xsd.limitType(minimum=0, maximum=n_repetitions - 1, center=0)

    if dim_type is not None and 'echo' in dim_type:
        n_echoes = data_shape[dim_type.index('echo')]  # type: ignore[index]
        if n_echoes > 1:
            encoding_limits.contrast = xsd.limitType(minimum=0, maximum=n_echoes - 1, center=0)

    return encoding_limits


def _build_experimental_conditions(dataset: object) -> xsd.experimentalConditionsType:
    """Build experimental conditions (Larmor frequency)."""
    larmor_freq_mhz = get_float(dataset, 'PVM_FrqWork', 'PVM_FrqRef', 'VisuAcqImagingFrequency')
    h1_frequency_hz = int(larmor_freq_mhz * 1e6) if larmor_freq_mhz is not None else 0
    return xsd.experimentalConditionsType(H1resonanceFrequency_Hz=h1_frequency_hz)


def _build_acquisition_system_information(dataset: object) -> xsd.acquisitionSystemInformationType:
    """Build acquisition system information (vendor, field strength, channels)."""
    field_strength = get_float(dataset, 'PVM_MagnetStrength', 'BF1')
    n_channels = get_int(dataset, 'PVM_EncNReceivers') or 1
    return xsd.acquisitionSystemInformationType(
        systemVendor='Bruker',
        systemFieldStrength_T=field_strength,
        receiverChannels=n_channels,
    )


def _build_measurement_information(dataset: object) -> xsd.measurementInformationType:
    """Build measurement information (patient position, protocol name)."""
    measurement_information = xsd.measurementInformationType()

    subject_position = get_str(dataset, 'SUBJECT_position')
    if subject_position and subject_position in BRUKER_TO_PATIENT_POSITION:
        measurement_information.patientPosition = BRUKER_TO_PATIENT_POSITION[subject_position]

    sequence_name = get_str(dataset, 'PULPROG', 'Method', 'ACQ_scan_name')
    if sequence_name:
        measurement_information.protocolName = sequence_name

    return measurement_information


def _build_subject_information(dataset: object) -> xsd.subjectInformationType | None:
    """Build subject information (patient name)."""
    patient_name = get_str(dataset, 'SUBJECT_name_string', 'VisuSubjectName', 'SUBJECT_id')
    return xsd.subjectInformationType(patientName=patient_name) if patient_name else None


def _build_sequence_parameters(dataset: object) -> xsd.sequenceParametersType | None:
    """Build sequence parameters (TR, TE, flip angle)."""
    tr = get_float(dataset, 'PVM_RepetitionTime')
    te = get_float(dataset, 'PVM_EchoTime', 'EffectiveTE')
    flip_angle = get_float(dataset, 'PVM_ExcPulseAngle', 'ExcPulse1.Flipangle')

    if not any(v is not None for v in (tr, te, flip_angle)):
        return None

    return xsd.sequenceParametersType(
        TR=[tr] if tr is not None else None,
        TE=[te] if te is not None else None,
        flipAngle_deg=[flip_angle] if flip_angle is not None else None,
    )

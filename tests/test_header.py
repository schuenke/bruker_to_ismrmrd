"""Tests for bruker_to_ismrmrd.header."""

from __future__ import annotations

import numpy as np
import pytest
from ismrmrd import xsd

from bruker_to_ismrmrd.header import build_ismrmrd_header
from tests.conftest import FakeDataset

# ---------------------------------------------------------------------------
# build_ismrmrd_header - encoding
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderEncoding:
    """Tests for encoding section of ``build_ismrmrd_header``."""

    def test_build_ismrmrd_header_enc_matrix(self, cartesian_dataset: FakeDataset) -> None:
        """Verify encoded matrix size from PVM_EncMatrix."""
        header = build_ismrmrd_header(cartesian_dataset)
        enc = header.encoding[0]
        assert enc.encodedSpace.matrixSize.x == 256
        assert enc.encodedSpace.matrixSize.y == 128
        assert enc.encodedSpace.matrixSize.z == 1

    def test_build_ismrmrd_header_fov(self, cartesian_dataset: FakeDataset) -> None:
        """Verify field of view from PVM_Fov and PVM_SliceThick."""
        header = build_ismrmrd_header(cartesian_dataset)
        enc = header.encoding[0]
        assert enc.encodedSpace.fieldOfView_mm.x == pytest.approx(200.0)
        assert enc.encodedSpace.fieldOfView_mm.y == pytest.approx(200.0)
        # 2D with slice thickness
        assert enc.encodedSpace.fieldOfView_mm.z == pytest.approx(2.0)

    def test_build_ismrmrd_header_fallback_to_data_shape(self) -> None:
        """Fall back to data_shape/dim_type when PVM_EncMatrix is missing."""
        ds = FakeDataset(
            params={
                'PVM_Fov': np.array([100.0, 100.0]),
                'PVM_FrqWork': np.array([300.0]),
                'PVM_MagnetStrength': 7.0,
                'PVM_EncNReceivers': 1,
                'PULPROG': '<FLASH>',
            },
            scheme_id='CART_2D',
        )
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'channel']
        data_shape = (64, 32, 3, 1)
        header = build_ismrmrd_header(ds, data_shape=data_shape, dim_type=dim_type)
        enc = header.encoding[0]
        assert enc.encodedSpace.matrixSize.x == 64
        assert enc.encodedSpace.matrixSize.y == 32
        assert enc.encodedSpace.matrixSize.z == 1

    def test_build_ismrmrd_header_missing_encoding_raises(self) -> None:
        """Raise ValueError when PVM_EncMatrix is missing and no data_shape."""
        ds = FakeDataset(
            params={'PVM_FrqWork': np.array([300.0])},
            scheme_id='CART_2D',
        )
        with pytest.raises(ValueError, match='Missing Bruker parameter'):
            build_ismrmrd_header(ds)

    def test_build_ismrmrd_header_fov_missing_defaults_to_ones(self) -> None:
        """Default FOV to 1.0 mm when PVM_Fov is missing."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([128, 64]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        enc = header.encoding[0]
        assert enc.encodedSpace.fieldOfView_mm.x == pytest.approx(1.0)

    def test_build_ismrmrd_header_3d_encoding(self) -> None:
        """Build correct 3D encoding from a three-element PVM_EncMatrix."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([128, 64, 32]),
                'PVM_Fov': np.array([100.0, 100.0, 50.0]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_3D',
        )
        header = build_ismrmrd_header(ds)
        enc = header.encoding[0]
        assert enc.encodedSpace.matrixSize.z == 32
        assert enc.encodedSpace.fieldOfView_mm.z == pytest.approx(50.0)

    def test_build_ismrmrd_header_trajectory_cartesian(self, cartesian_dataset: FakeDataset) -> None:
        """Set CARTESIAN trajectory for a Cartesian scheme."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.encoding[0].trajectory == xsd.trajectoryType.CARTESIAN

    def test_build_ismrmrd_header_trajectory_spiral(self) -> None:
        """Set SPIRAL trajectory for a spiral scheme."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([128, 128]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='SPIRAL',
        )
        header = build_ismrmrd_header(ds)
        assert header.encoding[0].trajectory == xsd.trajectoryType.SPIRAL

    def test_build_ismrmrd_header_unknown_scheme_trajectory_other(self) -> None:
        """Default to OTHER trajectory for an unknown scheme."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([128, 128]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='ZTE',
        )
        header = build_ismrmrd_header(ds)
        assert header.encoding[0].trajectory == xsd.trajectoryType.OTHER

    def test_build_ismrmrd_header_echo_train_length(self, cartesian_dataset: FakeDataset) -> None:
        """Derive echo train length from PVM_RareFactor."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.encoding[0].echoTrainLength == 4


# ---------------------------------------------------------------------------
# build_ismrmrd_header - encoding limits
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderEncodingLimits:
    """Tests for encoding limits in the header."""

    def test_build_ismrmrd_header_k1_limits(self, cartesian_dataset: FakeDataset) -> None:
        """Verify kspace_encoding_step_1 limits from PVM_EncMatrix."""
        header = build_ismrmrd_header(cartesian_dataset)
        k1 = header.encoding[0].encodingLimits.kspace_encoding_step_1
        assert k1.minimum == 0
        assert k1.maximum == 127
        assert k1.center == 64

    def test_build_ismrmrd_header_slice_limits(self, cartesian_dataset: FakeDataset) -> None:
        """Derive slice limits from PVM_SPackArrNSlices."""
        header = build_ismrmrd_header(cartesian_dataset)
        slc = header.encoding[0].encodingLimits.slice
        assert slc is not None
        assert slc.maximum == 2  # n_slices=3 -> max=2

    def test_build_ismrmrd_header_slice_limits_from_dim_type(self) -> None:
        """Override slice count from the 'object' dimension in dim_type."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
                'PVM_SPackArrNSlices': 1,
            },
            scheme_id='CART_2D',
        )
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'channel']
        data_shape = (64, 64, 5, 2)
        header = build_ismrmrd_header(ds, data_shape=data_shape, dim_type=dim_type)
        slc = header.encoding[0].encodingLimits.slice
        assert slc is not None
        assert slc.maximum == 4

    def test_build_ismrmrd_header_repetition_limits(self) -> None:
        """Derive repetition limits from NR parameter."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
                'NR': 5,
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        rep = header.encoding[0].encodingLimits.repetition
        assert rep is not None
        assert rep.maximum == 4

    def test_build_ismrmrd_header_echo_limits(self) -> None:
        """Derive contrast limits from the 'echo' dimension in dim_type."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_2D',
        )
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'echo', 'channel']
        data_shape = (64, 64, 4, 1)
        header = build_ismrmrd_header(ds, data_shape=data_shape, dim_type=dim_type)
        contrast = header.encoding[0].encodingLimits.contrast
        assert contrast is not None
        assert contrast.maximum == 3

    def test_build_ismrmrd_header_no_repetition_limit_when_single(self) -> None:
        """Omit repetition limits when NR is 1."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
                'NR': 1,
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.encoding[0].encodingLimits.repetition is None

    def test_build_ismrmrd_header_repetition_from_dim_type(self) -> None:
        """Override repetition count from the 'repetition' dimension in dim_type."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
                'NR': 1,
            },
            scheme_id='CART_2D',
        )
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'repetition', 'channel']
        data_shape = (64, 64, 3, 1)
        header = build_ismrmrd_header(ds, data_shape=data_shape, dim_type=dim_type)
        rep = header.encoding[0].encodingLimits.repetition
        assert rep is not None
        assert rep.maximum == 2


# ---------------------------------------------------------------------------
# build_ismrmrd_header - experimental conditions
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderExpConditions:
    """Tests for experimental conditions section."""

    def test_build_ismrmrd_header_larmor_frequency(self, cartesian_dataset: FakeDataset) -> None:
        """Convert PVM_FrqWork from MHz to Hz for the Larmor frequency."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.experimentalConditions.H1resonanceFrequency_Hz == 300_000_000

    def test_build_ismrmrd_header_larmor_frequency_missing(self) -> None:
        """Default Larmor frequency to 0 when PVM_FrqWork is missing."""
        ds = FakeDataset(
            params={'PVM_EncMatrix': np.array([64, 64])},
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.experimentalConditions.H1resonanceFrequency_Hz == 0


# ---------------------------------------------------------------------------
# build_ismrmrd_header - acquisition system info
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderAcqSystemInfo:
    """Tests for acquisition system information."""

    def test_build_ismrmrd_header_vendor(self, cartesian_dataset: FakeDataset) -> None:
        """Set the system vendor to Bruker."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.acquisitionSystemInformation.systemVendor == 'Bruker'

    def test_build_ismrmrd_header_field_strength(self, cartesian_dataset: FakeDataset) -> None:
        """Derive field strength from PVM_MagnetStrength."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.acquisitionSystemInformation.systemFieldStrength_T == pytest.approx(7.0)

    def test_build_ismrmrd_header_receiver_channels(self, cartesian_dataset: FakeDataset) -> None:
        """Derive receiver channel count from PVM_EncNReceivers."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.acquisitionSystemInformation.receiverChannels == 1

    def test_build_ismrmrd_header_channels_default(self) -> None:
        """Default receiver channels to 1 when PVM_EncNReceivers is missing."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.acquisitionSystemInformation.receiverChannels == 1


# ---------------------------------------------------------------------------
# build_ismrmrd_header - measurement information
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderMeasInfo:
    """Tests for measurement information."""

    def test_build_ismrmrd_header_patient_position(self, cartesian_dataset: FakeDataset) -> None:
        """Map SUBJECT_position to the ISMRMRD patient position."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.measurementInformation.patientPosition == xsd.patientPositionType.HFS

    def test_build_ismrmrd_header_protocol_name(self, cartesian_dataset: FakeDataset) -> None:
        """Derive protocol name from PULPROG."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.measurementInformation.protocolName == 'RARE'

    def test_build_ismrmrd_header_no_position(self) -> None:
        """Leave patient position as None when SUBJECT_position is missing."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.measurementInformation.patientPosition is None


# ---------------------------------------------------------------------------
# build_ismrmrd_header - subject information
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderSubjectInfo:
    """Tests for subject information."""

    def test_build_ismrmrd_header_subject_name(self, cartesian_dataset: FakeDataset) -> None:
        """Derive patient name from SUBJECT_name_string."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.subjectInformation is not None
        assert header.subjectInformation.patientName == 'TestSubject'

    def test_build_ismrmrd_header_no_subject(self) -> None:
        """Omit subjectInformation when SUBJECT_name_string is missing."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.subjectInformation is None


# ---------------------------------------------------------------------------
# build_ismrmrd_header - sequence parameters
# ---------------------------------------------------------------------------


class TestBuildIsmrmrdHeaderSeqParams:
    """Tests for sequence parameters."""

    def test_build_ismrmrd_header_tr(self, cartesian_dataset: FakeDataset) -> None:
        """Derive TR from PVM_RepetitionTime."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.sequenceParameters is not None
        assert [pytest.approx(2500.0)] == header.sequenceParameters.TR

    def test_build_ismrmrd_header_te(self, cartesian_dataset: FakeDataset) -> None:
        """Derive TE from PVM_EchoTime."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert [pytest.approx(30.0)] == header.sequenceParameters.TE

    def test_build_ismrmrd_header_flip_angle(self, cartesian_dataset: FakeDataset) -> None:
        """Derive flip angle from PVM_ExcPulseAngle."""
        header = build_ismrmrd_header(cartesian_dataset)
        assert header.sequenceParameters.flipAngle_deg == [pytest.approx(90.0)]

    def test_build_ismrmrd_header_no_sequence_params(self) -> None:
        """Omit sequenceParameters when TR/TE/flip are all missing."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.sequenceParameters is None

    def test_build_ismrmrd_header_partial_sequence_params(self) -> None:
        """Include sequenceParameters with only TR when TE is missing."""
        ds = FakeDataset(
            params={
                'PVM_EncMatrix': np.array([64, 64]),
                'PVM_FrqWork': np.array([300.0]),
                'PVM_RepetitionTime': 1000.0,
            },
            scheme_id='CART_2D',
        )
        header = build_ismrmrd_header(ds)
        assert header.sequenceParameters is not None
        assert [pytest.approx(1000.0)] == header.sequenceParameters.TR
        assert header.sequenceParameters.TE is None

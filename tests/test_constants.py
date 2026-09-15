"""Tests for bruker_to_ismrmrd.constants."""

from __future__ import annotations

import numpy as np
from ismrmrd import xsd

from bruker_to_ismrmrd.constants import (
    BRUKER_TO_PATIENT_POSITION,
    DIMTYPE_TO_IDX,
    POSITION_MATRICES,
    SCHEME_TO_TRAJECTORY,
    resolve_scheme_id,
)
from tests.conftest import FakeDataset

# ---------------------------------------------------------------------------
# Static mapping tables
# ---------------------------------------------------------------------------


class TestPositionMatrices:
    """Tests for POSITION_MATRICES."""

    def test_position_matrices_shape(self) -> None:
        """Verify that all position matrices are 3x3 float32."""
        for key, mat in POSITION_MATRICES.items():
            assert mat.shape == (3, 3), f'{key} has wrong shape'
            assert mat.dtype == np.float32

    def test_position_matrices_known_entry(self) -> None:
        """Verify the Head_Supine position matrix."""
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        np.testing.assert_array_equal(POSITION_MATRICES['Head_Supine'], expected)

    def test_position_matrices_pv7_entry(self) -> None:
        """Verify the SUBJ_POS_Supine position matrix matches Head_Supine."""
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        np.testing.assert_array_equal(POSITION_MATRICES['SUBJ_POS_Supine'], expected)


class TestBrukerToPatientPosition:
    """Tests for BRUKER_TO_PATIENT_POSITION."""

    def test_bruker_to_patient_position_head_supine(self) -> None:
        """Map Head_Supine to HFS."""
        assert BRUKER_TO_PATIENT_POSITION['Head_Supine'] == xsd.patientPositionType.HFS

    def test_bruker_to_patient_position_tail_supine(self) -> None:
        """Map Tail_Supine to FFS."""
        assert BRUKER_TO_PATIENT_POSITION['Tail_Supine'] == xsd.patientPositionType.FFS

    def test_bruker_to_patient_position_pv7_format(self) -> None:
        """Map PV7-style SUBJ_POS_Prone to HFP."""
        assert BRUKER_TO_PATIENT_POSITION['SUBJ_POS_Prone'] == xsd.patientPositionType.HFP


class TestSchemeToTrajectory:
    """Tests for SCHEME_TO_TRAJECTORY."""

    def test_scheme_cartesian_2d(self) -> None:
        """Map CART_2D to CARTESIAN trajectory."""
        assert SCHEME_TO_TRAJECTORY['CART_2D'] == xsd.trajectoryType.CARTESIAN

    def test_scheme_cartesian_3d(self) -> None:
        """Map CART_3D to CARTESIAN trajectory."""
        assert SCHEME_TO_TRAJECTORY['CART_3D'] == xsd.trajectoryType.CARTESIAN

    def test_scheme_radial(self) -> None:
        """Map RADIAL to RADIAL trajectory."""
        assert SCHEME_TO_TRAJECTORY['RADIAL'] == xsd.trajectoryType.RADIAL

    def test_scheme_spiral(self) -> None:
        """Map SPIRAL to SPIRAL trajectory."""
        assert SCHEME_TO_TRAJECTORY['SPIRAL'] == xsd.trajectoryType.SPIRAL

    def test_scheme_epi(self) -> None:
        """Map EPI to EPI trajectory."""
        assert SCHEME_TO_TRAJECTORY['EPI'] == xsd.trajectoryType.EPI

    def test_scheme_depi(self) -> None:
        """Map dEPI to EPI trajectory."""
        assert SCHEME_TO_TRAJECTORY['dEPI'] == xsd.trajectoryType.EPI


class TestDimtypeToIdx:
    """Tests for DIMTYPE_TO_IDX."""

    def test_dimtype_to_idx_keys(self) -> None:
        """Verify all expected dimension types are present."""
        expected_keys = {'k_space_encode_step_1', 'k_space_encode_step_2', 'object', 'repetition', 'echo'}
        assert set(DIMTYPE_TO_IDX.keys()) == expected_keys

    def test_dimtype_to_idx_object_maps_to_slice(self) -> None:
        """Map the 'object' dimension type to 'slice'."""
        assert DIMTYPE_TO_IDX['object'] == 'slice'


# ---------------------------------------------------------------------------
# resolve_scheme_id
# ---------------------------------------------------------------------------


class TestResolveSchemeId:
    """Tests for ``resolve_scheme_id``."""

    def test_resolve_scheme_id_from_attribute(self) -> None:
        """Return the scheme_id attribute directly when set."""
        ds = FakeDataset(scheme_id='SPIRAL')
        assert resolve_scheme_id(ds) == 'SPIRAL'

    def test_resolve_scheme_id_fallback_infer(self) -> None:
        """Fall back to ``_infer_scheme_id()`` when ``scheme_id`` is None."""
        ds = FakeDataset(scheme_id=None)
        ds._infer_scheme_id = lambda: 'RADIAL'  # type: ignore[attr-defined]
        assert resolve_scheme_id(ds) == 'RADIAL'

    def test_resolve_scheme_id_no_attribute_defaults_cart2d(self) -> None:
        """Default to CART_2D when the dataset has no scheme_id attribute."""
        assert resolve_scheme_id(object()) == 'CART_2D'

    def test_resolve_scheme_id_infer_raises_defaults_cart2d(self) -> None:
        """Default to CART_2D when ``_infer_scheme_id()`` raises ``AttributeError``."""
        ds = FakeDataset(scheme_id=None)
        # no _infer_scheme_id method -> AttributeError is suppressed
        assert resolve_scheme_id(ds) == 'CART_2D'

    def test_resolve_scheme_id_infer_returns_none_defaults_cart2d(self) -> None:
        """Default to CART_2D when ``_infer_scheme_id()`` returns None."""
        ds = FakeDataset(scheme_id=None)
        ds._infer_scheme_id = lambda: None  # type: ignore[attr-defined]
        assert resolve_scheme_id(ds) == 'CART_2D'

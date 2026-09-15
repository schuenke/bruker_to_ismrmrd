"""Tests for bruker_to_ismrmrd.geometry."""

from __future__ import annotations

import numpy as np

from bruker_to_ismrmrd.geometry import (
    _compute_directions,
    _compute_position,
    _select_frames,
    compute_orientation_and_position,
)
from tests.conftest import FakeDataset

# ---------------------------------------------------------------------------
# _select_frames
# ---------------------------------------------------------------------------


class TestSelectFrames:
    """Tests for ``_select_frames``."""

    def test_select_frames_single_frame_tiles(self) -> None:
        """Tile a single frame across all acquisitions."""
        values = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        result = _select_frames(values, width=9, n_acquisitions=5, n_objects=1)
        assert result is not None
        assert result.shape == (5, 9)
        np.testing.assert_array_equal(result[0], result[4])

    def test_select_frames_matching_acquisitions(self) -> None:
        """Return frames directly when count matches n_acquisitions."""
        values = np.arange(27, dtype=np.float32)  # 3 frames x 9
        result = _select_frames(values, width=9, n_acquisitions=3, n_objects=1)
        assert result is not None
        assert result.shape == (3, 9)

    def test_select_frames_per_object_tiles(self) -> None:
        """Tile per-object frames across repeated acquisitions."""
        values = np.arange(18, dtype=np.float32)  # 2 frames x 9
        result = _select_frames(values, width=9, n_acquisitions=6, n_objects=2)
        assert result is not None
        assert result.shape == (6, 9)
        # Pattern repeats: objects 0,1 repeated 3 times
        np.testing.assert_array_equal(result[0], result[2])
        np.testing.assert_array_equal(result[1], result[3])

    def test_select_frames_incompatible_returns_none(self) -> None:
        """Return None when frames cannot be evenly distributed."""
        values = np.arange(18, dtype=np.float32)  # 2 frames
        result = _select_frames(values, width=9, n_acquisitions=5, n_objects=3)
        assert result is None

    def test_select_frames_position_width_3(self) -> None:
        """Handle position vectors with width=3."""
        values = np.array([1, 2, 3], dtype=np.float32)
        result = _select_frames(values, width=3, n_acquisitions=4, n_objects=1)
        assert result is not None
        assert result.shape == (4, 3)


# ---------------------------------------------------------------------------
# _compute_directions
# ---------------------------------------------------------------------------


class TestComputeDirections:
    """Tests for ``_compute_directions``."""

    def test_compute_directions_identity_orientation(self) -> None:
        """Return axis-aligned directions for an identity orientation matrix."""
        ds = FakeDataset(params={'VisuCoreOrientation': np.eye(3, dtype=np.float32).flatten()})
        subject_rot = np.eye(3, dtype=np.float32)
        rd, pd, sd = _compute_directions(ds, subject_rot, n_acquisitions=1, n_objects=1)
        assert rd is not None
        assert pd is not None
        assert sd is not None
        np.testing.assert_allclose(rd[0], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(pd[0], [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(sd[0], [0, 0, 1], atol=1e-6)

    def test_compute_directions_fallback_to_grad_orient(self) -> None:
        """Fall back to PVM_SPackArrGradOrient when VisuCoreOrientation is missing."""
        orient = np.eye(3, dtype=np.float32).flatten()
        ds = FakeDataset(params={'PVM_SPackArrGradOrient': orient})
        subject_rot = np.eye(3, dtype=np.float32)
        rd, _pd, _sd = _compute_directions(ds, subject_rot, n_acquisitions=1, n_objects=1)
        assert rd is not None

    def test_compute_directions_no_orientation_returns_none(self) -> None:
        """Return None triplet when no orientation data is available."""
        ds = FakeDataset()
        subject_rot = np.eye(3, dtype=np.float32)
        rd, pd, sd = _compute_directions(ds, subject_rot, n_acquisitions=1, n_objects=1)
        assert rd is None
        assert pd is None
        assert sd is None

    def test_compute_directions_empty_orientation_returns_none(self) -> None:
        """Return None when the orientation array is empty."""
        ds = FakeDataset(params={'VisuCoreOrientation': np.array([], dtype=np.float32)})
        subject_rot = np.eye(3, dtype=np.float32)
        rd, _pd, _sd = _compute_directions(ds, subject_rot, n_acquisitions=1, n_objects=1)
        assert rd is None

    def test_compute_directions_with_subject_rotation(self) -> None:
        """Apply the subject rotation matrix to the orientation."""
        # Head_Supine rotation flips x and z
        orient = np.eye(3, dtype=np.float32).flatten()
        ds = FakeDataset(params={'VisuCoreOrientation': orient})
        subject_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        rd, _pd, _sd = _compute_directions(ds, subject_rot, n_acquisitions=1, n_objects=1)
        assert rd is not None
        np.testing.assert_allclose(rd[0], [-1, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# _compute_position
# ---------------------------------------------------------------------------


class TestComputePosition:
    """Tests for ``_compute_position``."""

    def test_compute_position_available(self) -> None:
        """Return tiled position vectors when VisuCorePosition is present."""
        ds = FakeDataset(params={'VisuCorePosition': np.array([10, 20, 30], dtype=np.float32)})
        pos = _compute_position(ds, n_acquisitions=2, n_objects=1)
        assert pos is not None
        assert pos.shape == (2, 3)
        np.testing.assert_allclose(pos[0], [10, 20, 30])

    def test_compute_position_missing_returns_none(self) -> None:
        """Return None when VisuCorePosition is missing."""
        ds = FakeDataset()
        pos = _compute_position(ds, n_acquisitions=2, n_objects=1)
        assert pos is None


# ---------------------------------------------------------------------------
# compute_orientation_and_position
# ---------------------------------------------------------------------------


class TestComputeOrientationAndPosition:
    """Tests for ``compute_orientation_and_position``."""

    def test_compute_orientation_and_position_known_position(self) -> None:
        """Compute directions and position for a known subject position."""
        orient = np.eye(3, dtype=np.float32).flatten()
        ds = FakeDataset(
            params={
                'SUBJECT_position': 'Head_Supine',
                'VisuCoreOrientation': orient,
                'VisuCorePosition': np.array([1, 2, 3], dtype=np.float32),
            }
        )
        rd, _pd, _sd, pos = compute_orientation_and_position(ds, n_acquisitions=1, n_objects=1)
        assert rd is not None
        assert pos is not None

    def test_compute_orientation_and_position_unknown_position(self) -> None:
        """Default to identity rotation for an unknown SUBJECT_position."""
        orient = np.eye(3, dtype=np.float32).flatten()
        ds = FakeDataset(
            params={
                'SUBJECT_position': 'Unknown_Position',
                'VisuCoreOrientation': orient,
            }
        )
        rd, _pd, _sd, _pos = compute_orientation_and_position(ds, n_acquisitions=1, n_objects=1)
        assert rd is not None
        # Identity rotation applied, so identity orientation stays as-is
        np.testing.assert_allclose(rd[0], [1, 0, 0], atol=1e-6)

    def test_compute_orientation_and_position_no_subject_position(self) -> None:
        """Return directions but no position when SUBJECT_position is absent."""
        orient = np.eye(3, dtype=np.float32).flatten()
        ds = FakeDataset(params={'VisuCoreOrientation': orient})
        rd, _pd, _sd, pos = compute_orientation_and_position(ds, n_acquisitions=1, n_objects=1)
        assert rd is not None
        assert pos is None

    def test_compute_orientation_and_position_multi_object(self) -> None:
        """Tile orientation and position across multiple objects."""
        # 2 slices x 9 elements = 18
        orient = np.concatenate([np.eye(3, dtype=np.float32).flatten(), np.eye(3, dtype=np.float32).flatten()])
        position = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        ds = FakeDataset(
            params={
                'VisuCoreOrientation': orient,
                'VisuCorePosition': position,
            }
        )
        # 6 total acquisitions, 2 objects -> tiled
        rd, _pd, _sd, pos = compute_orientation_and_position(ds, n_acquisitions=6, n_objects=2)
        assert rd is not None
        assert rd.shape == (6, 3)
        assert pos is not None
        assert pos.shape == (6, 3)

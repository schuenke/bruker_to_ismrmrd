"""Tests for bruker_to_ismrmrd.acquisitions."""

from __future__ import annotations

import ismrmrd
import numpy as np
import pytest

from bruker_to_ismrmrd.acquisitions import (
    _compute_flags,
    _extract_readout,
    _get_loop_axes,
    _set_orientation,
    build_acquisitions,
)
from tests.conftest import FakeDataset

# ---------------------------------------------------------------------------
# _get_loop_axes
# ---------------------------------------------------------------------------


class TestGetLoopAxes:
    """Tests for ``_get_loop_axes``."""

    def test_get_loop_axes_standard(self) -> None:
        """Identify loop axes from a standard 4D dim_type."""
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'channel']
        shape = (64, 32, 3, 2)
        result = _get_loop_axes(dim_type, k0_axis=0, ch_axis=3, shape=shape)
        assert len(result) == 2
        assert result[0] == (1, 'k_space_encode_step_1', 32)
        assert result[1] == (2, 'object', 3)

    def test_get_loop_axes_no_channel(self) -> None:
        """Identify loop axes when no channel dimension is present."""
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1']
        shape = (64, 32)
        result = _get_loop_axes(dim_type, k0_axis=0, ch_axis=None, shape=shape)
        assert len(result) == 1
        assert result[0] == (1, 'k_space_encode_step_1', 32)


# ---------------------------------------------------------------------------
# _extract_readout
# ---------------------------------------------------------------------------


class TestExtractReadout:
    """Tests for ``_extract_readout``."""

    def test_extract_readout_2d_with_channel(self) -> None:
        """Extract a 2D readout line with channel and verify shape and indices."""
        # shape: (k0=4, ch=2, k1=3)
        data = np.arange(24, dtype=np.complex64).reshape(4, 2, 3)
        loop_axes = [(2, 'k_space_encode_step_1', 3)]
        readout, idx_vals = _extract_readout(
            data, k0_axis=0, ch_axis=1, n_coils=2, n_k0=4, loop_axes=loop_axes, multi_idx=(1,)
        )
        assert readout.shape == (2, 4)
        assert idx_vals == {'k_space_encode_step_1': 1}

    def test_extract_readout_channel_after_k0(self) -> None:
        """Transpose readout when k0 axis precedes channel axis."""
        # shape: (k0=4, ch=2) with loop selecting the slice
        data = np.arange(24, dtype=np.complex64).reshape(4, 2, 3)
        loop_axes = [(2, 'k_space_encode_step_1', 3)]
        readout, _ = _extract_readout(
            data, k0_axis=0, ch_axis=1, n_coils=2, n_k0=4, loop_axes=loop_axes, multi_idx=(0,)
        )
        assert readout.shape == (2, 4)

    def test_extract_readout_no_channel(self) -> None:
        """Extract a single-channel readout and reshape to (1, n_k0)."""
        data = np.arange(12, dtype=np.complex64).reshape(4, 3)
        loop_axes = [(1, 'k_space_encode_step_1', 3)]
        readout, idx_vals = _extract_readout(
            data, k0_axis=0, ch_axis=None, n_coils=1, n_k0=4, loop_axes=loop_axes, multi_idx=(2,)
        )
        assert readout.shape == (1, 4)
        assert idx_vals == {'k_space_encode_step_1': 2}

    def test_extract_readout_channel_before_k0(self) -> None:
        """Keep readout as-is when channel axis precedes k0 axis."""
        # Shape: (ch=2, k0=4, k1=3) -> k0_axis=1, ch_axis=0
        data = np.arange(24, dtype=np.complex64).reshape(2, 4, 3)
        loop_axes = [(2, 'k_space_encode_step_1', 3)]
        readout, _ = _extract_readout(
            data, k0_axis=1, ch_axis=0, n_coils=2, n_k0=4, loop_axes=loop_axes, multi_idx=(0,)
        )
        assert readout.shape == (2, 4)

    def test_extract_readout_multidim_reshape(self) -> None:
        """Fall back to reshape when ch_axis is None and ndim exceeds 1."""
        # Data shape: (k0=4, extra=2) with ch_axis=None.
        # 'extra' is a loop dim but we don't index it -> readout_line is 2D (4, 2).
        # n_coils=2, n_k0=4 so reshape(2,4) succeeds.
        data = np.arange(8, dtype=np.complex64).reshape(4, 2)
        loop_axes: list[tuple[int, str, int]] = []  # no loop indexing
        readout, _ = _extract_readout(
            data, k0_axis=0, ch_axis=None, n_coils=2, n_k0=4, loop_axes=loop_axes, multi_idx=()
        )
        assert readout.shape == (2, 4)


# ---------------------------------------------------------------------------
# _compute_flags
# ---------------------------------------------------------------------------


class TestComputeFlags:
    """Tests for ``_compute_flags``."""

    def test_compute_flags_first_encode_first_object(self) -> None:
        """Set FIRST_IN_SLICE and FIRST_IN_REPETITION flags for index 0."""
        flags = _compute_flags(
            {'k_space_encode_step_1': 0, 'k_space_encode_step_2': 0, 'object': 0}, n_k1=4, n_k2=1, n_obj=2
        )
        assert flags & ismrmrd.ACQ_FIRST_IN_SLICE
        assert flags & ismrmrd.ACQ_FIRST_IN_REPETITION

    def test_compute_flags_last_encode_last_object(self) -> None:
        """Set LAST_IN_SLICE and LAST_IN_REPETITION flags for final indices."""
        flags = _compute_flags(
            {'k_space_encode_step_1': 3, 'k_space_encode_step_2': 0, 'object': 1}, n_k1=4, n_k2=1, n_obj=2
        )
        assert flags & ismrmrd.ACQ_LAST_IN_SLICE
        assert flags & ismrmrd.ACQ_LAST_IN_REPETITION

    def test_compute_flags_middle_encode(self) -> None:
        """Clear FIRST/LAST flags for a middle encode step."""
        flags = _compute_flags({'k_space_encode_step_1': 2, 'object': 0}, n_k1=4, n_k2=1, n_obj=2)
        assert not (flags & ismrmrd.ACQ_FIRST_IN_SLICE)
        assert not (flags & ismrmrd.ACQ_LAST_IN_SLICE)

    def test_compute_flags_defaults_when_missing(self) -> None:
        """Default to index 0 when k_space_encode_step_1/2 are missing."""
        flags = _compute_flags({}, n_k1=1, n_k2=1, n_obj=1)
        assert flags & ismrmrd.ACQ_FIRST_IN_SLICE
        assert flags & ismrmrd.ACQ_LAST_IN_SLICE
        assert flags & ismrmrd.ACQ_FIRST_IN_REPETITION
        assert flags & ismrmrd.ACQ_LAST_IN_REPETITION


# ---------------------------------------------------------------------------
# _set_orientation
# ---------------------------------------------------------------------------


class TestSetOrientation:
    """Tests for ``_set_orientation``."""

    def test_set_orientation_with_directions(self) -> None:
        """Assign read, phase, slice directions and position from arrays."""
        acq = ismrmrd.Acquisition()
        acq.resize(number_of_samples=4, active_channels=1, trajectory_dimensions=0)
        read_dir = np.array([[0.5, 0.5, 0]], dtype=np.float32)
        phase_dir = np.array([[-0.5, 0.5, 0]], dtype=np.float32)
        slice_dir = np.array([[0, 0, 1]], dtype=np.float32)
        position = np.array([[10, 20, 30]], dtype=np.float32)
        _set_orientation(acq, 0, read_dir, phase_dir, slice_dir, position)
        np.testing.assert_allclose(acq.read_dir, [0.5, 0.5, 0])
        np.testing.assert_allclose(acq.position, [10, 20, 30])

    def test_set_orientation_defaults_identity(self) -> None:
        """Default to identity directions when all inputs are None."""
        acq = ismrmrd.Acquisition()
        acq.resize(number_of_samples=4, active_channels=1, trajectory_dimensions=0)
        _set_orientation(acq, 0, None, None, None, None)
        np.testing.assert_allclose(acq.read_dir, [1, 0, 0])
        np.testing.assert_allclose(acq.phase_dir, [0, 1, 0])
        np.testing.assert_allclose(acq.slice_dir, [0, 0, 1])


# ---------------------------------------------------------------------------
# build_acquisitions
# ---------------------------------------------------------------------------


class TestBuildAcquisitions:
    """Tests for ``build_acquisitions``."""

    def test_build_acquisitions_basic_cartesian(self) -> None:
        """Build acquisitions from simple 2D Cartesian data (k0=8, k1=4, ch=1)."""
        ds = FakeDataset(params={'PVM_DigDw': 5.0})
        data = np.ones((8, 4, 1), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'channel']
        acqs = build_acquisitions(ds, data, dim_type)
        assert len(acqs) == 4  # one per k1 line
        assert acqs[0].number_of_samples == 8
        assert acqs[0].active_channels == 1
        assert acqs[0].center_sample == 4
        assert acqs[0].sample_time_us == pytest.approx(5.0)

    def test_build_acquisitions_missing_k0_raises(self) -> None:
        """Raise ValueError when k_space_encode_step_0 is missing from dim_type."""
        ds = FakeDataset()
        data = np.ones((8, 4), dtype=np.complex64)
        with pytest.raises(ValueError, match='No k_space_encode_step_0'):
            build_acquisitions(ds, data, ['x', 'y'])

    def test_build_acquisitions_encoding_indices(self) -> None:
        """Assign encoding indices following np.ndindex iteration order."""
        ds = FakeDataset(params={'PVM_DigDw': 5.0})
        data = np.ones((8, 4, 2, 1), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'channel']
        acqs = build_acquisitions(ds, data, dim_type)
        assert len(acqs) == 8  # 4 k1 * 2 slices
        # np.ndindex iterates C-order: (k1=0,obj=0),(k1=0,obj=1),(k1=1,obj=0),...
        assert acqs[0].idx.kspace_encode_step_1 == 0
        assert acqs[0].idx.slice == 0
        assert acqs[1].idx.kspace_encode_step_1 == 0
        assert acqs[1].idx.slice == 1
        assert acqs[2].idx.kspace_encode_step_1 == 1
        assert acqs[2].idx.slice == 0

    def test_build_acquisitions_flags_first_last(self) -> None:
        """Set FIRST_IN_SLICE and LAST_IN_SLICE flags on boundary acquisitions."""
        ds = FakeDataset(params={'PVM_DigDw': 5.0})
        data = np.ones((8, 4, 1), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'channel']
        acqs = build_acquisitions(ds, data, dim_type)
        assert acqs[0].flags & ismrmrd.ACQ_FIRST_IN_SLICE
        assert acqs[3].flags & ismrmrd.ACQ_LAST_IN_SLICE

    def test_build_acquisitions_with_trajectory(self) -> None:
        """Attach per-acquisition trajectory columns from a 3D trajectory array."""
        ds = FakeDataset(params={'PVM_DigDw': 5.0})
        data = np.ones((8, 4, 1), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'channel']
        traj = np.random.default_rng(42).random((2, 8, 4)).astype(np.float32)
        acqs = build_acquisitions(ds, data, dim_type, traj_data=traj)
        assert acqs[0].traj.shape == (8, 2)
        np.testing.assert_allclose(acqs[0].traj, traj[:, :8, 0].T)

    def test_build_acquisitions_no_channel_dim(self) -> None:
        """Infer a single channel when dim_type has no 'channel' entry."""
        ds = FakeDataset(params={'PVM_DigDw': 5.0})
        data = np.ones((8, 4), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1']
        acqs = build_acquisitions(ds, data, dim_type)
        assert len(acqs) == 4
        assert acqs[0].active_channels == 1

    def test_build_acquisitions_no_dwell_time(self) -> None:
        """Leave sample_time_us at 0 when PVM_DigDw is missing."""
        ds = FakeDataset()
        data = np.ones((8, 4, 1), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'channel']
        acqs = build_acquisitions(ds, data, dim_type)
        assert acqs[0].sample_time_us == pytest.approx(0.0)

    def test_build_acquisitions_multi_coil(self) -> None:
        """Build multi-channel acquisitions from data with a channel dimension."""
        ds = FakeDataset(params={'PVM_DigDw': 5.0})
        data = np.ones((8, 2, 4, 1), dtype=np.complex64)
        dim_type = ['k_space_encode_step_0', 'channel', 'k_space_encode_step_1', 'object']
        acqs = build_acquisitions(ds, data, dim_type)
        # 4 k1 lines * 1 object
        assert len(acqs) == 4
        assert acqs[0].active_channels == 2

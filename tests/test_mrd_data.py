"""Tests for bruker_to_ismrmrd.mrd_data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ismrmrd
import numpy as np
from ismrmrd import xsd

from bruker_to_ismrmrd.mrd_data import MrdData


def _make_header() -> xsd.ismrmrdschema.ismrmrdHeader:
    """Create a minimal valid ISMRMRD header."""
    return xsd.ismrmrdschema.ismrmrdHeader(
        encoding=[
            xsd.encodingType(
                trajectory=xsd.trajectoryType.CARTESIAN,
                encodedSpace=xsd.encodingSpaceType(
                    matrixSize=xsd.matrixSizeType(x=64, y=32, z=1),
                    fieldOfView_mm=xsd.fieldOfViewMm(x=200.0, y=200.0, z=5.0),
                ),
                reconSpace=xsd.encodingSpaceType(
                    matrixSize=xsd.matrixSizeType(x=64, y=32, z=1),
                    fieldOfView_mm=xsd.fieldOfViewMm(x=200.0, y=200.0, z=5.0),
                ),
                encodingLimits=xsd.encodingLimitsType(
                    kspace_encoding_step_1=xsd.limitType(minimum=0, maximum=31, center=16),
                ),
            )
        ],
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=128000000),
    )


def _make_acquisitions(n: int = 4, n_samples: int = 64) -> list[ismrmrd.Acquisition]:
    """Create a list of simple acquisitions."""
    rng = np.random.default_rng(42)
    acqs = []
    for k1 in range(n):
        acq = ismrmrd.Acquisition()
        acq.resize(number_of_samples=n_samples, active_channels=1, trajectory_dimensions=0)
        acq.data[:] = rng.standard_normal((1, n_samples)).astype(np.complex64)
        acq.idx.kspace_encode_step_1 = k1
        acqs.append(acq)
    return acqs


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMrdDataConstruction:
    """Tests for ``MrdData`` construction."""

    def test_mrd_data_minimal(self) -> None:
        """Construct an MrdData with only a header."""
        header = _make_header()
        mrd = MrdData(header=header)
        assert mrd.header is header
        assert mrd.acquisitions == []
        assert mrd.waveforms == []
        assert mrd.images == {}

    def test_mrd_data_with_acquisitions(self) -> None:
        """Construct an MrdData with acquisitions."""
        header = _make_header()
        acqs = _make_acquisitions(2)
        mrd = MrdData(header=header, acquisitions=acqs)
        assert len(mrd.acquisitions) == 2

    def test_mrd_data_with_waveforms(self) -> None:
        """Construct an MrdData with waveform data."""
        header = _make_header()
        wf = ismrmrd.Waveform()
        mrd = MrdData(header=header, waveforms=[wf])
        assert len(mrd.waveforms) == 1


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestMrdDataSave:
    """Tests for ``MrdData.save``."""

    def test_save_creates_file(self) -> None:
        """Create an MRD file on disk."""
        mrd = MrdData(header=_make_header(), acquisitions=_make_acquisitions(2))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mrd.save(Path(tmpdir) / 'out.mrd')
            assert path.exists()

    def test_save_creates_parent_directories(self) -> None:
        """Create intermediate directories when saving."""
        mrd = MrdData(header=_make_header(), acquisitions=_make_acquisitions(1))
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / 'a' / 'b' / 'out.mrd'
            path = mrd.save(nested)
            assert path.exists()

    def test_save_returns_resolved_path(self) -> None:
        """Return a resolved Path object after saving."""
        mrd = MrdData(header=_make_header())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mrd.save(Path(tmpdir) / 'test.mrd')
            assert isinstance(path, Path)

    def test_save_with_string_path(self) -> None:
        """Accept a string path and create the MRD file."""
        mrd = MrdData(header=_make_header(), acquisitions=_make_acquisitions(1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mrd.save(str(Path(tmpdir) / 'test.mrd'))
            assert path.exists()

    def test_save_empty_acquisitions(self) -> None:
        """Save an MrdData with no acquisitions."""
        mrd = MrdData(header=_make_header())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mrd.save(Path(tmpdir) / 'empty.mrd')
            assert path.exists()

    def test_save_custom_dataset_name(self) -> None:
        """Save and reload with a custom dataset name."""
        mrd = MrdData(header=_make_header(), acquisitions=_make_acquisitions(1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mrd.save(Path(tmpdir) / 'custom.mrd', dataset_name='my_dataset')
            assert path.exists()
            loaded = MrdData.from_file(path, dataset_name='my_dataset')
            assert len(loaded.acquisitions) == 1


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


class TestMrdDataFromFile:
    """Tests for ``MrdData.from_file``."""

    def test_from_file_header_roundtrip(self) -> None:
        """Preserve the ISMRMRD header XML through a save/load cycle."""
        original = MrdData(header=_make_header(), acquisitions=_make_acquisitions(3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'rt.mrd'
            original.save(path)
            loaded = MrdData.from_file(path)
            assert xsd.ToXML(loaded.header) == xsd.ToXML(original.header)

    def test_from_file_acquisitions_data_roundtrip(self) -> None:
        """Preserve acquisition data through a save/load cycle."""
        original = MrdData(header=_make_header(), acquisitions=_make_acquisitions(4))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'rt.mrd'
            original.save(path)
            loaded = MrdData.from_file(path)
            assert len(loaded.acquisitions) == 4
            for orig, load in zip(original.acquisitions, loaded.acquisitions, strict=True):
                np.testing.assert_array_equal(orig.data, load.data)

    def test_from_file_acquisitions_index_roundtrip(self) -> None:
        """Preserve encoding indices through a save/load cycle."""
        original = MrdData(header=_make_header(), acquisitions=_make_acquisitions(4))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'rt.mrd'
            original.save(path)
            loaded = MrdData.from_file(path)
            for orig, load in zip(original.acquisitions, loaded.acquisitions, strict=True):
                assert orig.idx.kspace_encode_step_1 == load.idx.kspace_encode_step_1

    def test_from_file_with_trajectory(self) -> None:
        """Preserve trajectory data through a save/load cycle."""
        header = _make_header()
        rng = np.random.default_rng(7)
        acq = ismrmrd.Acquisition()
        acq.resize(number_of_samples=64, active_channels=1, trajectory_dimensions=2)
        acq.data[:] = rng.standard_normal((1, 64)).astype(np.complex64)
        acq.traj[:] = rng.random((64, 2)).astype(np.float32)
        original = MrdData(header=header, acquisitions=[acq])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'traj.mrd'
            original.save(path)
            loaded = MrdData.from_file(path)
            np.testing.assert_allclose(loaded.acquisitions[0].traj, original.acquisitions[0].traj)

    def test_from_file_empty_acquisitions(self) -> None:
        """Load an MrdData with no acquisitions, waveforms, or images."""
        original = MrdData(header=_make_header())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'empty.mrd'
            original.save(path)
            loaded = MrdData.from_file(path)
            assert loaded.acquisitions == []
            assert loaded.waveforms == []
            assert loaded.images == {}

    def test_from_file_string_path(self) -> None:
        """Accept a string path for loading."""
        original = MrdData(header=_make_header(), acquisitions=_make_acquisitions(1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'str.mrd'
            original.save(path)
            loaded = MrdData.from_file(str(path))
            assert len(loaded.acquisitions) == 1

    def test_from_file_waveform_roundtrip(self) -> None:
        """Preserve waveform data through a save/load cycle."""
        header = _make_header()
        wf = ismrmrd.Waveform()
        wf.resize(16, 1)
        wf.data[:] = np.arange(16, dtype=np.uint32).reshape(1, 16)
        original = MrdData(header=header, waveforms=[wf])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'wf.mrd'
            original.save(path)
            loaded = MrdData.from_file(path)
            assert len(loaded.waveforms) == 1
            np.testing.assert_array_equal(loaded.waveforms[0].data, original.waveforms[0].data)

    def test_save_with_images(self) -> None:
        """Verify that images can be saved (loading has an ismrmrd-python bug)."""
        header = _make_header()
        img = ismrmrd.Image()
        img.resize(1, 1, 4, 4)
        img.data[:] = np.arange(16, dtype=np.complex64).reshape(1, 1, 4, 4)
        original = MrdData(header=header, images={'image_0': [img]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'img.mrd'
            result = original.save(path)
            assert result.exists()

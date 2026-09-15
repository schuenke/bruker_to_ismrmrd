"""Tests for bruker_to_ismrmrd.converter."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bruker_to_ismrmrd.converter import (
    _extract_kspace,
    _extract_noncartesian_rawdata,
    _load_auxiliary_parameters,
    _load_trajectory,
    _set_acquisition_time,
    convert,
)
from bruker_to_ismrmrd.mrd_data import MrdData
from tests.conftest import FakeDataset

# ---------------------------------------------------------------------------
# _load_auxiliary_parameters
# ---------------------------------------------------------------------------


class TestLoadAuxiliaryParameters:
    """Tests for ``_load_auxiliary_parameters``."""

    def test_load_auxiliary_parameters_subject_present(self, tmp_path: Path) -> None:
        """Load the subject file into _parameters when it exists."""
        experiment_dir = tmp_path / 'study' / '1'
        experiment_dir.mkdir(parents=True)
        subject_path = tmp_path / 'study' / 'subject'
        subject_path.write_text('##TITLE=subject\n##END=\n')

        ds = FakeDataset()
        mock_jcampdx = MagicMock(return_value='subject_params')
        _load_auxiliary_parameters(ds, experiment_dir, mock_jcampdx)
        assert 'subject' in ds._parameters

    def test_load_auxiliary_parameters_no_subject(self, tmp_path: Path) -> None:
        """Leave _parameters unchanged when the subject file is missing."""
        experiment_dir = tmp_path / 'study' / '1'
        experiment_dir.mkdir(parents=True)

        ds = FakeDataset()
        mock_jcampdx = MagicMock()
        _load_auxiliary_parameters(ds, experiment_dir, mock_jcampdx)
        assert 'subject' not in ds._parameters

    def test_load_auxiliary_parameters_visu_pars_present(self, tmp_path: Path) -> None:
        """Load the visu_pars file when it exists."""
        experiment_dir = tmp_path / 'study' / '1'
        pdata_dir = experiment_dir / 'pdata' / '1'
        pdata_dir.mkdir(parents=True)
        (pdata_dir / 'visu_pars').write_text('##TITLE=visu\n##END=\n')

        ds = FakeDataset()
        mock_jcampdx = MagicMock(return_value='visu_params')
        _load_auxiliary_parameters(ds, experiment_dir, mock_jcampdx)
        assert 'visu_pars' in ds._parameters

    def test_load_auxiliary_parameters_visu_pars_already_loaded(self, tmp_path: Path) -> None:
        """Skip loading visu_pars when it is already present in _parameters."""
        experiment_dir = tmp_path / 'study' / '1'
        pdata_dir = experiment_dir / 'pdata' / '1'
        pdata_dir.mkdir(parents=True)
        (pdata_dir / 'visu_pars').write_text('##TITLE=visu\n##END=\n')

        ds = FakeDataset()
        ds._parameters['visu_pars'] = 'existing'
        mock_jcampdx = MagicMock(return_value='new_params')
        _load_auxiliary_parameters(ds, experiment_dir, mock_jcampdx)
        assert ds._parameters['visu_pars'] == 'existing'


# ---------------------------------------------------------------------------
# _extract_kspace
# ---------------------------------------------------------------------------


class TestExtractKspace:
    """Tests for ``_extract_kspace``."""

    def test_extract_kspace_fid_dataset(self) -> None:
        """Return k-space and dim_type directly from a fid dataset."""
        data = np.ones((8, 4, 1), dtype=np.complex64)
        ds = FakeDataset(
            type='fid',
            kspace=data,
            dim_type=['k_space_encode_step_0', 'k_space_encode_step_1', 'channel'],
        )
        result_data, result_dim = _extract_kspace(ds, scheme_id='CART_2D')
        np.testing.assert_array_equal(result_data, data)
        assert result_dim == ['k_space_encode_step_0', 'k_space_encode_step_1', 'channel']

    def test_extract_kspace_rawdata_noncartesian_spiral(self) -> None:
        """Delegate to _extract_noncartesian_rawdata for non-Cartesian rawdata."""
        raw = np.ones((16, 4, 1), dtype=np.complex64)
        ds = FakeDataset(type='rawdata', raw=raw)
        ds.set_param('NI', 1)
        ds.set_param('NR', 1)
        _result_data, result_dim = _extract_kspace(ds, scheme_id='SPIRAL')
        assert 'k_space_encode_step_0' in result_dim

    def test_extract_kspace_rawdata_cartesian_with_dim_type_fix(self) -> None:
        """Fix PV360 dim_type replacing 'sample' with 'k_space_encode_step_0'."""
        data = np.ones((8, 4, 1, 1, 2), dtype=np.complex64)
        ds = FakeDataset(
            type='rawdata',
            kspace=data,
            dim_type=['sample', 'channel', 'scan', 'object', 'repetition'],
        )
        ds.set_param('ACQ_dim', 2)
        _result_data, result_dim = _extract_kspace(ds, scheme_id='CART_2D')
        assert result_dim[0] == 'k_space_encode_step_0'

    def test_extract_kspace_rawdata_3d_cartesian(self) -> None:
        """Remap scan to k_space_encode_step_2 for 3D Cartesian rawdata."""
        data = np.ones((8, 4, 2, 1, 1), dtype=np.complex64)
        ds = FakeDataset(
            type='rawdata',
            kspace=data,
            dim_type=['sample', 'channel', 'scan', 'object', 'repetition'],
        )
        ds.set_param('ACQ_dim', 3)
        _, result_dim = _extract_kspace(ds, scheme_id='CART_2D')
        assert result_dim == [
            'k_space_encode_step_0',
            'k_space_encode_step_1',
            'k_space_encode_step_2',
            'repetition',
            'channel',
        ]


# ---------------------------------------------------------------------------
# _extract_noncartesian_rawdata
# ---------------------------------------------------------------------------


class TestExtractNoncartesianRawdata:
    """Tests for ``_extract_noncartesian_rawdata``."""

    def test_extract_noncartesian_rawdata_reshape(self) -> None:
        """Reshape raw data into (samples, projections, objects, repetitions, channels)."""
        # raw shape: (n_samples=16, n_scans=6, n_receivers=2)
        # NI=2, NR=1 -> n_projections = 6 / (2*1) = 3
        raw = np.arange(192, dtype=np.complex64).reshape(16, 6, 2)
        ds = FakeDataset(raw=raw)
        ds.set_param('NI', 2)
        ds.set_param('NR', 1)
        data, dim_type = _extract_noncartesian_rawdata(ds)
        assert data.shape == (16, 3, 2, 1, 2)
        assert dim_type == ['k_space_encode_step_0', 'k_space_encode_step_1', 'object', 'repetition', 'channel']

    def test_extract_noncartesian_rawdata_defaults(self) -> None:
        """Default NI and NR to 1 when missing."""
        raw = np.ones((8, 4, 1), dtype=np.complex64)
        ds = FakeDataset(raw=raw)
        data, _ = _extract_noncartesian_rawdata(ds)
        assert data.shape == (8, 4, 1, 1, 1)


# ---------------------------------------------------------------------------
# _load_trajectory
# ---------------------------------------------------------------------------


class TestLoadTrajectory:
    """Tests for ``_load_trajectory``."""

    def test_load_trajectory_available(self) -> None:
        """Scale trajectory by the encoding matrix when both are present."""
        traj_raw = np.random.default_rng(42).random((2, 64, 10)).astype(np.float32)
        expected = traj_raw.copy()
        expected[0] *= 128
        expected[1] *= 128
        ds = FakeDataset(_traj=traj_raw, traj=traj_raw)
        ds.set_param('PVM_EncMatrix', np.array([128, 128]))
        result = _load_trajectory(ds)
        assert result is not None
        assert result.shape == traj_raw.shape
        # Should be scaled by enc_matrix
        np.testing.assert_allclose(result[0], expected[0])
        np.testing.assert_allclose(result[1], expected[1])

    def test_load_trajectory_none(self) -> None:
        """Return None when _traj is explicitly None."""
        ds = FakeDataset(_traj=None)
        result = _load_trajectory(ds)
        assert result is None

    def test_load_trajectory_no_traj_attr(self) -> None:
        """Return None when the dataset has no _traj attribute."""
        ds = FakeDataset()
        result = _load_trajectory(ds)
        assert result is None

    def test_load_trajectory_no_enc_matrix(self) -> None:
        """Return unscaled trajectory when PVM_EncMatrix is missing."""
        traj = np.ones((2, 8, 4), dtype=np.float32) * 0.5
        ds = FakeDataset(_traj=traj, traj=traj)
        result = _load_trajectory(ds)
        assert result is not None
        np.testing.assert_allclose(result, traj)


# ---------------------------------------------------------------------------
# _set_acquisition_time
# ---------------------------------------------------------------------------


class TestSetAcquisitionTime:
    """Tests for ``_set_acquisition_time``."""

    def test_set_acquisition_time_from_acq_time(self) -> None:
        """Parse ACQ_time and set seriesDate/seriesTime on the header."""
        ds = FakeDataset(params={'ACQ_time': '<2024-06-15T14:30:00,000000+0200>'})
        header = MagicMock()
        header.measurementInformation = MagicMock()
        _set_acquisition_time(ds, header)
        assert header.measurementInformation.seriesDate.year == 2024
        assert header.measurementInformation.seriesDate.month == 6
        assert header.measurementInformation.seriesTime.hour == 14

    def test_set_acquisition_time_fallback_to_dataset_date(self) -> None:
        """Fall back to the dataset's date attribute when ACQ_time is missing."""
        ds = FakeDataset(date=dt.datetime(2023, 3, 1, 10, 0, 0))
        header = MagicMock()
        header.measurementInformation = MagicMock()
        _set_acquisition_time(ds, header)
        assert header.measurementInformation.seriesDate.year == 2023

    def test_set_acquisition_time_none(self) -> None:
        """Leave the header unchanged when no time information is available."""
        ds = FakeDataset()
        header = MagicMock()
        header.measurementInformation = MagicMock()
        _set_acquisition_time(ds, header)
        # seriesDate was never assigned (no setattr call beyond mock init)
        assert not hasattr(header.measurementInformation.seriesDate, 'year') or isinstance(
            header.measurementInformation.seriesDate, MagicMock
        )

    def test_set_acquisition_time_no_measurement_info(self) -> None:
        """Do nothing when measurementInformation is None."""
        ds = FakeDataset(params={'ACQ_time': '<2024-01-01T00:00:00,000000+0000>'})
        header = MagicMock()
        header.measurementInformation = None
        # Should not raise
        _set_acquisition_time(ds, header)


# ---------------------------------------------------------------------------
# convert (integration-style with mocks)
# ---------------------------------------------------------------------------


class TestConvert:
    """Tests for ``convert``."""

    @pytest.fixture
    def _mock_bruker(self, tmp_path: Path) -> tuple[Path, MagicMock]:
        """Set up a mock brukerapi Dataset for convert()."""
        # Create minimal experiment directory structure
        exp_dir = tmp_path / 'study' / '1'
        exp_dir.mkdir(parents=True)
        fid_path = exp_dir / 'fid'
        fid_path.write_bytes(b'\x00' * 16)

        mock_dataset = MagicMock()
        mock_dataset.path = str(fid_path)
        mock_dataset.type = 'fid'
        mock_dataset.scheme_id = 'CART_2D'
        mock_dataset._traj = None
        mock_dataset._parameters = {}

        # K-space data: (k0=8, k1=4, ch=1)
        mock_dataset.kspace = np.ones((8, 4, 1), dtype=np.complex64)
        mock_dataset.dim_type = ['k_space_encode_step_0', 'k_space_encode_step_1', 'channel']

        # Parameters needed for header
        param_values = {
            'PVM_EncMatrix': np.array([8, 4]),
            'PVM_Fov': np.array([100.0, 50.0]),
            'PVM_FrqWork': np.array([300.0]),
            'PVM_MagnetStrength': 7.0,
            'PVM_EncNReceivers': 1,
            'PULPROG': '<FLASH>',
            'PVM_DigDw': 5.0,
            'ACQ_time': '<2024-01-15T10:30:00,500000+0100>',
        }

        class ParamObj:
            """Wrapper exposing a value attribute for mock parameters."""

            def __init__(self, v: object) -> None:
                """Store the parameter value."""
                self.value = v

        def getitem(name: str) -> ParamObj:
            if name in param_values:
                return ParamObj(param_values[name])
            raise KeyError(name)

        mock_dataset.__getitem__ = MagicMock(side_effect=getitem)
        mock_dataset.__contains__ = MagicMock(side_effect=lambda n: n in param_values)

        return fid_path, mock_dataset

    @patch('brukerapi.jcampdx.JCAMPDX')
    @patch('brukerapi.dataset.Dataset')
    def test_convert_returns_mrd_data(
        self, mock_dataset_cls: MagicMock, _mock_jcampdx_cls: MagicMock, _mock_bruker: tuple[Path, MagicMock]
    ) -> None:
        """Return an MrdData with the expected number of acquisitions."""
        fid_path, mock_dataset = _mock_bruker
        mock_dataset_cls.return_value = mock_dataset
        result = convert(fid_path)
        assert isinstance(result, MrdData)
        assert len(result.acquisitions) == 4

    @patch('brukerapi.jcampdx.JCAMPDX')
    @patch('brukerapi.dataset.Dataset')
    def test_convert_header_encoding(
        self, mock_dataset_cls: MagicMock, _mock_jcampdx_cls: MagicMock, _mock_bruker: tuple[Path, MagicMock]
    ) -> None:
        """Populate the ISMRMRD header encoding from Bruker parameters."""
        fid_path, mock_dataset = _mock_bruker
        mock_dataset_cls.return_value = mock_dataset
        result = convert(fid_path)
        enc = result.header.encoding[0]
        assert enc.encodedSpace.matrixSize.x == 8
        assert enc.encodedSpace.matrixSize.y == 4

    @patch('brukerapi.jcampdx.JCAMPDX')
    @patch('brukerapi.dataset.Dataset')
    def test_convert_acquisition_datetime_in_header(
        self, mock_dataset_cls: MagicMock, _mock_jcampdx_cls: MagicMock, _mock_bruker: tuple[Path, MagicMock]
    ) -> None:
        """Store ACQ_time in the header's measurementInformation."""
        fid_path, mock_dataset = _mock_bruker
        mock_dataset_cls.return_value = mock_dataset
        result = convert(fid_path)
        mi = result.header.measurementInformation
        assert mi is not None
        assert mi.seriesDate is not None
        assert mi.seriesDate.year == 2024

    @patch('brukerapi.jcampdx.JCAMPDX')
    @patch('brukerapi.dataset.Dataset')
    def test_convert_with_trajectory(
        self, mock_dataset_cls: MagicMock, _mock_jcampdx_cls: MagicMock, _mock_bruker: tuple[Path, MagicMock]
    ) -> None:
        """Attach trajectory data to acquisitions for non-Cartesian schemes."""
        fid_path, mock_dataset = _mock_bruker
        mock_dataset_cls.return_value = mock_dataset

        # Add trajectory
        traj = np.random.default_rng(42).random((2, 8, 4)).astype(np.float32) * 0.5
        mock_dataset._traj = traj
        mock_dataset.traj = traj
        mock_dataset.scheme_id = 'SPIRAL'

        result = convert(fid_path)
        assert result.acquisitions[0].traj.shape[1] == 2  # 2D trajectory

    @patch('brukerapi.jcampdx.JCAMPDX')
    @patch('brukerapi.dataset.Dataset')
    def test_convert_trajectory_truncation(
        self, mock_dataset_cls: MagicMock, _mock_jcampdx_cls: MagicMock, _mock_bruker: tuple[Path, MagicMock]
    ) -> None:
        """Truncate readout data when trajectory has fewer samples."""
        fid_path, mock_dataset = _mock_bruker
        mock_dataset_cls.return_value = mock_dataset

        # Trajectory with 6 samples vs 8 readout points
        traj = np.random.default_rng(42).random((2, 6, 4)).astype(np.float32) * 0.5
        mock_dataset._traj = traj
        mock_dataset.traj = traj
        mock_dataset.scheme_id = 'SPIRAL'

        # We need to add PVM_SpiralPreSize to the param lookup
        original_getitem = mock_dataset.__getitem__.side_effect

        def new_getitem(name: str) -> object:
            if name == 'PVM_SpiralPreSize':

                class P:
                    """Param wrapper."""

                    value = 1

                return P()
            return original_getitem(name)

        mock_dataset.__getitem__.side_effect = new_getitem

        result = convert(fid_path)
        # Readout should be truncated to 6 samples
        assert result.acquisitions[0].number_of_samples == 6

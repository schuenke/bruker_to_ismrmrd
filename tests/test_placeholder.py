"""Smoke tests to verify the package structure and imports."""

import datetime as dt
import tempfile
from pathlib import Path

import ismrmrd
import numpy as np
from ismrmrd import xsd

from bruker_to_ismrmrd import MrdData


def test_import() -> None:
    """Test that the package can be imported."""
    import bruker_to_ismrmrd

    assert bruker_to_ismrmrd is not None


def test_public_api() -> None:
    """Test that the public API is accessible."""
    from bruker_to_ismrmrd import MrdData, convert

    assert callable(convert)
    assert MrdData is not None


def test_parse_bruker_datetime() -> None:
    """Test parsing of Bruker timestamp formats."""
    from bruker_to_ismrmrd.parameters import parse_bruker_datetime

    # PV360 ISO format
    result = parse_bruker_datetime('<2024-01-15T10:30:00,500000+0100>')
    assert result is not None
    assert isinstance(result, dt.datetime)
    assert result.year == 2024
    assert result.month == 1
    assert result.day == 15
    assert result.tzinfo is None  # should be timezone-naive

    # Older PV5/6/7 format
    result = parse_bruker_datetime('10:30:00 15 Jan 2024')
    assert result is not None
    assert result.year == 2024

    # Invalid / missing
    assert parse_bruker_datetime(None) is None
    assert parse_bruker_datetime('') is None
    assert parse_bruker_datetime('not-a-date') is None


def test_resolve_scheme_id_fallback() -> None:
    """Test that resolve_scheme_id returns CART_2D for unknown datasets."""
    from bruker_to_ismrmrd.constants import resolve_scheme_id

    # A plain object has no scheme_id or _infer_scheme_id
    assert resolve_scheme_id(object()) == 'CART_2D'


def _create_minimal_mrd_data() -> 'MrdData':
    """Create a minimal MrdData instance for testing."""
    from bruker_to_ismrmrd import MrdData

    header = xsd.ismrmrdschema.ismrmrdHeader(
        encoding=[
            xsd.encodingType(
                trajectory=xsd.trajectoryType.CARTESIAN,
                encodedSpace=xsd.encodingSpaceType(
                    matrixSize=xsd.matrixSizeType(x=256, y=128, z=1),
                    fieldOfView_mm=xsd.fieldOfViewMm(x=200.0, y=200.0, z=5.0),
                ),
                reconSpace=xsd.encodingSpaceType(
                    matrixSize=xsd.matrixSizeType(x=256, y=128, z=1),
                    fieldOfView_mm=xsd.fieldOfViewMm(x=200.0, y=200.0, z=5.0),
                ),
                encodingLimits=xsd.encodingLimitsType(
                    kspace_encoding_step_1=xsd.limitType(minimum=0, maximum=127, center=64),
                ),
            )
        ],
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=128000000),
    )

    # Create two simple acquisitions
    acquisitions = []
    for k1 in range(2):
        acq = ismrmrd.Acquisition()
        acq.resize(number_of_samples=256, active_channels=1, trajectory_dimensions=0)
        acq.data[:] = np.random.default_rng(k1).standard_normal((1, 256)).astype(np.complex64)
        acq.idx.kspace_encode_step_1 = k1
        acquisitions.append(acq)

    return MrdData(header=header, acquisitions=acquisitions)


def test_mrd_data_save_and_load() -> None:
    """Test MrdData round-trip: save to file, load back, compare."""
    from bruker_to_ismrmrd import MrdData

    original = _create_minimal_mrd_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        mrd_path = Path(tmpdir) / 'test.mrd'

        # Save
        result_path = original.save(mrd_path)
        assert result_path.exists()

        # Load back
        loaded = MrdData.from_file(mrd_path)

        # Compare header
        assert xsd.ToXML(loaded.header) == xsd.ToXML(original.header)

        # Compare acquisitions
        assert len(loaded.acquisitions) == len(original.acquisitions)
        for orig_acq, load_acq in zip(original.acquisitions, loaded.acquisitions, strict=True):
            assert orig_acq.idx.kspace_encode_step_1 == load_acq.idx.kspace_encode_step_1
            np.testing.assert_array_equal(orig_acq.data, load_acq.data)

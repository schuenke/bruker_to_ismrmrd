"""Tests for bruker_to_ismrmrd.parameters."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from bruker_to_ismrmrd.parameters import get_array, get_float, get_int, get_param, get_str, parse_bruker_datetime
from tests.conftest import FakeDataset

# ---------------------------------------------------------------------------
# get_param
# ---------------------------------------------------------------------------


class TestGetParam:
    """Tests for ``get_param``."""

    def test_get_param_first_match(self, fake_dataset: FakeDataset) -> None:
        """Return the first matching parameter when multiple names are given."""
        fake_dataset.set_param('A', 42)
        fake_dataset.set_param('B', 99)
        assert get_param(fake_dataset, 'A', 'B') == 42

    def test_get_param_fallback_to_second(self, fake_dataset: FakeDataset) -> None:
        """Fall back to the second name when the first is missing."""
        fake_dataset.set_param('B', 99)
        assert get_param(fake_dataset, 'A', 'B') == 99

    def test_get_param_missing_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is not present."""
        assert get_param(fake_dataset, 'MISSING') is None

    def test_get_param_exception_during_access(self) -> None:
        """Suppress exceptions raised during parameter access and return None."""

        class BadDataset:
            """Dataset that raises on item access."""

            def __getitem__(self, item: str) -> None:
                """Raise RuntimeError for any item access."""
                raise RuntimeError('broken')

        assert get_param(BadDataset(), 'X') is None


# ---------------------------------------------------------------------------
# get_float
# ---------------------------------------------------------------------------


class TestGetFloat:
    """Tests for ``get_float``."""

    def test_get_float_scalar(self, fake_dataset: FakeDataset) -> None:
        """Return a scalar float value."""
        fake_dataset.set_param('PVM_DigDw', 10.5)
        assert get_float(fake_dataset, 'PVM_DigDw') == pytest.approx(10.5)

    def test_get_float_from_array(self, fake_dataset: FakeDataset) -> None:
        """Extract the first element from a single-element array."""
        fake_dataset.set_param('PVM_FrqWork', np.array([300.15]))
        assert get_float(fake_dataset, 'PVM_FrqWork') == pytest.approx(300.15)

    def test_get_float_missing_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is missing."""
        assert get_float(fake_dataset, 'MISSING') is None

    def test_get_float_empty_array_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is an empty array."""
        fake_dataset.set_param('EMPTY', np.array([]))
        assert get_float(fake_dataset, 'EMPTY') is None


# ---------------------------------------------------------------------------
# get_int
# ---------------------------------------------------------------------------


class TestGetInt:
    """Tests for ``get_int``."""

    def test_get_int_scalar(self, fake_dataset: FakeDataset) -> None:
        """Return a scalar integer value."""
        fake_dataset.set_param('NR', 5)
        assert get_int(fake_dataset, 'NR') == 5

    def test_get_int_from_float(self, fake_dataset: FakeDataset) -> None:
        """Truncate a float to int."""
        fake_dataset.set_param('NI', 3.0)
        assert get_int(fake_dataset, 'NI') == 3

    def test_get_int_from_array(self, fake_dataset: FakeDataset) -> None:
        """Extract the first element from a single-element array as int."""
        fake_dataset.set_param('ACQ_dim', np.array([2]))
        assert get_int(fake_dataset, 'ACQ_dim') == 2

    def test_get_int_missing_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is missing."""
        assert get_int(fake_dataset, 'MISSING') is None

    def test_get_int_empty_array_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is an empty array."""
        fake_dataset.set_param('EMPTY', np.array([]))
        assert get_int(fake_dataset, 'EMPTY') is None


# ---------------------------------------------------------------------------
# get_str
# ---------------------------------------------------------------------------


class TestGetStr:
    """Tests for ``get_str``."""

    def test_get_str_plain(self, fake_dataset: FakeDataset) -> None:
        """Return a plain string value."""
        fake_dataset.set_param('PULPROG', 'RARE')
        assert get_str(fake_dataset, 'PULPROG') == 'RARE'

    def test_get_str_strips_angle_brackets(self, fake_dataset: FakeDataset) -> None:
        """Strip surrounding angle brackets from the value."""
        fake_dataset.set_param('PULPROG', '<RARE>')
        assert get_str(fake_dataset, 'PULPROG') == 'RARE'

    def test_get_str_missing_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is missing."""
        assert get_str(fake_dataset, 'MISSING') is None

    def test_get_str_from_numeric_value(self, fake_dataset: FakeDataset) -> None:
        """Convert a numeric value to its string representation."""
        fake_dataset.set_param('SomeNum', 42)
        result = get_str(fake_dataset, 'SomeNum')
        assert result == '42'

    def test_get_str_empty_array_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is an empty array."""
        fake_dataset.set_param('EMPTY', np.array([]))
        assert get_str(fake_dataset, 'EMPTY') is None

    def test_get_str_from_array(self, fake_dataset: FakeDataset) -> None:
        """Extract and strip a string from a single-element array."""
        fake_dataset.set_param('Method', np.array(['<EPI>']))
        assert get_str(fake_dataset, 'Method') == 'EPI'


# ---------------------------------------------------------------------------
# get_array
# ---------------------------------------------------------------------------


class TestGetArray:
    """Tests for ``get_array``."""

    def test_get_array_returns_ndarray(self, fake_dataset: FakeDataset) -> None:
        """Return the parameter as a NumPy array."""
        fake_dataset.set_param('PVM_EncMatrix', np.array([256, 128]))
        result = get_array(fake_dataset, 'PVM_EncMatrix')
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [256, 128])

    def test_get_array_with_dtype(self, fake_dataset: FakeDataset) -> None:
        """Cast the result to the requested dtype."""
        fake_dataset.set_param('PVM_Matrix', np.array([64, 64]))
        result = get_array(fake_dataset, 'PVM_Matrix', dtype=np.float32)
        assert result is not None
        assert result.dtype == np.float32

    def test_get_array_missing_returns_none(self, fake_dataset: FakeDataset) -> None:
        """Return None when the parameter is missing."""
        assert get_array(fake_dataset, 'MISSING') is None

    def test_get_array_no_dtype(self, fake_dataset: FakeDataset) -> None:
        """Preserve the original dtype when none is specified."""
        fake_dataset.set_param('DATA', np.array([1.0, 2.0, 3.0]))
        result = get_array(fake_dataset, 'DATA')
        assert result is not None
        assert result.dtype == np.float64

    def test_get_array_fallback_names(self, fake_dataset: FakeDataset) -> None:
        """Fall back to the second name when the first is missing."""
        fake_dataset.set_param('PVM_Matrix', np.array([32, 32]))
        result = get_array(fake_dataset, 'PVM_EncMatrix', 'PVM_Matrix')
        assert result is not None
        np.testing.assert_array_equal(result, [32, 32])


# ---------------------------------------------------------------------------
# parse_bruker_datetime
# ---------------------------------------------------------------------------


class TestParseBrukerDatetime:
    """Tests for ``parse_bruker_datetime``."""

    def test_parse_bruker_datetime_iso_format(self) -> None:
        """Parse an ISO-formatted Bruker timestamp with timezone."""
        result = parse_bruker_datetime('<2024-01-15T10:30:00,500000+0100>')
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo is None

    def test_parse_bruker_datetime_legacy_format(self) -> None:
        """Parse a legacy Bruker timestamp without timezone."""
        result = parse_bruker_datetime('10:30:00 15 Jan 2024')
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo is None

    def test_parse_bruker_datetime_none_returns_none(self) -> None:
        """Return None when the input is None."""
        assert parse_bruker_datetime(None) is None

    def test_parse_bruker_datetime_empty_returns_none(self) -> None:
        """Return None when the input is an empty string."""
        assert parse_bruker_datetime('') is None

    def test_parse_bruker_datetime_invalid_returns_none(self) -> None:
        """Return None when the input is not a valid date string."""
        assert parse_bruker_datetime('not-a-date') is None

    def test_parse_bruker_datetime_no_timezone(self) -> None:
        """Return a naive datetime for legacy format without timezone info."""
        result = parse_bruker_datetime('14:00:00 01 Mar 2023')
        assert result is not None
        assert result.tzinfo is None
        assert result == dt.datetime(2023, 3, 1, 14, 0, 0)

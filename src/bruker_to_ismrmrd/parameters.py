"""Helpers for reading Bruker parameters from a brukerapi Dataset.

These wrappers try multiple parameter names (to handle differences between
ParaVision versions) and return ``None`` when a parameter is missing rather
than raising an exception.
"""

from __future__ import annotations

import contextlib
import datetime as dt

import numpy as np


def get_param(dataset: object, *names: str) -> object | None:
    """Read the first available parameter from a brukerapi Dataset.

    Parameters
    ----------
    dataset
        A ``brukerapi.dataset.Dataset`` instance.
    names
        Parameter names to try (first match wins).

    Returns
    -------
        The parameter value, or ``None`` if not found.
    """
    for name in names:
        with contextlib.suppress(Exception):
            return dataset[name].value  # type: ignore[index]
    return None


def get_float(dataset: object, *names: str) -> float | None:
    """Read the first available parameter as a float."""
    val = get_param(dataset, *names)
    if val is None:
        return None
    arr = np.asarray(val)
    return float(arr.reshape(-1)[0]) if arr.size > 0 else None


def get_int(dataset: object, *names: str) -> int | None:
    """Read the first available parameter as an integer."""
    val = get_param(dataset, *names)
    if val is None:
        return None
    arr = np.asarray(val)
    return int(arr.reshape(-1)[0]) if arr.size > 0 else None


def get_str(dataset: object, *names: str) -> str | None:
    """Read the first available parameter as a string."""
    val = get_param(dataset, *names)
    if val is None:
        return None
    if isinstance(val, str):
        return val.strip('<>')
    arr = np.asarray(val)
    return str(arr.reshape(-1)[0]).strip('<>') if arr.size > 0 else None


def get_array(dataset: object, *names: str, dtype: type | None = None) -> np.ndarray | None:
    """Read the first available parameter as a numpy array."""
    val = get_param(dataset, *names)
    if val is None:
        return None
    arr = np.asarray(val)
    return arr.astype(dtype) if dtype is not None else arr


def parse_bruker_datetime(value: str | None) -> dt.datetime | None:
    """Parse a Bruker acquisition time string.

    Handles both PV-360 ISO format (``2024-01-01T12:00:00,000+0100``)
    and older PV5/6/7 format (``12:00:00 01 Jan 2024``).

    Parameters
    ----------
    value
        Raw timestamp string from ``ACQ_time`` or similar.

    Returns
    -------
        Timezone-naive datetime, or ``None`` if parsing fails.
    """
    if not value:
        return None
    for fmt in ('%Y-%m-%dT%H:%M:%S,%f%z', '%H:%M:%S %d %b %Y'):
        try:
            timestamp = dt.datetime.strptime(value.strip('<>'), fmt)
        except ValueError:
            continue
        return timestamp.replace(tzinfo=None) if timestamp.tzinfo is not None else timestamp
    return None

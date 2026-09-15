"""PyTest configuration and shared fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


@dataclass
class FakeParam:
    """Mimics a brukerapi JCAMPDX parameter (has a ``.value`` attribute)."""

    value: Any


class FakeDataset:
    """Lightweight mock of ``brukerapi.dataset.Dataset``.

    Supports the ``dataset[name].value`` access pattern used by the
    parameter helpers and the conversion pipeline.
    """

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: object) -> None:
        """Initialize the fake dataset with optional parameters and attributes."""
        self._params: dict[str, FakeParam] = {}
        self._parameters: dict[str, Any] = {}
        if params:
            for k, v in params.items():
                self._params[k] = FakeParam(v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item: str) -> FakeParam:
        """Return the parameter for *item* or raise ``KeyError``."""
        if item in self._params:
            return self._params[item]
        raise KeyError(item)

    def __contains__(self, item: str) -> bool:
        """Check whether *item* exists in the parameter store."""
        return item in self._params

    def set_param(self, name: str, value: object) -> None:
        """Set a parameter after construction."""
        self._params[name] = FakeParam(value)


@pytest.fixture
def fake_dataset() -> FakeDataset:
    """Return a bare ``FakeDataset`` with no parameters."""
    return FakeDataset()


@pytest.fixture
def cartesian_dataset() -> FakeDataset:
    """Return a FakeDataset pre-loaded with typical Cartesian 2D parameters."""
    return FakeDataset(
        params={
            'PVM_EncMatrix': np.array([256, 128]),
            'PVM_Fov': np.array([200.0, 200.0]),
            'PVM_SliceThick': 2.0,
            'PVM_SPackArrNSlices': 3,
            'NR': 1,
            'PVM_FrqWork': np.array([300.0]),
            'PVM_MagnetStrength': 7.0,
            'PVM_EncNReceivers': 1,
            'SUBJECT_position': 'Head_Supine',
            'PULPROG': '<RARE>',
            'SUBJECT_name_string': 'TestSubject',
            'PVM_RepetitionTime': 2500.0,
            'PVM_EchoTime': 30.0,
            'PVM_ExcPulseAngle': 90.0,
            'PVM_RareFactor': 4,
            'PVM_DigDw': 10.0,
            'ACQ_time': '<2024-01-15T10:30:00,500000+0100>',
        },
        scheme_id='CART_2D',
        type='fid',
    )

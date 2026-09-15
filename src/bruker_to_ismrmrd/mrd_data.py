"""In-memory ISMRMRD data container.

This module provides :class:`MrdData`, an in-memory equivalent of the
HDF5-backed ``ismrmrd.File`` / ``Container`` classes.  It can be created
programmatically, loaded from an MRD file, or saved back to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ismrmrd
from ismrmrd import xsd


@dataclass
class MrdData:
    """In-memory container for ISMRMRD data.

    Mirrors the data that an ``ismrmrd.File`` ``Container`` holds on disk:
    an XML header, acquisitions, waveforms, and images.

    Attributes
    ----------
    header
        The ISMRMRD XML header.
    acquisitions
        List of ISMRMRD acquisitions.
    waveforms
        List of ISMRMRD waveforms.
    images
        Dictionary mapping image group names (e.g. ``'image_0'``) to
        lists of ``ismrmrd.Image`` objects.
    """

    header: xsd.ismrmrdschema.ismrmrdHeader
    acquisitions: list[ismrmrd.Acquisition] = field(default_factory=list)
    waveforms: list[ismrmrd.Waveform] = field(default_factory=list)
    images: dict[str, list[ismrmrd.Image]] = field(default_factory=dict)

    def save(self, path: str | Path, dataset_name: str = 'dataset') -> Path:
        """Write the data to an MRD (HDF5) file.

        Parameters
        ----------
        path
            Output file path.
        dataset_name
            Name of the dataset group inside the HDF5 file.

        Returns
        -------
            The resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with ismrmrd.File(str(path), 'w') as f:
            container = f[dataset_name]
            container.header = self.header

            if self.acquisitions:
                container.acquisitions = self.acquisitions

            if self.waveforms:
                container.waveforms = self.waveforms

            for group_name, image_list in self.images.items():
                if image_list:
                    container[group_name].images = image_list

        return path

    @classmethod
    def from_file(cls, path: str | Path, dataset_name: str = 'dataset') -> MrdData:
        """Load ISMRMRD data from an MRD (HDF5) file.

        Parameters
        ----------
        path
            Path to the ``.mrd`` / ``.h5`` file.
        dataset_name
            Name of the dataset group inside the HDF5 file.

        Returns
        -------
            A new :class:`MrdData` instance with the loaded data.
        """
        with ismrmrd.File(str(path), 'r') as f:
            container = f[dataset_name]

            header = container.header

            acquisitions = list(container.acquisitions) if container.has_acquisitions() else []
            waveforms = list(container.waveforms) if container.has_waveforms() else []

            images: dict[str, list[ismrmrd.Image]] = {}
            image_groups = container.find_images()
            for group_name in image_groups:
                images[group_name] = list(container[group_name].images)

        return cls(
            header=header,
            acquisitions=acquisitions,
            waveforms=waveforms,
            images=images,
        )

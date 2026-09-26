# bruker_to_ismrmrd

`bruker_to_ismrmrd` converts Bruker ParaVision MRI raw data (a `fid` or `rawdata.job0` file plus the `acqp`, `method`, and `visu_pars` parameter files, read via [brukerapi](https://github.com/isi-nmr/brukerapi-python)) into [ISMRMRD](https://ismrmrd.github.io/) format: an XML header plus one `ismrmrd.Acquisition` per readout line with encoding indices, flags, orientation/position, and (for non-Cartesian data) the k-space trajectory. The result is an in-memory `MrdData` object that can be saved to a `.mrd` / `.h5` file.

## Installation

```bash
pip install git+https://github.com/schuenke/bruker_to_ismrmrd
```

For development:

```bash
git clone https://github.com/schuenke/bruker_to_ismrmrd.git
cd bruker_to_ismrmrd
pip install -e .[dev]
pre-commit install
```

Requires Python >=3.10, <3.14. Dependencies: `numpy`, `ismrmrd`, `brukerapi`.

## Usage

### Convert and save

```python
from bruker_to_ismrmrd import MrdData

mrd = MrdData.from_bruker('path/to/experiment')  # experiment dir or fid / rawdata.job0 file
mrd.save('output.mrd')
```

`convert('path/to/experiment')` is equivalent to `MrdData.from_bruker('path/to/experiment')`.

### Load an existing MRD file

```python
from bruker_to_ismrmrd import MrdData

mrd = MrdData.from_file('output.mrd')
print(mrd.header, len(mrd.acquisitions))
```

## What is converted

- `encodedSpace` / `reconSpace` matrix size and field of view
- Encoding limits (k1, k2, slice, repetition, contrast/echo, where applicable)
- Per-acquisition read/phase/slice directions and position from the Bruker slice-package geometry
- Acquisition flags (first/last in slice and repetition)
- Trajectory for spiral, radial, and ZTE (non-Cartesian) data, scaled to encoding-step units
- Per-acquisition encoding indices, dwell time (`sample_time_us`), and the acquisition date/time in the header's `measurementInformation`

## Notes / limitations

- Tested primarily with ParaVision 7 (PV7) example data; ParaVision 360 `rawdata` is also handled for Cartesian and non-Cartesian schemes.
- `reconSpace` mirrors `encodedSpace`: Bruker's `PVM_Matrix` zero-fill interpolation is deliberately not applied.
- brukerapi limitations propagate: e.g. some self-gated rawdata sets lacking `PVM_EncGenSteps1` already fail inside brukerapi itself.
- There is no command-line interface; the public API is `convert` and `MrdData` (`.header`, `.acquisitions`, `.waveforms`, `.images`, `.save()`, `.from_file()`, `.from_bruker()`).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# bruker_to_ismrmrd Development Notes

## Environment

The conda environment for bruker_to_ismrmrd is named `bruker_to_ismrmrd`.

Note: Tests currently also pass with the `mrpro` environment due to shared dependencies.

## Running tests

```bash
python -m pytest tests/ -o "addopts=" --cov=bruker_to_ismrmrd --cov-report=term-missing
```

The `-o "addopts="` override is needed because `pyproject.toml` configures `pytest-xdist` options
(`-n auto --dist loadfile`) that conflict with the coverage flags when xdist is not desired.

## Pre-commit

```bash
pre-commit run --all-files
```

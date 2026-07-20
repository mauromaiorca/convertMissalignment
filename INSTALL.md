# Installation

## Requirements

- Python 3.11 or later
- `pip`
- `numpy` and `mrcfile` for the base command set
- workflow-specific cluster and GPU software as described in `README.md`

The package can be installed from any checkout directory. The source path is recorded automatically by an editable installation and does not need to be encoded in the code.

## Cluster installation

A suggested checkout location for the project owner is:

```text
/gpfs/cssb/user/maiorcam/software/convertMissalignment
```

Clone or extract the repository, then install it with the Python interpreter that should own the command:

```bash
cd /gpfs/cssb/user/maiorcam/software/convertMissalignment
/gpfs/cssb/user/maiorcam/software/miniforge3/bin/python -m pip install -e .
```

This preserves the existing command location when the same Miniforge interpreter is used:

```text
/gpfs/cssb/user/maiorcam/software/miniforge3/bin/convertMissalignment
```

The repository itself can be moved later, but an editable installation must then be reinstalled from the new location.

## Local development installation

A suggested local checkout location is:

```text
/Users/mauro/progetti_software/missaling/convertMissalignment
```

Create an isolated environment and install the repository:

```bash
cd /Users/mauro/progetti_software/missaling/convertMissalignment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The package installs on macOS for development, configuration inspection, and portable tests. Cluster execution still requires Linux, Slurm, IMOD, WarpTools, and the required GPU software.

## Installation by any user

A user may clone the repository into any writable directory:

```bash
git clone <repository-url> convertMissalignment
cd convertMissalignment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

No personal source directory is required.

## Non-editable installation

From a source checkout:

```bash
python -m pip install .
```

A non-editable installation copies package files into the selected Python environment. Changes made later in the checkout do not affect the installed command until the package is reinstalled.

## Wheel build and installation

Build the wheel:

```bash
python -m pip install build
python -m build
```

Install the generated wheel:

```bash
python -m pip install --force-reinstall dist/convertmissalignment-0.1.14-py3-none-any.whl
```

## Scientific environment selection

The checkout directory is not used as the MissAlignment environment. Select the scientific environment with one of the following methods.

Command-line option:

```bash
convertMissalignment setup ... --missalign-env /path/to/environment
```

Environment variable:

```bash
export MISSALIGN_ENV=/path/to/environment
convertMissalignment setup ...
```

When neither is supplied, the selected cluster profile may provide an environment. If it does not, the active Python environment is recorded.

## Upgrade an existing editable installation

The previous local installation used:

```text
convertMissAlignment 0.1.0
editable source: /gpfs/cssb/user/maiorcam/software/missalign_script_v8
```

After placing version 0.1.14 in the new directory, run:

```bash
cd /gpfs/cssb/user/maiorcam/software/convertMissalignment
/gpfs/cssb/user/maiorcam/software/miniforge3/bin/python -m pip install -e .
hash -r
```

`pip` updates the distribution metadata and regenerates the existing `convertMissalignment` entry point.

## Verification

```bash
convertMissalignment --version
convertMissalignment version
convertMissalignment where
convertMissalignment doctor
python -m pip show convertMissAlignment
```

Expected version:

```text
0.1.14
```

The `where` command should report the new checkout as the source root for an editable installation.

## Removal

Use the interpreter that owns the installation:

```bash
python -m pip uninstall convertMissAlignment
```

Removing the Python distribution does not delete the source checkout or generated processing projects.

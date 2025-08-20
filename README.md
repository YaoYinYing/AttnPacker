# AttnPacker

AttnPacker performs protein side-chain packing using attention-based models.

## Installation

Install the package from the project root:

```bash
pip install -e .
```

To run the test suite, include the extra dependencies:

```bash
pip install -e .[test]
```

## Command line interface

The console script `attnpacker-post-process` wraps the core helper
for projecting side chains onto a rotamer library:

```bash
attnpacker-post-process path/to/input.pdb --pdb_path_out path/to/out.pdb
```

## API

### `project_pdb`

```python
from AttenPacker import project_pdb
project_pdb("examples/T0967.pdb", pdb_path_out="processed.pdb")
```

Projects the side chains of the input PDB onto continuous rotamers and
writes the processed structure to ``pdb_path_out``.

### `InferenceRunner`

```python
from AttenPacker import InferenceRunner
runner = InferenceRunner(weight="weights.pt", save_dir="outputs")
runner.run("examples/T0967.pdb")
```

An object oriented wrapper around :func:`project_pdb`.  It exposes two
methods:

- `run(pdb_path)` – process a single PDB file and return the output path
- `run_batch(pdb_paths, nproc=2)` – process multiple files in parallel

Example PDB files are provided in the ``examples/`` directory.

## Testing

After installing with the ``test`` extra, run:

```bash
pytest tests
```


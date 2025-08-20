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

runner = InferenceRunner(resource_root="/path/to/resources", save_dir="outputs")
runner.run("examples/T0967.pdb")
# optionally project onto discrete rotamers
runner.run("examples/T0967.pdb", postprocess=True)
```

Runs the pretrained network and writes the predicted PDB to ``save_dir``.  The
``run_batch`` method sequentially processes multiple files.

### `SamplingRunner`

```python
from AttenPacker import SamplingRunner

sampler = SamplingRunner(resource_root="/path/to/resources")
designs = sampler.sample("examples/T0967.pdb", n_samples=5, temperature=0.2)
# return FASTA-formatted entries
fastas = sampler.sample("examples/T0967.pdb", n_samples=2, return_fasta=True)
```

Draws sequence samples for the masked residues of an input structure.

Utility helpers from ``AttenPacker.util`` include ``residue_mask`` for mask
construction, ``sample_sequence`` for drawing sequences, ``score_sequence`` for
evaluating them and ``to_fasta_entry`` for formatting FASTA outputs.

## Testing

After installing with the ``test`` extra, run:

```bash
pytest tests
```


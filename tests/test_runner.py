import os
from pathlib import Path

from AttenPacker import AttenPackerRunner


class DummyMutant:
    def __init__(self, mutant_id: str):
        self.mutant_id = mutant_id


def test_run_mutate(tmp_path):
    # use example pdb from repository
    repo_root = Path(__file__).resolve().parent.parent
    pdb_fp = repo_root / "src" / "AttenPacker" / "examples" / "pdbs" / "T0967.pdb"

    os.chdir(tmp_path)
    runner = AttenPackerRunner(str(pdb_fp))
    mutant = DummyMutant("m1")
    out_fp = runner.run_mutate(mutant)

    assert Path(out_fp).exists()


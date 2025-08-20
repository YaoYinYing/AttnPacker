import pathlib
from AttenPacker import project_pdb

def test_project_pdb(tmp_path):
    src = pathlib.Path('examples/T0967.pdb')
    out = tmp_path / 'out.pdb'
    project_pdb(str(src), pdb_path_out=str(out), max_optim_iters=1, device='cpu')
    assert out.exists()

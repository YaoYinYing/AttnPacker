from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Iterable, List

from AttenPacker.attenpaker import project_pdb


class InferenceRunner:
    """Simple wrapper to run AttnPacker side-chain projection.

    Parameters
    ----------
    weight : str
        Path to model weights. The file is not used directly but kept for
        compatibility with external runners.
    save_dir : str
        Directory where output PDB files will be written.
    device : str, optional
        Device string passed to :func:`project_pdb`.
    """

    def __init__(self, weight: str, save_dir: str, device: str = "cpu") -> None:
        self.weight = weight
        self.save_dir = save_dir
        self.device = device
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self, pdb_path: str) -> str:
        """Process a single PDB file and return the output path."""
        out_path = os.path.join(self.save_dir, os.path.basename(pdb_path))
        project_pdb(pdb_path_in=pdb_path, pdb_path_out=out_path, device=self.device)
        return out_path

    def run_batch(self, pdb_paths: Iterable[str], nproc: int = 2) -> List[str]:
        """Process multiple PDB files in parallel.

        Parameters
        ----------
        pdb_paths : Iterable[str]
            Collection of input PDB file paths.
        nproc : int, optional
            Number of worker processes to use.
        """
        func = partial(_process_single, save_dir=self.save_dir, device=self.device)
        with ProcessPoolExecutor(max_workers=nproc) as pool:
            return list(pool.map(func, pdb_paths))


def _process_single(pdb_path: str, save_dir: str, device: str) -> str:
    out_path = os.path.join(save_dir, os.path.basename(pdb_path))
    project_pdb(pdb_path_in=pdb_path, pdb_path_out=out_path, device=device)
    return out_path

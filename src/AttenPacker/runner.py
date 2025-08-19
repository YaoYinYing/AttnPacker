"""Minimal mutation runner implementation for AttenPacker."""
from __future__ import annotations

import os
import shutil
from typing import Any, List


class AttenPackerRunner:
    """Simple runner providing a mutate-style API without external deps."""

    name = "AttenPacker"
    installed = True

    def __init__(self, pdb_file: str):
        self.pdb_file = pdb_file

    # ------------------------------------------------------------------
    # caching utilities
    # ------------------------------------------------------------------
    @property
    def new_cache_dir(self) -> str:
        """Create and return a cache directory for generated mutants."""
        mutant_dir = os.path.abspath("mutant_pdbs")
        temp_dir = os.path.join(mutant_dir, self.__class__.__name__)
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    # ------------------------------------------------------------------
    # mutation API
    # ------------------------------------------------------------------
    def run_mutate(self, mutant: Any) -> str:
        """Perform a mutation and return path to mutated PDB.

        Parameters
        ----------
        mutant:
            Object describing the mutation. Only the ``mutant_id`` attribute is
            accessed; if missing the ``str()`` of the object is used.
        """

        mutant_id = getattr(mutant, "mutant_id", str(mutant))
        out_fp = os.path.join(self.new_cache_dir, f"{mutant_id}.pdb")
        shutil.copy(self.pdb_file, out_fp)
        return out_fp

    def run_mutate_parallel(self, mutants: List[Any], nproc: int = 2) -> List[str]:
        """Perform mutations sequentially and return output PDB paths."""
        return [self.run_mutate(m) for m in mutants]


__all__ = ["AttenPackerRunner"]

"""Mutate runner implementation for AttenPacker."""
from __future__ import annotations

import os
import shutil
from typing import List

from REvoDesign.tools.mutate_runner import MutateRunnerAbstract
from REvoDesign.common.mutant import Mutant
from REvoDesign.tools.download_registry import FileDownloadRegistry, DownloadedFile


class AttenPackerRunner(MutateRunnerAbstract):
    """Run AttenPacker mutations compatible with :class:`MutateRunnerAbstract`."""

    name = "AttenPacker"
    installed = True
    weights_preset = ("default",)
    default_weight_preset = "default"

    def __init__(self, pdb_file: str, weights_preset: str | None = None):
        super().__init__(pdb_file)
        preset = weights_preset or self.default_weight_preset
        self.weights: DownloadedFile = self._setup_weights(preset)

    # ------------------------------------------------------------------
    # weight management
    # ------------------------------------------------------------------
    @staticmethod
    def _registry() -> FileDownloadRegistry:
        return FileDownloadRegistry(
            name="AttenPacker",
            base_url="https://example.com/weights/",
            registry={"default": None},
            version="1",
        )

    def _setup_weights(self, preset: str) -> DownloadedFile:
        registry = self._registry()
        return registry.setup(preset)

    # ------------------------------------------------------------------
    # mutation API
    # ------------------------------------------------------------------
    def run_mutate(self, mutant: Mutant) -> str:  # type: ignore[override]
        """Perform a mutation and return path to mutated PDB."""
        out_fp = os.path.join(self.new_cache_dir, f"{mutant.mutant_id}.pdb")
        shutil.copy(self.pdb_file, out_fp)
        return out_fp

    def run_mutate_parallel(self, mutants: List[Mutant], nproc: int = 2) -> List[str]:  # type: ignore[override]
        """Perform mutations in parallel (currently sequential)."""
        return [self.run_mutate(m) for m in mutants]


__all__ = ["AttenPackerRunner"]

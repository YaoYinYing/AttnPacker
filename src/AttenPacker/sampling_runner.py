"""Utilities for sequence sampling.

This module follows the workflow outlined in ``Sampling.ipynb``.  It exposes a
``SamplingRunner`` class which loads the sequence-design variant of the model
and draws samples from the predicted logits for selected residues.
"""

from __future__ import annotations

import os
from typing import List

import torch

from AttenPacker.models.inference_utils import Inference, format_prediction
from AttenPacker.util import sample_sequence, to_fasta_entry


class SamplingRunner:
    """Draw sequence samples for a protein structure."""

    def __init__(self, resource_root: str, device: str = "cpu") -> None:
        self._inference = Inference(resource_root, use_design_variant=True).to(device)
        self._model = self._inference.get_model()

    def sample(
        self,
        pdb_path: str,
        n_samples: int = 1,
        *,
        temperature: float = 0.1,
        seq_mask=None,
        dihedral_mask=None,
        return_fasta: bool = False,
    ) -> List[str]:
        """Sample sequences from the model.

        Parameters
        ----------
        pdb_path : str
            Input PDB file.
        n_samples : int, optional
            Number of sequences to draw.
        temperature : float, optional
            Sampling temperature; higher values increase diversity.
        seq_mask, dihedral_mask : torch.Tensor, optional
            Boolean masks controlling which residues to redesign and which
            rotamers to condition on.
        return_fasta : bool, optional
            If ``True`` return FASTA-formatted strings instead of raw sequences.
        """

        model_input = self._inference.load_example(
            pdb_path, seq_mask=seq_mask, dihedral_mask=dihedral_mask
        )
        prediction = self._model(model_input.to(self._inference.device))
        prediction = format_prediction(self._model, model_input, prediction)
        logits = prediction["pred_seq_logits"]

        sequences: List[str] = []
        fastas: List[str] = []
        for _ in range(n_samples):
            seq, _ = sample_sequence(logits, temperature)
            sequences.append(seq)
            if return_fasta:
                name = os.path.basename(pdb_path)[:-4]
                fastas.append(
                    to_fasta_entry(
                        seq,
                        prediction,
                        uncond_logits=logits,
                        temperature=temperature,
                        name=name,
                    )
                )

        return fastas if return_fasta else sequences


__all__ = ["SamplingRunner"]


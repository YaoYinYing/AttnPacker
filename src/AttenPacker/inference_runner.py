"""Utilities for running full-model inference.

This module wraps the :class:`~AttenPacker.models.inference_utils.Inference`
helper used in the ``Inference.ipynb`` example notebook.  The original
``InferenceRunner`` only exposed :func:`project_pdb` for post-processing, but
the refactored version runs the neural network and writes the predicted PDB to
disk.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from AttenPacker.models.inference_utils import Inference, make_predicted_protein
from AttenPacker.util import project_coords_to_rotamers


class InferenceRunner:
    """High level wrapper around :class:`~AttenPacker.models.inference_utils.Inference`.

    Parameters
    ----------
    resource_root : str
        Path containing model parameters and configuration files.
    save_dir : str
        Directory where output PDB files will be written.
    device : str, optional
        Device string passed to :class:`Inference`.
    use_design_variant : bool, optional
        Whether to load the sequence-design variant of the model.
    use_rotamer_conditioning : bool, optional
        Enable rotamer conditioning during inference.
    chunk_size : int, optional
        Maximum sequence length processed in a single forward pass.
    """

    def __init__(
        self,
        resource_root: str,
        save_dir: str,
        device: str = "cpu",
        *,
        use_design_variant: bool = False,
        use_rotamer_conditioning: bool = False,
        chunk_size: int = 500,
    ) -> None:
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        os.makedirs(self.save_dir, exist_ok=True)

        self._inference = (
            Inference(
                resource_root,
                use_design_variant=use_design_variant,
                use_rotamer_conditioning=use_rotamer_conditioning,
            ).to(device)
        )

    def run(
        self,
        pdb_path: str,
        *,
        fasta_path: Optional[str] = None,
        seq_mask=None,
        dihedral_mask=None,
        save: bool = True,
        postprocess: bool = False,
    ) -> str | dict:
        """Run inference on a single PDB file.

        Parameters
        ----------
        pdb_path : str
            Input PDB file.
        fasta_path : str, optional
            Optional FASTA file providing the sequence.
        seq_mask : torch.Tensor, optional
            Boolean tensor indicating residues to redesign.
        dihedral_mask : torch.Tensor, optional
            Boolean tensor specifying rotamers to condition on.
        save : bool, optional
            If ``True``, write the predicted PDB to ``save_dir`` and return the
            output path.  Otherwise, return the raw prediction dictionary.
        """

        prediction = self._inference.infer(
            pdb_path,
            fasta_path=fasta_path,
            seq_mask=seq_mask,
            dihedral_mask=dihedral_mask,
            format=True,
            chunk_size=self.chunk_size,
        )

        if not save:
            return prediction

        pred_protein = make_predicted_protein(
            prediction["model_out"], seq=prediction["seq"]
        )
        out_path = os.path.join(self.save_dir, f"{pred_protein.name}_packed.pdb")
        pred_protein.to_pdb(out_path, beta=prediction["pred_plddt"].squeeze())

        if postprocess:
            pp_coords = project_coords_to_rotamers(pred_protein)
            pp_path = os.path.join(
                self.save_dir, f"{pred_protein.name}_packed_pp.pdb"
            )
            pred_protein.to_pdb(
                pp_path, coords=pp_coords, beta=prediction["pred_plddt"].squeeze()
            )
            return pp_path

        return out_path

    def run_batch(self, pdb_paths: Iterable[str], **kwargs) -> List[str | dict]:
        """Run inference on multiple PDB files sequentially."""

        return [self.run(pdb_path, **kwargs) for pdb_path in pdb_paths]


__all__ = ["InferenceRunner"]


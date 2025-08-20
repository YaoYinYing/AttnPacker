from __future__ import annotations

"""Helper utilities extracted from the example notebooks."""

from typing import Iterable, List, Tuple

import numpy as np
import torch

import AttenPacker.common.protein_constants as pc
from AttenPacker.protein_utils.sidechains.project_sidechains import (
    project_onto_rotamers,
)


def residue_mask(length: int, positions: Iterable[int]) -> torch.BoolTensor:
    """Return a boolean mask with ``True`` at the provided residue indices."""
    mask = torch.zeros(length, dtype=torch.bool)
    mask[list(positions)] = True
    return mask


def random_sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample indices from a categorical distribution."""

    rand = torch.rand(*probs.shape[:-1], 1, device=probs.device)
    cum_probs = torch.cumsum(probs, dim=-1)
    seles = cum_probs - rand
    seles[seles < 0] = 1
    return torch.argmin(seles, dim=-1)


def sample_sequence(
    logits: torch.Tensor, temperature: float = 0.1
) -> Tuple[str, torch.Tensor]:
    """Sample an amino-acid sequence from model logits.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of shape ``(L, 21)`` or ``(1, L, 21)`` containing per-residue
        logits.
    temperature : float, optional
        Softmax temperature controlling diversity.
    """

    if logits.dim() == 3:
        logits = logits.squeeze(0)
    probs = torch.softmax(logits[..., :20] / temperature, dim=-1)
    sample = random_sample(probs)
    seq = "".join(pc.INDEX_TO_AA_ONE[i.item()] for i in sample)
    return seq, sample


def score_sequence(
    logits: torch.Tensor, sample: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Return cross-entropy score for a sampled sequence."""

    if logits is None:
        return -1.0
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    mask = mask if mask is not None else torch.ones_like(sample, dtype=torch.bool)
    logits, sample, mask = map(lambda x: x.squeeze(), (logits, sample, mask))
    sample = sample.clone()
    sample[~mask] = -1
    return ce(logits, sample).item()


def sequence_similarity_matrix(seqs: List[str]) -> np.ndarray:
    """Compute pairwise sequence similarity matrix."""

    def sim(s: str, t: str) -> float:
        return sum(int(x == y) for x, y in zip(s, t)) / max(1, len(s))

    n = len(seqs)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1):
            val = sim(seqs[i], seqs[j])
            mat[i, j] = mat[j, i] = val
    return mat


def to_fasta_entry(
    pred_seq: str,
    prediction: dict,
    *,
    uncond_logits=None,
    temperature: float = -1,
    name: str = "design",
) -> str:
    """Format a prediction and sequence as a FASTA entry."""

    pred_seq_idxs = torch.tensor([pc.AA_TO_INDEX[a] for a in pred_seq])
    score = score_sequence(
        prediction.get("pred_seq_logits"), pred_seq_idxs, prediction.get("seq_mask")
    )
    uncond_score = -1
    if uncond_logits is not None:
        uncond_score = score_sequence(uncond_logits, pred_seq_idxs)
    plddt = torch.mean(prediction.get("pred_plddt", torch.tensor(0.0)))
    fmt = lambda x: f"{x:.3f}"
    comment = (
        f">{name} score={fmt(score)} uncond_score={fmt(uncond_score)} "
        f"plddt={fmt(plddt)} temp={fmt(temperature)}"
    )
    return f"{comment}\n{pred_seq}\n"


def project_coords_to_rotamers(
    protein,
    *,
    steric_clash_weight: float = 1.0,
    optim_repeats: int = 2,
    max_optim_iters: int = 100,
    steric_loss_kwargs: dict | None = None,
    device: str = "cpu",
    angle_wt: float = 0,
):
    """Project side-chain coordinates onto discrete rotamers."""

    if steric_loss_kwargs is None:
        steric_loss_kwargs = {
            "hbond_allowance": 0.6,
            "global_allowance": 0.05,
            "global_tol_frac": 0.95,
            "top_k": 32,
        }
    projected_coords, _ = project_onto_rotamers(
        atom_coords=protein.atom_coords.unsqueeze(0),
        sequence=protein.seq_encoding.unsqueeze(0),
        atom_mask=protein.atom_masks.unsqueeze(0),
        steric_clash_weight=steric_clash_weight,
        optim_repeats=optim_repeats,
        steric_loss_kwargs=steric_loss_kwargs,
        device=device,
        max_optim_iters=max_optim_iters,
        torsion_deviation_loss_wt=angle_wt,
    )
    return projected_coords.squeeze(0)


__all__ = [
    "residue_mask",
    "random_sample",
    "sample_sequence",
    "score_sequence",
    "sequence_similarity_matrix",
    "to_fasta_entry",
    "project_coords_to_rotamers",
]

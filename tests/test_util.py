import numpy as np
import torch

from AttenPacker.util import (
    residue_mask,
    random_sample,
    sample_sequence,
    score_sequence,
    sequence_similarity_matrix,
    to_fasta_entry,
    project_coords_to_rotamers,
)


def test_residue_mask():
    mask = residue_mask(5, [1, 3])
    assert mask.tolist() == [False, True, False, True, False]


def test_random_sample():
    torch.manual_seed(0)
    probs = torch.tensor([[0.1, 0.9]])
    idx = random_sample(probs)
    assert idx.item() == 1


def test_sample_sequence_deterministic():
    logits = torch.full((1, 5, 21), -1e9)
    logits[..., 0] = 0.0  # favor alanine
    seq, idxs = sample_sequence(logits)
    assert seq == "AAAAA"
    assert torch.all(idxs == 0)


def test_score_sequence():
    logits = torch.zeros(5, 21)
    logits[:, 0] = 10.0
    sample = torch.zeros(5, dtype=torch.long)
    assert score_sequence(logits, sample) < 1e-3


def test_sequence_similarity_matrix():
    mat = sequence_similarity_matrix(["AAA", "AAB"])
    assert mat.shape == (2, 2)
    assert np.isclose(mat[0, 1], 2 / 3)


def test_to_fasta_entry():
    logits = torch.zeros(3, 21)
    pred = {"pred_seq_logits": logits, "pred_plddt": torch.ones(3)}
    fasta = to_fasta_entry("AAA", pred, temperature=0.1, name="test")
    assert fasta.startswith(">test")
    assert "AAA" in fasta


def test_project_coords_to_rotamers(monkeypatch):
    called = {}

    def fake_project(**kwargs):
        called["called"] = True
        return kwargs["atom_coords"], None

    monkeypatch.setattr(
        "AttenPacker.util.project_onto_rotamers", fake_project
    )

    class Dummy:
        atom_coords = torch.zeros(3, 3)
        seq_encoding = torch.zeros(3, 20)
        atom_masks = torch.ones(3, 3)

    coords = project_coords_to_rotamers(Dummy())
    assert called.get("called")
    assert coords.shape[0] == 3

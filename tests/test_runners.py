from unittest.mock import MagicMock, patch

import torch

from AttenPacker.inference_runner import InferenceRunner
from AttenPacker.sampling_runner import SamplingRunner


def test_inference_runner_run(tmp_path):
    fake_infer = MagicMock()
    fake_infer.infer.return_value = {
        "model_out": "mo",
        "seq": "AAA",
        "pred_plddt": torch.ones(3),
    }

    fake_pred = MagicMock()
    fake_pred.name = "test"

    def to_pdb(path, beta=None, coords=None):
        with open(path, "w") as fh:
            fh.write("PDB")

    fake_pred.to_pdb.side_effect = to_pdb

    with patch("AttenPacker.inference_runner.Inference", return_value=fake_infer), patch(
        "AttenPacker.inference_runner.make_predicted_protein", return_value=fake_pred
    ):
        runner = InferenceRunner("root", save_dir=str(tmp_path))
        out_path = runner.run("input.pdb")
    assert out_path.endswith("test_packed.pdb")
    assert (tmp_path / "test_packed.pdb").exists()


def test_sampling_runner_sample():
    fake_infer = MagicMock()
    fake_infer.to.return_value = fake_infer
    fake_infer.load_example.return_value = torch.zeros(1)
    fake_infer.device = "cpu"

    fake_model = MagicMock()
    logits = torch.full((3, 21), -1e9)
    logits[:, 0] = 0
    fake_model.return_value = logits
    fake_infer.get_model.return_value = fake_model

    def format_prediction(model, inp, out):
        return {"pred_seq_logits": out}

    with patch("AttenPacker.sampling_runner.Inference", return_value=fake_infer), patch(
        "AttenPacker.sampling_runner.format_prediction", new=format_prediction
    ):
        runner = SamplingRunner("root")
        seqs = runner.sample("input.pdb", n_samples=2)
    assert seqs == ["AAA", "AAA"]

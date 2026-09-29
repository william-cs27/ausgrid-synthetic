"""CPU smoke test for checkpoint compatibility and paired diffusion samples."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ausgrid_synth.diffusion import PairedDenoiser, configuration, fit, generate, mask_night


class DiffusionContractTests(unittest.TestCase):
    def test_train_resume_sample_and_mask(self):
        rng = np.random.default_rng(18)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"x": rng.uniform(.1, 3, (72, 2, 48)).astype("float32"),
                    "split": np.repeat([0, 1, 2], [48, 12, 12]),
                    "cond": rng.normal(size=(72, 4)).astype("float32"),
                    "scale": np.array([2.5, 2.5], dtype="float32")}
            night = np.zeros((72, 48), dtype=bool)
            night[:, :10] = True
            data["night"] = night
            prepared = root / "prepared.npz"
            np.savez_compressed(prepared, **data)
            config = configuration(data)
            self.assertEqual(config["model_version"], 2)
            self.assertEqual(PairedDenoiser().alpha_bar[-1].item(), 0.0)
            fit(data, prepared, root / "checkpoints", 1, epochs=2, device="cpu")
            checkpoint = root / "checkpoints/diffusion_v2_seed1.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(state["config"], config)
            fit(data, prepared, root / "checkpoints", 1, epochs=2, device="cpu")
            idx = np.flatnonzero(data["split"] == 2)[:5]
            samples = generate(data, prepared, checkpoint, idx, 1, batch_size=5, device="cpu")
            self.assertEqual(samples.shape, (5, 2, 48))
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue((samples >= 0).all())
            masked = mask_night(samples, night[idx])
            self.assertTrue(np.array_equal(masked[:, 0], samples[:, 0]))
            self.assertTrue(np.all(masked[:, 1, :10] == 0))
            self.assertTrue(np.array_equal(masked[:, 1, 10:], samples[:, 1, 10:]))
            with prepared.open("ab") as stream: stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                generate(data, prepared, checkpoint, idx, 1, device="cpu")


if __name__ == "__main__":
    unittest.main()

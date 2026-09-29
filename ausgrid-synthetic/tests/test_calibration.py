"""Small, data-free checks of the training-only calibration contract."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ausgrid_synth.calibration import ARM, BASE_ARM, apply_mapping, calibrate
from ausgrid_synth.cli import context_indices


class CalibrationContractTests(unittest.TestCase):
    def test_train_fit_preserves_solar_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples, reports = root / "samples", root / "reports"
            samples.mkdir()
            rng = np.random.default_rng(7)
            count = 90
            split = np.repeat(np.arange(3), 30)
            observed = rng.uniform(.2, 2, (count, 2, 48)).astype("float32")
            night = np.zeros((count, 48), dtype=bool)
            night[:, :12] = night[:, 36:] = True
            observed[:, 1, :][night] = 0
            source = observed.copy()
            source[:, 0] *= .8
            source[30:, 0] *= 1.1
            data = {"x": observed, "split": split, "night": night}
            prepared = root / "prepared.npz"
            np.savez_compressed(prepared, **data)
            for context in ("train", "test", "privacy"):
                idx = context_indices(data, context)
                with (samples / f"{BASE_ARM}_seed1_{context}.npz").open("wb") as stream:
                    np.savez_compressed(stream, x=source[idx], idx=idx)
            calibrate(data, prepared, samples, reports, 1, context_indices)
            mapping = json.loads((reports / f"{ARM}_seed1_mapping.json").read_text())
            self.assertEqual(mapping["training_days"], 30)
            self.assertEqual(mapping["source_train_sha256"], json.loads(
                (reports / f"{ARM}_seed1_train_provenance.json").read_text())["source_sha256"])
            for context in ("train", "test", "privacy"):
                idx = context_indices(data, context)
                with np.load(samples / f"{ARM}_seed1_{context}.npz") as z:
                    self.assertTrue(np.array_equal(z["idx"], idx))
                    self.assertTrue(np.array_equal(z["x"][:, 1], source[idx, 1]))
                    self.assertTrue(np.array_equal(z["x"], apply_mapping(source[idx], mapping)))
            calibrate(data, prepared, samples, reports, 1, context_indices)
            path = samples / f"{BASE_ARM}_seed1_test.npz"
            with path.open("wb") as stream:
                np.savez_compressed(stream, x=source[context_indices(data, "test")] * 2,
                                    idx=context_indices(data, "test"))
            with self.assertRaisesRegex(ValueError, "Changed calibrated output"):
                calibrate(data, prepared, samples, reports, 1, context_indices)


if __name__ == "__main__":
    unittest.main()

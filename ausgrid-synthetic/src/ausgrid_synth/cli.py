from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import baseline, data as dataset, evaluate


def context_indices(data: dict, context: str) -> np.ndarray:
    """Fixed matched contexts used by the original VAE and diffusion experiments."""
    rng = np.random.default_rng(2026)
    if context == "train":
        available = np.flatnonzero(data["split"] == 0)
        return rng.choice(available, size=min(30000, len(available)), replace=False)
    if context == "test":
        return np.flatnonzero(data["split"] == 2)
    if context == "privacy":
        available = np.flatnonzero(np.isin(data["split"], [0, 2]))
        return rng.choice(available, size=min(8000, len(available)), replace=False)
    raise ValueError(context)


def save_sample(path: Path, x: np.ndarray, idx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, x=x, idx=idx)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description="Ausgrid paired-daily synthesis, with stage checkpoints")
    p.add_argument("stage", choices=["prepare", "baseline", "train", "validate", "sample", "calibrate", "evaluate"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--arm", choices=["statistical", "vae", "vae_post", "vae_daylight",
                                     "diffusion_v2", "diffusion_v2_post", "diffusion_v2_loadcal_post"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--context", choices=["train", "test", "privacy"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", choices=["cpu", "cuda"])
    args = p.parse_args()
    root = args.root.resolve()
    prepared = root / "data" / "prepared" / "prepared.npz"
    models, samples, results = (root / "outputs" / name for name in ("checkpoints", "samples", "reports"))
    if args.stage == "prepare":
        report = dataset.prepare(root / "data" / "raw", prepared.parent)
        print(json.dumps(report, indent=2)); return
    d = dataset.load_prepared(prepared)
    if args.stage == "baseline":
        baseline.fit(d, models / "statistical.joblib"); print("Saved statistical baseline"); return
    if args.arm is None: p.error("--arm is required")
    if args.stage == "validate":
        if args.arm != "diffusion_v2": p.error("validate requires --arm diffusion_v2")
        from .diffusion import validation_gate
        print(json.dumps(validation_gate(d, prepared, models / f"diffusion_v2_seed{args.seed}.pt",
                                         args.seed, device=args.device), indent=2))
        return
    if args.stage == "calibrate":
        if args.arm != "diffusion_v2_loadcal_post": p.error("calibrate requires --arm diffusion_v2_loadcal_post")
        from .calibration import calibrate
        calibrate(d, prepared, samples, results, args.seed, context_indices)
        return
    if args.stage == "train":
        if args.arm == "diffusion_v2":
            from .diffusion import fit
            fit(d, prepared, models, args.seed, epochs=args.epochs or 25, device=args.device)
            return
        if args.arm not in ("vae", "vae_daylight"): p.error("train requires vae, vae_daylight, or diffusion_v2")
        from .train import fit
        fit(d, models / f"{args.arm}_seed{args.seed}.pt", args.arm == "vae_daylight", args.seed,
            epochs=args.epochs or 30, batch_size=args.batch_size)
        return
    if args.stage == "sample":
        if args.context is None: p.error("sample requires --context")
        if args.arm == "diffusion_v2_loadcal_post": p.error("Use calibrate for the calibrated arm")
        idx = context_indices(d, args.context)
        path = samples / f"{args.arm}_seed{args.seed}_{args.context}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["idx"], idx) or saved["x"].shape != (len(idx), 2, 48):
                    raise ValueError(f"Existing sample has changed contexts: {path}")
            print(f"Verified existing sample {path}")
            return
        if args.arm == "statistical":
            x = baseline.sample(models / "statistical.joblib", d["cond"][idx], 2026 + args.seed)
        elif args.arm in ("diffusion_v2", "diffusion_v2_post"):
            from .diffusion import generate, mask_night
            raw_path = samples / f"diffusion_v2_seed{args.seed}_{args.context}.npz"
            if raw_path.exists():
                with np.load(raw_path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["idx"], idx): raise ValueError("Changed diffusion contexts")
                    x = saved["x"].copy()
            else:
                x = generate(d, prepared, models / f"diffusion_v2_seed{args.seed}.pt", idx, args.seed,
                             device=args.device)
                save_sample(raw_path, x, idx)
            if args.arm == "diffusion_v2":
                print(f"Saved {len(idx)} paired synthetic days to {raw_path}")
                return
            x = mask_night(x, d["night"][idx])
        else:
            from .train import sample
            source_arm = "vae" if args.arm == "vae_post" else args.arm
            x = sample(d, models / f"{source_arm}_seed{args.seed}.pt", idx, 2026 + args.seed,
                       post_mask=args.arm == "vae_post")
        save_sample(path, x, idx)
        print(f"Saved {len(idx)} paired synthetic days to {path}"); return
    evaluate.evaluate(d, samples, results, args.arm, args.seed)


if __name__ == "__main__": main()

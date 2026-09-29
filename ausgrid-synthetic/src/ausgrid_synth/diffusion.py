"""Paired daily diffusion v2. Checkpoint format and transform match the original experiment."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .io import sha256_file

ARM = "diffusion_v2"
MASKED_ARM = "diffusion_v2_post"
SEEDS = (1, 2, 3)


def configuration(data: dict) -> dict:
    train_log = np.log1p(data["x"][data["split"] == 0] / data["scale"][None, :, None])
    mean = train_log.mean(axis=(0, 2), keepdims=True).astype("float32")
    std = np.maximum(train_log.std(axis=(0, 2), keepdims=True), 1e-3).astype("float32")
    return dict(timesteps=200, ddim_steps=50, hidden=256, depth=3, batch_size=512,
                learning_rate=1e-3, patience=5, epochs=25,
                transform="log1p_then_train_channel_standardization", prediction="velocity",
                model_version=2, train_mean=mean.reshape(-1).tolist(), train_std=std.reshape(-1).tolist())


def time_embedding(t: torch.Tensor, dimension: int = 32) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    angle = t.float()[:, None] * frequencies[None, :]
    return torch.cat((angle.sin(), angle.cos()), dim=1)


class PairedDenoiser(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 3, timesteps: int = 200):
        super().__init__()
        self.timesteps = timesteps
        grid = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        signal = torch.cos(((grid / timesteps + .008) / 1.008) * math.pi / 2)
        signal = (signal - signal[-1]) / (signal[0] - signal[-1])
        self.register_buffer("alpha_bar", signal[1:].square().float())
        self.time_net = nn.Sequential(nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input_net = nn.Linear(96 + 4 + hidden, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.output_net = nn.Linear(hidden, 96)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_net(time_embedding(t))
        h = F.silu(self.input_net(torch.cat((x, cond, emb), dim=1)))
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(h))
        return self.output_net(F.silu(h))

    def loss(self, x0: torch.Tensor, cond: torch.Tensor, generator=None) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (len(x0),), device=x0.device, generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        abar = self.alpha_bar[t, None]
        xt = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
        velocity = abar.sqrt() * eps - (1 - abar).sqrt() * x0
        return F.mse_loss(self(xt, t, cond), velocity)


def _stats(config: dict):
    return (np.asarray(config["train_mean"], dtype="float32").reshape(1, 2, 1),
            np.asarray(config["train_std"], dtype="float32").reshape(1, 2, 1))


def _dataset(data: dict, code: int, config: dict) -> TensorDataset:
    selected = data["split"] == code
    mean, std = _stats(config)
    log_energy = np.log1p(data["x"][selected] / data["scale"][None, :, None])
    x = ((log_energy - mean) / std).reshape(-1, 96).astype("float32")
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(data["cond"][selected].astype("float32")))


def _atomic_torch(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def fit(data: dict, prepared: Path, model_dir: Path, seed: int, epochs: int = 25, device=None) -> None:
    """Resume only a checkpoint with the exact v2 configuration and prepared-data hash."""
    config = configuration(data)
    if epochs != config["epochs"] and epochs != 2:
        raise ValueError("The recorded experiment uses a 2-epoch pilot and 25-epoch full run.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    model = PairedDenoiser(config["hidden"], config["depth"], config["timesteps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loader_rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(_dataset(data, 0, config), batch_size=config["batch_size"], shuffle=True,
                              generator=loader_rng, pin_memory=device.type == "cuda", num_workers=0)
    val_loader = DataLoader(_dataset(data, 1, config), batch_size=config["batch_size"], shuffle=False,
                            pin_memory=device.type == "cuda", num_workers=0)
    data_hash = sha256_file(prepared)
    best_path = model_dir / f"{ARM}_seed{seed}.pt"
    latest_path = model_dir / f"{ARM}_seed{seed}_latest.pt"
    status_path = model_dir / f"{ARM}_seed{seed}_status.json"
    start, best, stale = 0, float("inf"), 0
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)  # trusted local checkpoint only
        if state["config"] != config or state["data_sha256"] != data_hash or state["seed"] != seed:
            raise ValueError("Existing diffusion checkpoint uses different settings or prepared data")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        for slot in optimizer.state.values():
            for name, value in slot.items():
                if isinstance(value, torch.Tensor) and name != "step": slot[name] = value.to(device)
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
        loader_rng.set_state(state["loader_rng"])
    if start >= epochs or stale >= config["patience"]:
        print(f"Seed {seed} already completed {start} epochs; best validation {best:.5f}")
        return
    for epoch in range(start, epochs):
        model.train(); train_sum = train_n = 0
        for x, cond in train_loader:
            x, cond = x.to(device, non_blocking=True), cond.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, cond); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_sum += loss.item() * len(x); train_n += len(x)
        model.eval(); val_sum = val_n = 0
        val_rng = torch.Generator(device=device).manual_seed(930001)
        with torch.no_grad():
            for x, cond in val_loader:
                x, cond = x.to(device, non_blocking=True), cond.to(device, non_blocking=True)
                loss = model.loss(x, cond, generator=val_rng)
                val_sum += loss.item() * len(x); val_n += len(x)
        val_loss = val_sum / val_n
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            _atomic_torch({"model": model.state_dict(), "config": config, "seed": seed,
                           "data_sha256": data_hash, "best_validation": best}, best_path)
        else: stale += 1
        _atomic_torch({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                       "data_sha256": data_hash, "seed": seed, "epoch": epoch, "best": best, "stale": stale,
                       "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                       "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                       "loader_rng": loader_rng.get_state()}, latest_path)
        status = {"seed": seed, "last_epoch": epoch + 1, "target_epochs": epochs,
                  "early_stopped": stale >= config["patience"], "best_validation": best,
                  "data_sha256": data_hash, "config": config}
        tmp = status_path.with_suffix(".tmp"); tmp.write_text(json.dumps(status, indent=2)); tmp.replace(status_path)
        print(f"seed={seed} epoch={epoch+1} train={train_sum/train_n:.5f} val={val_loss:.5f} best={best:.5f}", flush=True)
        if stale >= config["patience"]: break


@torch.inference_mode()
def generate(data: dict, prepared: Path, checkpoint: Path, indices: np.ndarray,
             seed: int, batch_size: int = 2048, device=None) -> np.ndarray:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = configuration(data)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # trusted local checkpoint only
    if state["data_sha256"] != sha256_file(prepared) or state["config"] != config or state["seed"] != seed:
        raise ValueError("Diffusion checkpoint does not match this data/configuration/seed")
    model = PairedDenoiser(config["hidden"], config["depth"], config["timesteps"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    steps = np.rint(np.linspace(config["timesteps"] - 1, 0, config["ddim_steps"])).astype(int)
    assert len(np.unique(steps)) == len(steps) and steps[-1] == 0
    generator = torch.Generator(device=device).manual_seed(2026 + seed)
    batches = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        cond = torch.from_numpy(data["cond"][chunk].astype("float32")).to(device)
        x = torch.randn((len(chunk), 96), device=device, generator=generator)
        for j, step in enumerate(steps):
            t = torch.full((len(chunk),), int(step), device=device, dtype=torch.long)
            velocity = model(x, t, cond)
            a = model.alpha_bar[int(step)]
            clean = a.sqrt() * x - (1 - a).sqrt() * velocity
            eps = (1 - a).sqrt() * x + a.sqrt() * velocity
            if j + 1 < len(steps):
                next_a = model.alpha_bar[int(steps[j + 1])]
                x = next_a.sqrt() * clean + (1 - next_a).sqrt() * eps
            else: x = clean
            if not torch.isfinite(x).all(): raise ValueError("Nonfinite diffusion sample")
        batches.append(x.reshape(-1, 2, 48).cpu().numpy())
    latent = np.concatenate(batches)
    mean, std = _stats(config)
    log_energy = latent * std + mean
    if not np.isfinite(log_energy).all() or np.max(log_energy) > 8:
        raise ValueError("Extreme or nonfinite generated log energy")
    energy = np.expm1(np.maximum(log_energy, 0)) * data["scale"][None, :, None]
    ceiling = 10 * np.max(data["x"][data["split"] == 0], axis=(0, 2))
    if not np.isfinite(energy).all() or (np.max(energy, axis=(0, 2)) > ceiling).any():
        raise ValueError("Extreme generated interval energy")
    return energy.astype("float32")


def validation_gate(data: dict, prepared: Path, checkpoint: Path, seed: int = 1, device=None) -> dict:
    val = np.flatnonzero(data["split"] == 1)
    indices = np.random.default_rng(2026).choice(val, size=min(2048, len(val)), replace=False)
    probe = generate(data, prepared, checkpoint, indices, seed, batch_size=512, device=device)
    result = {}
    for channel, label in enumerate(("load", "solar")):
        real_q95 = float(np.quantile(data["x"][indices, channel].max(axis=1), .95))
        synth_q95 = float(np.quantile(probe[:, channel].max(axis=1), .95))
        result[label] = {"real_peak_q95": real_q95, "synthetic_peak_q95": synth_q95}
        if synth_q95 > 3 * max(real_q95, .01):
            raise ValueError(f"Validation peak gate failed for {label}: {synth_q95:.3f} vs {real_q95:.3f}")
    return result


def mask_night(x: np.ndarray, night: np.ndarray) -> np.ndarray:
    result = x.copy()
    result[:, 1, :][night] = 0
    return result

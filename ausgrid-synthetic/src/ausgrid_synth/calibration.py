"""Exploratory training-only daily load-total calibration for diffusion v2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import sha256_file

BASE_ARM = "diffusion_v2_post"
ARM = "diffusion_v2_loadcal_post"
METHOD = "daily_gccl_training_ecdf_ratio_tail_v1"
QUANTILES = np.linspace(.001, .999, 999, dtype="float64")


def fit_mapping(source: np.ndarray, observed: np.ndarray, *, seed: int, data_sha256: str,
                source_train_sha256: str) -> dict:
    generated = source[:, 0, :].sum(axis=1, dtype=np.float64)
    real = observed[:, 0, :].sum(axis=1, dtype=np.float64)
    if not np.isfinite(generated).all() or not np.isfinite(real).all() or generated.max() <= 0 or real.max() <= 0:
        raise ValueError("Expected finite, nonzero training daily totals")
    src_q = np.quantile(generated, QUANTILES)
    dst_q = np.quantile(real, QUANTILES)
    unique, first = np.unique(src_q, return_index=True)
    positive = unique > 0
    unique, target = unique[positive], dst_q[first][positive]
    if len(unique) <= 1 or np.any(np.diff(target) < 0): raise ValueError("Invalid empirical mapping")
    return {"method": METHOD, "seed": seed, "data_sha256": data_sha256,
            "source_arm": BASE_ARM, "source_train_sha256": source_train_sha256,
            "src_q": unique.tolist(), "dst_q": target.tolist(),
            "source_zero_days": int((generated == 0).sum()), "real_zero_days": int((real == 0).sum()),
            "training_days": len(source)}


def apply_mapping(source: np.ndarray, mapping: dict) -> np.ndarray:
    total = source[:, 0, :].sum(axis=1, dtype=np.float64)
    src = np.asarray(mapping["src_q"], dtype=np.float64)
    dst = np.asarray(mapping["dst_q"], dtype=np.float64)
    mapped = np.interp(total, src, dst)
    mapped = np.where(total < src[0], total * dst[0] / src[0], mapped)
    mapped = np.where(total > src[-1], total * dst[-1] / src[-1], mapped)
    ratio = np.divide(mapped, total, out=np.zeros_like(mapped), where=total > 0)
    result = source.copy()
    result[:, 0, :] = (source[:, 0, :].astype(np.float64) * ratio[:, None]).astype("float32")
    if not np.array_equal(result[:, 1, :], source[:, 1, :]) or not np.isfinite(result).all() or np.min(result) < 0:
        raise ValueError("Invalid calibrated sample")
    return result


def _read_sample(path: Path, expected_idx: np.ndarray):
    with np.load(path, allow_pickle=False) as z:
        x, idx = z["x"], z["idx"]
    if x.shape != (len(idx), 2, 48) or not np.array_equal(idx, expected_idx) or not np.isfinite(x).all() or np.min(x) < 0:
        raise ValueError(f"Invalid or mismatched source: {path}")
    return x


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


def calibrate(data: dict, prepared: Path, samples: Path, reports: Path, seed: int,
              context_indices) -> None:
    """Fit on training households and apply unchanged mapping to three matched contexts."""
    reports.mkdir(parents=True, exist_ok=True)
    data_hash = sha256_file(prepared)
    train_path = samples / f"{BASE_ARM}_seed{seed}_train.npz"
    train_idx = context_indices(data, "train")
    train = _read_sample(train_path, train_idx)
    if np.any(train[:, 1, :][data["night"][train_idx]] != 0): raise ValueError("Expected masked source GG")
    expected = fit_mapping(train, data["x"][train_idx], seed=seed, data_sha256=data_hash,
                           source_train_sha256=sha256_file(train_path))
    mapping_path = reports / f"{ARM}_seed{seed}_mapping.json"
    if mapping_path.exists():
        if json.loads(mapping_path.read_text()) != expected: raise ValueError("Mapping differs; choose a new arm name")
    else: _write_json(mapping_path, expected)
    for context in ("train", "test", "privacy"):
        idx = context_indices(data, context)
        source_path = samples / f"{BASE_ARM}_seed{seed}_{context}.npz"
        output_path = samples / f"{ARM}_seed{seed}_{context}.npz"
        provenance_path = reports / f"{ARM}_seed{seed}_{context}_provenance.json"
        source_hash = sha256_file(source_path)
        info = {"method": METHOD, "data_sha256": data_hash, "source_sha256": source_hash,
                "mapping_sha256": sha256_file(mapping_path)}
        if output_path.exists() and provenance_path.exists():
            saved = _read_sample(output_path, idx)
            old = json.loads(provenance_path.read_text())
            if any(old.get(key) != val for key, val in info.items()) or old.get("output_sha256") != sha256_file(output_path):
                raise ValueError(f"Changed calibrated output: {output_path}")
            continue
        if provenance_path.exists(): raise ValueError(f"Missing sample for provenance: {output_path}")
        source = _read_sample(source_path, idx)
        if np.any(source[:, 1, :][data["night"][idx]] != 0): raise ValueError("Expected masked source GG")
        adjusted = apply_mapping(source, expected)
        max_load = float(adjusted[:, 0, :].max())
        if max_load > 10 * data["x"][data["split"] == 0, 0, :].max():
            raise ValueError("Extreme load after calibration")
        if output_path.exists():
            if not np.array_equal(_read_sample(output_path, idx), adjusted):
                raise ValueError(f"Incomplete sample differs: {output_path}")
        else:
            tmp = output_path.with_suffix(".tmp")
            with tmp.open("wb") as stream: np.savez_compressed(stream, x=adjusted, idx=idx)
            tmp.replace(output_path)
        _write_json(provenance_path, {**info, "output_sha256": sha256_file(output_path),
                    "zero_source_days": int((source[:, 0, :].sum(axis=1) == 0).sum()),
                    "max_load_interval_kwh": max_load})
        print(f"Saved {output_path}")

# Ausgrid paired daily synthetic profiles

This project generates independent paired daily profiles of **recorded total consumption** (`GC + CL` where a household has controlled-load records) and **gross solar generation** (`GG`). Each channel has 48 half-hour kWh intervals. The current generator is conditional diffusion v2 with a conservative solar-night mask (`diffusion_v2_post`). A training-only daily load-total calibration is available as an exploratory postprocessing arm. Earlier statistical and VAE experiments remain for comparison; the original diffusion run failed a load-peak plausibility check.

## Repository layout

The project files are in the inner `ausgrid-synthetic/` directory of this repository. Run commands from that directory (the one containing `src/`, `requirements.txt`, and this README).

| Path | Purpose |
| --- | --- |
| `src/ausgrid_synth/data.py` | Raw CSV preparation, exclusions, household split, conditions, solar-night mask |
| `src/ausgrid_synth/diffusion.py` | Diffusion v2 model, checkpointed training, DDIM sampling, validation peak gate |
| `src/ausgrid_synth/calibration.py` | Optional training-only daily load-total mapping |
| `src/ausgrid_synth/cli.py` | Preparation, training, validation, sampling, calibration, evaluation |
| `src/ausgrid_synth/evaluate.py` | Shared fidelity, utility, and bounded disclosure probe |
| `notebooks/Ausgrid_Diffusion_Colab.ipynb` | Current training, sampling, evaluation, and profile review |
| `notebooks/Ausgrid_Diffusion_Load_Calibration.ipynb` | Optional follow-up mapping and paired comparison |
| `notebooks/Ausgrid_Colab.ipynb`, `Ausgrid_Jupyter.ipynb` | Historical statistical and VAE workflows |
| `notebooks/Ausgrid_Monthly_*.ipynb` | Separate monthly-data research track; not inputs to the daily diffusion model |
| `outputs/reports/` | Recorded experiment reports; checkpoints and samples are local runtime artifacts |

## Reproduce the daily experiment

Install `requirements.txt` in a Python environment with PyTorch. A CUDA GPU is recommended for training and full sampling. Place the three original `Solar home 2010-2011.csv`, `2011-2012.csv`, and `2012-2013.csv` files in `data/raw/` (retain the descriptive first row). The current repository contains copies, but confirm their provenance and license before redistributing data. Work from the inner project directory:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m ausgrid_synth.cli prepare
python -m ausgrid_synth.cli train --arm diffusion_v2 --seed 1 --epochs 2
python -m ausgrid_synth.cli train --arm diffusion_v2 --seed 1 --epochs 25
python -m ausgrid_synth.cli validate --arm diffusion_v2 --seed 1
```

The two-epoch pilot resumes into the fixed 25-epoch run. After the validation peak gate passes, repeat `train --arm diffusion_v2 --epochs 25` for seeds 2 and 3, then run:

```bash
for seed in 1 2 3; do
  for context in train test privacy; do
    python -m ausgrid_synth.cli sample --arm diffusion_v2_post --seed "$seed" --context "$context"
  done
  python -m ausgrid_synth.cli evaluate --arm diffusion_v2_post --seed "$seed"
done
```

`diffusion_v2_post` uses the **same** model and generated profiles as `diffusion_v2`, setting GG to zero only for the conservative night mask. The sample command saves the raw diffusion sample before deriving the masked one. Reruns verify the existing sample's context indices. Checkpoints resume only when the fixed configuration and prepared-data SHA-256 match. The `validate` stage gates the validation-household peak q95 at no more than 3× its real counterpart. This gate catches the original runaway failure; it does not establish model quality.

For optional training-only load calibration, generate all three masked contexts for each seed first:

```bash
for seed in 1 2 3; do
  python -m ausgrid_synth.cli calibrate --arm diffusion_v2_loadcal_post --seed "$seed"
  python -m ausgrid_synth.cli evaluate --arm diffusion_v2_loadcal_post --seed "$seed"
done
```

Or open `notebooks/Ausgrid_Diffusion_Colab.ipynb`, followed by `notebooks/Ausgrid_Diffusion_Load_Calibration.ipynb`. They locate the project from a local notebook folder or the documented Google Drive location, use the same CLI, and show the recorded comparison. If using Colab, mount Drive from the setup cell when prompted. The prepared `.npz`, sample `.npz`, and `.pt` checkpoint files are not tracked; create them locally from the raw CSVs and training run. Reports alone cannot regenerate samples. Do not load checkpoints from untrusted sources.

## Data and evaluation contract

- Households without any CL records contribute GC. Households with CL records contribute `GC + CL` only on days with GC, GG, and CL present. Missing CL is never set to zero for those households. Estimated-quality, missing, negative, and daylight-saving transition days are excluded. Inspect `data/prepared/audit.json` after preparation.
- Original interval order runs from `0:30` (00:00–00:30) through final `0:00` (23:30–24:00). The fixed split is 180/60/60 train/validation/test **households**, approximately stratified by panel capacity. Scaling and model fitting use train households. Calendar and log panel capacity supply four conditions; household ID, weather, postcode, and real test load are not generator inputs.
- The model jointly generates 96 values, using train-channel `log1p` transforms, velocity prediction with a zero-terminal-signal cosine schedule, and 50 deterministic DDIM steps. Nonnegative energy is enforced at inverse transform; extreme or nonfinite output aborts instead of being silently upper-clipped. The night mask uses approximate Sydney solar elevation below −9°, not an exact roof-level PV calculation or a kWp energy cap.
- Matched train, test, and privacy context indices are fixed across arms. The shared evaluator reports load and solar daily/peak Wasserstein distances, profile errors, nighttime GG, a real-test morning-to-afternoon Ridge MAE trained on 30,000 matched synthetic versus real train days, and a bounded nearest-profile household membership AUC. Its AUC is **not** a privacy guarantee. Generated sample archives contain `idx` linked to internal household data and should not be released as anonymized output. These are independent daily samples without longitudinal household identities.

## What the saved reports support

Three seeds share one household split. Mean scores below are calculated from the tracked `outputs/reports/*_seed{1,2,3}.json` files; lower W1 and MAE is favorable. They describe the recorded run, not a new result from this refactor.

| Arm | Load daily W1 (kWh) | Load peak W1 (kWh/interval) | Solar daily W1 (kWh) | Synthetic-to-real MAE (kWh/interval) | Membership AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| VAE + night mask | 0.668 | 0.415 | 0.289 | 0.300 | 0.524 |
| Diffusion v2 + night mask | 1.015 | 0.065 | 0.133 | 0.272 | 0.532 |
| Diffusion v2 + load calibration + night mask | 0.747 | 0.065 | 0.133 | 0.273 | 0.531 |

Diffusion v2 improves peak fidelity, solar daily fidelity, and downstream utility against the VAE in these reports. The VAE has a lower **load daily W1**. Calibration improves diffusion's load daily W1, with small changes to peak fidelity and utility; it was developed after viewing the same held-out test split. The original diffusion arm had implausible peaks (seed 1 generated load peak q95 ≈463 kWh per interval against ≈3.10 real) and is not the current model. Treat all method selection on this already-inspected test split as exploratory; confirm on untouched households before making a final claim.

The monthly notebooks and outputs address a separate 2007–2014 monthly dataset. Their VAE results should not be combined with the daily 48-interval scores above.

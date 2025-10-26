<div align="center">

# Universal RL Trader + Multi‑Ticker News + Markov Regimes

*(Boolean‑safety • Stable Markov injection • News *k* floor • Diagnostics)*

[![python](https://img.shields.io/badge/Python-3.10%7C3.11-blue.svg)](https://www.python.org/) [![pytorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/) [![hydra](https://img.shields.io/badge/Config-Editable-green.svg)]() [![dvc](https://img.shields.io/badge/DVC-Optional-6f42c1.svg)](https://dvc.org/) [![docker](https://img.shields.io/badge/Docker-Optional-2496ed.svg)](https://www.docker.com/)

</div>

> **TL;DR**
>
> End‑to‑end research template for learning a trading policy from price/volume + technical features, enriched with **news‑driven factors** (surprise, impact tiers) and **Markov regime** posteriors. Implements an **LSTM‑based PPO** agent, observation/reward normalization, curriculum training, and a reproducible inference pipeline with diagnostics.

---

## Table of Contents

* [Features](#features)
* [Project Structure](#project-structure)
* [Setup](#setup)

  * [1) Python](#1-python)
  * [2) (Optional) DVC + DagsHub](#2-optional-dvc--dagshub)
  * [3) (Optional) Docker](#3-optional-docker)
* [Data: News CSV expectations](#data-news-csv-expectations)
* [Quickstart](#quickstart)
* [Training](#training)
* [Evaluation & Inference](#evaluation--inference)
* [Configuration & Toggles](#configuration--toggles)
* [Git Workflow (feature branches)](#git-workflow-feature-branches)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [License](#license)

---

## Features

* **Multi‑ticker basket** (AAPL, MSFT, GOOGL, AMZN, META by default).
* **News factor** built from daily *ImpactSurprise* with clamped *k* (optionally calibrated per ticker).
* **Markov regimes** (3 states) with stable posterior filtering → soft policy adaptation.
* **LSTM backbone** for policy/value with PPO (GAE, ratio clipping, vf clipping, entropy decay, early stop by KL).
* **Observation & reward normalization** (shared RMS) to stabilize learning.
* **Vol‑targeted position scaling** + trend gates (ADX/EMA gap) + long‑only option.
* **Curriculum training** across staged windows + periodic basket validation + best checkpointing.
* **Diagnostics**: Sharpe, MDD, ROI; benchmark curve; top news attached to position changes.
* **Performance niceties**: AMP autocast, optional `torch.compile`, cosine LR schedulers.

## Project Structure

```
repo/
├─ news_score/                    # your news CSVs (see schema below)
│  ├─ Apple_news_score_last.csv
│  ├─ MSFT_score_last.csv
│  └─ ...
├─ src/
│  ├─ train.py                    # curriculum training loop (PPO)
│  ├─ infer.py                    # inference & diagnostics per ticker
│  ├─ data_prep.py                # yfinance download + feature engineering
│  ├─ news_factors.py             # CSV → daily impacts → factor map
│  ├─ markov.py                   # regime fit + posterior filter
│  ├─ env.py                      # trading gym env + RMS wrappers
│  └─ models.py                   # LSTM Actor/Critic, memory, agent
├─ configs/                       # (optional) YAML for hyperparams
│  └─ default.yaml
├─ requirements.txt
├─ Dockerfile                     # optional (see below)
├─ dvc.yaml                       # optional DVC pipeline
├─ .dvcignore                     # optional DVC ignore
└─ README.md
```

> Your current code can live in a single script as well; the split above is a suggested refactor for maintainability.

## Setup

### 1) Python

```bash
# Create env (Conda shown; you can use venv instead)
conda create -n rltrader python=3.11 -y
conda activate rltrader

# Install deps
pip install -r requirements.txt
# Minimal requirements include:
# torch, numpy, pandas, yfinance, ta, scikit-learn, gym
```

> **macOS M‑series**: prefer the official PyTorch install command for Metal if you want GPU‑acceleration; CPU works fine too.

### 2) (Optional) DVC + DagsHub

This repo can version large datasets/artifacts via **DVC** while keeping code on GitHub/DagsHub.

```bash
# Init (once)
dvc init

git add .dvc .dvcignore
git commit -m "chore(dvc): init"

# Set DVC remote to DagsHub (replace USER/REPO)
dvc remote add -d dagshub https://dagshub.com/USER/REPO.dvc
# If using auth tokens:
dvc remote modify dagshub auth basic
dvc remote modify dagshub user <your_username>
dvc remote modify dagshub password <your_token>

# Track big directories
dvc add news_score
# or individual files: dvc add news_score/Apple_news_score_last.csv

git add news_score.dvc .gitignore
git commit -m "data: track news_score via dvc"

# Push data to DagsHub storage
dvc push

# Pull data later on a new machine
dvc pull
```

**Notes**

* `git remote` vs `dvc remote`: Git pushes code/metadata; DVC pushes the actual file contents to object storage. You usually want **both**: `git push` and `dvc push`.
* If Git push is rejected ("fetch first"), run `git pull --rebase` then retry.

### 3) (Optional) Docker

```bash
# Build
docker build -t rltrader:latest .
# Run
docker run --rm -it -v "$PWD:/work" -w /work rltrader:latest python src/train.py
```

**Common macOS fix**: if you see *Cannot connect to the Docker daemon*, open the Docker Desktop app first.

## Data: News CSV expectations

Your **news_score** CSVs can have flexible column names; the loader will try to auto‑detect. Minimum required fields (names may vary, see mapping below):

* **date/time**: `date` | `datetime` | `pubDate` | `published_at` | `time` | `timestamp` (UTC or local; UTC preferred)
* **title**: `title` | `headline`
* **url**: `url` | `link`
* **label** *(sentiment)*: `label` | `sentiment` | `class` | `polarity` → values mapped to `{bullish,bearish,neutral}`
* *(optional)* **impact tier/weight**: `impact_tier` or `impact_strength`/`tier`; `impact_weight` or `weight`

The pipeline will:

1. Convert to **NY local time** and map each item to the **trading day** (T‑day) with post‑4pm rollover to next day.
2. Deduplicate by `(trading_day, title_canon, domain)` keeping the *strongest* item.
3. Aggregate per day → compute `ImpactSurprise` (EMA residual) and construct a **news factor** (1+k·tanh(x/scale)) clamped to ±`cap`.

## Quickstart

```bash
# 1) Prepare news (paths are in code; adjust as needed)
mkdir -p news_score
# Put your CSVs into news_score/ (see Data section)

# 2) Train (curriculum + periodic validation + best checkpoint)
python src/train.py

# 3) Inference on basket (uses best checkpoint if found)
python src/infer.py --tickers AAPL MSFT GOOGL AMZN META \
  --split test --attach-news --save-csv runs/bt/
```

Outputs include per‑ticker ROI/Sharpe/MDD and an optional CSV with `Equity`, `Benchmark`, `Position`, and a `TopNewsAtChange` column.

## Training

* **Curriculum**: three stages over different windows with periodic validation on a basket.
* **Validation metric**: mean(Sharpe − 0.5·|MDD|) across tickers.
* **Checkpoints**: actor/critic `.pt` saved under `tmp/<run_id>/` when validation improves.
* **Stability**: AMP, optional `torch.compile`, RMS normalization for obs/rewards, entropy decay.

Key hyperparameters (see code):

* `SEQ_LEN`, `BATCH_SIZE`, `N_EPOCHS`, `ENTROPY_COEF`, `policy_clip`, `target_kl`, `vf_clip`, `max_grad_norm`.
* Position/penalties: `PERCENT_CAPITAL`, `TARGET_DAILY_VOL`, `TURNOVER_COST`, `HOLD_COST`, `SMOOTH_COST`, `LEVER_CAP`.
* Gates: `TREND_GATE`, `ADX_ENTER/ADX_FULL`, `TREND_STRENGTH_CLIP`, `LONG_ONLY`.

## Evaluation & Inference

```bash
# Single ticker
env PYTHONPATH=src python src/infer.py --ticker NVDA --split test --csv out/nvda_bt.csv

# Basket + calibrated news k (per ticker)
python src/infer.py --tickers AAPL MSFT ... --calibrate-k
```

What you get:

* **ROI**: `(Equity[-1]-1)*100`.
* **Sharpe**: daily mean/std × √252.
* **MDD**: min of `Equity/peak - 1`.
* **Benchmark**: buy‑and‑hold curve on the same date range.
* **Top news at position changes** to inspect narrative triggers.

## Configuration & Toggles

Most switches are top‑level constants or CLI flags you can expose:

* `FORCE_NEWS_K` (or per‑ticker calibration with `--calibrate-k`).
* `ENABLE_SURPRISE_SOFTGATE` to cap exposure on big negative surprises in low‑trend regimes.
* `WEAKEN_RAW_GATES_FOR_TEST` to relax ADX/EMA gating during inference.
* Markov usage can be disabled by skipping feature injection.

> Tip: Move these into `configs/default.yaml` and parse via `omegaconf`/Hydra if you want structured experiments.

## Git Workflow (feature branches)

```bash
# Create a dedicated branch for experiments
git checkout -b feat/ppo-news-markov

# Commit clearly and often
git add -A
git commit -m "feat(ppo): add news factor with k‑floor and regime soft‑gating"

# Push and open a PR to main (protected)
git push origin feat/ppo-news-markov
```

**Style**: use conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`) and small, focused PRs. Enable branch protection on `main` and require review before merge.

## Troubleshooting

* **Docker**: `Cannot connect to the Docker daemon` → start Docker Desktop on macOS, or `sudo service docker start` on Linux.
* **DVC**: files “push to DagsHub” vs “Git remote” confusion → remember to run **both** `git push` and `dvc push`. Use `dvc remote list` to verify remotes; `dvc pull` to download.
* **Git push rejected** (`fetch first`) → `git pull --rebase`, resolve conflicts, retry.
* **OpenAI key not set** → make sure `.env` is loaded (e.g., `python-dotenv`) and **do not** hardcode keys in source. Verify with:

  ```python
  import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv("OPENAI_API_KEY")))
  ```
* **Missing columns in news CSV** → see *Data* mapping; the loader attempts reasonable defaults.

## Roadmap

* [ ] Config YAML + Hydra CLI.
* [ ] TensorBoard/Weights & Biases logging.
* [ ] Expand asset universe (ETFs, FX) & multi‑asset allocation.
* [ ] Walk‑forward cross‑validation & robust metrics.
* [ ] More expressive regime models (HMM/GMM) with online updates.
* [ ] Risk overlay: circuit breakers, MDD floors, dynamic capital fraction.

## License

TBD (MIT/Apache‑2.0 recommended for research templates).

---

### Citation

If you use this repo in academic work, please cite it informally as:

> *Universal RL Trader with News and Markov Regimes* (2025). Research template implementing LSTM‑PPO with news factors and regime adaptation.

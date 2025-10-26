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
TRADING_AGENT3/
├─ .venv/                     # local virtualenv (optional)
├─ news_score/
│  ├─ AMZN_score_last.csv
│  ├─ Apple_news_score_last.csv
│  ├─ GOOGLE_score_last.csv   # note: GOOGLE (not GOOGL)
│  ├─ META_score_last.csv
│  └─ MSFT_score_last.csv
├─ tmp/                       # checkpoints/outputs created by the run
├─ agent.py                   # main script (training + inference + diagnostics)
├─ requirements.txt
├─ Dockerfile                 # container build (primary)
├─ .dockerfile                # alt/legacy dockerfile (unused; keep or remove)
├─ .dockerignore
├─ .gitignore
└─ Makefile                   # optional shortcuts (if used)
```

> This layout assumes a **single main script** (`agent.py`) that runs end‑to‑end (train → validate → inference). You can refactor into modules later if desired.

## Setup

### 1) Python

````bash
# Create and activate a local venv (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt
```bash
# Create env (Conda shown; you can use venv instead)
conda create -n rltrader python=3.11 -y
conda activate rltrader

# Install deps
pip install -r requirements.txt
# Minimal requirements include:
# torch, numpy, pandas, yfinance, ta, scikit-learn, gym
````

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
# Build (uses Dockerfile in repo root)
docker build -t trading_agent3:latest .

# Run, mounting the local folder so the script can see your news CSVs
docker run --rm -it \
  -v "$PWD:/app" \
  -w /app \
  trading_agent3:latest \
  python agent.py
```

**macOS tip**: if you see *Cannot connect to the Docker daemon*, open Docker Desktop first.bash

# Build

docker build -t rltrader:latest .

# Run

docker run --rm -it -v "$PWD:/work" -w /work rltrader:latest python src/train.py

````
**Common macOS fix**: if you see *Cannot connect to the Docker daemon*, open the Docker Desktop app first.

## Data: News CSV expectations
The code expects CSVs inside **news_score/** with these file names (as in your repo):

- `Apple_news_score_last.csv`, `MSFT_score_last.csv`, `GOOGLE_score_last.csv`, `AMZN_score_last.csv`, `META_score_last.csv`

Column mapping is flexible; at minimum provide: date/time, title, url, and label (bullish/bearish/neutral). Optional: impact tier/weight. See details below if your columns differ.

## Quickstart
```bash
# 1) Ensure news CSVs exist
ls news_score/*.csv

# 2) Run the main script (does training + inference + prints ROI/Sharpe/MDD)
python agent.py

# Outputs: checkpoints and optional CSVs under tmp/<run_id>/
````

If you want to skip training and only run inference with existing weights, adapt `agent.py` flags (if present) or set the booleans at the top of the script. (By default the script trains and then runs inference on the basket.)
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

After the run, check the console for per‑ticker metrics like:

* **ROI** `(Equity[-1]-1)*100`
* **Sharpe** daily mean/std × √252
* **MDD** min drawdown of equity

CSV backtests may be written under `tmp/<run_id>/bt_<TICKER>.csv` if enabled in `agent.py` (search for `to_csv`).

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

* **Docker**: `Cannot connect to the Docker daemon` → start Docker Desktop (macOS) or `sudo service docker start` (Linux).
* **Requirements build error during `pip install -r requirements.txt` in Docker**: make sure `requirements.txt` lists compatible versions for your Python base image. Rebuild with `--no-cache` if you tweaked it.
* **News file path mismatch**: the code uses `news_score/GOOGLE_score_last.csv` (GOOGLE, not GOOGL). Ensure names match exactly.
* **Git push rejected (fetch first)**: `git pull --rebase origin main`, resolve conflicts, then `git push`.
* **OpenAI key not set (if you use LLM scoring elsewhere)**: load `.env` properly (do not hardcode keys). Verify:

  ```python
  import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('OPENAI_API_KEY')))
  ```

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


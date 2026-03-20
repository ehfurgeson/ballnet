"""
scripts/04_calibrate.py
=======================
Compute empirical calibration buckets for the active model on the validation set.
Must be run after training (03_train.py).

For each stat, the script:
  1. Collects (predicted_value, actual_value) pairs from the validation set
  2. Bins predictions into 1-unit integer buckets (0-1, 1-2, 2-3, …)
  3. Dynamically merges adjacent sparse buckets until every bucket has ≥ MIN_SAMPLES
  4. Saves the merged bucket structure to conformal_residuals.pkl

Input:  data/ (all tensor artifacts), models/<ACTIVE_MODEL>/model.pth
Output: models/<ACTIVE_MODEL>/conformal_residuals.pkl
        → {"version": "empirical_v1",
            "PTS": [{"lo": 0, "hi": 5, "actuals": [...]}, ...],
            "AST": [...], ...}

At inference, given a point estimate ŷ and a threshold t:
    bucket = lookup bucket containing ŷ
    P(stat > t) = mean(array(bucket["actuals"]) > t)
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (DATA_DIR, MODEL_DIR, GATV2_SRC, FEATURE_COLS,
                    PREDICTION_COLS, PRED_INDICES, SEQ_LENGTH,
                    SPLIT_TRAIN, SPLIT_VAL)

sys.path.insert(0, str(GATV2_SRC))
from gatv2tcn import GATv2TCN  # pyright: ignore[reportMissingImports]

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [calibrate]  %(message)s")
log = logging.getLogger(__name__)

N_IN     = len(FEATURE_COLS) + 2 + 2   # 13 stats + 2 team emb + 2 pos emb = 17
N_OUT    = len(PREDICTION_COLS)
SPLIT_TR = SPLIT_TRAIN
SPLIT_VAL= SPLIT_VAL

# ── LOG_TRANSFORM calibration flag ──────────────────────────────────
# When log1p is applied to BOTH pred and target in the training loss,
# the model outputs real-space values. No expm1 needed before bucketing.
LOG_TRANSFORM = False

# Minimum number of actuals required per bucket. Sparse buckets are dynamically
# merged with the adjacent bucket (left or right, whichever is smaller) until
# every surviving bucket meets this threshold.
MIN_SAMPLES = 50


# ─────────────────────────────────────────────────────────────────
# Dynamic bucket merging
# ─────────────────────────────────────────────────────────────────

def merge_sparse_buckets(bucket_dict: dict, min_samples: int = MIN_SAMPLES) -> list:
    """
    Input:  {int_lo: [actual, actual, ...]}  — 1-unit integer buckets
    Output: sorted list of {"lo": int, "hi": int, "actuals": [...]}

    Greedy merge: repeatedly find the smallest bucket (by sample count) and
    merge it with whichever of its two neighbors is smaller. Repeat until all
    remaining buckets have ≥ min_samples.
    """
    if not bucket_dict:
        return []

    # Build initial unit-width groups
    groups = [
        {"lo": k, "hi": k + 1, "actuals": list(bucket_dict[k])}
        for k in sorted(bucket_dict.keys())
    ]

    changed = True
    while changed:
        changed = False
        # Find the first bucket below min_samples
        for i, g in enumerate(groups):
            if len(g["actuals"]) >= min_samples:
                continue  # this bucket is fine

            # Merge with the smaller of the two neighbours
            if len(groups) == 1:
                break  # Only one bucket left, nothing to merge

            left_n  = len(groups[i - 1]["actuals"]) if i > 0             else float("inf")
            right_n = len(groups[i + 1]["actuals"]) if i < len(groups) - 1 else float("inf")

            if left_n <= right_n and i > 0:
                # Absorb current into left neighbour
                groups[i - 1]["actuals"].extend(g["actuals"])
                groups[i - 1]["hi"] = g["hi"]
                groups.pop(i)
            else:
                # Absorb current into right neighbour
                groups[i + 1]["actuals"] = g["actuals"] + groups[i + 1]["actuals"]
                groups[i + 1]["lo"] = g["lo"]
                groups.pop(i)

            changed = True
            break  # restart scan after any merge

    return groups


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────

def build_edges(G, player_ids, device):
    nd = {pid: i for i, pid in enumerate(player_ids)}
    edges = list(G.edges())
    if not edges:
        n = len(player_ids)
        return torch.stack([torch.arange(n), torch.arange(n)]).long().to(device)
    s, d = zip(*edges)
    s = [nd.get(x, 0) for x in s]; d = [nd.get(x, 0) for x in d]
    return torch.stack([torch.LongTensor(s + d), torch.LongTensor(d + s)]).to(device)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run():
    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    log.info(f"Device: {device}")

    X_seq        = pickle.load(open(DATA_DIR / "X_seq.pkl",         "rb"))
    G_seq        = pickle.load(open(DATA_DIR / "G_seq.pkl",         "rb"))
    player_ids   = pickle.load(open(DATA_DIR / "player_ids.pkl",    "rb"))
    team_temporal= pickle.load(open(DATA_DIR / "team_temporal.pkl", "rb"))
    pos_temporal = pickle.load(open(DATA_DIR / "pos_temporal.pkl",  "rb"))
    n_teams      = pickle.load(open(DATA_DIR / "n_teams.pkl",       "rb"))
    X_raw        = pickle.load(open(DATA_DIR / "X_raw.pkl", "rb"))

    D, P, _ = X_seq.shape
    val_start = int(D * SPLIT_TR)
    val_end   = int(D * SPLIT_VAL)

    n_pos = 3
    team_emb = torch.nn.Linear(n_teams, 2).to(device)
    pos_emb  = torch.nn.Linear(n_pos,    2).to(device)
    model    = GATv2TCN(
        in_channels        = N_IN,
        out_channels       = N_OUT,
        len_input          = SEQ_LENGTH,
        len_output         = 1,
        temporal_filter    = 64,
        out_gatv2conv      = 32,
        dropout_tcn        = 0.25,
        dropout_gatv2conv  = 0.5,
        head_gatv2conv     = 4,
    ).to(device)

    team_emb.load_state_dict(torch.load(MODEL_DIR / "team_emb.pth", map_location=device))
    pos_emb.load_state_dict(torch.load(MODEL_DIR / "pos_emb.pth",   map_location=device))
    model.load_state_dict(torch.load(MODEL_DIR / "model.pth",       map_location=device))
    model.eval(); team_emb.eval(); pos_emb.eval()

    log.info(f"Running on val days {val_start}–{val_end} ({val_end - val_start} days)")

    # raw_buckets[stat][int_lo] → list of actual values
    raw_buckets: dict = {stat: defaultdict(list) for stat in PREDICTION_COLS}

    for di in range(max(val_start, SEQ_LENGTH), val_end):
        team_t = torch.FloatTensor(team_temporal[di]).to(device)
        pos_t  = torch.FloatTensor(pos_temporal[di]).to(device)
        with torch.no_grad():
            tv = team_emb(team_t)
            pv = pos_emb(pos_t)
            Xl = []
            for abs_day in range(di - SEQ_LENGTH + 1, di + 1):
                x_t = torch.cat([torch.FloatTensor(X_seq[abs_day]).to(device), tv, pv], dim=1)
                Xl.append(x_t)
            x_input = torch.stack(Xl, dim=-1)[None, ...]
            g_edges = [build_edges(G_seq[d], player_ids, device)
                       for d in range(di - SEQ_LENGTH + 1, di + 1)]
            pred_raw = model(x_input, g_edges)[0].cpu().numpy()  # (P, N_OUT)
        if LOG_TRANSFORM:
            # Convert log-space model output back to real stat units for bucket boundaries
            pred_raw = np.expm1(np.maximum(pred_raw, 0))

        if di + 1 >= len(X_seq):
            continue

        for pidx in range(P):
            if (X_raw[di + 1, pidx] == 0).all():
                continue  # player didn't play on target day

            actual_raw = X_raw[di + 1, pidx, PRED_INDICES]
            for si, stat in enumerate(PREDICTION_COLS):
                pred_val   = float(pred_raw[pidx, si])
                actual_val = float(actual_raw[si])
                bucket_lo  = int(np.floor(pred_val))   # e.g. pred=14.3 → bucket 14
                raw_buckets[stat][bucket_lo].append(actual_val)

    # ── Dynamic merge + save ─────────────────────────────────────────────────
    output = {"version": "empirical_v1"}

    log.info("\nBucket summary after dynamic merge (MIN_SAMPLES=%d):", MIN_SAMPLES)
    log.info(f"  {'Stat':<6}  {'Buckets':>7}  {'Total n':>8}  {'Range':>14}  Description")
    log.info(f"  {'─'*65}")

    for stat in PREDICTION_COLS:
        groups = merge_sparse_buckets(raw_buckets[stat], min_samples=MIN_SAMPLES)
        output[stat] = groups

        total_n   = sum(len(g["actuals"]) for g in groups)
        lo_vals   = [g["lo"] for g in groups]
        hi_vals   = [g["hi"] for g in groups]
        pred_range = f"{min(lo_vals)}–{max(hi_vals)}"

        log.info(f"  {stat:<6}  {len(groups):>7}  {total_n:>8}  {pred_range:>14}")
        for g in groups:
            arr = np.array(g["actuals"])
            log.info(f"    pred [{g['lo']:>3}–{g['hi']:>3})  n={len(arr):5d}  "
                     f"mean_actual={arr.mean():+.2f}  std={arr.std():.2f}  "
                     f"p5={np.percentile(arr,5):.1f}  p95={np.percentile(arr,95):.1f}")

    out_path = MODEL_DIR / "conformal_residuals.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    log.info(f"\n✓ Saved empirical buckets to {out_path}")


if __name__ == "__main__":
    run()

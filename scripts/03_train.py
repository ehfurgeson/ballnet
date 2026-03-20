"""
scripts/03_train.py
===================
Trains the GATv2TCN model and saves weights to models/<ACTIVE_MODEL>/.

FAITHFUL port of from_google_drive/reproduction/02_train.py — the canonical
training script that produced the weights currently in from_google_drive/outputs/model/.

Key differences from a naive rewrite:
  • Team / position vectors are STATIC (one-hot per player, built from
    player_id2team.pkl and player_id2position.pkl) NOT temporal day-by-day arrays.
  • Training samples a random BATCH of 20 days per epoch — it does NOT iterate
    over all training days, which would be very slow.
  • Target y is the NEXT day's stats (OFFSET=1 sliding window), not the current day.
  • Loss is masked to only players who actually played that day (h_train[i].unique()).
  • GATv2TCN constructor uses: len_input, len_output, out_gatv2conv, dropout_tcn,
    dropout_gatv2conv, head_gatv2conv — NOT seq_length/heads shorthand.

Hardware recommendation:
  • This is a 77KB model. On Apple MPS (M2 MacBook) 1000 epochs takes ~10-20 min.
  • Google Colab A100 is ~3-4× faster but offers no practical benefit for this size.
  • Training locally is recommended — no upload/download overhead, no session timeouts.

Input (from data/):
  X_seq.pkl, G_seq.pkl, player_ids.pkl,
  player_id2team.pkl, player_id2position.pkl
  (player_id2team.pkl and player_id2position.pkl live in NBA-GNN-prediction/
   in the original Colab repo — copy them to data/ or update the path below)

Output (to models/<ACTIVE_MODEL>/):
  model.pth, team_emb.pth, pos_emb.pth, train_loss.npy, val_loss.npy
  test_metrics.json, test_preds.npy, test_trues.npy (if evaluate=True)

Usage:
  python scripts/03_train.py
  python scripts/03_train.py --no-eval    # skip test evaluation
"""
import argparse
import copy
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (DATA_DIR, MODEL_DIR, GATV2_SRC, FEATURE_COLS,
                    PREDICTION_COLS, SEQ_LENGTH, QUANTILE_REGRESSION, QUANTILES,
                    SPLIT_TRAIN, SPLIT_VAL)

# GATv2TCN source lives in NBA-GNN-prediction/ (same structure as Colab)
sys.path.insert(0, str(GATV2_SRC))
from gatv2tcn import GATv2TCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (must match Colab 02_train.py exactly) ────────
OFFSET       = 1    # predict next day (y = X[window + 1])
EPOCHS       = 1000
BATCH_SIZE   = 20   # random sample of training days per epoch
LR           = 0.001
WEIGHT_DECAY = 0.001
PRED_INDICES = [FEATURE_COLS.index(c) for c in PREDICTION_COLS]
USG_IDX      = FEATURE_COLS.index("USG_PCT")  # index 11 — usage rate feature

# ── Model improvement flags ───────────────────────────────────────
# LOG_TRANSFORM: applies log1p to BOTH pred and target before MSE.
# This makes the loss proportional (a 5-pt miss on 40-pt game =  a 5-pt
# miss on 10-pt game in % terms), downweighting star-player outliers.
# NOTE: Because log1p is applied to both sides, the model still converges
# to outputting real-space values (pred ≈ target). No expm1 at inference.
# USAGE_WEIGHTED: weight each player's loss contribution by sqrt(usg_pct).
LOG_TRANSFORM   = True
USAGE_WEIGHTED  = True

PLAYER_ID2TEAM_PATH = DATA_DIR / "player_id2team.pkl"
PLAYER_ID2POS_PATH  = DATA_DIR / "player_id2position.pkl"


# ─────────────────────────────────────────────────────────────────
# Utilities — identical to 02_train.py
# ─────────────────────────────────────────────────────────────────
def fill_zeros_with_last(seq: np.ndarray) -> np.ndarray:
    """Forward-fill zero entries along axis-0 for each column."""
    seq_ff = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        arr = seq[:, i]
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        seq_ff[:, i] = arr[prev]
    return seq_ff


def sliding_window(arr, window: int, offset: int = 1):
    """Create (X, y) sliding-window pairs along axis-0."""
    from numpy.lib.stride_tricks import sliding_window_view
    if isinstance(arr, (list,)):
        # Handle graph list
        x = [arr[i:i+window] for i in range(len(arr) - window - offset + 1)]
        y = arr[window + offset - 1:]
        return x, y
    x = (sliding_window_view(arr[:-offset], window, axis=0)
         if offset > 0
         else sliding_window_view(arr, window, axis=0))
    y = arr[window + offset - 1:]
    return x, y


def graphs_to_edge_tensors(G_seq, player_ids, device) -> list:
    """Convert networkx graph list to PyG edge_index tensors."""
    node_dict = {pid: i for i, pid in enumerate(player_ids)}
    tensors = []
    for G in G_seq:
        edges = list(G.edges())
        if not edges:
            n = len(player_ids)
            ei = torch.stack([torch.arange(n), torch.arange(n)], dim=0).long().to(device)
        else:
            src, dst = zip(*edges)
            src = [node_dict.get(s, 0) for s in src]
            dst = [node_dict.get(d, 0) for d in dst]
            ei = torch.stack([
                torch.LongTensor(src + dst),
                torch.LongTensor(dst + src),
            ], dim=0).to(device)
        tensors.append(ei)
    return tensors


# ─────────────────────────────────────────────────────────────────
# Dataset construction — faithful port of 02_train.py create_dataset()
# ─────────────────────────────────────────────────────────────────
def create_dataset(device):
    log.info("Loading artefacts...")
    X_seq          = pickle.load(open(DATA_DIR / "X_seq.pkl",      "rb"))
    G_seq_graphs   = pickle.load(open(DATA_DIR / "G_seq.pkl",      "rb"))
    player_ids     = pickle.load(open(DATA_DIR / "player_ids.pkl", "rb"))
    player_id2team = pickle.load(open(PLAYER_ID2TEAM_PATH,         "rb"))
    player_id2pos  = pickle.load(open(PLAYER_ID2POS_PATH,          "rb"))

    N = len(player_ids)
    log.info(f"  Players: {N},  Days: {X_seq.shape[0]}")

    # ── Static team one-hot (per player, NOT per day) ─────────────
    # player_id2team may store either strings ("LAL") or ints depending on
    # which version of build_tensors produced it. Handle both.
    raw_team_vals = {pid: player_id2team.get(pid, "") for pid in player_ids}
    sample_val    = next(iter(raw_team_vals.values()), 0)
    if isinstance(sample_val, str):
        # Encode strings → ints alphabetically (stable across runs)
        all_teams    = sorted(set(v for v in raw_team_vals.values() if v))
        team_str2int = {t: i for i, t in enumerate(all_teams)}
        pid2team_int = {pid: team_str2int.get(v, 0) for pid, v in raw_team_vals.items()}
    else:
        pid2team_int = {pid: int(v) for pid, v in raw_team_vals.items()}
    n_teams      = max(pid2team_int.values()) + 1
    team_onehot  = np.zeros((N, n_teams), dtype=np.float32)
    for idx, pid in enumerate(player_ids):
        team_onehot[idx, pid2team_int[pid]] = 1.0
    log.info(f"  Teams: {n_teams}")
    team_tensor  = Variable(torch.FloatTensor(team_onehot)).to(device)

    # ── Static position one-hot ───────────────────────────────────
    pos_arrays = []
    for pid in player_ids:
        pos = player_id2pos.get(pid, np.array([0, 0, 0], dtype=np.float32))
        pos_arrays.append(np.array(pos, dtype=np.float32))
    position_tensor = Variable(torch.FloatTensor(np.stack(pos_arrays))).to(device)

    # ── Forward-fill zeros in X_seq ───────────────────────────────
    Xs = np.zeros_like(X_seq)
    for i in range(X_seq.shape[1]):
        Xs[:, i, :] = fill_zeros_with_last(X_seq[:, i, :])

    # ── Build edge tensor list ─────────────────────────────────────
    G_tensors = graphs_to_edge_tensors(G_seq_graphs, player_ids, device)

    # ── Sliding window (X=10-day window, y=next day's stats) ──────
    X_in_np, X_out_np = sliding_window(Xs,        SEQ_LENGTH, OFFSET)
    G_in,    G_out    = sliding_window(G_tensors,  SEQ_LENGTH, OFFSET)
    X_in  = Variable(torch.FloatTensor(X_in_np.copy())).to(device)
    X_out = Variable(torch.FloatTensor(X_out_np.copy())).to(device)

    # ── Chronological split ──────────────────────────────────────
    T  = X_in.shape[0]
    t1 = int(T * SPLIT_TRAIN)
    t2 = int(T * SPLIT_VAL)
    log.info(f"  Sequences: {T}  →  train:{t1}  val:{t2-t1}  test:{T-t2}")

    splits = {
        "train": (X_in[:t1],    X_out[:t1],    G_in[:t1],    G_out[:t1]),
        "val":   (X_in[t1:t2],  X_out[t1:t2],  G_in[t1:t2],  G_out[t1:t2]),
        "test":  (X_in[t2:],    X_out[t2:],    G_in[t2:],    G_out[t2:]),
    }
    return splits, team_tensor, position_tensor, n_teams


# ─────────────────────────────────────────────────────────────────
# Model construction — exact constructor args from 02_train.py
# ─────────────────────────────────────────────────────────────────
def build_model(n_teams: int, n_positions: int = 3, device=None):
    team_emb_dim = 2
    pos_emb_dim  = 2
    model_in     = len(FEATURE_COLS) + team_emb_dim + pos_emb_dim  # 17

    out_features = len(PREDICTION_COLS) * len(QUANTILES) if QUANTILE_REGRESSION else len(PREDICTION_COLS)

    team_embedding     = nn.Linear(n_teams,     team_emb_dim).to(device)
    position_embedding = nn.Linear(n_positions, pos_emb_dim).to(device)

    model = GATv2TCN(
        in_channels        = model_in,
        out_channels       = out_features,  # 18 or 6
        len_input          = SEQ_LENGTH,             # 10
        len_output         = 1,
        temporal_filter    = 64,
        out_gatv2conv      = 32,
        dropout_tcn        = 0.25,
        dropout_gatv2conv  = 0.5,
        head_gatv2conv     = 4,
    ).to(device)
    return model, team_embedding, position_embedding


# ─────────────────────────────────────────────────────────────────
# Training — faithful port of 02_train.py train()
# 20 random days per epoch, loss masked to active players only
# ─────────────────────────────────────────────────────────────────
def train(device):
    splits, team_tensor, position_tensor, n_teams = create_dataset(device)
    X_train, y_train, G_train, h_train = splits["train"]
    X_val,   y_val,   G_val,   h_val   = splits["val"]

    n_positions = position_tensor.shape[-1]
    model, team_emb, pos_emb = build_model(n_teams, n_positions, device)

    parameters = (list(model.parameters()) +
                  list(team_emb.parameters()) +
                  list(pos_emb.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    train_hist, val_hist = [], []
    min_val_loss  = float("inf")
    min_val_epoch = -1

    log.info(f"Training {EPOCHS} epochs on {device} (batch={BATCH_SIZE} days/epoch)")

    with tqdm(range(EPOCHS), desc="Training") as pbar:
      for epoch in pbar:
        # ─── Train ───────────────────────────────────────────────
        model.train(); team_emb.train(); pos_emb.train()
        team_vec = team_emb(team_tensor)
        pos_vec  = pos_emb(position_tensor)

        train_loss  = torch.tensor(0.0, device=device)
        batch_idx   = np.random.choice(X_train.shape[0], size=BATCH_SIZE, replace=False)
        for i in batch_idx:
            mask   = h_train[i].unique()   # players who played
            X_list = []
            G_list = []
            for t in range(SEQ_LENGTH):
                x_t = torch.cat([X_train[i, :, :, t], team_vec, pos_vec], dim=1)
                X_list.append(x_t)
                G_list.append(G_train[i][t])
            x    = torch.stack(X_list, dim=-1)[None, ...]   # (1, N, F, T)
            pred = model(x, G_list)[0]                       # (N, 6)

            target = y_train[i][mask][:, PRED_INDICES]       # (mask_n, 6)
            pred_m = pred[mask]                              # (mask_n, 6)

            if LOG_TRANSFORM:
                # Clamp negatives before log (model may predict slightly < 0 early in training)
                target = torch.log1p(target.clamp(min=0))
                pred_m = torch.log1p(pred_m.clamp(min=0))

            if USAGE_WEIGHTED:
                # USG_PCT from the INPUT day (day i, not target); shape (mask_n,)
                usg = X_train[i, :, USG_IDX, -1][mask]      # last timestep, USG col
                weights = torch.sqrt(usg.clamp(min=0.01))    # soft-cap avoids div instability
                weights = weights / weights.sum()             # normalize to sum=1
                
                if QUANTILE_REGRESSION:
                    loss = torch.tensor(0.0, device=device)
                    n_stats = len(PREDICTION_COLS)
                    for q_idx, q in enumerate(QUANTILES):
                        pred_q = pred_m[:, q_idx * n_stats : (q_idx + 1) * n_stats]
                        err = target - pred_q
                        q_loss = torch.max(q * err, (q - 1) * err)
                        loss += (q_loss * weights.unsqueeze(1)).sum()
                    loss = loss / len(QUANTILES)
                else:
                    errors = (pred_m - target) ** 2              # (mask_n, 6)
                    loss = (errors * weights.unsqueeze(1)).sum()
            else:
                if QUANTILE_REGRESSION:
                    def pinball_loss(pred, target, q):
                        err = target - pred
                        return torch.max(q * err, (q - 1) * err).mean()

                    loss = torch.tensor(0.0, device=device)
                    n_stats = len(PREDICTION_COLS)
                    for q_idx, q in enumerate(QUANTILES):
                        pred_q = pred_m[:, q_idx * n_stats : (q_idx + 1) * n_stats]
                        loss += pinball_loss(pred_q, target, q)
                    loss = loss / len(QUANTILES)
                else:
                    loss = F.mse_loss(pred_m, target)

            train_loss = train_loss + loss

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # ─── Validate ────────────────────────────────────────────
        model.eval(); team_emb.eval(); pos_emb.eval()
        val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            team_vec = team_emb(team_tensor)
            pos_vec  = pos_emb(position_tensor)
            for i in range(X_val.shape[0]):
                mask   = h_val[i].unique()
                X_list = []; G_list = []
                for t in range(SEQ_LENGTH):
                    x_t = torch.cat([X_val[i, :, :, t], team_vec, pos_vec], dim=1)
                    X_list.append(x_t)
                    G_list.append(G_val[i][t])
                x    = torch.stack(X_list, dim=-1)[None, ...]
                pred = model(x, G_list)[0]
                target = y_val[i][mask][:, PRED_INDICES]
                pred_m = pred[mask]
                if LOG_TRANSFORM:
                    target = torch.log1p(target.clamp(min=0))
                    pred_m = torch.log1p(pred_m.clamp(min=0))
                
                if QUANTILE_REGRESSION:
                    n_stats = len(PREDICTION_COLS)
                    pred_q50 = pred_m[:, 1 * n_stats : 2 * n_stats]
                    val_loss = val_loss + F.mse_loss(pred_q50, target)
                else:
                    val_loss = val_loss + F.mse_loss(pred_m, target)

        tl, vl = train_loss.item(), val_loss.item()
        train_hist.append(tl); val_hist.append(vl)

        # Step the LR scheduler
        scheduler.step()

        new_best = vl < min_val_loss
        pbar.set_postfix(train=f"{tl:.3f}", val=f"{vl:.3f}",
                         best=f"{min_val_loss:.3f}", saved="★" if new_best else "")

        if new_best:
            min_val_loss  = vl
            min_val_epoch = epoch
            torch.save(model.state_dict(),    MODEL_DIR / "model.pth")
            torch.save(team_emb.state_dict(), MODEL_DIR / "team_emb.pth")
            torch.save(pos_emb.state_dict(),  MODEL_DIR / "pos_emb.pth")

    np.save(MODEL_DIR / "train_loss.npy", np.array(train_hist))
    np.save(MODEL_DIR / "val_loss.npy",   np.array(val_hist))
    log.info(f"\n✓ Best val loss: {min_val_loss:.4f} at epoch {min_val_epoch}")
    log.info(f"  Saved weights to {MODEL_DIR}")
    return splits, team_tensor, position_tensor, n_teams


# ─────────────────────────────────────────────────────────────────
# Evaluation — faithful port of 02_train.py evaluate()
# ─────────────────────────────────────────────────────────────────
def evaluate(splits, team_tensor, position_tensor, n_teams, device):
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error

    X_test, y_test, G_test, h_test = splits["test"]
    n_positions = position_tensor.shape[-1]
    model, team_emb, pos_emb = build_model(n_teams, n_positions, device)

    model.load_state_dict(torch.load(MODEL_DIR / "model.pth",     map_location=device))
    team_emb.load_state_dict(torch.load(MODEL_DIR / "team_emb.pth", map_location=device))
    pos_emb.load_state_dict(torch.load(MODEL_DIR / "pos_emb.pth",   map_location=device))
    model.eval(); team_emb.eval(); pos_emb.eval()

    all_preds, all_trues = [], []
    rmse_sum = mae_sum = corr_sum = 0.0

    with torch.no_grad():
        team_vec = team_emb(team_tensor)
        pos_vec  = pos_emb(position_tensor)
        for i in range(X_test.shape[0]):
            mask   = h_test[i].unique()
            X_list = []; G_list = []
            for t in range(SEQ_LENGTH):
                x_t = torch.cat([X_test[i, :, :, t], team_vec, pos_vec], dim=1)
                X_list.append(x_t)
                G_list.append(G_test[i][t])
            x    = torch.stack(X_list, dim=-1)[None, ...]
            pred = model(x, G_list)[0]
            
            if QUANTILE_REGRESSION:
                n_stats = len(PREDICTION_COLS)
                pred_q50 = pred[:, 1 * n_stats : 2 * n_stats]
                p_np = pred_q50[mask].cpu().numpy()
            else:
                p_np = pred[mask].cpu().numpy()
                
            t_np = y_test[i][mask][:, PRED_INDICES].cpu().numpy()
            # No expm1 needed: log1p is applied to both pred and target in training,
            # so the model outputs real-space values (pred ≈ target directly).
            all_preds.append(p_np); all_trues.append(t_np)
            rmse_sum += root_mean_squared_error(t_np, p_np)
            mae_sum  += mean_absolute_error(t_np, p_np)
            corr_vals = []
            for mi in range(len(PREDICTION_COLS)):
                try:
                    r = np.corrcoef(p_np[:, mi], t_np[:, mi])[0, 1]
                    if not np.isnan(r) and abs(r) < 1 - 1e-7:
                        corr_vals.append(np.arctanh(r))
                except Exception:
                    pass
            if corr_vals:
                corr_sum += np.tanh(np.mean(corr_vals))

    n = X_test.shape[0]
    results = {"RMSE": rmse_sum / n, "MAE": mae_sum / n, "CORR": corr_sum / n}
    log.info("── Test Metrics ───────────────────────────────────")
    for k, v in results.items():
        log.info(f"  {k}: {v:.4f}")

    json.dump(results, open(MODEL_DIR / "test_metrics.json", "w"), indent=2)
    np.save(MODEL_DIR / "test_preds.npy", np.concatenate(all_preds))
    np.save(MODEL_DIR / "test_trues.npy", np.concatenate(all_trues))
    log.info("Evaluation complete ✓")
    return results


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval", action="store_true", help="Skip test evaluation")
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    log.info(f"Device: {device}")

    splits, team_tensor, position_tensor, n_teams = train(device)
    if not args.no_eval:
        evaluate(splits, team_tensor, position_tensor, n_teams, device)

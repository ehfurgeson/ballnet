"""
scripts/07_extract_game_embeddings.py
======================================
Splices the GATv2TCN backbone before its final output layer to extract
latent player embeddings, pools them by team per game, then trains and
evaluates downstream game-outcome prediction heads.

Architecture splice point (GATv2TCN forward):
  Input (1, P, 17, 10)
    → GATv2Conv × 10 timesteps → (P, 128, 10)
    → _time_convolution + residual → LayerNorm → (1, 10, P, 64)  ← SPLICE
    → _final_conv → (P, 6) per-player stat predictions

Game feature vector (65-dim):
  diff = emb_A − emb_B          (64,)  antisymmetric team embedding delta
  is_home_a                     (1,)   1 if team_a is home, 0 if team_b is home
  Label = 1 if team_a won, 0 if team_b won.

Step A — Fetch home/away info (LeagueGameFinder, 4 API calls for 4 seasons).
Step B — Extract embeddings for all game days → data/game_embeddings.parquet.
Step C — Train logistic regression + MLP heads on train split, report val/test.
Step D — High-confidence (≥65% predicted probability) margin analysis.

Usage:
  python scripts/07_extract_game_embeddings.py              # full run (fresh weights)
  python scripts/07_extract_game_embeddings.py --skip-extract   # use cached embeddings
  python scripts/07_extract_game_embeddings.py --cutoff 2026-03-05
"""
import argparse
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (DATA_DIR, MODEL_DIR, GATV2_SRC, FEATURE_COLS,
                    PREDICTION_COLS, PRED_INDICES, SEQ_LENGTH, BACKTEST_DIR)

sys.path.insert(0, str(GATV2_SRC))
from gatv2tcn import GATv2TCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [07_emb]  %(message)s")
log = logging.getLogger(__name__)

BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
EMBED_CACHE    = DATA_DIR / "game_embeddings.parquet"
HOME_CACHE     = DATA_DIR / "game_home_teams.parquet"    # {GAME_ID: home_team_abbr}
EMB_DIM        = 64     # temporal_filter
GAME_DIM       = 65     # EMB_DIM + 1 (is_home_a flag)
CONF_THRESHOLD = 0.65   # minimum predicted confidence for high-confidence analysis

# Train / val / test splits (match 03_train.py)
SPLIT_TR  = 0.50
SPLIT_VAL = 0.75

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


# ─────────────────────────────────────────────────────────────────
# Step A — Home/away lookup via nba_api LeagueGameFinder
# ─────────────────────────────────────────────────────────────────

def fetch_home_teams(use_cache: bool = True) -> Dict[str, str]:
    """
    Returns {GAME_ID: home_team_abbreviation} for all seasons in SEASONS.
    Uses MATCHUP field: 'TEAM vs. OPP' means TEAM is home, 'TEAM @ OPP' means away.
    Results cached to HOME_CACHE for subsequent runs.
    """
    if use_cache and HOME_CACHE.exists():
        df = pd.read_parquet(HOME_CACHE)
        log.info(f"  Loaded home-team lookup from cache ({len(df)} games).")
        return dict(zip(df["GAME_ID"], df["home_team"]))

    from nba_api.stats.endpoints import leaguegamefinder
    import time

    rows = []
    for season in SEASONS:
        log.info(f"  LeagueGameFinder: fetching {season}...")
        try:
            lgf   = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=30,
            )
            games = lgf.get_data_frames()[0]
            home  = games[games["MATCHUP"].str.contains(" vs. ", na=False)]
            for _, row in home[["GAME_ID", "TEAM_ABBREVIATION"]].iterrows():
                rows.append({"GAME_ID": str(row["GAME_ID"]), "home_team": row["TEAM_ABBREVIATION"]})
            log.info(f"    → {len(home)} home games found")
            time.sleep(0.6)
        except Exception as e:
            log.warning(f"    {season}: LeagueGameFinder failed ({e})")

    df = pd.DataFrame(rows).drop_duplicates("GAME_ID")
    df.to_parquet(HOME_CACHE, index=False)
    log.info(f"  ✓ Cached {len(df)} game home-team lookups to {HOME_CACHE}")
    return dict(zip(df["GAME_ID"], df["home_team"]))


# ─────────────────────────────────────────────────────────────────
# Step B — Embedding extraction
# ─────────────────────────────────────────────────────────────────

class BackboneWithEmbedding(nn.Module):
    """
    Wraps a trained GATv2TCN to intercept the (P, 64) layer-norm embedding
    immediately before _final_conv.  Call forward() normally, then
    get_embedding() to retrieve the (P, 64) tensor for that pass.
    """
    def __init__(self, model: GATv2TCN):
        super().__init__()
        self.m    = model
        self._emb: Optional[torch.Tensor] = None

    def forward(self, X: torch.Tensor, edge_index) -> torch.Tensor:
        m    = self.m
        X_hat = []
        for t in range(len(edge_index)):
            X_hat.append(
                torch.unsqueeze(
                    m._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index[t]), -1
                )
            )
        X_hat = F.relu(torch.cat(X_hat, dim=-1))[None, ...]          # (1, P, 128, 10)
        X_hat = m._time_convolution(X_hat.permute(0, 2, 1, 3))        # (1, 64, P, 10)
        X_res = m._residual_convolution(X.permute(0, 2, 1, 3))        # (1, 64, P, 10)
        X_norm = m._layer_norm(F.relu(X_res + X_hat).permute(0, 3, 2, 1))  # (1, 10, P, 64)
        self._emb = X_norm[0, -1, :, :].detach().cpu()                # (P, 64) last timestep
        out = m._final_conv(X_norm)
        return out.permute(0, 2, 1, 3)[..., -1]                       # (1, P, 6)

    def get_embedding(self) -> torch.Tensor:
        if self._emb is None:
            raise RuntimeError("Call forward() before get_embedding().")
        return self._emb


def load_backbone(device: torch.device, n_teams: int
                  ) -> Tuple[BackboneWithEmbedding, nn.Linear, nn.Linear]:
    team_emb = nn.Linear(n_teams, 2).to(device)
    pos_emb  = nn.Linear(3, 2).to(device)
    model    = GATv2TCN(
        in_channels=len(FEATURE_COLS) + 4, out_channels=len(PREDICTION_COLS),
        len_input=SEQ_LENGTH, len_output=1, temporal_filter=EMB_DIM,
        out_gatv2conv=32, dropout_tcn=0.25, dropout_gatv2conv=0.5, head_gatv2conv=4,
    ).to(device)
    team_emb.load_state_dict(torch.load(MODEL_DIR / "team_emb.pth", map_location=device))
    pos_emb.load_state_dict( torch.load(MODEL_DIR / "pos_emb.pth",  map_location=device))
    model.load_state_dict(   torch.load(MODEL_DIR / "model.pth",    map_location=device))
    backbone = BackboneWithEmbedding(model).to(device)
    backbone.eval(); team_emb.eval(); pos_emb.eval()
    return backbone, team_emb, pos_emb


def build_edges(G, player_ids, device):
    nd    = {pid: i for i, pid in enumerate(player_ids)}
    edges = list(G.edges())
    if not edges:
        n = len(player_ids)
        return torch.stack([torch.arange(n), torch.arange(n)]).long().to(device)
    s, d = zip(*edges)
    s = [nd.get(x, 0) for x in s]; d = [nd.get(x, 0) for x in d]
    return torch.stack([torch.LongTensor(s + d), torch.LongTensor(d + s)]).to(device)


def extract_embeddings(cutoff: str, home_lookup: Dict[str, str]) -> pd.DataFrame:
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    log.info(f"Device: {device}")

    X_seq         = pickle.load(open(DATA_DIR / "X_seq.pkl",         "rb"))
    X_raw         = pickle.load(open(DATA_DIR / "X_raw.pkl",         "rb"))
    G_seq         = pickle.load(open(DATA_DIR / "G_seq.pkl",         "rb"))
    player_ids    = pickle.load(open(DATA_DIR / "player_ids.pkl",    "rb"))
    game_dates    = pickle.load(open(DATA_DIR / "game_dates.pkl",    "rb"))
    team_temporal = pickle.load(open(DATA_DIR / "team_temporal.pkl", "rb"))
    pos_temporal  = pickle.load(open(DATA_DIR / "pos_temporal.pkl",  "rb"))
    n_teams       = pickle.load(open(DATA_DIR / "n_teams.pkl",       "rb"))

    D, P = X_seq.shape[0], X_seq.shape[1]
    log.info(f"Dataset: {D} days, {P} players, model: {MODEL_DIR.name}")

    boxscores = pd.read_parquet(DATA_DIR / "raw_boxscores.parquet")
    boxscores["GAME_DATE"] = boxscores["GAME_DATE"].astype(str)

    actual = (
        boxscores.groupby(["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"], as_index=False)["PTS"].sum()
    )
    game_results = []
    for gid, grp in actual.groupby("GAME_ID"):
        rows = grp.sort_values("TEAM_ABBREVIATION").reset_index(drop=True)
        if len(rows) != 2:
            continue
        ta, pa = rows["TEAM_ABBREVIATION"].iloc[0], float(rows["PTS"].iloc[0])
        tb, pb = rows["TEAM_ABBREVIATION"].iloc[1], float(rows["PTS"].iloc[1])
        home   = home_lookup.get(str(gid), "")
        is_home_a = int(home == ta) if home else -1   # -1 = unknown
        game_results.append({
            "GAME_ID":    gid, "GAME_DATE": rows["GAME_DATE"].iloc[0],
            "team_a": ta, "team_b": tb,
            "pts_a": pa, "pts_b": pb,
            "label_a": int(pa > pb), "margin": pa - pb,
            "is_home_a": is_home_a,
        })
    game_results_df = pd.DataFrame(game_results)
    home_known_pct = (game_results_df["is_home_a"] >= 0).mean()
    log.info(f"Home/away known for {home_known_pct*100:.1f}% of games")

    backbone, team_emb, pos_emb = load_backbone(device, n_teams)

    train_end = int(D * SPLIT_TR)
    val_end   = int(D * SPLIT_VAL)
    def _split(di): return "train" if di < train_end else ("val" if di < val_end else "test")

    records = []
    skipped = {"cutoff": 0, "noplayers": 0, "nomatch": 0}

    for day_idx in range(SEQ_LENGTH, D - 1):
        game_date_str = game_dates[day_idx + 1]
        if game_date_str > cutoff:
            skipped["cutoff"] += 1; continue

        day_games = game_results_df[game_results_df["GAME_DATE"] == game_date_str]
        if day_games.empty:
            skipped["nomatch"] += 1; continue

        active_pidxs = [i for i in range(P) if not (X_raw[day_idx + 1, i] == 0).all()]
        if not active_pidxs:
            skipped["noplayers"] += 1; continue

        team_t = torch.FloatTensor(team_temporal[day_idx]).to(device)
        pos_t  = torch.FloatTensor(pos_temporal[day_idx]).to(device)
        with torch.no_grad():
            tv = team_emb(team_t); pv = pos_emb(pos_t)
            Xl = []
            for abs_day in range(day_idx - SEQ_LENGTH + 1, day_idx + 1):
                Xl.append(torch.cat([torch.FloatTensor(X_seq[abs_day]).to(device), tv, pv], dim=1))
            x_input  = torch.stack(Xl, dim=-1)[None, ...]
            g_window = [build_edges(G_seq[d], player_ids, device)
                        for d in range(day_idx - SEQ_LENGTH + 1, day_idx + 1)]
            _ = backbone(x_input, g_window)
            emb = backbone.get_embedding()   # (P, 64) CPU

        day_bs = boxscores[boxscores["GAME_DATE"] == game_date_str]
        player_team_today = dict(zip(day_bs["PLAYER_ID"].astype(int), day_bs["TEAM_ABBREVIATION"]))
        player_game_today = dict(zip(day_bs["PLAYER_ID"].astype(int), day_bs["GAME_ID"].astype(str)))

        team_game_embs: Dict[Tuple[str, str], List[np.ndarray]] = {}
        for pidx in active_pidxs:
            pid  = int(player_ids[pidx])
            team = player_team_today.get(pid)
            gid  = player_game_today.get(pid)
            if team and gid:
                key = (gid, team)
                if key not in team_game_embs:
                    team_game_embs[key] = []
                team_game_embs[key].append(emb[pidx].numpy())

        split = _split(day_idx + 1)
        for _, game_row in day_games.iterrows():
            gid = str(game_row["GAME_ID"])
            ta  = game_row["team_a"]; tb = game_row["team_b"]
            embs_a = team_game_embs.get((gid, ta), [])
            embs_b = team_game_embs.get((gid, tb), [])
            if len(embs_a) < 3 or len(embs_b) < 3:
                continue
            pool_a = np.stack(embs_a).mean(axis=0)
            pool_b = np.stack(embs_b).mean(axis=0)
            diff   = pool_a - pool_b
            row = {
                "game_date": game_date_str, "game_id": gid,
                "team_a": ta, "team_b": tb,
                "pts_a": game_row["pts_a"], "pts_b": game_row["pts_b"],
                "label_a": game_row["label_a"], "margin": game_row["margin"],
                "is_home_a": int(game_row["is_home_a"]),
                "split": split,
                "n_players_a": len(embs_a), "n_players_b": len(embs_b),
            }
            for i in range(EMB_DIM):
                row[f"emb_a_{i}"] = float(pool_a[i])
                row[f"emb_b_{i}"] = float(pool_b[i])
                row[f"diff_{i}"]  = float(diff[i])
            records.append(row)

    log.info(f"Extracted {len(records)} games | skipped: {skipped}")
    df = pd.DataFrame(records)
    df.to_parquet(EMBED_CACHE, index=False)
    log.info(f"✓ Saved to {EMBED_CACHE} | splits: {df['split'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────
# Step C — Downstream head training
# ─────────────────────────────────────────────────────────────────

def get_xy(df: pd.DataFrame, split: str,
           include_home: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for a split. X is 65-dim [diff(64) | is_home_a(1)]."""
    sub  = df[df["split"] == split]
    # Exclude games where home team is unknown (is_home_a == -1)
    if include_home:
        sub = sub[sub["is_home_a"] >= 0]
    cols = [f"diff_{i}" for i in range(EMB_DIM)]
    X = sub[cols].values.astype(np.float32)
    if include_home:
        X = np.concatenate([X, sub[["is_home_a"]].values.astype(np.float32)], axis=1)
    y = sub["label_a"].values.astype(np.float32)
    return X, y


def run_logistic_probe(df: pd.DataFrame) -> Tuple:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss

    log.info("\n── Logistic Regression (linear probe, L2 regularized, 65-dim) ──")

    X_tr, y_tr = get_xy(df, "train")
    X_va, y_va = get_xy(df, "val")
    X_te, y_te = get_xy(df, "test")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegressionCV(
        cv=5, max_iter=2000, solver="lbfgs",
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        scoring="accuracy", random_state=42,
    )
    clf.fit(X_tr_s, y_tr)
    log.info(f"  Best C: {clf.C_[0]:.4f}")

    for name, Xs, y in [("train", X_tr_s, y_tr), ("val", X_va_s, y_va), ("test", X_te_s, y_te)]:
        acc = accuracy_score(y, clf.predict(Xs))
        ll  = log_loss(y, clf.predict_proba(Xs)[:, 1])
        log.info(f"  [{name:5}] Accuracy: {acc*100:.1f}%  Log-loss: {ll:.4f}  (n={len(y)})")

    # Confidence calibration table
    log.info("\n  Confidence calibration (test split):")
    probs_te  = clf.predict_proba(X_te_s)[:, 1]
    conf      = np.maximum(probs_te, 1 - probs_te)
    preds_te  = (probs_te >= 0.5).astype(int)
    correct   = (preds_te == y_te.astype(int))
    bins      = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
    bin_labels= ["50–55%", "55–60%", "60–65%", "65–70%", "70–80%", "80%+"]
    for i, lbl in enumerate(bin_labels):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() < 3:
            continue
        emp = float(correct[mask].mean())
        avg_c = float(conf[mask].mean())
        log.info(f"    Confidence {lbl:>8}: empirical acc = {emp*100:.1f}%  "
                 f"avg_conf = {avg_c*100:.1f}%  (n = {mask.sum()})")

    return clf, scaler, probs_te, y_te


def run_mlp_probe(df: pd.DataFrame, epochs: int = 150, hidden: int = 64,
                  dropout: float = 0.35) -> Tuple:
    log.info("\n── MLP Probe (2-layer, 65-dim input, BCE loss) ──")

    X_tr, y_tr = get_xy(df, "train")
    X_va, y_va = get_xy(df, "val")
    X_te, y_te = get_xy(df, "test")

    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - mu) / sd; X_va_s = (X_va - mu) / sd; X_te_s = (X_te - mu) / sd

    X_tr_t = torch.FloatTensor(X_tr_s); y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1)
    X_va_t = torch.FloatTensor(X_va_s); y_va_t = torch.FloatTensor(y_va).unsqueeze(1)
    X_te_t = torch.FloatTensor(X_te_s); y_te_t = torch.FloatTensor(y_te).unsqueeze(1)

    class GameHeadMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(GAME_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
        def forward(self, x): return self.net(x)

    mlp    = GameHeadMLP()
    optim  = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn= nn.BCEWithLogitsLoss()

    best_val_acc = 0.0; best_state = None; no_improve = 0; patience = 30

    for epoch in range(1, epochs + 1):
        mlp.train()
        optim.zero_grad()
        loss = loss_fn(mlp(X_tr_t), y_tr_t)
        loss.backward(); optim.step(); sched.step()

        mlp.eval()
        with torch.no_grad():
            val_acc = float((torch.sigmoid(mlp(X_va_t)) >= 0.5).float().eq(y_va_t).float().mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_state = {k: v.clone() for k, v in mlp.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stop at epoch {epoch}"); break
        if epoch % 25 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:>4}/{epochs}  loss={loss.item():.4f}  val_acc={val_acc*100:.1f}%")

    if best_state:
        mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        for name, Xt, yt in [("train", X_tr_t, y_tr_t), ("val", X_va_t, y_va_t), ("test", X_te_t, y_te_t)]:
            acc = float((torch.sigmoid(mlp(Xt)) >= 0.5).float().eq(yt).float().mean())
            log.info(f"  [{name:5}] Accuracy: {acc*100:.1f}%  (n={len(yt)})")

    return mlp, (mu, sd)


# ─────────────────────────────────────────────────────────────────
# Step D — High-confidence margin analysis
# ─────────────────────────────────────────────────────────────────

def analyse_high_confidence_margins(df: pd.DataFrame, clf, scaler) -> None:
    """
    For the test split, compare:
      - High-confidence games (LR predicted prob ≥ CONF_THRESHOLD for either team)
      - vs. the rest
    on actual game margin, accuracy, and season distribution.
    """
    log.info(f"\n── High-Confidence Analysis (≥{CONF_THRESHOLD*100:.0f}% predicted probability) ──")

    te_full  = df[df["split"] == "test"].copy()
    te       = te_full[te_full["is_home_a"] >= 0].copy()
    cols     = [f"diff_{i}" for i in range(EMB_DIM)] + ["is_home_a"]
    X_te     = scaler.transform(te[cols].values.astype(np.float32))
    probs_a  = clf.predict_proba(X_te)[:, 1]         # P(team_a wins)
    conf     = np.maximum(probs_a, 1 - probs_a)       # max(P_a, P_b)
    preds    = (probs_a >= 0.5).astype(int)
    correct  = (preds == te["label_a"].values.astype(int))

    te["conf"]       = conf
    te["pred_winner_a"] = preds
    te["correct"]    = correct
    te["margin_abs"] = te["margin"].abs()

    # Split into high / low confidence
    hi = te[te["conf"] >= CONF_THRESHOLD]
    lo = te[te["conf"] <  CONF_THRESHOLD]

    n_hi = len(hi); n_lo = len(lo)
    log.info(f"\n  High-confidence games (≥{CONF_THRESHOLD*100:.0f}%): {n_hi} ({n_hi/len(te)*100:.1f}% of test)")
    log.info(f"  Low-confidence games  (< {CONF_THRESHOLD*100:.0f}%): {n_lo}")

    if n_hi > 0:
        log.info(f"\n  Accuracy:")
        log.info(f"    High-confidence: {hi['correct'].mean()*100:.1f}%  "
                 f"(avg predicted conf {hi['conf'].mean()*100:.1f}%)")
        log.info(f"    Low-confidence:  {lo['correct'].mean()*100:.1f}%  "
                 f"(avg predicted conf {lo['conf'].mean()*100:.1f}%)")

        log.info(f"\n  Actual margin distribution for HIGH-confidence calls:")
        log.info(f"    Mean actual margin:   {hi['margin_abs'].mean():.1f} pts")
        log.info(f"    Median actual margin: {hi['margin_abs'].median():.1f} pts")

        log.info(f"\n  Actual margin distribution for LOW-confidence calls:")
        log.info(f"    Mean actual margin:   {lo['margin_abs'].mean():.1f} pts")
        log.info(f"    Median actual margin: {lo['margin_abs'].median():.1f} pts")

        # Margin bucket breakdown for high-confidence games
        log.info(f"\n  High-confidence calls by actual game margin:")
        margin_bins   = [0, 5, 10, 15, 20, float("inf")]
        margin_labels = ["< 5", " 5–10", "10–15", "15–20", "  20+"]
        for lo_b, hi_b, lbl in zip(margin_bins, margin_bins[1:], margin_labels):
            mask = (hi["margin_abs"] >= lo_b) & (hi["margin_abs"] < hi_b)
            if mask.sum() == 0:
                continue
            sub_acc = hi.loc[mask, "correct"].mean()
            log.info(f"    Actual margin {lbl} pts: {mask.sum():>4} games  "
                     f"accuracy = {sub_acc*100:.1f}%")

        # Season breakdown
        log.info(f"\n  High-confidence calls by season:")
        hi_copy = hi.copy()
        hi_copy["season"] = hi_copy["game_date"].apply(
            lambda d: f"{d[:4]}-{str(int(d[:4])+1)[-2:]}" if int(d[5:7]) >= 10
                      else f"{int(d[:4])-1}-{d[2:4]}"
        )
        for season, grp in hi_copy.groupby("season"):
            log.info(f"    {season}: {grp['correct'].mean()*100:.1f}%  (n={len(grp)})")

        # Most-confident correct and incorrect calls
        log.info(f"\n  Top 5 most-confident CORRECT calls (test set):")
        top_correct = hi[hi["correct"]].nlargest(5, "conf")
        for _, r in top_correct.iterrows():
            winner = r["team_a"] if r["pred_winner_a"] else r["team_b"]
            log.info(f"    {r['game_date']}  {r['team_a']} vs {r['team_b']}  "
                     f"predicted {winner} ({r['conf']*100:.1f}% conf)  "
                     f"actual margin = {r['margin_abs']:.0f} pts")

        log.info(f"\n  Top 5 most-confident WRONG calls (test set):")
        top_wrong = hi[~hi["correct"]].nlargest(5, "conf")
        for _, r in top_wrong.iterrows():
            pred = r["team_a"] if r["pred_winner_a"] else r["team_b"]
            actual = r["team_a"] if r["label_a"] else r["team_b"]
            log.info(f"    {r['game_date']}  {r['team_a']} vs {r['team_b']}  "
                     f"predicted {pred} ({r['conf']*100:.1f}% conf)  "
                     f"actual winner = {actual}  margin = {r['margin_abs']:.0f} pts")

        # Key question: do high-confidence wrong calls tend to be close games (upsets)?
        wrong_hi   = hi[~hi["correct"]]["margin_abs"]
        correct_hi = hi[ hi["correct"]]["margin_abs"]
        log.info(f"\n  Avg actual margin of HIGH-conf games:")
        log.info(f"    Correct predictions: {correct_hi.mean():.1f} pts")
        log.info(f"    Wrong predictions:   {wrong_hi.mean():.1f} pts")
        if wrong_hi.mean() < correct_hi.mean() - 3:
            log.info(f"    → Wrong calls tend to be UPSETS (close games model got wrong)")
        else:
            log.info(f"    → Wrong calls are NOT systematically closer — model has blind spots")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true",
                        help="Use cached data/game_embeddings.parquet.")
    parser.add_argument("--skip-home-cache", action="store_true",
                        help="Re-fetch home/away from LeagueGameFinder even if cache exists.")
    parser.add_argument("--cutoff", default="2026-03-05",
                        help="Only include game dates ≤ this. Default: 2026-03-05")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    log.info(f"Model:   {MODEL_DIR.name}")
    log.info(f"Cutoff:  {args.cutoff}")

    # ── Step A ────────────────────────────────────────────────────
    log.info("\nStep A — Fetching home/away lookup...")
    home_lookup = fetch_home_teams(use_cache=not args.skip_home_cache)
    log.info(f"  {len(home_lookup)} games with home-team data")

    # ── Step B ────────────────────────────────────────────────────
    if args.skip_extract and EMBED_CACHE.exists():
        log.info(f"\nStep B — Loading cached embeddings from {EMBED_CACHE}")
        df = pd.read_parquet(EMBED_CACHE)
        log.info(f"  {len(df)} games | {df['split'].value_counts().to_dict()}")
    else:
        log.info("\nStep B — Extracting embeddings from backbone...")
        df = extract_embeddings(cutoff=args.cutoff, home_lookup=home_lookup)

    if df.empty:
        log.error("No games extracted."); return

    # ── Step C ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Step C — Training downstream heads (65-dim: diff + is_home_a)")
    log.info("=" * 60)

    clf, scaler, probs_te, y_te = run_logistic_probe(df)
    mlp, norm = run_mlp_probe(df, epochs=args.epochs, hidden=args.hidden)

    # ── Step D ────────────────────────────────────────────────────
    analyse_high_confidence_margins(df, clf, scaler)

    log.info(f"\n✓ Embeddings cached at: {EMBED_CACHE}")
    log.info(f"  Re-run with --skip-extract to retrain heads only.")
    log.info(f"  Re-run without flags after new model weights to re-extract.")


if __name__ == "__main__":
    main()

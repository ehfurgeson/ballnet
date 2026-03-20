"""
predictor.py — Core GATv2TCN Predictor
=======================================
Shared by live.py and backtest.py. Never import from scripts/.

Usage:
    from predictor import GATv2Predictor
    p = GATv2Predictor()
    p.setup()   # loads all artifacts from data/ and models/<ACTIVE_MODEL>/
    p_over = p.predict_conformal_probability(player_id=203500, stat="PTS", threshold=22.5)
"""
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import (DATA_DIR, MODEL_DIR, GATV2_SRC, FEATURE_COLS,
                    PREDICTION_COLS, PRED_INDICES, SEQ_LENGTH, QUANTILE_REGRESSION)

# ── LOG_TRANSFORM inference flag ──────────────────────────────────
# When log1p is applied to BOTH pred and target in the training loss
# (the current implementation in 03_train.py), the model learns to output
# real-space values (pred ≈ target). No expm1 inversion is needed.
# Only set True if the model was trained with log1p on TARGETS ONLY.
LOG_TRANSFORM = False

sys.path.insert(0, str(GATV2_SRC))
from gatv2tcn import GATv2TCN


class GATv2Predictor:
    """
    Loads the trained GATv2TCN model and all inference artefacts.
    Active model is determined by config.ACTIVE_MODEL → config.MODEL_DIR.

    Methods:
      setup()                                          — load everything from disk
      get_data_freshness() → str                       — most recent game date loaded
      get_recent_games_count(player_id, n_days=7) → int
      predict_point_estimate(player_id, stat) → float
      predict_conformal_probability(player_id, stat, threshold) → dict
    """

    N_IN  = len(FEATURE_COLS) + 2 + 2   # 13 stats + 2 team emb + 2 pos emb
    N_OUT = len(PREDICTION_COLS) * 3 if QUANTILE_REGRESSION else len(PREDICTION_COLS)
    N_POS = 3

    def __init__(self):
        self.setup_complete = False
        self.device = (
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )

    # ─────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────
    def setup(self) -> None:
        """Load all artifacts and model weights into memory."""
        # ── Data artefacts ────────────────────────────────────────
        self.X_seq         = pickle.load(open(DATA_DIR / "X_seq.pkl",         "rb"))
        self.G_seq         = pickle.load(open(DATA_DIR / "G_seq.pkl",         "rb"))
        self.player_ids    = pickle.load(open(DATA_DIR / "player_ids.pkl",    "rb"))
        self.game_dates    = pickle.load(open(DATA_DIR / "game_dates.pkl",    "rb"))
        self.team_temporal = pickle.load(open(DATA_DIR / "team_temporal.pkl", "rb"))
        self.pos_temporal  = pickle.load(open(DATA_DIR / "pos_temporal.pkl",  "rb"))
        self.n_teams       = pickle.load(open(DATA_DIR / "n_teams.pkl",       "rb"))
        self.mu_per_day    = np.load(DATA_DIR / "mu_per_day.npy")
        self.sd_per_day    = np.load(DATA_DIR / "sd_per_day.npy")

        self.pid_to_idx    = {pid: i for i, pid in enumerate(self.player_ids)}

        # ── Graph edges ───────────────────────────────────────────
        self.G_edges = [self._graph_to_edges(G) for G in self.G_seq]

        # ── Empirical calibration buckets ─────────────────────────
        # Format: {"version": "empirical_v1", "PTS": [{"lo":0,"hi":5,"actuals":[...]}, ...}, ...}
        # Legacy formats (tiered residuals, flat residuals) are detected and handled.
        try:
            _cal_data = pickle.load(
                open(MODEL_DIR / "conformal_residuals.pkl", "rb")
            )
            if isinstance(_cal_data, dict) and _cal_data.get("version") == "empirical_v1":
                # Current empirical bucket format
                self.empirical_buckets: Dict[str, list] = {
                    stat: _cal_data[stat] for stat in PREDICTION_COLS if stat in _cal_data
                }
                # Pre-compute sorted lo-edge arrays for fast binary search
                import bisect
                self._bucket_lo_edges: Dict[str, list] = {
                    stat: [b["lo"] for b in buckets]
                    for stat, buckets in self.empirical_buckets.items()
                }
                self._bisect = bisect
            elif isinstance(_cal_data, dict) and "residuals" in _cal_data:
                # Previous tiered-residual format — wrap into synthetic buckets by tier name
                # so get_residual_std() still works; predict_conformal_probability will warn.
                self.empirical_buckets = _cal_data  # store raw for compat
                self._bucket_lo_edges  = {}
            else:
                # Legacy flat {stat: [residuals]} format
                self.empirical_buckets = _cal_data
                self._bucket_lo_edges  = {}
        except FileNotFoundError:
            self.empirical_buckets = {}
            self._bucket_lo_edges = {}
            if not QUANTILE_REGRESSION:
                print(f"[Warning] conformal_residuals.pkl not found in {MODEL_DIR.name}.")

        # ── Model weights ─────────────────────────────────────────
        self.team_embedding = torch.nn.Linear(self.n_teams, 2).to(self.device)
        self.pos_embedding  = torch.nn.Linear(self.N_POS,   2).to(self.device)
        self.model          = GATv2TCN(
            in_channels        = self.N_IN,
            out_channels       = self.N_OUT,
            len_input          = SEQ_LENGTH,
            len_output         = 1,
            temporal_filter    = 64,
            out_gatv2conv      = 32,
            dropout_tcn        = 0.25,
            dropout_gatv2conv  = 0.5,
            head_gatv2conv     = 4,
        ).to(self.device)

        self.team_embedding.load_state_dict(
            torch.load(MODEL_DIR / "team_emb.pth", map_location=self.device))
        self.pos_embedding.load_state_dict(
            torch.load(MODEL_DIR / "pos_emb.pth",  map_location=self.device))
        self.model.load_state_dict(
            torch.load(MODEL_DIR / "model.pth",     map_location=self.device))

        self.setup_complete = True
        print(f"[Predictor] Loaded model from {MODEL_DIR.name}. "
              f"Data: {len(self.game_dates)} days, most recent: {self.game_dates[-1]}")

    def _ensure_setup(self):
        if not self.setup_complete:
            self.setup()

    # ─────────────────────────────────────────────────────────────
    # Empirical bucket helpers
    # ─────────────────────────────────────────────────────────────

    def _get_bucket(self, stat: str, pred_val: float) -> Optional[dict]:
        """
        Binary-search for the merged bucket whose [lo, hi) range contains pred_val.
        Returns the bucket dict {"lo", "hi", "actuals"}, or None if unavailable.
        """
        buckets   = self.empirical_buckets.get(stat)
        lo_edges  = self._bucket_lo_edges.get(stat)
        if not buckets or lo_edges is None:
            return None
        # bisect_right gives the insertion point; step back one to get the containing bucket
        idx = self._bisect.bisect_right(lo_edges, pred_val) - 1
        idx = max(0, min(idx, len(buckets) - 1))
        return buckets[idx]

    def get_residual_std(self, stat: str) -> float:
        """
        Return the std of all actual values across all buckets for this stat.
        Used by quantile_test.py and live.py as the SD-distance denominator.
        """
        self._ensure_setup()
        buckets = self.empirical_buckets.get(stat, [])
        if not buckets or not isinstance(buckets, list):
            return 1.0
        all_actuals = []
        for b in buckets:
            if isinstance(b, dict) and "actuals" in b:
                all_actuals.extend(b["actuals"])
        return float(np.std(all_actuals)) if all_actuals else 1.0

    # ─────────────────────────────────────────────────────────────
    # Utility helpers
    # ─────────────────────────────────────────────────────────────
    def _graph_to_edges(self, G):
        nd = {pid: i for i, pid in enumerate(self.player_ids)}
        edges = list(G.edges())
        if not edges:
            n = len(self.player_ids)
            return torch.stack([torch.arange(n), torch.arange(n)]).long().to(self.device)
        s, d = zip(*edges)
        s = [nd.get(x, 0) for x in s]; d = [nd.get(x, 0) for x in d]
        return torch.stack([torch.LongTensor(s + d), torch.LongTensor(d + s)]).to(self.device)

    def _build_input(self, day_idx: int):
        """Build (1, P, N_IN, SEQ_LENGTH) input tensor for a given day index."""
        team_t = torch.FloatTensor(self.team_temporal[day_idx]).to(self.device)
        pos_t  = torch.FloatTensor(self.pos_temporal[day_idx]).to(self.device)
        with torch.no_grad():
            tv = self.team_embedding(team_t)
            pv = self.pos_embedding(pos_t)
            Xl = []
            for abs_day in range(day_idx - SEQ_LENGTH + 1, day_idx + 1):
                x_t = torch.cat([
                    torch.FloatTensor(self.X_seq[abs_day]).to(self.device), tv, pv
                ], dim=1)
                Xl.append(x_t)
            x_input = torch.stack(Xl, dim=-1)[None, ...]
        g_window = self.G_edges[day_idx - SEQ_LENGTH + 1 : day_idx + 1]
        return x_input, g_window

    def get_data_freshness(self) -> str:
        """Returns the most recent game date in the loaded dataset."""
        self._ensure_setup()
        return self.game_dates[-1] if self.game_dates else "unknown"

    def get_recent_games_count(self, player_id: int, n_days: int = 7) -> int:
        """
        Returns how many of the last n_days had non-zero stats for this player.
        A count of 0 over 7 days is a strong signal of injury / suspension.
        Uses X_seq (raw forward-filled) — non-zero means the player actually played.
        """
        self._ensure_setup()
        pidx = self.pid_to_idx.get(player_id)
        if pidx is None:
            return 0
        latest = len(self.game_dates) - 1
        start  = max(0, latest - n_days + 1)
        window = self.X_seq[start : latest + 1, pidx, :]
        return int((window != 0).any(axis=-1).sum())

    # ─────────────────────────────────────────────────────────────
    # Day-level batch inference  (fast path — use this in backtest)
    # ─────────────────────────────────────────────────────────────
    def _get_day_idx_for_date(self, game_date: str) -> Optional[int]:
        """Return the index into game_dates for a YYYY-MM-DD string, or None."""
        self._ensure_setup()
        try:
            return self.game_dates.index(game_date)
        except ValueError:
            return None

    def predict_all_for_day(self, day_idx: int) -> Optional[np.ndarray]:
        """
        Single eval-mode forward pass for ALL players on day_idx.

        Returns an (P, N_OUT) array in raw stat units (de-normalized),
        or None if day_idx < SEQ_LENGTH.

        Result is cached — repeated calls for the same day are free.
        Backtest usage:
            pred_matrix = predictor.predict_all_for_day(day_idx)
            player_pts  = pred_matrix[pid_to_idx[player_id], PREDICTION_COLS.index("PTS")]
        """
        self._ensure_setup()
        if not hasattr(self, "_day_cache"):
            self._day_cache: dict = {}
        if day_idx in self._day_cache:
            return self._day_cache[day_idx]
        if day_idx < SEQ_LENGTH:
            return None
        x_input, g_window = self._build_input(day_idx)
        self.model.eval(); self.team_embedding.eval(); self.pos_embedding.eval()
        with torch.no_grad():
            pred_raw = self.model(x_input, g_window)[0].cpu().numpy()  # (P, N_OUT)
        if LOG_TRANSFORM:
            pred_raw = np.expm1(np.maximum(pred_raw, 0))  # invert log1p; clamp negatives first
        self._day_cache[day_idx] = pred_raw
        return pred_raw

    def predict_all_mc_for_day(
        self, day_idx: int, n_samples: int = 20
    ) -> Optional[np.ndarray]:
        """
        Run `n_samples` MC-dropout forward passes for ALL players on day_idx.

        Returns shape (n_samples, P, N_OUT) in raw stat units, or None if too early.
        Cached per (day_idx, n_samples) — repeated calls are free.

        Each player's slice across the n_samples axis gives the same distribution
        as running n_samples separate per-player passes, but ~P× faster because
        the full 805-node graph is evaluated once per sample instead of per player.
        """
        self._ensure_setup()
        cache_key = (day_idx, n_samples)
        if not hasattr(self, "_mc_cache"):
            self._mc_cache: dict = {}
        if cache_key in self._mc_cache:
            return self._mc_cache[cache_key]
        if day_idx < SEQ_LENGTH:
            return None
        x_input, g_window = self._build_input(day_idx)
        self.model.train(); self.team_embedding.train(); self.pos_embedding.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred_raw = self.model(x_input, g_window)[0].cpu().numpy()
                if LOG_TRANSFORM:
                    pred_raw = np.expm1(np.maximum(pred_raw, 0))
                samples.append(pred_raw)
        result = np.stack(samples, axis=0)  # (n_samples, P, N_OUT)
        self._mc_cache[cache_key] = result
        return result

    def clear_day_cache(self):
        """Free cached day predictions (call between days if RAM is tight)."""
        self._day_cache = getattr(self, "_day_cache", {})
        self._mc_cache  = getattr(self, "_mc_cache",  {})
        self._day_cache.clear()
        self._mc_cache.clear()

    # ─────────────────────────────────────────────────────────────
    # Per-player inference (convenience wrappers — used by live.py)
    # ─────────────────────────────────────────────────────────────
    def predict_point_estimate(self, player_id: int, stat: str) -> Optional[float]:
        """Single point estimate using model.eval() (no dropout)."""
        self._ensure_setup()
        pidx = self.pid_to_idx.get(player_id)
        if pidx is None or stat not in PREDICTION_COLS:
            return None
        si        = PREDICTION_COLS.index(stat)
        latest    = len(self.game_dates) - 1
        x_input, g_window = self._build_input(latest)
        self.model.eval(); self.team_embedding.eval(); self.pos_embedding.eval()
        with torch.no_grad():
            pred_raw = self.model(x_input, g_window)[0].cpu().numpy()
        if LOG_TRANSFORM:
            pred_raw = np.expm1(np.maximum(pred_raw, 0))
            
        if QUANTILE_REGRESSION:
            n_stats = len(PREDICTION_COLS)
            return float(pred_raw[pidx, 1 * n_stats + si])
        else:
            return float(pred_raw[pidx, si])

    def predict_conformal_probability(
        self, player_id: int, stat: str, threshold: float
    ) -> Dict[str, float]:
        """
        Empirical over/under probability via bucket lookup.

        1. Single deterministic forward pass (model.eval) → point estimate ŷ
        2. Binary-search the empirical bucket that contains ŷ
        3. P(over) = mean(actuals_in_bucket > threshold)

        No MC-dropout, no residual sampling, no bias term — the bucket's
        empirical distribution of true outcomes already captures all these effects.
        """
        self._ensure_setup()
        pidx = self.pid_to_idx.get(player_id)
        if pidx is None or stat not in PREDICTION_COLS:
            return {"p_over": 0.0, "p_under": 0.0}

        si       = PREDICTION_COLS.index(stat)
        latest   = len(self.game_dates) - 1
        x_input, g_window = self._build_input(latest)

        self.model.eval(); self.team_embedding.eval(); self.pos_embedding.eval()
        with torch.no_grad():
            pred_raw = self.model(x_input, g_window)[0].cpu().numpy()
        if LOG_TRANSFORM:
            pred_raw = np.expm1(np.maximum(pred_raw, 0))
            
        if QUANTILE_REGRESSION:
            import scipy.stats
            n_stats = len(PREDICTION_COLS)
            q10 = float(pred_raw[pidx, 0 * n_stats + si])
            q50 = float(pred_raw[pidx, 1 * n_stats + si])
            q90 = float(pred_raw[pidx, 2 * n_stats + si])
            std = (q90 - q10) / 2.563
            p_over = float(scipy.stats.norm.sf(threshold, loc=q50, scale=std))
            return {"p_over": p_over, "p_under": 1.0 - p_over}

        point_est = float(pred_raw[pidx, si])

        bucket = self._get_bucket(stat, point_est)
        if bucket is None or not bucket["actuals"]:
            return {"p_over": 0.0, "p_under": 0.0}

        actuals = np.array(bucket["actuals"])
        p_over  = float(np.mean(actuals > threshold))
        return {"p_over": p_over, "p_under": 1.0 - p_over}

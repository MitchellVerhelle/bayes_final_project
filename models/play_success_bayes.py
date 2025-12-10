# models/play_success_bayes.py

from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pymc as pm

from .play_success_base import PlaySuccessModel


class BayesianPlaySuccessModel(PlaySuccessModel):
    """
    Bayesian logistic regression for play success.

    - Fit on play-level table (one row per game_id/play_id).
    - For visualization, can produce frame-level P(success) by building
      frame-level features and applying the same logistic model.

    feature_cols:
      names of columns in both:
        - play_df (for fitting)
        - per-frame feature table (for frame_prob_dict).
    """

    def __init__(
        self,
        feature_cols: List[str],
        intercept_scale: float = 5.0,
        coef_scale: float = 5.0,
        nu: float = 3.0,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        chains: int = 2
    ):
        self.feature_cols = feature_cols
        self.intercept_scale = float(intercept_scale)
        self.coef_scale = float(coef_scale)
        self.nu = float(nu)
        self.draws = int(draws)
        self.tune = int(tune)
        self.target_accept = float(target_accept)
        self.chains = int(chains)
        self._feat_mean: np.ndarray | None = None,
        self._feat_std: np.ndarray | None = None,

        self.model: pm.Model | None = None
        self.trace: pm.InferenceData | None = None

        # flattened posterior samples for fast prediction
        self._beta0_samples: np.ndarray | None = None
        self._beta_samples: np.ndarray | None = None

    def fit(self, play_df: pd.DataFrame) -> None:
        """
        Fit Bayesian logistic regression.

        play_df must contain:
            - 'play_success' (0/1)
            - all columns listed in self.feature_cols.
        """
        X_raw = play_df[self.feature_cols].to_numpy(dtype=float)
        y = play_df["play_success"].to_numpy(dtype=int)
        n_features = X_raw.shape[1]

        # --- standardize features for better mixing ---
        self._feat_mean = X_raw.mean(axis=0)
        self._feat_std = X_raw.std(axis=0)
        # avoid zeros
        self._feat_std[self._feat_std == 0] = 1.0
        X = (X_raw - self._feat_mean) / self._feat_std

        with pm.Model() as model:
            beta0 = pm.Normal(
                "beta0",
                mu=0.0,
                sigma=2.0,   # tighter, weakly informative prior
            )
            beta = pm.Normal(
                "beta",
                mu=0.0,
                sigma=1.0,
                shape=n_features,
            )

            logits = beta0 + pm.math.dot(X, beta)
            p = pm.Deterministic("p", pm.math.sigmoid(logits))

            pm.Bernoulli("y_obs", p=p, observed=y)

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                target_accept=self.target_accept,
                chains=self.chains,
            )

        self.model = model
        self.trace = trace

        beta0_vals = trace.posterior["beta0"].values  # (chains, draws)
        beta_vals = trace.posterior["beta"].values    # (chains, draws, n_features)

        self._beta0_samples = beta0_vals.reshape(-1)
        self._beta_samples = beta_vals.reshape(-1, n_features)

    def prob_by_sequence(
        self,
        frames_pre: Dict[int, pd.DataFrame],
        frames_post: Dict[int, pd.DataFrame],
    ) -> Dict[int, float]:
        pre_list  = [frames_pre[k]  for k in sorted(frames_pre.keys())]
        post_list = [frames_post[k] for k in sorted(frames_post.keys())]

        seq: List[Tuple[str, pd.DataFrame]] = []
        for d in pre_list:
            seq.append(("pre", d))
        for d in post_list:
            seq.append(("post", d))

        p_by_step: Dict[int, float] = {}
        prefix_frames: List[pd.DataFrame] = []

        for i, (phase, d_frame) in enumerate(seq):
            prefix_frames.append(d_frame)
            prefix_df = pd.concat(prefix_frames, ignore_index=True)

            feat_row = self._build_prefix_features(prefix_df)
            p = float(self.predict_proba_row(feat_row))  # or _predict_proba_from_features
            p_by_step[i] = p

        return p_by_step


    def _predict_proba_from_features(
        self,
        X_new: np.ndarray,
    ) -> np.ndarray:
        """
        Posterior mean probability for each row in X_new.
        """
        if self._beta0_samples is None or self._beta_samples is None:
            raise RuntimeError("Call fit(...) before predicting.")

        X_new = np.asarray(X_new, dtype=float)
        # apply same z-scoring as in fit
        if self._feat_mean is not None and self._feat_std is not None:
            X_new = (X_new - self._feat_mean) / self._feat_std

        beta0 = self._beta0_samples      # (n_samples,)
        beta = self._beta_samples        # (n_samples, n_features)
        n_samples = beta0.shape[0]
        n_rows = X_new.shape[0]

        probs = np.zeros((n_samples, n_rows), dtype=float)
        for i in range(n_samples):
            logits = beta0[i] + X_new @ beta[i]
            probs[i, :] = 1.0 / (1.0 + np.exp(-logits))

        return probs.mean(axis=0)


    # ------------------------------------------------------------------
    # Frame-level feature builder for one play
    # ------------------------------------------------------------------
    def _build_frame_features_for_play(self, d_play: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-frame features for one play.

        Expects columns:
            frame_id, player_role, player_side,
            x, y, ball_land_x, ball_land_y, absolute_yardline_number
        """
        d_play = d_play.sort_values("frame_id").copy()

        # Target receiver per frame
        target = d_play[d_play["player_role"] == "Targeted Receiver"]
        d_t = (
            target[["frame_id", "x", "y", "ball_land_x", "ball_land_y"]]
            .rename(columns={"x": "tgt_x", "y": "tgt_y"})
        )

        # Base frame-level table
        d_frames = (
            d_play[["frame_id", "absolute_yardline_number"]]
            .drop_duplicates("frame_id")
            .copy()
        )
        d_frames = d_frames.merge(d_t, on="frame_id", how="left")

        # distance from target to ball landing
        d_frames["dist_target_to_land"] = np.sqrt(
            (d_frames["tgt_x"] - d_frames["ball_land_x"]) ** 2 +
            (d_frames["tgt_y"] - d_frames["ball_land_y"]) ** 2
        )

        # defenders near landing point per frame
        def_rows = []
        for f, g in d_play.groupby("frame_id", sort=True):
            defenders = g[g["player_side"] == "Defense"]
            if defenders.empty:
                n_close = 0
            else:
                dx = defenders["x"] - g["ball_land_x"].iloc[0]
                dy = defenders["y"] - g["ball_land_y"].iloc[0]
                d_def = np.sqrt(dx**2 + dy**2)
                n_close = int((d_def <= 5.0).sum())
            def_rows.append({"frame_id": f, "num_defenders_close": n_close})

        d_defc = pd.DataFrame(def_rows)
        d_frames = d_frames.merge(d_defc, on="frame_id", how="left")

        d_frames["abs_yardline_at_throw"] = d_frames["absolute_yardline_number"]

        # keep only needed feature columns + frame_id
        cols = ["frame_id"] + self.feature_cols
        return d_frames[cols].copy()
    def _build_prefix_features(self, d_prefix: pd.DataFrame) -> pd.DataFrame:
        """
        Build a single feature row using ONLY frames in d_prefix
        (all frames with frame_id <= some f_max).

        Returns a 1-row DataFrame with *all* columns in self.feature_cols:
          - abs_yardline_at_throw
          - dist_target_to_land
          - num_defenders_close
          - breakaway
          - tackle_range
          - red_zone
          - ball_x
          - ball_y
          - dist_to_nearest_defender
          - dist_to_ball_land_position
          - dist_to_bounds
        """
        d_prefix = d_prefix.sort_values("frame_id").copy()
        frames = d_prefix["frame_id"].unique()
        if len(frames) == 0:
            raise ValueError("d_prefix has no frames")

        f_max = frames.max()
        g = d_prefix[d_prefix["frame_id"] == f_max]

        # --- red_zone: from FIRST frame of the play (prefix includes it) ---
        first_frame = frames.min()
        first_rows = d_prefix[d_prefix["frame_id"] == first_frame]
        abs_yardline_start = float(first_rows["absolute_yardline_number"].iloc[0])
        red_zone = int(abs_yardline_start <= 10.0)

        # --- absolute yardline at current frame ---
        abs_yardline_at_throw = float(g["absolute_yardline_number"].iloc[0])

        # --- ball landing coordinates (constant for the play) ---
        ball_x_land = float(d_prefix["ball_land_x"].iloc[0])
        ball_y_land = float(d_prefix["ball_land_y"].iloc[0])

        # --- target receiver at current frame ---
        tgt = g[g["player_role"] == "Targeted Receiver"]
        if not tgt.empty:
            tgt_x = float(tgt["x"].iloc[0])
            tgt_y = float(tgt["y"].iloc[0])
        else:
            tgt_x = np.nan
            tgt_y = np.nan

        # distance from target to ball landing
        if not np.isnan(tgt_x):
            dist_target_to_land = float(
                np.sqrt((tgt_x - ball_x_land) ** 2 + (tgt_y - ball_y_land) ** 2)
            )
            dist_to_ball_land_position = dist_target_to_land
        else:
            # treat as "far"
            dist_target_to_land = 50.0
            dist_to_ball_land_position = 50.0

        # --- defenders at current frame ---
        defenders = g[g["player_side"] == "Defense"]
        if defenders.empty or np.isnan(tgt_x):
            dist_nearest_def = 50.0
            tackle_range = 0
        else:
            dx_def = defenders["x"].to_numpy(dtype=float) - tgt_x
            dy_def = defenders["y"].to_numpy(dtype=float) - tgt_y
            dists_def = np.sqrt(dx_def ** 2 + dy_def ** 2)
            dist_nearest_def = float(dists_def.min())
            tackle_range = int(dist_nearest_def <= 2.0)

        # defenders close to landing point (num_defenders_close, radius=5)
        if defenders.empty:
            num_def_close = 0
        else:
            dx_land = defenders["x"].to_numpy(dtype=float) - ball_x_land
            dy_land = defenders["y"].to_numpy(dtype=float) - ball_y_land
            d_def_land = np.sqrt(dx_land ** 2 + dy_land ** 2)
            num_def_close = int((d_def_land <= 5.0).sum())

        # --- breakaway (post-style definition, but using ONLY current frame) ---
        # 1 if target is closer to ball_land than every other player in frame
        if not tgt.empty:
            dx_tgt_ball = tgt_x - ball_x_land
            dy_tgt_ball = tgt_y - ball_y_land
            dist_tgt_ball = np.sqrt(dx_tgt_ball ** 2 + dy_tgt_ball ** 2)

            others = g[g["nfl_id"] != tgt["nfl_id"].iloc[0]]
            if not others.empty:
                dx_o = others["x"].to_numpy(dtype=float) - ball_x_land
                dy_o = others["y"].to_numpy(dtype=float) - ball_y_land
                d_o = np.sqrt(dx_o ** 2 + dy_o ** 2)
                breakaway = int(dist_tgt_ball < d_o.min())
            else:
                breakaway = 1
        else:
            breakaway = 0

        # --- ball_x, ball_y ---
        # Approximation:
        #  - if Passer present in current frame: ball at QB
        #  - else: ball at landing point (as if it's arrived)
        qb = g[g["player_role"] == "Passer"]
        if not qb.empty:
            ball_x = float(qb["x"].iloc[0])
            ball_y = float(qb["y"].iloc[0])
        else:
            ball_x = ball_x_land
            ball_y = ball_y_land

        # --- dist_to_bounds (short axis) ---
        field_width = 53.3
        if not np.isnan(tgt_y):
            dist_to_bounds = float(min(tgt_y, field_width - tgt_y))
        else:
            dist_to_bounds = field_width / 2.0

        row = {
            "abs_yardline_at_throw": abs_yardline_at_throw,
            "dist_target_to_land": dist_target_to_land,
            "num_defenders_close": num_def_close,
            "breakaway": breakaway,
            "tackle_range": tackle_range,
            "red_zone": red_zone,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "dist_to_nearest_defender": dist_nearest_def,
            "dist_to_ball_land_position": dist_to_ball_land_position,
            "dist_to_bounds": dist_to_bounds,
        }

        return pd.DataFrame([row])

    def frame_prob_dict(self, d_play: pd.DataFrame, debug: bool = False) -> Dict[int, float]:
        """
        For a single play's input DataFrame (all frames),
        return dict: frame_id -> P(success | data up to and including that frame).

        Implementation:
        - For each frame f:
            prefix = all rows with frame_id <= f
            feat_row = _build_prefix_features(prefix)
            p_f = P(success | feat_row) using the logistic posterior
        """
        d_play = d_play.sort_values("frame_id").copy()
        frame_ids = sorted(d_play["frame_id"].unique())

        probs_by_frame: Dict[int, float] = {}

        for f in frame_ids:
            prefix = d_play[d_play["frame_id"] <= f]
            feat_row = self._build_prefix_features(prefix)
            X_new = feat_row[self.feature_cols].to_numpy(dtype=float)

            mean_prob = self._predict_proba_from_features(X_new)[0]
            probs_by_frame[f] = float(mean_prob)

        if debug:
            print(f"{len(probs_by_frame.keys())} frames")
        return probs_by_frame



    def predict_proba_row(self, row: pd.Series | pd.DataFrame) -> float:
        """
        Compute posterior mean P(success) for a single feature row.

        row:
          - either a pandas Series with keys in self.feature_cols
          - or a single-row DataFrame with those columns
        """
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise ValueError("predict_proba_row expects a single row.")
            row = row.iloc[0]

        # extract feature vector in the right column order
        X_new = row[self.feature_cols].to_numpy(dtype=float).reshape(1, -1)

        prob = self._predict_proba_from_features(X_new)[0]
        return float(prob)


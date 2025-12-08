"""Hierarchical Bayesian extension of the kinematic model for player movement.

This model extends the basic Bayesian kinematic model by adding hierarchical
structure that allows for:
- Player-specific noise scales (some players are more predictable than others)
- Position-specific biases (WRs, RBs, CBs may have different movement patterns)
- Global priors that share information across all players/positions

Input:
    - position:        (x, y)_t              (yard, yard)
    - speed:           s                     (yard/s)
    - acceleration:    a                     (yard/s^2)
    - direction:       theta                 (deg)
    - player metadata: nfl_id, player_position, etc.
    - observed next-step positions:
          x_next_obs, y_next_obs             (yard, yard)

Output:
    - Posterior predictive distribution over (x, y)_{t+dt}
    - Hierarchical posterior samples for noise parameters
    - Mean or sampled predictions attached to a DataFrame

Algorithm:
    - Step 1: Compute deterministic kinematic prediction:
                (mu_x, mu_y) = f(x_t, y_t, s, a, theta)
    
    - Step 2: Hierarchical structure:
                # Global noise scales
                sigma_x_global ~ HalfNormal(1.0)
                sigma_y_global ~ HalfNormal(1.0)
                
                # Position-specific deviations
                for each position:
                    sigma_x_pos ~ Normal(sigma_x_global, 0.3)
                    sigma_y_pos ~ Normal(sigma_y_global, 0.3)
                    bias_x_pos ~ Normal(0, 0.2)
                    bias_y_pos ~ Normal(0, 0.2)
                
                # Player-specific deviations (nested in position)
                for each player:
                    sigma_x_player ~ Normal(sigma_x_pos[player.position], 0.2)
                    sigma_y_player ~ Normal(sigma_y_pos[player.position], 0.2)
    
    - Step 3: Likelihood:
                x_next_obs ~ Normal(mu_x + bias_x_pos, sigma_x_player)
                y_next_obs ~ Normal(mu_y + bias_y_pos, sigma_y_player)
    
    - Step 4: Posterior inference via PyMC NUTS
    
    - Step 5: Prediction with hierarchical uncertainty
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .kinematic import KinematicModel
from .bayes_model_base import BayesianMovementModel


class HierarchicalBayesianKinematicModel(BayesianMovementModel):
    def __init__(self, fps: float = 10.0, use_accel: bool = True, name: Optional[str] = None):
        super().__init__(name=name or "hierarchical_bayes_kinematic")
        self.base = KinematicModel(fps=fps, use_accel=use_accel)
        self.trace: Optional[pm.backends.base.MultiTrace] = None
        self.model: Optional[pm.Model] = None
        self.player_to_idx: Dict[int, int] = {}
        self.position_to_idx: Dict[str, int] = {}
        self.idx_to_position: Dict[int, str] = {}

    def fit_bayes(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        s_col: str = "s",
        a_col: str = "a",
        dir_col: str = "dir",
        x_next_col: str = "x_next",
        y_next_col: str = "y_next",
        player_id_col: str = "nfl_id",
        position_col: str = "player_position",
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        chains: int = 2,
        use_position_hierarchy: bool = True,
        use_player_hierarchy: bool = False,  # Can be expensive with many players
        # Speedup options
        max_samples: Optional[int] = None,  # Subsample data for faster training
        use_vi: bool = False,  # Use Variational Inference instead of MCMC (much faster)
        vi_n: int = 10000,  # Number of VI iterations if use_vi=True
    ) -> None:
        """
        Fit the hierarchical Bayesian kinematic model.
        
        Args:
            use_position_hierarchy: If True, model position-specific parameters
            use_player_hierarchy: If True, model player-specific parameters (nested in position)
            max_samples: If set, randomly subsample this many rows for faster training
            use_vi: If True, use Variational Inference (ADVI) instead of MCMC (much faster, less accurate)
            vi_n: Number of iterations for VI if use_vi=True
        """
        # Subsample data if requested (speedup #1)
        if max_samples is not None and len(df) > max_samples:
            print(f"[HierarchicalBayesianKinematicModel] Subsampling from {len(df):,} to {max_samples:,} rows for faster training")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        d = df[dir_col].to_numpy(dtype=float)

        x_next_obs = df[x_next_col].to_numpy(dtype=float)
        y_next_obs = df[y_next_col].to_numpy(dtype=float)

        # Compute kinematic predictions
        mu_x, mu_y = self.base._step_array(x, y, s, a, d)

        # Build position and player mappings
        if use_position_hierarchy and position_col in df.columns:
            positions = df[position_col].fillna("Unknown").astype(str)
            unique_positions = sorted(positions.unique())
            self.position_to_idx = {pos: i for i, pos in enumerate(unique_positions)}
            self.idx_to_position = {i: pos for pos, i in self.position_to_idx.items()}
            position_idx = np.array([self.position_to_idx[p] for p in positions])
            n_positions = len(unique_positions)
        else:
            position_idx = None
            n_positions = 0

        if use_player_hierarchy and player_id_col in df.columns:
            player_ids = df[player_id_col].astype(int)
            unique_players = sorted(player_ids.unique())
            self.player_to_idx = {pid: i for i, pid in enumerate(unique_players)}
            player_idx = np.array([self.player_to_idx[pid] for pid in player_ids])
            n_players = len(unique_players)
        else:
            player_idx = None
            n_players = 0

        with pm.Model() as model:
            # Global noise scales
            sigma_x_global = pm.HalfNormal("sigma_x_global", sigma=1.0)
            sigma_y_global = pm.HalfNormal("sigma_y_global", sigma=1.0)

            if use_position_hierarchy and n_positions > 0:
                # Position-specific noise scales (hierarchical)
                sigma_x_pos = pm.Normal(
                    "sigma_x_pos",
                    mu=sigma_x_global,
                    sigma=0.3,
                    shape=n_positions,
                )
                sigma_y_pos = pm.Normal(
                    "sigma_y_pos",
                    mu=sigma_y_global,
                    sigma=0.3,
                    shape=n_positions,
                )
                # Position-specific biases
                bias_x_pos = pm.Normal("bias_x_pos", mu=0.0, sigma=0.2, shape=n_positions)
                bias_y_pos = pm.Normal("bias_y_pos", mu=0.0, sigma=0.2, shape=n_positions)

                # Select position-specific parameters for each observation
                sigma_x_obs = sigma_x_pos[position_idx]
                sigma_y_obs = sigma_y_pos[position_idx]
                bias_x_obs = bias_x_pos[position_idx]
                bias_y_obs = bias_y_pos[position_idx]
            else:
                # No position hierarchy, use global
                sigma_x_obs = sigma_x_global
                sigma_y_obs = sigma_y_global
                bias_x_obs = 0.0
                bias_y_obs = 0.0

            if use_player_hierarchy and n_players > 0 and use_position_hierarchy:
                # Player-specific noise (nested in position)
                # For each player, get their position
                player_positions = df.groupby(player_id_col)[position_col].first()
                player_pos_idx = np.array([
                    self.position_to_idx.get(str(p), 0) 
                    for p in player_positions.index
                ])
                
                sigma_x_player = pm.Normal(
                    "sigma_x_player",
                    mu=sigma_x_pos[player_pos_idx],
                    sigma=0.2,
                    shape=n_players,
                )
                sigma_y_player = pm.Normal(
                    "sigma_y_player",
                    mu=sigma_y_pos[player_pos_idx],
                    sigma=0.2,
                    shape=n_players,
                )
                
                # Select player-specific parameters
                sigma_x_obs = sigma_x_player[player_idx]
                sigma_y_obs = sigma_y_player[player_idx]
            elif use_player_hierarchy and n_players > 0:
                # Player hierarchy without position hierarchy
                sigma_x_player = pm.Normal(
                    "sigma_x_player",
                    mu=sigma_x_global,
                    sigma=0.2,
                    shape=n_players,
                )
                sigma_y_player = pm.Normal(
                    "sigma_y_player",
                    mu=sigma_y_global,
                    sigma=0.2,
                    shape=n_players,
                )
                sigma_x_obs = sigma_x_player[player_idx]
                sigma_y_obs = sigma_y_player[player_idx]

            # Likelihood
            pm.Normal(
                "x_next",
                mu=mu_x + bias_x_obs,
                sigma=pt.abs(sigma_x_obs),  # Ensure positive
                observed=x_next_obs,
            )
            pm.Normal(
                "y_next",
                mu=mu_y + bias_y_obs,
                sigma=pt.abs(sigma_y_obs),  # Ensure positive
                observed=y_next_obs,
            )

            # Speedup #3: Use Variational Inference instead of MCMC
            if use_vi:
                print(f"[HierarchicalBayesianKinematicModel] Using Variational Inference (ADVI) with {vi_n} iterations...")
                approx = pm.fit(
                    method='advi',
                    n=vi_n,
                    progressbar=True,
                )
                trace = approx.sample(draws=draws)
                trace = pm.to_inferencedata(trace)
            else:
                # Standard MCMC sampling (slower but more accurate)
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    target_accept=target_accept,
                    chains=chains,
                    return_inferencedata=True,
                    progressbar=True,
                )

        self.model = model
        self.trace = trace

    def posterior_samples_for_rows(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        x_col: str = "x",
        y_col: str = "y",
        s_col: str = "s",
        a_col: str = "a",
        dir_col: str = "dir",
        player_id_col: str = "nfl_id",
        position_col: str = "player_position",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate posterior samples for next-step positions for each row.
        
        Uses hierarchical posterior means for noise scales.
        """
        if self.trace is None or self.model is None:
            raise RuntimeError("Call fit_bayes(...) before posterior sampling.")

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        d = df[dir_col].to_numpy(dtype=float)

        mu_x, mu_y = self.base._step_array(x, y, s, a, d)

        n_rows = len(df)

        # Get posterior means for noise scales
        # Try to use position/player-specific if available, otherwise global
        sigma_x_vals = np.zeros(n_rows)
        sigma_y_vals = np.zeros(n_rows)
        bias_x_vals = np.zeros(n_rows)
        bias_y_vals = np.zeros(n_rows)

        # Check what parameters are in the trace
        trace_vars = list(self.trace.posterior.data_vars.keys())

        if "sigma_x_pos" in trace_vars and position_col in df.columns:
            # Use position-specific parameters
            positions = df[position_col].fillna("Unknown").astype(str)
            for i, pos in enumerate(positions):
                if pos in self.position_to_idx:
                    pos_idx = self.position_to_idx[pos]
                    sigma_x_vals[i] = float(self.trace.posterior["sigma_x_pos"].mean().values[pos_idx])
                    sigma_y_vals[i] = float(self.trace.posterior["sigma_y_pos"].mean().values[pos_idx])
                    if "bias_x_pos" in trace_vars:
                        bias_x_vals[i] = float(self.trace.posterior["bias_x_pos"].mean().values[pos_idx])
                    if "bias_y_pos" in trace_vars:
                        bias_y_vals[i] = float(self.trace.posterior["bias_y_pos"].mean().values[pos_idx])
                else:
                    # Fallback to global
                    sigma_x_vals[i] = float(self.trace.posterior["sigma_x_global"].mean())
                    sigma_y_vals[i] = float(self.trace.posterior["sigma_y_global"].mean())
        elif "sigma_x_player" in trace_vars and player_id_col in df.columns:
            # Use player-specific parameters
            player_ids = df[player_id_col].astype(int)
            for i, pid in enumerate(player_ids):
                if pid in self.player_to_idx:
                    player_idx = self.player_to_idx[pid]
                    sigma_x_vals[i] = float(self.trace.posterior["sigma_x_player"].mean().values[player_idx])
                    sigma_y_vals[i] = float(self.trace.posterior["sigma_y_player"].mean().values[player_idx])
                else:
                    # Fallback to global
                    sigma_x_vals[i] = float(self.trace.posterior["sigma_x_global"].mean())
                    sigma_y_vals[i] = float(self.trace.posterior["sigma_y_global"].mean())
        else:
            # Use global parameters
            sigma_x_global = float(self.trace.posterior["sigma_x_global"].mean())
            sigma_y_global = float(self.trace.posterior["sigma_y_global"].mean())
            sigma_x_vals[:] = sigma_x_global
            sigma_y_vals[:] = sigma_y_global

        # Generate samples
        x_samps = np.random.normal(
            mu_x + bias_x_vals,
            np.abs(sigma_x_vals),
            size=(n_samples, n_rows),
        )
        y_samps = np.random.normal(
            mu_y + bias_y_vals,
            np.abs(sigma_y_vals),
            size=(n_samples, n_rows),
        )

        return x_samps, y_samps


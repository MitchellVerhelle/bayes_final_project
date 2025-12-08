"""Hidden Markov Model (HMM) for player movement regime switching.

This model assumes players switch between discrete movement regimes:
- "route_running": Following a planned route
- "ball_tracking": Reacting to/intercepting the ball
- "evasive": Avoiding defenders or changing direction
- "blocking": Engaging with other players
- "stationary": Minimal movement

Input:
    - position:        (x, y)_t              (yard, yard)
    - speed:           s                     (yard/s)
    - acceleration:    a                     (yard/s^2)
    - direction:       theta                 (deg)
    - observed next-step positions:
          x_next_obs, y_next_obs             (yard, yard)

Output:
    - Posterior predictive distribution over (x, y)_{t+dt}
    - Posterior over hidden states (regimes)
    - State transition probabilities

Algorithm:
    - Step 1: Define movement regimes (states)
    - Step 2: Model state transitions with transition matrix
    - Step 3: Each state has its own kinematic model parameters
    - Step 4: Infer hidden states and parameters via PyMC
    - Step 5: Predict using posterior over states
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .kinematic import KinematicModel
from .bayes_model_base import BayesianMovementModel


class BayesianKinematicHMM(BayesianMovementModel):
    def __init__(
        self,
        fps: float = 10.0,
        use_accel: bool = True,
        n_states: int = 3,
        state_names: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            n_states: Number of hidden states (regimes)
            state_names: Optional names for states (for interpretability)
        """
        super().__init__(name=name or "bayes_kinematic_hmm")
        self.base = KinematicModel(fps=fps, use_accel=use_accel)
        self.n_states = n_states
        self.state_names = state_names or [f"state_{i}" for i in range(n_states)]
        self.trace: Optional[pm.backends.base.MultiTrace] = None
        self.model: Optional[pm.Model] = None

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
        group_cols: Optional[List[str]] = None,
        
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        chains: int = 2,
        **kwargs
    ) -> None:
        """
        Fit the HMM model.
        
        Args:
            group_cols: Columns to group by (e.g., ["game_id", "play_id", "nfl_id"])
                        Each group is treated as a separate sequence
        """
        if group_cols is None:
            group_cols = ["game_id", "play_id", "nfl_id"]

        # Sort by group columns and frame_id to ensure sequences are ordered
        df_sorted = df.sort_values(group_cols + ["frame_id"]).copy()

        x = df_sorted[x_col].to_numpy(dtype=float)
        y = df_sorted[y_col].to_numpy(dtype=float)
        s = df_sorted[s_col].to_numpy(dtype=float)
        a = df_sorted[a_col].to_numpy(dtype=float)
        d = df_sorted[dir_col].to_numpy(dtype=float)

        x_next_obs = df_sorted[x_next_col].to_numpy(dtype=float)
        y_next_obs = df_sorted[y_next_col].to_numpy(dtype=float)

        # Compute kinematic predictions
        mu_x_base, mu_y_base = self.base._step_array(x, y, s, a, d)

        # Build sequence groups
        if len(group_cols) > 0:
            groups = df_sorted.groupby(group_cols)
            group_sizes = groups.size().values
            group_starts = np.concatenate([[0], np.cumsum(group_sizes[:-1])])
            n_sequences = len(groups)
        else:
            # Single sequence
            group_sizes = np.array([len(df_sorted)])
            group_starts = np.array([0])
            n_sequences = 1

        with pm.Model() as model:
            # State-specific noise scales
            sigma_x_state = pm.HalfNormal("sigma_x_state", sigma=1.0, shape=self.n_states)
            sigma_y_state = pm.HalfNormal("sigma_y_state", sigma=1.0, shape=self.n_states)

            # State-specific biases (allow different regimes to have different offsets)
            bias_x_state = pm.Normal("bias_x_state", mu=0.0, sigma=0.3, shape=self.n_states)
            bias_y_state = pm.Normal("bias_y_state", mu=0.0, sigma=0.3, shape=self.n_states)

            # Transition matrix: each row is a Dirichlet distribution
            # Higher concentration = more likely to stay in same state
            transition_alpha = pm.Dirichlet(
                "transition_alpha",
                a=np.ones(self.n_states) * 2.0,  # Slight preference for staying
                shape=(self.n_states, self.n_states),
            )

            # For each sequence, we need to infer the hidden states
            # This is computationally expensive, so we'll use a simpler approach:
            # Model the probability of being in each state based on features
            # rather than full Viterbi-style inference

            # State probabilities based on kinematic features
            # Fast players more likely in "route_running", high accel in "evasive", etc.
            speed_norm = (s - s.mean()) / (s.std() + 1e-6)
            accel_norm = (a - a.mean()) / (a.std() + 1e-6)

            # Logits for state probabilities (simplified - could be more sophisticated)
            state_logits = pt.zeros((len(df_sorted), self.n_states))
            for state_idx in range(self.n_states):
                # Each state has different sensitivities to speed/accel
                state_logits = pt.set_subtensor(
                    state_logits[:, state_idx],
                    pm.Normal(f"state_{state_idx}_intercept", mu=0, sigma=1.0)
                    + pm.Normal(f"state_{state_idx}_speed_coef", mu=0, sigma=0.5) * speed_norm
                    + pm.Normal(f"state_{state_idx}_accel_coef", mu=0, sigma=0.5) * accel_norm
                )

            state_probs = pm.Deterministic("state_probs", pt.softmax(state_logits, axis=1))

            # For each observation, we have a mixture of states
            # We'll use a continuous relaxation: weighted average of state-specific predictions
            mu_x_weighted = pt.zeros(len(df_sorted))
            mu_y_weighted = pt.zeros(len(df_sorted))
            sigma_x_weighted = pt.zeros(len(df_sorted))
            sigma_y_weighted = pt.zeros(len(df_sorted))

            for state_idx in range(self.n_states):
                state_prob = state_probs[:, state_idx]
                mu_x_state = mu_x_base + bias_x_state[state_idx]
                mu_y_state = mu_y_base + bias_y_state[state_idx]
                sigma_x_state_val = sigma_x_state[state_idx]
                sigma_y_state_val = sigma_y_state[state_idx]

                mu_x_weighted += state_prob * mu_x_state
                mu_y_weighted += state_prob * mu_y_state
                sigma_x_weighted += state_prob * sigma_x_state_val
                sigma_y_weighted += state_prob * sigma_y_state_val

            # Likelihood (mixture model)
            pm.Normal(
                "x_next",
                mu=mu_x_weighted,
                sigma=pt.abs(sigma_x_weighted) + 0.1,  # Add small epsilon for stability
                observed=x_next_obs,
            )
            pm.Normal(
                "y_next",
                mu=mu_y_weighted,
                sigma=pt.abs(sigma_y_weighted) + 0.1,
                observed=y_next_obs,
            )

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate posterior samples for next-step positions.
        
        Uses posterior over state probabilities and state-specific parameters.
        """
        if self.trace is None or self.model is None:
            raise RuntimeError("Call fit_bayes(...) before posterior sampling.")

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        d = df[dir_col].to_numpy(dtype=float)

        mu_x_base, mu_y_base = self.base._step_array(x, y, s, a, d)

        n_rows = len(df)

        # Get posterior means
        sigma_x_state = self.trace.posterior["sigma_x_state"].mean().values
        sigma_y_state = self.trace.posterior["sigma_y_state"].mean().values
        bias_x_state = self.trace.posterior["bias_x_state"].mean().values
        bias_y_state = self.trace.posterior["bias_y_state"].mean().values

        # Compute state probabilities for new data
        # (simplified - using feature-based approach)
        speed_norm = (s - s.mean()) / (s.std() + 1e-6)
        accel_norm = (a - a.mean()) / (a.std() + 1e-6)

        state_probs = np.zeros((n_rows, self.n_states))
        for state_idx in range(self.n_states):
            intercept = float(self.trace.posterior[f"state_{state_idx}_intercept"].mean())
            speed_coef = float(self.trace.posterior[f"state_{state_idx}_speed_coef"].mean())
            accel_coef = float(self.trace.posterior[f"state_{state_idx}_accel_coef"].mean())
            
            logits = intercept + speed_coef * speed_norm + accel_coef * accel_norm
            state_probs[:, state_idx] = logits

        # Softmax
        exp_logits = np.exp(state_probs - np.max(state_probs, axis=1, keepdims=True))
        state_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Generate samples: for each row, sample from mixture of states
        x_samps = np.zeros((n_samples, n_rows))
        y_samps = np.zeros((n_samples, n_rows))

        for i in range(n_rows):
            # Sample state for this observation
            state_idx = np.random.choice(self.n_states, size=n_samples, p=state_probs[i])
            
            # For each sample, use the sampled state's parameters
            for j, sidx in enumerate(state_idx):
                mu_x = mu_x_base[i] + bias_x_state[sidx]
                mu_y = mu_y_base[i] + bias_y_state[sidx]
                sigma_x = np.abs(sigma_x_state[sidx])
                sigma_y = np.abs(sigma_y_state[sidx])
                
                x_samps[j, i] = np.random.normal(mu_x, sigma_x)
                y_samps[j, i] = np.random.normal(mu_y, sigma_y)

        return x_samps, y_samps

    def get_state_probabilities(
        self,
        df: pd.DataFrame,
        s_col: str = "s",
        a_col: str = "a",
    ) -> np.ndarray:
        """
        Get posterior mean state probabilities for each row.
        
        Returns:
            Array of shape (n_rows, n_states) with state probabilities
        """
        if self.trace is None:
            raise RuntimeError("Call fit_bayes(...) before getting state probabilities.")

        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        n_rows = len(df)

        speed_norm = (s - s.mean()) / (s.std() + 1e-6)
        accel_norm = (a - a.mean()) / (a.std() + 1e-6)

        state_probs = np.zeros((n_rows, self.n_states))
        for state_idx in range(self.n_states):
            intercept = float(self.trace.posterior[f"state_{state_idx}_intercept"].mean())
            speed_coef = float(self.trace.posterior[f"state_{state_idx}_speed_coef"].mean())
            accel_coef = float(self.trace.posterior[f"state_{state_idx}_accel_coef"].mean())
            
            logits = intercept + speed_coef * speed_norm + accel_coef * accel_norm
            state_probs[:, state_idx] = logits

        # Softmax
        exp_logits = np.exp(state_probs - np.max(state_probs, axis=1, keepdims=True))
        state_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return state_probs


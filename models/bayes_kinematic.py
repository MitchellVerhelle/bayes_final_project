"""Bayesian kinematic model explanation:

Input:
    - position:        (x, y)_t              (yard, yard)
    - speed:           s                     (yard/s)
    - acceleration:    a                     (yard/s^2)
    - direction:       theta                 (deg)
    - observed next-step positions:
          x_next_obs, y_next_obs             (yard, yard)

Output:
    - Posterior predictive distribution over (x, y)_{t+dt}
    - Posterior samples for noise parameters (sigma_x, sigma_y)
    - Mean or sampled predictions attached to a DataFrame

Algorithm:
    - Step 1: Compute deterministic kinematic prediction:
                (mu_x, mu_y) = f(x_t, y_t, s, a, theta)
              using the baseline kinematic equation.

    - Step 2: Introduce probabilistic uncertainty via likelihood:
                x_next_obs ~ Normal(mu_x, sigma_x)
                y_next_obs ~ Normal(mu_y, sigma_y)

    - Step 3: Place priors on the noise scales:
                sigma_x ~ HalfNormal(1.0)
                sigma_y ~ HalfNormal(1.0)

    - Step 4: Construct a PyMC model and perform posterior sampling
              (NUTS or similar) to obtain posterior distributions over
              sigma_x, sigma_y and the latent noise structure.

    - Step 5: For prediction:
              - Either compute the posterior mean (default in
                predict_dataframe):
                    E[x_pred], E[y_pred]
              - Or generate full posterior predictive samples via
                posterior_samples_for_rows for uncertainty quantification.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pymc as pm

from .kinematic import KinematicModel
from .bayes_model_base import BayesianMovementModel


class BayesianKinematicModel(BayesianMovementModel):
    def __init__(self, fps: float = 10.0, use_accel: bool = True, name: Optional[str] = None):
        super().__init__(name=name or "bayes_kinematic")
        self.base = KinematicModel(fps=fps, use_accel=use_accel)
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
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        chains: int = 2,
        max_samples: Optional[int] = None,
        use_vi: bool = False,
        vi_n: int = 10000,
    ) -> None:
        """ Fit model on a step-level dataset. """
        if max_samples is not None and len(df) > max_samples:
            print(f"[BayesianKinematicModel] Subsampling from {len(df):,} to {max_samples:,} rows for faster training")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        d = df[dir_col].to_numpy(dtype=float)

        x_next_obs = df[x_next_col].to_numpy(dtype=float)
        y_next_obs = df[y_next_col].to_numpy(dtype=float)

        mu_x, mu_y = self.base._step_array(x, y, s, a, d)

        with pm.Model() as model:
            sigma_x = pm.HalfNormal("sigma_x", sigma=1.0)
            sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)

            pm.Normal("x_next", mu=mu_x, sigma=sigma_x, observed=x_next_obs)
            pm.Normal("y_next", mu=mu_y, sigma=sigma_y, observed=y_next_obs)

            # Variational Inference instead of MCMC if flagged
            if use_vi:
                print(f"[BayesianKinematicModel] Using Variational Inference (ADVI) with {vi_n} iterations...")
                approx = pm.fit(
                    method='advi',
                    n=vi_n,
                    progressbar=True,
                )
                trace_samples = approx.sample(draws=draws)
                try:
                    trace = az.convert_to_inference_data(trace_samples)
                except (AttributeError, TypeError):
                    try:
                        trace = az.from_pymc(trace_samples)
                    except AttributeError:
                        trace = az.convert_to_inference_data(trace_samples, group="posterior")
            else:
                # MCMC sampling
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
        """ Uses the deterministic kinematic mean plus Gaussian noise with sigma_x, sigma_y taken from the fitted posterior. """
        if self.trace is None or self.model is None:
            raise RuntimeError("Call fit_bayes(...) before posterior sampling.")

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        d = df[dir_col].to_numpy(dtype=float)

        mu_x, mu_y = self.base._step_array(x, y, s, a, d)

        # Posterior means of noise scales
        sigma_x = float(self.trace.posterior["sigma_x"].mean())
        sigma_y = float(self.trace.posterior["sigma_y"].mean())

        n_rows = len(df)
        x_samps = np.random.normal(mu_x, sigma_x, size=(n_samples, n_rows))
        y_samps = np.random.normal(mu_y, sigma_y, size=(n_samples, n_rows))
        return x_samps, y_samps

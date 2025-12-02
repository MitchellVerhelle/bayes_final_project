"""Abstract base class for Bayesian movement models.

Input:
    - Training data as a pandas DataFrame with required movement features
      and next-step targets, typically a step-level dataset derived from
      input_2023_w*.csv and output_2023_w*.csv, e.g.:
          game_id, play_id, nfl_id, frame_id,
          x, y, s, a, dir, ...
          x_next, y_next

    - New data as a pandas DataFrame for which posterior predictive
      positions should be generated, typically containing:
          game_id, play_id, nfl_id, frame_id,
          x, y, s, a, dir, ...

Output:
    - Posterior predictive summaries of next-step positions written to
      standardized columns (e.g., x_pred, y_pred), and optionally
      additional columns (e.g., quantiles) if subclasses choose.
    - Optional posterior samples for next-step positions for use in
      visualization or downstream analysis.
    - Predictions that can be converted to the official output format
      (game_id, play_id, nfl_id, frame_id, x, y) via MovementModel.to_output_format.

Behavior:
    - Extends MovementModel with Bayesian-specific methods for fitting and
      posterior prediction.
    - Requires subclasses to implement a Bayesian fitting procedure and a
      method for generating posterior predictive samples.
    - Provides a default predict_dataframe(...) that computes posterior
      means or medians over (x_{t+1}, y_{t+1}).
    - Provides helper methods to integrate posterior samples into
      real-time visualizations (e.g., plotting posterior clouds and means
      on a Matplotlib Axes during animations).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.axes

from .model_base import MovementModel


class BayesianMovementModel(MovementModel, ABC):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "bayesian_model")
        self.trace: Any = None
        self.model: Any = None

    # NEW: hook into generic pipeline
    def fit(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        s_col: str = "s",
        a_col: str = "a",
        dir_col: str = "dir",
        x_next_col: str = "x_next",
        y_next_col: str = "y_next",
        **kwargs: Any,
    ) -> None:
        """
        Generic fit hook for MovementModel-based pipelines.

        Just forwards to fit_bayes with reasonable defaults so that
        tools.pipeline.train_eval_model(...) works for Bayesian models too.
        """
        return self.fit_bayes(
            df,
            x_col=x_col,
            y_col=y_col,
            s_col=s_col,
            a_col=a_col,
            dir_col=dir_col,
            x_next_col=x_next_col,
            y_next_col=y_next_col,
            **kwargs,
        )

    @abstractmethod
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
        **kwargs: Any,
    ) -> None:
        """
        Fit the Bayesian model on a step-level dataset of consecutive positions.

        df is expected to contain:
            - current-step features: x, y, s, a, dir, ...
            - next-step targets:    x_next, y_next

        Subclasses should:
            - Construct a probabilistic model with priors (e.g. HalfNormal
              for noise scales) and a likelihood relating kinematic
              predictions to observed next-step positions.
            - Run posterior inference (e.g. MCMC or VI) and store state
              in self.model and self.trace.
        """
        ...

    @abstractmethod
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
        Generate posterior samples for next-step positions for each row.

        df is typically a subset of a step-level dataset (e.g., all rows
        for one frame, or a single player at a single frame).

        Returns:
            (x_samples, y_samples), each with shape (n_samples, n_rows),
            where each column corresponds to the posterior samples for
            a particular row in df.
        """
        ...

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        s_col: str = "s",
        a_col: str = "a",
        dir_col: str = "dir",
        out_x_col: str = "x_pred",
        out_y_col: str = "y_pred",
        summary: str = "mean",
        n_samples: int = 200,
    ) -> pd.DataFrame:
        """
        Compute posterior predictive summaries for each row.

        By default, returns posterior means or medians in x_pred, y_pred.
        The returned DataFrame preserves id columns such as:
            game_id, play_id, nfl_id, frame_id
        so predictions can be fed directly into visualization or formatted
        into competition-style output.
        """
        x_samps, y_samps = self.posterior_samples_for_rows(
            df, n_samples=n_samples,
            x_col=x_col, y_col=y_col, s_col=s_col, a_col=a_col, dir_col=dir_col,
        )

        if summary == "mean":
            x_summary = x_samps.mean(axis=0)
            y_summary = y_samps.mean(axis=0)
        elif summary == "median":
            x_summary = np.median(x_samps, axis=0)
            y_summary = np.median(y_samps, axis=0)
        else:
            raise ValueError(f"Unknown summary type: {summary}")

        out = df.copy()
        out[out_x_col] = x_summary
        out[out_y_col] = y_summary
        return out

    # --- Visualization helpers for real-time posterior integration ---

    def init_posterior_artists(
        self,
        ax: matplotlib.axes.Axes,
        sample_color: str = "purple",
        mean_color: str = "black",
        sample_size: float = 2.0,
        mean_size: float = 6.0,
        alpha: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Initialize scatter artists to visualize posterior samples and
        their mean on a Matplotlib Axes.

        Returns a dict of artists that can be updated frame-by-frame.
        """
        samples_artist, = ax.plot(
            [], [], "o",
            ms=sample_size,
            color=sample_color,
            alpha=alpha,
            label="Posterior samples",
        )
        mean_artist, = ax.plot(
            [], [], "o",
            ms=mean_size,
            color=mean_color,
            label="Posterior mean",
        )
        return {"samples": samples_artist, "mean": mean_artist}

    def update_posterior_artists(
        self,
        artists: Dict[str, Any],
        x_samples: np.ndarray,
        y_samples: np.ndarray,
    ) -> Tuple[Any, Any]:
        """
        Update posterior sample and mean artists for a single time step.

        Intended use:
            - Call posterior_samples_for_rows(...) for the relevant rows
            or a single player.
            - Extract that player's samples: x_samps[:, i], y_samps[:, i].
            - Call update_posterior_artists(...) inside a Matplotlib
            FuncAnimation update(...) to animate the posterior.
        """
        samples_artist = artists["samples"]
        mean_artist = artists["mean"]

        # x_samples, y_samples: 1D arrays of posterior draws for a single point
        samples_artist.set_data(x_samples, y_samples)

        mx = float(np.mean(x_samples))
        my = float(np.mean(y_samples))
        # set_data expects sequences, so wrap scalars in 1-element lists
        mean_artist.set_data([mx], [my])

        return samples_artist, mean_artist


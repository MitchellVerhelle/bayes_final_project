"""Baseline kinematic model for player movement.

Input:
    - position:     (x, y)_t         (yard, yard)
    - speed:        s                (yard/s)
    - acceleration: a                (yard/s^2)
    - direction:    theta            (deg)

Output:
    - position:     (x, y)_{t+dt}    (yard, yard)
      where dt = 1 / fps

Algorithm:
    - Step 1: Convert direction (theta) from degrees to radians.
    - Step 2: If acceleration is used, update speed using:
                v_next = s + a * dt
                v_avg  = 0.5 * (s + v_next)
              Otherwise, use v_avg = s.
    - Step 3: Compute displacement:
                dx = v_avg * cos(theta) * dt
                dy = v_avg * sin(theta) * dt
    - Step 4: Return updated position:
                x_{t+dt} = x_t + dx
                y_{t+dt} = y_t + dy

Notes:
    - This is a purely deterministic physics-based baseline.
    - No learning occurs; the model contains no trained parameters.
    - Used both as a benchmark and as a mean function for future Bayesian models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model_base import MovementModel


class KinematicModel(MovementModel):
    """
    Baseline kinematic model for player movement.

    Given position (x, y), speed s (yards/sec), acceleration a (yards/sec^2),
    and movement direction dir (degrees), predict next-step (x, y) assuming
    motion in a straight line over a fixed time step dt = 1 / fps.
    """

    def __init__(self, fps: float = 10.0, use_accel: bool = True, name: Optional[str] = None):
        super().__init__(name=name or "kinematic")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.use_accel = bool(use_accel)

    def _step_array(
        self,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        a: np.ndarray,
        direction_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = np.deg2rad(direction_deg)

        if self.use_accel:
            v_next = s + a * self.dt
            v_avg = 0.5 * (s + v_next)
        else:
            v_avg = s

        dx = v_avg * np.cos(theta) * self.dt
        dy = v_avg * np.sin(theta) * self.dt

        return x + dx, y + dy

    def predict_step(
        self,
        x: float,
        y: float,
        s: float,
        a: float,
        direction_deg: float,
    ) -> tuple[float, float]:
        x_arr = np.array([x], dtype=float)
        y_arr = np.array([y], dtype=float)
        s_arr = np.array([s], dtype=float)
        a_arr = np.array([a], dtype=float)
        d_arr = np.array([direction_deg], dtype=float)

        x_next, y_next = self._step_array(x_arr, y_arr, s_arr, a_arr, d_arr)
        return float(x_next[0]), float(y_next[0])

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
    ) -> pd.DataFrame:
        """
        Predict next-step positions for every row in df.

        df is expected to contain columns:
            x_col, y_col, s_col, a_col, dir_col
        as in the NFL Big Data Bowl tracking input.
        """
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        s = df[s_col].to_numpy(dtype=float)
        a = df[a_col].to_numpy(dtype=float)
        d = df[dir_col].to_numpy(dtype=float)

        x_next, y_next = self._step_array(x, y, s, a, d)

        out = df.copy()
        out[out_x_col] = x_next
        out[out_y_col] = y_next
        return out

    def predict_path(
        self,
        df: pd.DataFrame,
        horizon: int,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convenience wrapper around MovementModel.rollout_horizon.
        """
        return self.rollout_horizon(df, horizon=horizon, **kwargs)
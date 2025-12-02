"""Kinematic model with simple boundary-aware direction adjustment.

Input:
    - position:     (x, y)_t
    - speed:        s
    - acceleration: a
    - direction:    theta (deg)

Output:
    - position:     (x, y)_{t+dt}

Algorithm:
    - Same straight-line kinematics as KinematicModel, but when a player is
      near the field boundary and heading outward, the motion direction is
      projected tangentially to the boundary so the player "slides" along it
      instead of stepping out of bounds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .model_base import MovementModel


class KinematicBoundaryModel(MovementModel):
    def __init__(
        self,
        fps: float = 10.0,
        use_accel: bool = True,
        name: Optional[str] = None,
        x_min: float = 0.0,
        x_max: float = 120.0,
        y_min: float = 0.0,
        y_max: float = 53.3,
        boundary_margin: float = 1.0,
    ):
        super().__init__(name=name or "kinematic_with_boundaries")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.use_accel = bool(use_accel)

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.boundary_margin = float(boundary_margin)

    def _adjust_direction_for_boundaries(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dx_dir: np.ndarray,
        dy_dir: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.boundary_margin <= 0:
            return dx_dir, dy_dir

        m = self.boundary_margin

        dx_dir = dx_dir.copy()
        dy_dir = dy_dir.copy()

        # sidelines (y)
        mask_y_low = (y <= self.y_min + m) & (dy_dir < 0)
        mask_y_high = (y >= self.y_max - m) & (dy_dir > 0)
        mask_y = mask_y_low | mask_y_high
        dy_dir[mask_y] = 0.0

        # end lines (x)
        mask_x_low = (x <= self.x_min + m) & (dx_dir < 0)
        mask_x_high = (x >= self.x_max - m) & (dx_dir > 0)
        mask_x = mask_x_low | mask_x_high
        dx_dir[mask_x] = 0.0

        # renormalize
        norm = np.sqrt(dx_dir**2 + dy_dir**2)
        zero_mask = norm == 0
        norm_safe = norm.copy()
        norm_safe[zero_mask] = 1.0

        dx_dir = dx_dir / norm_safe
        dy_dir = dy_dir / norm_safe
        dx_dir[zero_mask] = 1.0
        dy_dir[zero_mask] = 0.0

        return dx_dir, dy_dir

    def _step_array(
        self,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        a: np.ndarray,
        direction_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = np.deg2rad(direction_deg)
        dx_dir = np.cos(theta)
        dy_dir = np.sin(theta)

        dx_dir, dy_dir = self._adjust_direction_for_boundaries(x, y, dx_dir, dy_dir)

        if self.use_accel:
            v_next = s + a * self.dt
            v_avg = 0.5 * (s + v_next)
        else:
            v_avg = s

        dx = v_avg * dx_dir * self.dt
        dy = v_avg * dy_dir * self.dt

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

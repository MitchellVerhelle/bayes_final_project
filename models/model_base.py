"""Contains abstract class for all prediction models.

Input:
    - A pandas DataFrame containing required movement features, typically
      derived from the NFL Big Data Bowl input files, e.g. columns:
          game_id, play_id, nfl_id, frame_id,
          x, y, s, a, dir, ...
      plus optional next-step targets:
          x_next, y_next
      when training or evaluating RMSE.

Output:
    - Predicted next-step positions written to standardized columns
      (x_pred, y_pred) in a DataFrame that still contains all original
      identifier columns (game_id, play_id, nfl_id, frame_id, ...).
    - Standardized RMSE evaluation metrics for comparison across models.
    - Optional helper to format predictions into the competition-style
      output schema:
          game_id, play_id, nfl_id, frame_id, x, y

Behavior:
    - Defines the unified interface that all movement models must implement.
    - Provides an abstract predict_dataframe(...) method that each model
      overrides with its own prediction logic.
    - Provides a default fit(...) (no-op) and a shared RMSE evaluator to
      ensure every model is measured consistently on a step-level dataset.
    - Provides a helper to convert model predictions into the official
      output format used by output_2023_w*.csv and final submissions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterable, List
import pandas as pd
import numpy as np


class MovementModel(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Optional training hook.

        df is typically a step-level dataset containing both features
        (x, y, s, a, dir, ...) and next-step targets (x_next, y_next).
        Baseline / analytic models may ignore this and override predict only.
        """
        return

    @abstractmethod
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
        Given a DataFrame of features at time t, return a copy with
        predicted next-step positions in out_x_col, out_y_col.

        The returned DataFrame should preserve identifier columns such as:
            game_id, play_id, nfl_id, frame_id
        so that predictions can be mapped back into the competition
        output format.
        """
        ...
    

    def rollout_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        *,
        id_cols: Iterable[str] = ("game_id", "play_id", "nfl_id", "frame_id"),
        x_col: str = "x",
        y_col: str = "y",
        s_col: str = "s",
        a_col: str = "a",
        dir_col: str = "dir",
        out_x_col: str = "x_pred",
        out_y_col: str = "y_pred",
        step_col: str = "h",
    ) -> pd.DataFrame:
        """
        Deterministic multi-step rollout using repeated one-step predictions.

        - df: state at time t (one row per player at that frame).
        - horizon: number of steps ahead (H). We generate predictions for
          t+1,...,t+H by repeatedly calling predict_dataframe and feeding
          predictions back in as the new x,y.

        Returns a long-form DataFrame with columns:
            id_cols..., step_col, x_pred, y_pred

        Subclasses that have their own multi-step logic can override this
        method entirely.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        id_cols = list(id_cols)
        missing_ids = [c for c in id_cols if c not in df.columns]
        if missing_ids:
            raise ValueError(f"Missing id columns in df: {missing_ids}")

        # Don't mutate caller's df
        current = df.copy()
        records: list[pd.DataFrame] = []

        for h in range(1, horizon + 1):
            step_pred = self.predict_dataframe(
                current,
                x_col=x_col,
                y_col=y_col,
                s_col=s_col,
                a_col=a_col,
                dir_col=dir_col,
                out_x_col=out_x_col,
                out_y_col=out_y_col,
            )

            rec = step_pred[id_cols + [out_x_col, out_y_col]].copy()
            rec[step_col] = h
            rec = rec.rename(columns={out_x_col: "x_pred", out_y_col: "y_pred"})
            records.append(rec)

            # Feed predictions back as new state for next step
            current = current.copy()
            current[x_col] = step_pred[out_x_col].to_numpy()
            current[y_col] = step_pred[out_y_col].to_numpy()

        out = pd.concat(records, ignore_index=True)
        return out

    def to_output_from_rollout(
        self,
        rollout_df: pd.DataFrame,
        *,
        id_cols: Iterable[str] = ("game_id", "play_id", "nfl_id", "frame_id"),
        step_col: str = "h",
        step: int = 1,
        x_pred_col: str = "x_pred",
        y_pred_col: str = "y_pred",
    ) -> pd.DataFrame:
        """
        Convert a multi-step rollout into competition-style output for a
        specific step (typically step=1).

        rollout_df is the long-form output from rollout_horizon.
        """
        if step_col not in rollout_df.columns:
            raise ValueError(f"rollout_df is missing step column '{step_col}'")

        subset = rollout_df[rollout_df[step_col] == step].copy()
        if subset.empty:
            raise ValueError(f"No rows found for {step_col} == {step}")

        return self.to_output_format(
            subset,
            id_cols=id_cols,
            x_pred_col=x_pred_col,
            y_pred_col=y_pred_col,
        )

    def rmse(
        self,
        df: pd.DataFrame,
        x_true_col: str = "x_next",
        y_true_col: str = "y_next",
        x_pred_col: str = "x_pred",
        y_pred_col: str = "y_pred",
    ) -> dict:
        """
        Compute RMSE on a step-level dataset.

        Expected usage:
            - df contains true next-step columns (x_next, y_next)
              built from output_2023_w*.csv.
            - df also contains model predictions (x_pred, y_pred)
              produced by predict_dataframe(...).
        """
        x_true = df[x_true_col].to_numpy(dtype=float)
        y_true = df[y_true_col].to_numpy(dtype=float)
        x_pred = df[x_pred_col].to_numpy(dtype=float)
        y_pred = df[y_pred_col].to_numpy(dtype=float)

        ex = x_true - x_pred
        ey = y_true - y_pred

        rmse_x = float(np.sqrt(np.mean(ex**2)))
        rmse_y = float(np.sqrt(np.mean(ey**2)))
        rmse_tot = float(np.sqrt(np.mean(ex**2 + ey**2)))

        return {
            "model": self.name,
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse": rmse_tot,
        }

    def to_output_format(
        self,
        df: pd.DataFrame,
        id_cols: Iterable[str] = ("game_id", "play_id", "nfl_id", "frame_id"),
        x_pred_col: str = "x_pred",
        y_pred_col: str = "y_pred",
    ) -> pd.DataFrame:
        """
        Format predictions into competition-style output format.

        Given a DataFrame that includes:
            - id columns: game_id, play_id, nfl_id, frame_id
            - predicted columns: x_pred, y_pred

        Returns a new DataFrame with columns:
            game_id, play_id, nfl_id, frame_id, x, y

        This matches the structure of output_2023_w*.csv and the required
        final submission format.
        """
        id_cols = list(id_cols)
        missing = [c for c in id_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing id columns in df: {missing}")

        if x_pred_col not in df.columns or y_pred_col not in df.columns:
            raise ValueError(
                f"Predicted columns '{x_pred_col}' and/or '{y_pred_col}' not found."
            )

        out = df[id_cols + [x_pred_col, y_pred_col]].copy()
        out = out.rename(columns={x_pred_col: "x", y_pred_col: "y"})
        return out

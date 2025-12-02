from __future__ import annotations

from typing import Iterable, Tuple, Dict, Optional, List
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from .model_base import MovementModel

def weeks_up_to(n: int) -> List[int]:
    """
    Convenience helper: weeks_up_to(3) -> [1, 2, 3].
    """
    if n < 1 or n > 18:
        raise ValueError("n must be between 1 and 18 (inclusive).")
    return list(range(1, n + 1))

def fit_model_up_to_week(
    model: MovementModel,
    max_train_week: int,
    *,
    source: str = "input",
    x_col: str = "x",
    y_col: str = "y",
    s_col: str = "s",
    a_col: str = "a",
    dir_col: str = "dir",
    **fit_kwargs,
) -> MovementModel:
    """
    Train (or fit_bayes) a MovementModel using weeks 1..max_train_week.
    """
    train_weeks = list(range(1, max_train_week + 1))
    print(f"[fit_model_up_to_week] Training on weeks: {train_weeks}")

    train_df = build_step_df(train_weeks, source=source)

    print(f"[fit_model_up_to_week] Fitting model '{model.name}' on "
          f"{len(train_df):,} rows...")
    model.fit(
        train_df,
        x_col=x_col,
        y_col=y_col,
        s_col=s_col,
        a_col=a_col,
        dir_col=dir_col,
        x_next_col="x_next",
        y_next_col="y_next",
        **fit_kwargs,
    )
    print(f"[fit_model_up_to_week] Done fitting '{model.name}'.")
    return model

def train_eval_until_week(
    model: MovementModel,
    max_train_week: int,
    *,
    test_weeks: Optional[Iterable[int]] = None,
    source: str = "input",
    x_col: str = "x",
    y_col: str = "y",
    s_col: str = "s",
    a_col: str = "a",
    dir_col: str = "dir",
    **fit_kwargs,
) -> Tuple[MovementModel, Dict[str, float]]:
    """
    Convenience wrapper around train_eval_model using weeks 1..max_train_week
    for training, and user-specified test_weeks (default: [max_train_week]).

    Example:
        train_eval_until_week(model, 3, test_weeks=[4,5])

    This works for both deterministic and Bayesian models.
    """
    train_weeks = weeks_up_to(max_train_week)
    if test_weeks is None:
        test_weeks = [max_train_week]

    # build data
    train_df = build_step_df(train_weeks, source=source)
    test_df = build_step_df(test_weeks, source=source)

    # fit (handles Bayesian via overridden fit)
    model.fit(
        train_df,
        x_col=x_col,
        y_col=y_col,
        s_col=s_col,
        a_col=a_col,
        dir_col=dir_col,
        x_next_col="x_next",
        y_next_col="y_next",
        **fit_kwargs,
    )

    # predict on test
    test_pred = model.predict_dataframe(
        test_df,
        x_col=x_col,
        y_col=y_col,
        s_col=s_col,
        a_col=a_col,
        dir_col=dir_col,
        out_x_col="x_pred",
        out_y_col="y_pred",
    )

    metrics = model.rmse(
        test_pred,
        x_true_col="x_next",
        y_true_col="y_next",
        x_pred_col="x_pred",
        y_pred_col="y_pred",
    )
    return model, metrics

def build_step_df_from_input(weeks: Iterable[int]) -> pd.DataFrame:
    dfs = []
    weeks = list(weeks)
    for w in tqdm(weeks, desc="Building step df (input)", unit="week"):
        df = pd.read_csv(f"train/input_2023_w{w:02d}.csv")
        df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

        df["x_next"] = df.groupby(
            ["game_id", "play_id", "nfl_id"]
        )["x"].shift(-1)
        df["y_next"] = df.groupby(
            ["game_id", "play_id", "nfl_id"]
        )["y"].shift(-1)

        step = df.dropna(subset=["x_next", "y_next"]).copy()
        step["week"] = w
        dfs.append(step)
    return pd.concat(dfs, ignore_index=True)


def build_step_df_from_output(weeks: Iterable[int]) -> pd.DataFrame:
    dfs = []
    weeks = list(weeks)
    for w in tqdm(weeks, desc="Building step df (output)", unit="week"):
        df = pd.read_csv(f"train/output_2023_w{w:02d}.csv")
        df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

        df["x_next"] = df.groupby(
            ["game_id", "play_id", "nfl_id"]
        )["x"].shift(-1)
        df["y_next"] = df.groupby(
            ["game_id", "play_id", "nfl_id"]
        )["y"].shift(-1)

        step = df.dropna(subset=["x_next", "y_next"]).copy()
        step["week"] = w
        dfs.append(step)
    return pd.concat(dfs, ignore_index=True)


def build_step_df(weeks: Iterable[int], source: str = "input") -> pd.DataFrame:
    if source == "input":
        return build_step_df_from_input(weeks)
    elif source == "output":
        return build_step_df_from_output(weeks)
    else:
        raise ValueError(f"Unknown source: {source}")


def train_eval_model(
    model: MovementModel,
    train_weeks: Iterable[int],
    test_weeks: Iterable[int],
    source: str = "input",
    x_col: str = "x",
    y_col: str = "y",
    s_col: str = "s",
    a_col: str = "a",
    dir_col: str = "dir",
) -> Tuple[MovementModel, Dict[str, float]]:
    train_df = build_step_df(train_weeks, source=source)
    test_df = build_step_df(test_weeks, source=source)

    model.fit(train_df, x_next_col="x_next", y_next_col="y_next")

    test_pred = model.predict_dataframe(
        test_df,
        x_col=x_col,
        y_col=y_col,
        s_col=s_col,
        a_col=a_col,
        dir_col=dir_col,
        out_x_col="x_pred",
        out_y_col="y_pred",
    )

    metrics = model.rmse(
        test_pred,
        x_true_col="x_next",
        y_true_col="y_next",
        x_pred_col="x_pred",
        y_pred_col="y_pred",
    )
    return model, metrics

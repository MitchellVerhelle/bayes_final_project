from __future__ import annotations

from typing import Iterable, Tuple, Dict, Optional, List, Union, Set
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from .model_base import MovementModel
from tools.tracking_utils import (
    load_play,
    frames_from_input,
    frames_from_output_merged,
)
from .play_success_bayes import BayesianPlaySuccessModel
from tools.tracking_utils import load_play, frames_from_input, frames_from_output_merged


def prob_by_step_for_play(
    ps_model: BayesianPlaySuccessModel,
    week: int,
    game_id: int,
    play_id: int,
) -> Dict[int, float]:
    """
    Return dict: step_index (0..T-1) -> P(success | frames up to this
    animation step), using BOTH pre- and post-throw frames.

    This matches exactly the sequence used by visualize_predictions.
    """
    # get the same d_in / d_out the visualizer uses
    input_path  = f"train/input_2023_w{week:02d}.csv"
    output_path = f"train/output_2023_w{week:02d}.csv"
    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)

    d_in = input_df[(input_df.game_id == game_id) & (input_df.play_id == play_id)]
    d_out = output_df[(output_df.game_id == game_id) & (output_df.play_id == play_id)]
    d_in = d_in.sort_values(["frame_id", "nfl_id"])
    d_out = d_out.sort_values(["frame_id", "nfl_id"])

    # build pre/post frame dicts
    frames_pre = frames_from_input(d_in)
    frames_post = frames_from_output_merged(d_in, d_out)

    # sanity: you should see both pre and post frame counts here
    print("pre frames:", len(frames_pre), "post frames:", len(frames_post))

    # let the Bayesian model build the combined sequence and prefix probs
    p_by_step = ps_model.prob_by_sequence(frames_pre, frames_post)

    return p_by_step

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

def label_plays_with_success(
    d_in: pd.DataFrame,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Play-level label: success if final targeted-receiver position
    is within 'threshold' yards of (ball_land_x, ball_land_y).
    """
    target = d_in[d_in["player_role"] == "Targeted Receiver"].copy()

    target["dist_to_ball_land"] = np.sqrt(
        (target["x"] - target["ball_land_x"]) ** 2 +
        (target["y"] - target["ball_land_y"]) ** 2
    )

    final_target = (
        target.sort_values("frame_id")
              .groupby(["game_id", "play_id"])
              .tail(1)
    )

    final_target["play_success"] = (
        final_target["dist_to_ball_land"] <= threshold
    ).astype(int)

    play_labels = final_target[["game_id", "play_id", "play_success"]].copy()
    return play_labels

def build_prefix_training_data(
    weeks: int | Iterable[int],
    model: BayesianPlaySuccessModel,
    threshold: float = 2.0,
    input_template: str = "train/input_2023_w{week:02d}.csv",
    output_template: str = "train/output_2023_w{week:02d}.csv",
) -> pd.DataFrame:
    """
    Build prefix-level training data over *both* pre- and post-throw frames.

    For each week, each play, each frame f in that play:
      - prefix = all frames with frame_id <= f  (pre + post)
      - feat_row = model._build_prefix_features(prefix)
      - label   = play_success for that play

    Returns a DataFrame with columns:
      game_id, play_id, frame_id, week, <model.feature_cols>, play_success
    """
    if isinstance(weeks, int):
        week_list = list(range(1, weeks + 1))
    else:
        week_list = list(weeks)

    all_rows: list[dict] = []

    for w in tqdm(week_list, desc="Weeks"):
        in_path = input_template.format(week=w)
        out_path = output_template.format(week=w)

        inp = pd.read_csv(in_path)
        out = pd.read_csv(out_path)

        play_labels = label_plays_with_success(inp, threshold=threshold)

        plays = list(
            inp[["game_id", "play_id"]]
            .drop_duplicates()
            .itertuples(index=False)
        )

        for game_id, play_id in tqdm(plays, desc=f"Plays (week {w})"):
            d_in = inp[(inp.game_id == game_id) & (inp.play_id == play_id)].copy()
            d_out = out[(out.game_id == game_id) & (out.play_id == play_id)].copy()

            frames_pre = frames_from_input(d_in)
            frames_post = frames_from_output_merged(d_in, d_out)

            # full play with frame_id
            d_play = pd.concat(
                [df.assign(frame_id=f)
                 for f, df in {**frames_pre, **frames_post}.items()],
                ignore_index=True,
            ).sort_values("frame_id")

            # play-level label
            row_label = play_labels[
                (play_labels.game_id == game_id) &
                (play_labels.play_id == play_id)
            ]
            if row_label.empty:
                continue
            y = int(row_label["play_success"].iloc[0])

            frame_ids = sorted(d_play["frame_id"].unique())

            for f in frame_ids:
                prefix = d_play[d_play["frame_id"] <= f]
                feat_df = model._build_prefix_features(prefix)
                feat_row = feat_df.iloc[0]

                record = {
                    "game_id": game_id,
                    "play_id": play_id,
                    "frame_id": f,
                    "week": w,
                    "play_success": y,
                }
                for col in model.feature_cols:
                    record[col] = feat_row[col]

                all_rows.append(record)

    train_df = pd.DataFrame(all_rows)
    cols = ["game_id", "play_id", "frame_id", "week"] + model.feature_cols + ["play_success"]
    return train_df[cols]


def build_play_frame_features(
    d_play: pd.DataFrame,
    post_frame_ids: Iterable[int],
    tackle_radius: float = 2.0,
    def_close_to_land_radius: float = 5.0,
    field_width: float = 53.3,
) -> pd.DataFrame:
    """
    Build per-frame features for a single play.

    Expected columns in d_play:
      - game_id, play_id
      - frame_id
      - player_role       (e.g. 'Passer', 'Targeted Receiver')
      - player_side       ('Offense' or 'Defense')
      - nfl_id
      - x, y
      - ball_land_x, ball_land_y
      - absolute_yardline_number

    post_frame_ids:
      Iterable of frame_id values that correspond to post-throw frames
      (all others are treated as pre-throw).

    Returns one row per frame_id with columns:
      frame_id,
      abs_yardline_at_throw,
      dist_target_to_land,
      num_defenders_close,
      breakaway,
      tackle_range,
      red_zone,
      ball_x,
      ball_y,
      dist_to_nearest_defender,
      dist_to_ball_land_position,
      dist_to_bounds
    """
    d_play = d_play.sort_values("frame_id").copy()
    post_frame_ids = set(post_frame_ids)

    frames = sorted(d_play["frame_id"].unique())
    if not frames:
        raise ValueError("d_play has no frames.")

    # --- play-level constants ---
    first_frame = frames[0]
    first_frame_rows = d_play[d_play["frame_id"] == first_frame]

    # red_zone: offense within 10 yards of end zone at start of play
    red_zone_flag = int(
        first_frame_rows["absolute_yardline_number"].iloc[0] <= 10.0
    )

    # ball landing coordinates (constant for play)
    ball_x_land = float(d_play["ball_land_x"].iloc[0])
    ball_y_land = float(d_play["ball_land_y"].iloc[0])

    # identify QB throw position from last PRE frame
    pre_frames = [f for f in frames if f not in post_frame_ids]
    if pre_frames:
        last_pre_frame = max(pre_frames)
        pre_last = d_play[d_play["frame_id"] == last_pre_frame]
        qb_pre = pre_last[pre_last["player_role"] == "Passer"]
        if not qb_pre.empty:
            qb_throw_x = float(qb_pre["x"].iloc[0])
            qb_throw_y = float(qb_pre["y"].iloc[0])
        else:
            off_last = pre_last[pre_last["player_side"] == "Offense"]
            qb_throw_x = float(off_last["x"].mean())
            qb_throw_y = float(off_last["y"].mean())
    else:
        last_pre_frame = None
        qb_throw_x = ball_x_land
        qb_throw_y = ball_y_land

    # precompute interpolation for ball in post frames
    post_sorted = sorted(f for f in frames if f in post_frame_ids)
    n_post = len(post_sorted)

    def ball_xy_for_frame(f: int, frame_df: pd.DataFrame) -> tuple[float, float]:
        # pre-throw: ball at QB position
        if f not in post_frame_ids or n_post == 0:
            qb = frame_df[frame_df["player_role"] == "Passer"]
            if not qb.empty:
                return float(qb["x"].iloc[0]), float(qb["y"].iloc[0])
            off = frame_df[frame_df["player_side"] == "Offense"]
            return float(off["x"].mean()), float(off["y"].mean())
        # post-throw: linearly interpolate from qb_throw -> ball_land
        idx = post_sorted.index(f)
        if n_post > 1:
            t = idx / (n_post - 1)
        else:
            t = 1.0
        bx = qb_throw_x + t * (ball_x_land - qb_throw_x)
        by = qb_throw_y + t * (ball_y_land - qb_throw_y)
        return bx, by

    rows = []

    for f in frames:
        g = d_play[d_play["frame_id"] == f]

        # Target receiver
        tgt = g[g["player_role"] == "Targeted Receiver"]
        if tgt.empty:
            # skip frames with no target receiver (edge case)
            continue
        tgt_x = float(tgt["x"].iloc[0])
        tgt_y = float(tgt["y"].iloc[0])

        # abs_yardline_at_throw: yardline at this frame
        abs_yardline = float(g["absolute_yardline_number"].iloc[0])

        # distance from target to ball landing
        dist_to_land = float(
            np.sqrt((tgt_x - ball_x_land) ** 2 + (tgt_y - ball_y_land) ** 2)
        )

        # defenders and distances (to target)
        defenders = g[g["player_side"] == "Defense"]
        if defenders.empty:
            dist_nearest_def = 100.0
        else:
            dx_def = defenders["x"].to_numpy(dtype=float) - tgt_x
            dy_def = defenders["y"].to_numpy(dtype=float) - tgt_y
            dists_def = np.sqrt(dx_def**2 + dy_def**2)
            dist_nearest_def = float(dists_def.min())

        # num_defenders_close: within def_close_to_land_radius of landing point
        if defenders.empty:
            num_def_close = 0
        else:
            dx_land = defenders["x"].to_numpy(dtype=float) - ball_x_land
            dy_land = defenders["y"].to_numpy(dtype=float) - ball_y_land
            d_land = np.sqrt(dx_land**2 + dy_land**2)
            num_def_close = int((d_land <= def_close_to_land_radius).sum())

        # breakaway (post-only):
        # 1 if target is closer to ball_land than every other player
        if f in post_frame_ids and not defenders.empty:
            dx_tgt = tgt_x - ball_x_land
            dy_tgt = tgt_y - ball_y_land
            dist_tgt_ball = np.sqrt(dx_tgt**2 + dy_tgt**2)

            others = g[g["nfl_id"] != tgt["nfl_id"].iloc[0]]
            if not others.empty:
                dx_o = others["x"].to_numpy(dtype=float) - ball_x_land
                dy_o = others["y"].to_numpy(dtype=float) - ball_y_land
                d_o = np.sqrt(dx_o**2 + dy_o**2)
                breakaway = int(dist_tgt_ball < d_o.min())
            else:
                breakaway = 1
        else:
            breakaway = 0

        # tackle_range: any defender within tackle_radius of target
        if np.isnan(dist_nearest_def):
            tackle_range = 0
        else:
            tackle_range = int(dist_nearest_def <= tackle_radius)

        red_zone = red_zone_flag

        # ball_x, ball_y
        ball_x, ball_y = ball_xy_for_frame(f, g)

        # dist_to_bounds: nearest sideline
        dist_to_bounds = float(min(tgt_y, field_width - tgt_y))

        rows.append(
            {
                "frame_id": f,
                "abs_yardline_at_throw": abs_yardline,
                "dist_target_to_land": dist_to_land,
                "num_defenders_close": num_def_close,
                "breakaway": breakaway,
                "tackle_range": tackle_range,
                "red_zone": red_zone,
                "ball_x": float(ball_x),
                "ball_y": float(ball_y),
                "dist_to_nearest_defender": dist_nearest_def,
                "dist_to_ball_land_position": dist_to_land,
                "dist_to_bounds": dist_to_bounds,
            }
        )

    feat_df = pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)
    return feat_df


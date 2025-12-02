import numpy as np
import pandas as pd
import importlib
from typing import Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.patches import Ellipse

from models.bayes_model_base import BayesianMovementModel
from . import tracking_utils, field_plot
importlib.reload(tracking_utils)
importlib.reload(field_plot)
from .tracking_utils import load_play, frames_from_input, frames_from_output_merged
from .field_plot import animate_full_play, animate_pre_play, draw_field

@dataclass
class PlayVizData:
    d_in: pd.DataFrame
    d_out_kine: pd.DataFrame
    frames_pre: dict
    frames_post: dict
    seq: list         # list[(phase, df)]
    t_seq: list       # ball interpolation t in [0,1] for post frames
    pred_ids: list
    side_by_pid: dict
    ball: tuple       # (bx, by)
    qb_throw: tuple   # (qb_throw_x, qb_throw_y)
    pos_labels: list | None
    use_bayesian_cones: bool


@dataclass
class Artists:
    fig: plt.Figure
    ax: plt.Axes
    off: any
    targ: any
    deff: any
    pred_off: any
    pred_def: any
    ball_target: any
    ball_in_air: any
    path_lines: list            # list[(pid, line)]
    cone_patches: dict          # pid -> [Ellipse,...]
    mean_markers: dict          # pid -> [Line2D,...]
    legend_handles: list

# Data prep helpers
def _prepare_play_frames(
    model,
    week: int,
    game_id: int,
    play_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load play and build pre/post frames with kinematics."""
    d_in, d_out = get_din_dout(week, game_id, play_id)
    d_in = d_in.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    d_out = d_out.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

    frames_pre = frames_from_input(d_in)

    fps = getattr(model, "fps", 10.0)
    d_out_kine = d_out.copy()
    g = d_out_kine.groupby(["game_id", "play_id", "nfl_id"], sort=False)

    d_out_kine["dx"] = g["x"].diff()
    d_out_kine["dy"] = g["y"].diff()
    d_out_kine["s"] = np.sqrt(d_out_kine["dx"] ** 2 + d_out_kine["dy"] ** 2) * fps
    d_out_kine["dir"] = np.degrees(np.arctan2(d_out_kine["dy"], d_out_kine["dx"]))
    d_out_kine["a"] = g["s"].diff() * fps
    d_out_kine = d_out_kine.drop(columns=["dx", "dy"])

    meta = d_in[
        [
            "nfl_id",
            "player_side",
            "player_position",
            "player_role",
            "player_to_predict",
        ]
    ].drop_duplicates("nfl_id")
    d_out_kine = d_out_kine.merge(meta, on="nfl_id", how="left")

    cols_post = [
        "game_id",
        "play_id",
        "frame_id",
        "nfl_id",
        "player_side",
        "player_position",
        "player_role",
        "x",
        "y",
        "s",
        "a",
        "dir",
        "player_to_predict",
    ]
    frames_post: dict[int, pd.DataFrame] = {}
    for f_id, g_f in d_out_kine.groupby("frame_id"):
        frames_post[f_id] = g_f[cols_post].copy()

    return d_in, d_out_kine, frames_pre, frames_post


def _compute_pred_ids_and_sides(d_in: pd.DataFrame) -> tuple[list, dict]:
    """Determine which players to predict and their sides."""
    base = d_in.copy()
    if "player_to_predict" in base.columns and base.player_to_predict.any():
        pred_mask_base = base.player_to_predict
    else:
        pred_mask_base = base.player_role == "Targeted Receiver"
        if not pred_mask_base.any():
            pred_mask_base = base.player_side == "Offense"

    pred_ids = (
        base.loc[pred_mask_base, "nfl_id"]
        .dropna()
        .unique()
        .tolist()
    )

    side_by_pid = {}
    if pred_ids:
        side_meta = base[["nfl_id", "player_side"]].drop_duplicates("nfl_id")
        side_by_pid = dict(zip(side_meta["nfl_id"], side_meta["player_side"]))

    return pred_ids, side_by_pid


def _build_sequence(
    frames_pre: dict,
    frames_post: dict,
    interval: int,
    pause_time: Optional[float],
    d_in: pd.DataFrame,
) -> tuple[list, list, tuple, tuple, list | None]:
    """Build (phase, df) sequence, t_seq, ball location, QB throw location, and pos labels."""
    pre_list = [frames_pre[k] for k in sorted(frames_pre.keys())]
    post_list = [frames_post[k] for k in sorted(frames_post.keys())] if frames_post else []

    if pause_time is not None and pause_time <= 0:
        pause_time = None

    seq: list[tuple[str, pd.DataFrame]] = []
    for d in pre_list:
        seq.append(("pre", d))
    if pause_time and pre_list:
        last_pre = pre_list[-1]
        n_frames_pause = round(interval / pause_time) if pause_time else 0
        for _ in range(n_frames_pause):
            seq.append(("pre_pause", last_pre))
    for d in post_list:
        seq.append(("post", d))

    # ball landing
    ball = (d_in.ball_land_x.iloc[0], d_in.ball_land_y.iloc[0])
    if ball is None:
        ball = (-999, -999)
    bx, by = ball

    # QB throw location (last pre-frame)
    if pre_list:
        last_pre = pre_list[-1]
        qb = last_pre[last_pre.player_role == "Passer"]
        if not qb.empty:
            qb_throw_x = qb.x.iloc[0]
            qb_throw_y = qb.y.iloc[0]
        else:
            off_last = last_pre[last_pre.player_side == "Offense"]
            qb_throw_x = off_last.x.mean()
            qb_throw_y = off_last.y.mean()
    else:
        qb_throw_x, qb_throw_y = bx, by

    # t in [0,1] for ball interpolation in post
    t_seq = []
    post_count = sum(1 for ph, _ in seq if ph == "post")
    current_post_idx = 0
    for ph, _ in seq:
        if ph == "post":
            if post_count > 1:
                t = current_post_idx / (post_count - 1)
            else:
                t = 1.0
            current_post_idx += 1
        else:
            t = None
        t_seq.append(t)

    # labels for title
    if "player_to_predict" in d_in.columns and d_in.player_to_predict.any():
        pos = (
            d_in.loc[d_in.player_to_predict, "player_position"]
            .dropna()
            .unique()
            .tolist()
        )
    else:
        pos = None

    qb_throw = (qb_throw_x, qb_throw_y)
    ball_xy = (bx, by)
    return seq, t_seq, ball_xy, qb_throw, pos


def _prepare_viz_data(
    model,
    week: int,
    game_id: int,
    play_id: int,
) -> PlayVizData:
    d_in, d_out_kine, frames_pre, frames_post = _prepare_play_frames(
        model, week, game_id, play_id
    )
    pred_ids, side_by_pid = _compute_pred_ids_and_sides(d_in)

    is_bayes_model = isinstance(model, BayesianMovementModel)
    has_trace = is_bayes_model and (getattr(model, "trace", None) is not None)
    use_bayesian_cones = has_trace

    # dummy interval/pause for sequence; real ones passed later, but we need structure here
    # (we'll rebuild sequence in visualize_predictions with true interval/pause)
    return PlayVizData(
        d_in=d_in,
        d_out_kine=d_out_kine,
        frames_pre=frames_pre,
        frames_post=frames_post,
        seq=[],          # to be filled in visualize_predictions
        t_seq=[],
        pred_ids=pred_ids,
        side_by_pid=side_by_pid,
        ball=(0.0, 0.0),  # placeholders
        qb_throw=(0.0, 0.0),
        pos_labels=None,
        use_bayesian_cones=use_bayesian_cones,
    )


# Artist initialization and reset
def _init_artists(
    data: PlayVizData,
    show_paths: bool,
    show_cones: bool,
    show_legend: bool,
    horizon: int,
) -> Artists:
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)

    off, = ax.plot([], [], "o", color="orange", label="Offense")
    targ, = ax.plot([], [], "o", color="red", label="Targeted receiver")
    deff, = ax.plot([], [], "s", color="blue", label="Defense")

    pred_off, = ax.plot([], [], "o", mfc="none", mec="lightblue", mew=1.0, label="Pred Offense")
    pred_def, = ax.plot([], [], "s", mfc="none", mec="lightblue", mew=1.0, label="Pred Defense")

    ball_target, = ax.plot([], [], "x", color="black", markersize=10, label="Ball target")
    ball_in_air, = ax.plot([], [], "o", mfc="none", mec="black", mew=1.5, label="Ball in air")

    proxy_path, = ax.plot([], [], "--", color="black", lw=1.5, alpha=0.9, label="Pred path")

    # paths
    path_lines = []
    if show_paths:
        for pid in data.pred_ids:
            line, = ax.plot(
                [],
                [],
                "--",
                lw=2.0,
                alpha=0.9,
                color="black",
                label=None,
            )
            path_lines.append((pid, line))

    # cones + mean markers per player
    cone_patches: dict[int, list[Ellipse]] = {}
    mean_markers: dict[int, list[plt.Line2D]] = {}

    if show_cones and data.use_bayesian_cones and data.pred_ids:
        for pid in data.pred_ids:
            side = data.side_by_pid.get(pid, "Offense")
            marker_style = "o" if side == "Offense" else "s"
            cone_list = []
            marker_list = []
            for _ in range(horizon):
                ell = Ellipse(
                    xy=(0.0, 0.0),
                    width=0.0,
                    height=0.0,
                    angle=0.0,
                    facecolor="grey",
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.0,
                    zorder=4,
                )
                ax.add_patch(ell)
                cone_list.append(ell)

                m, = ax.plot(
                    [],
                    [],
                    marker_style,
                    ms=6,
                    mfc="none",
                    mec="black",
                    alpha=0.0,
                    zorder=5,
                    label=None,
                )
                marker_list.append(m)

            cone_patches[pid] = cone_list
            mean_markers[pid] = marker_list

        # label one cone + mean for legend
        any_pid = data.pred_ids[0]
        cone_patches[any_pid][0].set_label("95% cone")
        mean_markers[any_pid][0].set_label("Pred mean")

    # legend
    legend_handles = [off, targ, deff, ball_target, ball_in_air]
    if show_paths:
        legend_handles.append(proxy_path)
    if show_cones and cone_patches:
        any_pid = data.pred_ids[0]
        legend_handles.append(cone_patches[any_pid][0])
    if show_cones and mean_markers:
        any_pid = data.pred_ids[0]
        legend_handles.append(mean_markers[any_pid][0])
    if show_legend:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8)

    return Artists(
        fig=fig,
        ax=ax,
        off=off,
        targ=targ,
        deff=deff,
        pred_off=pred_off,
        pred_def=pred_def,
        ball_target=ball_target,
        ball_in_air=ball_in_air,
        path_lines=path_lines,
        cone_patches=cone_patches,
        mean_markers=mean_markers,
        legend_handles=legend_handles,
    )


def _reset_prediction_artists(artists: Artists):
    # clear paths
    for _, line in artists.path_lines:
        line.set_data([], [])
    # clear cones
    for pid in artists.cone_patches:
        for ell in artists.cone_patches[pid]:
            ell.set_alpha(0.0)
            ell.width = 0.0
            ell.height = 0.0
    # clear mean markers
    for pid in artists.mean_markers:
        for m in artists.mean_markers[pid]:
            m.set_data([], [])
            m.set_alpha(0.0)

# Per-frame update helpers
def _update_pre_frame(
    d: pd.DataFrame,
    pos_labels: list | None,
    artists: Artists,
):
    o = d[d.player_side == "Offense"]
    df = d[d.player_side == "Defense"]
    tg = d[d.player_role == "Targeted Receiver"]
    p = d[d.player_to_predict] if "player_to_predict" in d.columns else d.iloc[0:0]
    p_off = p[p.player_side == "Offense"]
    p_def = p[p.player_side == "Defense"]

    artists.off.set_data(o.x.values, o.y.values)
    artists.deff.set_data(df.x.values, df.y.values)
    artists.targ.set_data(tg.x.values, tg.y.values)
    artists.pred_off.set_data(p_off.x.values, p_off.y.values)
    artists.pred_def.set_data(p_def.x.values, p_def.y.values)

    qb_frame = d[d.player_role == "Passer"]
    if not qb_frame.empty:
        qb_x = qb_frame.x.iloc[0]
        qb_y = qb_frame.y.iloc[0]
    else:
        off_frame = d[d.player_side == "Offense"]
        qb_x = off_frame.x.mean()
        qb_y = off_frame.y.mean()

    artists.ball_target.set_data([], [])
    artists.ball_in_air.set_data([qb_x], [qb_y])

    if pos_labels:
        artists.ax.set_title("Pre-throw: " + ", ".join(pos_labels))
    else:
        artists.ax.set_title("Pre-throw")


def _update_post_frame_draw_players_and_ball(
    d: pd.DataFrame,
    t: float | None,
    data: PlayVizData,
    artists: Artists,
):
    """Update actual players + ball for post frames (no predictions here)."""
    o = d[d.player_side == "Offense"]
    df = d[d.player_side == "Defense"]
    tg = d[d.player_role == "Targeted Receiver"]
    p = d[d.player_to_predict] if "player_to_predict" in d.columns else d.iloc[0:0]
    p_off = p[p.player_side == "Offense"]
    p_def = p[p.player_side == "Defense"]

    artists.off.set_data(o.x.values, o.y.values)
    artists.deff.set_data(df.x.values, df.y.values)
    artists.targ.set_data(tg.x.values, tg.y.values)
    artists.pred_off.set_data(p_off.x.values, p_off.y.values)
    artists.pred_def.set_data(p_def.x.values, p_def.y.values)

    bx, by = data.ball
    qb_throw_x, qb_throw_y = data.qb_throw

    artists.ball_target.set_data([bx], [by])
    if t is not None:
        cx = qb_throw_x + t * (bx - qb_throw_x)
        cy = qb_throw_y + t * (by - qb_throw_y)
        artists.ball_in_air.set_data([cx], [cy])
    else:
        artists.ball_in_air.set_data([], [])

    if data.pos_labels:
        artists.ax.set_title("Post-throw predictions: " + ", ".join(data.pos_labels))
    else:
        artists.ax.set_title("Post-throw predictions")


# Predictions and Cones
def _update_paths_from_rollout(
    model,
    base_state: pd.DataFrame,
    artists: Artists,
    horizon: int,
):
    """Use model.rollout_horizon to draw dashed paths from base_state."""
    if not artists.path_lines:
        return

    rollout = model.rollout_horizon(
        base_state,
        horizon=horizon,
        id_cols=("game_id", "play_id", "nfl_id", "frame_id"),
        x_col="x",
        y_col="y",
        s_col="s",
        a_col="a",
        dir_col="dir",
        step_col="h",
    )

    for pid, line in artists.path_lines:
        row0 = base_state[base_state.nfl_id == pid]
        if row0.empty:
            line.set_data([], [])
            continue

        x0 = float(row0.x.iloc[0])
        y0 = float(row0.y.iloc[0])

        roll_pid = rollout[rollout.nfl_id == pid].sort_values("h")
        if roll_pid.empty:
            line.set_data([x0], [y0])
            continue

        centers_x = [x0] + roll_pid["x_pred"].tolist()
        centers_y = [y0] + roll_pid["y_pred"].tolist()
        line.set_data(centers_x, centers_y)


def _build_multistep_states(
    model,
    base_state: pd.DataFrame,
    horizon: int,
) -> list[pd.DataFrame]:
    """Deterministic rollout using predict_dataframe, for cone centers."""
    states = [base_state.copy()]
    current = base_state.copy()
    for _ in range(1, horizon + 1):
        step_pred = model.predict_dataframe(
            current,
            x_col="x",
            y_col="y",
            s_col="s",
            a_col="a",
            dir_col="dir",
            out_x_col="x_pred",
            out_y_col="y_pred",
        )
        step = current.copy()
        step["x"] = step_pred["x_pred"].to_numpy()
        step["y"] = step_pred["y_pred"].to_numpy()
        states.append(step)
        current = step
    return states


def _update_bayesian_cones(
    model: BayesianMovementModel,
    states: list[pd.DataFrame],
    data: PlayVizData,
    artists: Artists,
    bayes_samples: int,
    horizon: int,
    confidence_pct: float = 0.95, # For cones
):
    if not artists.cone_patches or not artists.mean_markers:
        return

    max_h = min(horizon, len(states) - 1)
    min_size = 0.5  # yards

    for pid in data.pred_ids:
        if pid not in artists.cone_patches or pid not in artists.mean_markers:
            continue

        for h_step in range(1, max_h + 1):
            df_h = states[h_step]
            row_h = df_h[df_h.nfl_id == pid]
            if row_h.empty:
                continue

            x_samps, y_samps = model.posterior_samples_for_rows(
                row_h,
                n_samples=bayes_samples,
                x_col="x",
                y_col="y",
                s_col="s",
                a_col="a",
                dir_col="dir",
            )
            xs = x_samps[:, 0]
            ys = y_samps[:, 0]

            mx = float(xs.mean())
            my = float(ys.mean())

            lower_pct = (1-confidence_pct)/2
            upper_pct = 1-lower_pct
            x_lo, x_hi = np.quantile(xs, [lower_pct, upper_pct])
            y_lo, y_hi = np.quantile(ys, [lower_pct, upper_pct])

            width = max(float(x_hi - x_lo), min_size)
            height = max(float(y_hi - y_lo), min_size)

            base_alpha = 0.35
            alpha = max(0.1, base_alpha * (1.0 - (h_step - 1) / max_h))

            ell = artists.cone_patches[pid][h_step - 1]
            ell.center = (mx, my)
            ell.width = width
            ell.height = height
            ell.angle = 0.0
            ell.set_alpha(alpha)

            m = artists.mean_markers[pid][h_step - 1]
            m.set_data([mx], [my])
            m.set_alpha(min(alpha + 0.2, 0.9))


# visualize predictions
def visualize_predictions(
    model,
    week: int,
    game_id: int,
    play_id: int,
    horizon: int = 3,
    interval: int = 120,
    pause_time: Optional[float] = None,
    bayes_samples: int = 200,
    show_paths: bool = True,
    show_cones: bool = True,
    show_legend: bool = True,
    cone_pct: float = 0.95,
):
    """
    Animate a play like animate_week_play, plus model predictions AFTER the throw.

    Toggles:
      - show_paths: dashed prediction paths
      - show_cones: Bayesian 95% cones + mean markers (if model fitted)
      - show_legend: whether to draw a legend
    """
    # --- prep data ---
    data = _prepare_viz_data(model, week, game_id, play_id)
    seq, t_seq, ball_xy, qb_throw, pos_labels = _build_sequence(
        data.frames_pre,
        data.frames_post,
        interval=interval,
        pause_time=pause_time,
        d_in=data.d_in,
    )
    data.seq = seq
    data.t_seq = t_seq
    data.ball = ball_xy
    data.qb_throw = qb_throw
    data.pos_labels = pos_labels

    # --- init artists ---
    artists = _init_artists(
        data,
        show_paths=show_paths,
        show_cones=show_cones,
        show_legend=show_legend,
        horizon=horizon,
    )

    # maintain a predicted_state across post frames
    predicted_state: pd.DataFrame | None = None

    def init():
        artists.off.set_data([], [])
        artists.deff.set_data([], [])
        artists.targ.set_data([], [])
        artists.pred_off.set_data([], [])
        artists.pred_def.set_data([], [])
        artists.ball_target.set_data([], [])
        artists.ball_in_air.set_data([], [])
        _reset_prediction_artists(artists)
        return (
            artists.off,
            artists.deff,
            artists.targ,
            artists.pred_off,
            artists.pred_def,
            artists.ball_target,
            artists.ball_in_air,
            *(line for _, line in artists.path_lines),
            *(ell for pid in artists.cone_patches for ell in artists.cone_patches[pid]),
            *(m for pid in artists.mean_markers for m in artists.mean_markers[pid]),
        )

    def update(i):
        nonlocal predicted_state

        phase, d = data.seq[i]
        t = data.t_seq[i]

        if phase.startswith("pre"):
            _reset_prediction_artists(artists)
            _update_pre_frame(d, data.pos_labels, artists)
        else:
            # post
            _reset_prediction_artists(artists)
            _update_post_frame_draw_players_and_ball(d, t, data, artists)

            # seed predicted_state on first post frame
            if predicted_state is None:
                predicted_state = d.copy()

            # paths
            if show_paths:
                _update_paths_from_rollout(model, predicted_state, artists, horizon=horizon)

            # cones
            if (
                show_cones
                and data.use_bayesian_cones
                and isinstance(model, BayesianMovementModel)
            ):
                states = _build_multistep_states(model, predicted_state, horizon=horizon)
                _update_bayesian_cones(
                    model,
                    states,
                    data,
                    artists,
                    bayes_samples=bayes_samples,
                    horizon=horizon,
                    confidence_pct=cone_pct,
                )

            # move prediction base forward one actual frame
            predicted_state = d.copy()

        return (
            artists.off,
            artists.deff,
            artists.targ,
            artists.pred_off,
            artists.pred_def,
            artists.ball_target,
            artists.ball_in_air,
            *(line for _, line in artists.path_lines),
            *(ell for pid in artists.cone_patches for ell in artists.cone_patches[pid]),
            *(m for pid in artists.mean_markers for m in artists.mean_markers[pid]),
        )

    ani = anim.FuncAnimation(
        artists.fig,
        update,
        frames=len(data.seq),
        init_func=init,
        interval=interval,
        blit=True,
    )
    plt.close(artists.fig)
    return ani



def animate_week_play(week, game_id, play_id, interval=120, pre_only=False, pause_time: Optional[float]=False):
    """
    pause_time is None or number of seconds to pause.
    """
    d_in, d_out = get_din_dout(week, game_id, play_id)
    frames_pre  = frames_from_input(d_in)
    frames_post = frames_from_output_merged(d_in, d_out)
    ball = (d_in.ball_land_x.iloc[0], d_in.ball_land_y.iloc[0])

    if "player_to_predict" in d_in.columns:
        pos = (
            d_in.loc[d_in.player_to_predict, "player_position"]
                .dropna()
                .unique()
                .tolist()
        )
    else:
        pos = None

    if pre_only:
        return animate_pre_play(
            frames_pre,
            ball_land=ball,
            interval=interval,
            predict_positions=pos,
        )
    return animate_full_play(
        frames_pre,
        frames_post,
        ball_land=ball,
        interval=interval,
        predict_positions=pos,
        pause_time=pause_time,
    )

def get_din_dout(week, game_id, play_id):
    input_path  = f"train/input_2023_w{week:02d}.csv"
    output_path = f"train/output_2023_w{week:02d}.csv"
    inp = pd.read_csv(input_path)
    outp = pd.read_csv(output_path)
    return load_play(inp, outp, game_id, play_id)

def test_frame_alignment(week, idx, game_id, play_id):
    d_in, d_out = get_din_dout(week, game_id, play_id)

    print("Ball land:", d_in.ball_land_x.iloc[idx], d_in.ball_land_y.iloc[idx])

    d_out_trg = d_out.merge(
        d_in[["nfl_id", "player_role"]].drop_duplicates(), on="nfl_id", how="left"
    )
    trg_out = d_out_trg[d_out_trg.player_role == "Targeted Receiver"]
    trg_last = trg_out[trg_out.frame_id == trg_out.frame_id.max()]

    print("Target last frame x,y:")
    print(trg_last[["x", "y"]])
    print(f"Last frame ID: {trg_out.frame_id.max()}")
    print(f"Plotted ID: {trg_last.frame_id}")

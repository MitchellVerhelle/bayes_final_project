import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from typing import Optional

def draw_field(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("forestgreen")
    y0, y1 = ax.get_ylim()
    y_mid = 0.5 * (y0 + y1)
    dash_len = 4.0
    dash_y0 = y_mid - dash_len / 2
    dash_y1 = y_mid + dash_len / 2
    for x in range(0, 121):
        if x % 10 == 0 and 10 <= x <= 110:
            ax.plot([x, x], [y0, y1], color='white', lw=1.0)
        elif 10 < x < 110:
            ax.plot([x, x], [dash_y0, dash_y1], color='white', lw=0.5)

def animate_pre_play(frames_in, ball_land=None, interval=120, predict_positions=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)

    off, = ax.plot([], [], 'o', color='orange', label='Offense')
    targ, = ax.plot([], [], 'o', color='red', label='Targeted receiver')
    deff, = ax.plot([], [], 's', color='blue', label='Defense')
    football_img = mpimg.imread("images/football.png")
    football_icon = OffsetImage(football_img, zoom=0.05, alpha=0.95)
    ball_artist = AnnotationBbox(
        football_icon,
        ball_land if ball_land is not None else (-999, -999),
        frameon=False,
        box_alignment=(0.5, 0.5),
    )
    ax.add_artist(ball_artist)
    pred_off, = ax.plot([], [], 'o', mfc='none', mec='lightblue', mew=1.0, label='Pred Offense')
    pred_def, = ax.plot([], [], 's', mfc='none', mec='lightblue', mew=1.0, label='Pred Defense')

    if predict_positions:
        ax.set_title("Predicting: " + ", ".join(predict_positions))

    ax.legend(
        handles=[off, targ, deff],
        loc='upper right',
        framealpha=0.8
    )
    f_ids = sorted(frames_in.keys())

    def init():
        off.set_data([], [])
        deff.set_data([], [])
        targ.set_data([], [])
        pred_off.set_data([], [])
        pred_def.set_data([], [])
        return off, deff, targ, pred_off, pred_def, ball_artist

    def update(i):
        d = frames_in[f_ids[i]]
        o = d[d.player_side == "Offense"]
        df = d[d.player_side == "Defense"]
        tg = d[d.player_role == "Targeted Receiver"]
        p = d[d.player_to_predict]
        p_off = p[p.player_side == "Offense"]
        p_def = p[p.player_side == "Defense"]

        off.set_data(o.x.values,  o.y.values)
        deff.set_data(df.x.values, df.y.values)
        targ.set_data(tg.x.values, tg.y.values)
        pred_off.set_data(p_off.x.values, p_off.y.values)
        pred_def.set_data(p_def.x.values, p_def.y.values)

        return off, deff, targ, pred_off, pred_def, ball_artist

    ani = anim.FuncAnimation(
        fig, update, frames=len(f_ids),
        init_func=init, interval=interval, blit=True
    )
    plt.close(fig)
    return ani

def animate_full_play(frames_pre, frames_post, ball_land, interval=120, predict_positions=None, pause_time: Optional[float]=None):
    """animate_full_play animates teh full play and payses for pause_time seconds. (Or 0 if None)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)

    if pause_time is not None and pause_time <= 0:
        pause_time = None

    off,  = ax.plot([], [], 'o', color='orange', label='Offense')
    targ, = ax.plot([], [], 'o', color='red',    label='Targeted receiver')
    deff, = ax.plot([], [], 's', color='blue',   label='Defense')
    pred_off, = ax.plot([], [], 'o', mfc='none', mec='lightblue', mew=1.0, label='Pred Offense')
    pred_def, = ax.plot([], [], 's', mfc='none', mec='lightblue', mew=1.0, label='Pred Defense')
    ball_target, = ax.plot([], [], 'x', color='black', markersize=10, label='Ball target')
    ball, = ax.plot([], [], 'o', mfc='none', mec='black', mew=1.5, label='Ball in air')

    ax.legend(handles=[off, targ, deff, ball_target, ball], loc='upper right', framealpha=0.8)

    if predict_positions:
        ax.set_title("Pre-throw: " + ", ".join(predict_positions))
    else:
        ax.set_title("Pre-throw")

    pre_list  = [frames_pre[k]  for k in sorted(frames_pre.keys())]
    post_list = [frames_post[k] for k in sorted(frames_post.keys())] if frames_post else []

    # build (phase, frame_df) sequence
    seq = []
    for d in pre_list:
        seq.append(("pre", d))
    if pause_time and pre_list:
        last_pre = pre_list[-1]
        n_frames_pause = round(interval/pause_time) if pause_time else 0
        for _ in range(n_frames_pause):
            seq.append(("pre_pause", last_pre))
    for d in post_list:
        seq.append(("post", d))

    # Ball landing coordinates
    if ball_land is not None:
        bx, by = ball_land
    else:
        bx, by = -999, -999

    # QB throw location = Last known QB position in pre-play frames
    if pre_list:
        last_pre = pre_list[-1]
        qb = last_pre[last_pre.player_role == "Passer"]
        if not qb.empty:
            qb_throw_x = qb.x.iloc[0]
            qb_throw_y = qb.y.iloc[0]
        else:
            # crude fallback: mean of offense
            off_last = last_pre[last_pre.player_side == "Offense"]
            qb_throw_x = off_last.x.mean()
            qb_throw_y = off_last.y.mean()
    else:
        qb_throw_x, qb_throw_y = bx, by

    # precompute t values for ball in air
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

    def init():
        off.set_data([], [])
        deff.set_data([], [])
        targ.set_data([], [])
        pred_off.set_data([], [])
        pred_def.set_data([], [])
        ball_target.set_data([], [])
        ball.set_data([], [])
        return off, deff, targ, pred_off, pred_def, ball_target, ball

    def update(i):
        phase, d = seq[i]
        t = t_seq[i]

        o   = d[d.player_side == "Offense"]
        df  = d[d.player_side == "Defense"]
        tg  = d[d.player_role == "Targeted Receiver"]
        p   = d[d.player_to_predict]
        p_off = p[p.player_side == "Offense"]
        p_def = p[p.player_side == "Defense"]

        off.set_data(o.x.values,   o.y.values)
        deff.set_data(df.x.values, df.y.values)
        targ.set_data(tg.x.values, tg.y.values)
        pred_off.set_data(p_off.x.values, p_off.y.values)
        pred_def.set_data(p_def.x.values, p_def.y.values)

        if phase.startswith("pre"):
            # ball moves with QB in pre-frames
            qb_frame = d[d.player_role == "Passer"]
            if not qb_frame.empty:
                qb_x = qb_frame.x.iloc[0]
                qb_y = qb_frame.y.iloc[0]
            else:
                off_frame = d[d.player_side == "Offense"]
                qb_x = off_frame.x.mean()
                qb_y = off_frame.y.mean()

            ball_target.set_data([], [])
            ball.set_data([qb_x], [qb_y])
            
            if predict_positions:
                ax.set_title("Pre-throw: " + ", ".join(predict_positions))
            else:
                ax.set_title("Pre-throw")
        else:
            ball_target.set_data([bx], [by])
            if t is not None:
                cx = qb_throw_x + t * (bx - qb_throw_x)
                cy = qb_throw_y + t * (by - qb_throw_y)
                ball.set_data([cx], [cy])
            else:
                ball.set_data([], [])

            if predict_positions:
                ax.set_title("Post-throw: " + ", ".join(predict_positions))
            else:
                ax.set_title("Post-throw")

        return off, deff, targ, pred_off, pred_def, ball_target, ball

    ani = anim.FuncAnimation(
        fig, update, frames=len(seq),
        init_func=init, interval=interval, blit=True
    )
    plt.close(fig)
    return ani
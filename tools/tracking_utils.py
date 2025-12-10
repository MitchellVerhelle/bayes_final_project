import pandas as pd

def create_response_variable(df):
    df = df.copy()
    df["successful_play"] = df[""]

def load_play(input_df, output_df, game_id, play_id):
    d_in = input_df[(input_df.game_id == game_id) & (input_df.play_id == play_id)]
    d_out = output_df[(output_df.game_id == game_id) & (output_df.play_id == play_id)]
    d_in = d_in.sort_values(["frame_id", "nfl_id"])
    d_out = d_out.sort_values(["frame_id", "nfl_id"])
    return d_in, d_out

def frames_from_input(df):
    cols = [
        "game_id",
        "play_id",
        "frame_id",
        "nfl_id",
        "player_side",
        "player_position",
        "player_role",
        "player_to_predict",
        "x",
        "y",
        "absolute_yardline_number",
        "play_direction",
        "ball_land_x",
        "ball_land_y",
    ]
    cols = [c for c in cols if c in df.columns]

    frames = {}
    for f, d in df.groupby("frame_id"):
        frames[f] = d[cols].copy()
    return frames


def frames_from_output_merged(d_in, d_out):
    meta_cols = [
        "game_id",
        "play_id",
        "nfl_id",
        "player_side",
        "player_position",
        "player_role",
        "player_to_predict",
        "absolute_yardline_number",
        "play_direction",
        "ball_land_x",
        "ball_land_y",
    ]
    meta_cols = [c for c in meta_cols if c in d_in.columns]

    meta = (
        d_in[meta_cols]
        .drop_duplicates(["game_id", "play_id", "nfl_id"])
    )

    d = d_out.merge(meta, on=["game_id", "play_id", "nfl_id"], how="left")

    cols = [
        "game_id",
        "play_id",
        "frame_id",
        "nfl_id",
        "player_side",
        "player_position",
        "player_role",
        "player_to_predict",
        "x",
        "y",
        "absolute_yardline_number",
        "play_direction",
        "ball_land_x",
        "ball_land_y",
    ]
    cols = [c for c in cols if c in d.columns]

    frames = {}
    for f, g in d.groupby("frame_id"):
        frames[f] = g[cols].copy()
    return frames

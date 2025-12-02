import pandas as pd

def load_play(input_df, output_df, game_id, play_id):
    d_in = input_df[(input_df.game_id == game_id) & (input_df.play_id == play_id)]
    d_out = output_df[(output_df.game_id == game_id) & (output_df.play_id == play_id)]
    d_in = d_in.sort_values(["frame_id", "nfl_id"])
    d_out = d_out.sort_values(["frame_id", "nfl_id"])
    return d_in, d_out

def frames_from_input(df):
    cols = ["nfl_id","player_side","player_position","player_role","x","y","player_to_predict"]
    frames = {}
    for f, d in df.groupby("frame_id"):
        frames[f] = d[cols].copy()
    return frames

def frames_from_output_merged(d_in, d_out):
    meta = d_in[["nfl_id","player_side","player_position","player_role","player_to_predict"]].drop_duplicates("nfl_id")
    d = d_out.merge(meta, on="nfl_id", how="left")
    cols = ["nfl_id","player_side","player_position","player_role","x","y","player_to_predict"]
    frames = {}
    for f, g in d.groupby("frame_id"):
        frames[f] = g[cols].copy()
    return frames
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.model_base import MovementModel
from models.bayes_model_base import BayesianMovementModel


class DummyModel(MovementModel):
    # simple deterministic shift: x_next = x+1, y_next = y+2
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
        out = df.copy()
        out[out_x_col] = out[x_col] + 1.0
        out[out_y_col] = out[y_col] + 2.0
        return out


class DummyBayesModel(BayesianMovementModel):
    # posterior is just Normal around x+1,y+2 with fixed sigma
    def __init__(self, sigma: float = 0.1, name: str = "dummy_bayes"):
        super().__init__(name=name)
        self.sigma = float(sigma)

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
        **kwargs,
    ) -> None:
        # no-op for dummy
        self.trace = {"sigma": self.sigma}
        self.model = "dummy"

    def posterior_samples_for_rows(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        x_col: str = "x",
        y_col: str = "y",
        s_col: str = "s",
        a_col: str = "a",
        dir_col: str = "dir",
    ):
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        mu_x = x + 1.0
        mu_y = y + 2.0
        n_rows = len(df)
        x_samps = np.random.normal(mu_x, self.sigma, size=(n_samples, n_rows))
        y_samps = np.random.normal(mu_y, self.sigma, size=(n_samples, n_rows))
        return x_samps, y_samps


def make_step_df(n: int = 5) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "game_id": np.ones(n, dtype=int) * 1,
            "play_id": np.ones(n, dtype=int) * 10,
            "nfl_id": np.arange(n, dtype=int),
            "frame_id": np.arange(n, dtype=int),
            "x": np.linspace(10, 14, n),
            "y": np.linspace(20, 24, n),
            "s": 1.0,
            "a": 0.0,
            "dir": 0.0,
        }
    )
    df["x_next"] = df["x"] + 1.0
    df["y_next"] = df["y"] + 2.0
    return df


def test_dummy_model_predict_and_rmse():
    df = make_step_df()
    model = DummyModel(name="dummy")

    df_pred = model.predict_dataframe(df)
    assert "x_pred" in df_pred.columns
    assert "y_pred" in df_pred.columns

    assert np.allclose(df_pred["x_pred"].to_numpy(), df["x"].to_numpy() + 1.0)
    assert np.allclose(df_pred["y_pred"].to_numpy(), df["y"].to_numpy() + 2.0)

    metrics = model.rmse(df_pred, x_true_col="x_next", y_true_col="y_next")
    assert metrics["model"] == "dummy"
    assert np.isclose(metrics["rmse"], 0.0, atol=1e-9)


def test_to_output_format():
    df = make_step_df()
    model = DummyModel()
    df_pred = model.predict_dataframe(df)

    out = model.to_output_format(df_pred)
    expected_cols = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]
    assert list(out.columns) == expected_cols

    assert np.allclose(out["x"].to_numpy(), df_pred["x_pred"].to_numpy())
    assert np.allclose(out["y"].to_numpy(), df_pred["y_pred"].to_numpy())


def test_dummy_bayes_predict_shapes_and_summary():
    df = make_step_df()
    model = DummyBayesModel(sigma=0.1)

    model.fit_bayes(df)

    x_samps, y_samps = model.posterior_samples_for_rows(df, n_samples=50)
    assert x_samps.shape == (50, len(df))
    assert y_samps.shape == (50, len(df))

    df_mean = model.predict_dataframe(df, summary="mean")
    df_med = model.predict_dataframe(df, summary="median")

    assert "x_pred" in df_mean.columns and "y_pred" in df_mean.columns
    assert "x_pred" in df_med.columns and "y_pred" in df_med.columns

    assert np.allclose(
        df_mean["x_pred"].to_numpy(), df["x"].to_numpy() + 1.0, atol=0.3
    )
    assert np.allclose(
        df_mean["y_pred"].to_numpy(), df["y"].to_numpy() + 2.0, atol=0.3
    )


def test_bayes_posterior_artists_update():
    df = make_step_df(n=3)
    model = DummyBayesModel(sigma=0.1)

    fig, ax = plt.subplots()
    artists = model.init_posterior_artists(ax)

    x_samps, y_samps = model.posterior_samples_for_rows(df.iloc[[0]], n_samples=30)
    x0 = x_samps[:, 0]
    y0 = y_samps[:, 0]

    samples_artist, mean_artist = model.update_posterior_artists(
        artists, x0, y0
    )

    sx, sy = samples_artist.get_data()
    mx, my = mean_artist.get_data()

    assert len(sx) == len(x0)
    assert len(sy) == len(y0)
    assert np.isclose(mx, np.mean(x0))
    assert np.isclose(my, np.mean(y0))

    plt.close(fig)

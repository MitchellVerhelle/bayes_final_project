# models/play_success_base.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class PlaySuccessModel(ABC):
    """
    Base class for models that predict play-level success (0/1) and
    can be evaluated frame-by-frame for a single play.

    Responsibilities:
      - fit(...) on a play-level dataset
      - frame_prob_dict(d_play): frame_id -> P(success | frame t info)
    """

    @abstractmethod
    def fit(self, play_df: pd.DataFrame) -> None:
        """
        Fit the model on a play-level table.

        play_df must contain:
            - 'play_success' (0/1 label)
            - any model-specific feature columns.
        """
        ...

    @abstractmethod
    def frame_prob_dict(self, d_play: pd.DataFrame) -> Dict[int, float]:
        """
        Given all input frames for a single (game_id, play_id) from
        input_2023_w*.csv, return a mapping:

            frame_id -> P(success | features at that frame)

        d_play is expected to be the raw tracking rows (one per player-frame)
        for that play.
        """
        ...

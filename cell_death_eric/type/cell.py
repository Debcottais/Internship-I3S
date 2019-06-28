# from __future__ import annotations

import numpy as np_
from typing import Sequence, Tuple


coords_h = Tuple[np_.ndarray, np_.ndarray]


class cell_t:
    def __init__(self) -> None:
        #
        self.uid = -1  # 0 and up: unique ID within its frame
        # It corresponds to its position in its frame list of cells

        self.time_point = -1  # 0 and up: time point of the frame it belongs to
        self.position = ()  # (row, col): position of its centroid in the frame (not necessarily integers)
        self.pixels = None  # As returned by np_.nonzero: list of its composing pixels

        self.features = {}  # Dictionary of its feature values

    @classmethod
    def WithProperties(
        cls, uid: int, time_point: int, position: Sequence[float], pixels: coords_h
    ) -> 'cell_t':
        #
        instance = cls()

        instance.uid = uid
        instance.time_point = time_point
        instance.position = position
        instance.pixels = pixels

        return instance

    def __str__(self) -> str:
        #
        return (
            f"{self.uid} "
            f"in {self.time_point} "
            f"@ {self.position[0]}x{self.position[1]}"
        )

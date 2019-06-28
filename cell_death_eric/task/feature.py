import numpy as np_
from typing import Sequence, Tuple


coords_h = Tuple[np_.ndarray, np_.ndarray]


def MOMPLoc(cfp: np_.ndarray, momp: np_.ndarray, roi_coords: coords_h) -> float:
    #
    local_cfp = cfp[roi_coords]
    max_cfp_value = np_.max(local_cfp)
    max_map_at_coords = (cfp == max_cfp_value)[roi_coords]
    first_max_idx = np_.nonzero(max_map_at_coords)[0][0]

    row = roi_coords[0][first_max_idx]
    col = roi_coords[1][first_max_idx]
    roi_momp = momp[
        max(row - 2, 0) : min(row + 2, momp.shape[0] - 1),
        max(col - 2, 0) : min(col + 2, momp.shape[1] - 1),
    ]

    return np_.percentile(roi_momp, 5)


def Edginess(
    channel: np_.ndarray, roi_coords: coords_h, origin: Sequence[float]
) -> np_.ndarray:
    #
    row_shifts = (-1, 0, 1, -1, 1, -1, 0, 1)
    col_shifts = (-1, -1, -1, 0, 0, 1, 1, 1)
    profile_length = 5

    n_shifts = row_shifts.__len__()
    cardinal_jumps = np_.zeros(n_shifts, dtype=np_.float64)

    origin_row = int(round(origin[0]))
    origin_col = int(round(origin[1]))

    roi_map = np_.zeros(channel.shape, dtype=np_.bool)
    roi_map[roi_coords] = True
    rolling_profile = np_.empty(profile_length, dtype=channel.dtype)
    out_dist_threshold = (profile_length // 2) + 1

    for line_idx in range(n_shifts):
        row = origin_row
        col = origin_col
        out_dist = 0
        prof_idx = 0
        rolling_profile.fill(0)

        while out_dist < out_dist_threshold:
            prev_intensity = channel[row, col]

            row += row_shifts[line_idx]
            if (row < 0) or (row >= channel.shape[0]):
                break
            col += col_shifts[line_idx]
            if (col < 0) or (col >= channel.shape[1]):
                break

            prof_idx += 1
            rolling_profile[prof_idx % profile_length] = (
                prev_intensity - channel[row, col]
            )
            if not roi_map[row, col]:
                out_dist += 1

        cardinal_jumps[line_idx] = max(rolling_profile)

    return cardinal_jumps

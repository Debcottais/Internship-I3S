# from __future__ import annotations

from type.cell import cell_t
from task import feature as ft_, segmentation as sg_
from type.frame import frame_t
from type.tracks import tracks_t

import sys, os 
import imageio as io_
import matplotlib.pyplot as pl_
import networkx as nx_
import numpy as np_
import pathlib as ph_
import scipy.spatial.distance as dt_
from typing import Any, Callable, List, Sequence, Tuple, Union


class sequence_t:
    def __init__(self) -> None:
        #
        self.frames = {}  # Dictionary "channel" -> list of frames
        self.segm_channel = None  # Channel used for segmentation
        self.tracking = None

    @classmethod
    def FromTiffFile(
        cls,
        path: Union[str, ph_.Path],
        channel_names: Sequence[str],
        from_frame: int = 0,
        to_frame: int = 999999,
        post_processing_fct: Callable = None,
    ) -> 'sequence_t':
        #
        # Channel name == '___' => discarded
        #
        instance = cls()

        n_channels = channel_names.__len__()
        for name in channel_names:
            if name != "___":
                instance.frames[name] = []

        ch_idx = n_channels - 1
        time_point = -1
        frame_reader = io_.get_reader(path)

        for raw_frame in frame_reader:
            ch_idx += 1
            if ch_idx == n_channels:
                ch_idx = 0
                time_point += 1

            if time_point < from_frame:
                continue
            elif time_point > to_frame:
                break

            if post_processing_fct is not None:
                raw_frame = post_processing_fct(raw_frame)
            name = channel_names[ch_idx]
            frame = frame_t.WithProperties(name, time_point, raw_frame)
            if name != "___":
                instance.frames[name].append(frame)

        return instance

    def __str__(self) -> str:
        #
        self_as_str = ""

        for channel, frames in self.frames.items():
            initial_time_point = frames[0].time_point
            self_as_str += (
                f"[{channel}]\n"
                f"    {frames[0].size[1]}x{frames[0].size[0]}x"
                f"[{initial_time_point}..{frames[-1].time_point}]\n"
            )

            for f_idx, frame in enumerate(frames):
                if frame.cells is not None:
                    self_as_str += (
                        f"    {initial_time_point+f_idx}: c {frame.cells.__len__()}\n"
                    )

        return self_as_str

    def RegisterChannelsOn(self, ref_channel: str) -> None:
        #
        channel_names = tuple(self.frames.keys())

        ref_channel_idx = channel_names.index(ref_channel)
        other_channel_idc = set(range(channel_names.__len__())) - {ref_channel_idx}

        ref_frames = self.frames[ref_channel]

        for c_idx in other_channel_idc:
            floating_frames = self.frames[channel_names[c_idx]]
            for f_idx in range(ref_frames.__len__()):
                floating_frames[f_idx].RegisterOn(ref_frames[f_idx].contents)

    def SegmentCellsOnChannel(
        self,
        channel: str,
        min_area: int = 1,
        max_area: int = 999999,
        method: str = "sobel",
    ) -> None:
        #
        self.segm_channel = channel

        for frame in self.frames[channel]:
            frame.SegmentCells(min_area=min_area, max_area=max_area, method=method)

    def CellSegmentationAt(self, time_point: int, binary: bool = False) -> np_.ndarray:
        #
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        frames = self.frames[self.segm_channel]
        frame = frames[time_point - frames[0].time_point]

        return frame.CellSegmentation(binary=binary)

    def PlotSegmentations(self, show_figure: bool = True) -> None:
        #
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        segmentations = []
        from_frame = self.frames[self.segm_channel][0].time_point
        to_frame = self.frames[self.segm_channel][-1].time_point
        for time_point in range(from_frame, to_frame + 1):
            segmentations.append(self.CellSegmentationAt(time_point))

        sg_.PlotSegmentations(segmentations, show_figure=show_figure)

    def RootCells(self) -> List[cell_t]:
        #
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        return self.frames[self.segm_channel][0].cells

    def TrackCells(
        self, max_dist: float = np_.inf, bidirectional: bool = False
    ) -> None:
        #
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        self.tracking = tracks_t()
        frames = self.frames[self.segm_channel]

        for f_idx in range(1, frames.__len__()):
            prev_frame = frames[f_idx - 1]
            curr_frame = frames[f_idx]
            translation = curr_frame.TranslationWRT(prev_frame)

            prev_pos = prev_frame.CellPositions()
            curr_pos = curr_frame.CellPositions(offset=translation)
            all_dist = dt_.cdist(prev_pos, curr_pos)

            for curr_cell in curr_frame.cells:
                curr_uid = curr_cell.uid
                prev_uid = np_.argmin(all_dist[:, curr_uid])
                if all_dist[prev_uid, curr_uid] <= max_dist:
                    if bidirectional:
                        # sym=symmetric
                        sym_curr_uid = np_.argmin(all_dist[prev_uid, :])
                        if sym_curr_uid == curr_uid:
                            # Note: a cell has a unique next cell due to bidirectional constraint
                            prev_cell = prev_frame.cells[prev_uid]
                            self.tracking.add_edge(prev_cell, curr_cell)
                    else:
                        prev_cell = prev_frame.cells[prev_uid]
                        self.tracking.add_edge(prev_cell, curr_cell)

    def PlotTracking(self, show_figure: bool = True) -> None:
        #
        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")

        self.tracking.Plot(show_figure=show_figure)

    def CellFeatureNames(self) -> Tuple[str]:
        #
        return tuple(self.frames[self.segm_channel][0].cells[0].features.keys())

    def ComputeCellFeatures(self, three_channels: Sequence[str]) -> None:
        #
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        # Segmentation must have been performed on the first channel of the list
        if three_channels[0] != self.segm_channel:
            raise ValueError(
                f"{three_channels[0]}: First channel must be segmentation channel {self.segm_channel}"
            )

        for frame_ch1, frame_ch2, frame_ch3 in zip(
            self.frames[three_channels[0]],
            self.frames[three_channels[1]],
            self.frames[three_channels[2]],
        ):
            channel_1 = frame_ch1.contents
            channel_2 = frame_ch2.contents
            channel_3 = frame_ch3.contents

            for cell in frame_ch1.cells:
                intensities_1 = channel_1[cell.pixels]
                intensities_2 = channel_2[cell.pixels]

                cell.features["FRET"] = np_.median(intensities_2)
                cell.features["CFP-over-FRET"] = np_.median(
                    intensities_1 / intensities_2
                )

                if channel_3 is not None:
                    cell.features["MOMP"] = ft_.MOMPLoc(
                        channel_1, channel_3, cell.pixels
                    )
                    cell.features["RFP"] = np_.percentile(channel_3[cell.pixels], 80)

                cell.features["edge"] = ft_.Edginess(
                    channel_1, cell.pixels, cell.position
                )

    def CellFeatureEvolution(
        self, root_cell: cell_t, feature: str
    ) -> List[List[Tuple[int, Any]]]:
        #
        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")
        # self.segm_channel is necessarily not None

        evolution = []

        track = self.tracking.TrackContainingCell(root_cell)
        if track is not None:
            current_piece = []
            for from_cell, to_cell in nx_.dfs_edges(track, source=root_cell):
                if current_piece.__len__() == 0:
                    current_piece.append(
                        (from_cell.time_point, from_cell.features[feature])
                    )
                current_piece.append((to_cell.time_point, to_cell.features[feature]))

                if track.out_degree(to_cell) != 1:
                    evolution.append(current_piece)
                    current_piece = []

        return evolution

    def PlotCellFeatureEvolutions(
        self, cell_list: Sequence[cell_t], feature: str, show_figure: bool = True
    ) -> None:
        #
        figure = pl_.figure()
        axes = figure.gca()
        axes.set_title(feature)

        plots = []
        labels = []
        colors = "bgrcmyk"

        for root_cell in cell_list:
            color_idx = root_cell.uid % colors.__len__()
            subplot = None

            for piece in self.CellFeatureEvolution(root_cell, feature):
                if not (isinstance(piece[0][1], int) or isinstance(piece[0][1], float)):
                    break

                time_points = []
                feature_values = []
                for time_point, feature_value in piece:
                    time_points.append(time_point)
                    feature_values.append(feature_value)

                subplot = axes.plot(
                    time_points, feature_values, colors[color_idx] + "-x"
                )[0]

            if subplot is not None:
                plots.append(subplot)
                labels.append(f"root_cell {root_cell.uid}")

        if plots.__len__() > 0:
            axes.legend(handles=plots, labels=labels)
        else:
            pl_.close(figure)

        if show_figure:
            pl_.show()

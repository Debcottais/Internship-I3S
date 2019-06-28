# from __future__ import annotations

from task import segmentation as sg_
from type.cell import cell_t

import numpy as np_
import skimage.feature as ft_
import skimage.measure as ms_
import skimage.morphology as mp_
import skimage.transform as tf_
from typing import Sequence, Union


class frame_t:
    def __init__(self) -> None:
        #
        self.channel = ""  # Name of the channel of the frame
        self.time_point = -1  # 0 and up: time point of the frame
        self.size = None  # (height, width) of the frame
        self.contents = None  # numpy.ndarray of the frame contents

        self.cells = None  # List of segmented cells

    @classmethod
    def WithProperties(
        cls, channel: str, time_point: int, contents: np_.ndarray
    ) -> 'frame_t':
        #
        instance = cls()

        instance.channel = channel
        instance.time_point = time_point
        instance.size = contents.shape
        instance.contents = contents

        return instance

    def ClearContents(self) -> None:
        self.contents = None  # To free some memory up when segmentation done

    def TranslationWRT(self, ref_frame: Union[np_.ndarray, 'frame_t']) -> np_.ndarray:
        #
        if isinstance(ref_frame, str):
            ref_frame = ref_frame.contents

        return ft_.register_translation(
            ref_frame,
            self.contents,
            upsample_factor=8,
            space="real",
            return_error=False,
        )

    def RegisterOn(self, ref_frame: Union[np_.ndarray, 'frame_t']) -> None:
        #
        if isinstance(ref_frame, type('frame_t')):
            ref_frame = ref_frame.contents

        translation = self.TranslationWRT(ref_frame)
        translation = tf_.EuclideanTransform(translation=translation)
        self.contents = tf_.warp(self.contents, translation)

    def SegmentCells(
        self, min_area: int = 1, max_area: int = 999999, method: str = "sobel"
    ) -> None:
        #
        if method == "original":
            segmentation = sg_.CellSegmentation_Original(self.contents)
        elif method == "slic":
            segmentation = sg_.CellSegmentation_SLIC(self.contents)
        elif method == "contours":
            segmentation = sg_.CellSegmentation_Contours(self.contents)
        elif method == "sobel":
            segmentation = sg_.CellSegmentation_Sobel(self.contents)
        else:
            raise ValueError(f"{method}: Invalid segmentation method")

        # Segmentation labeling and cell instantiations
        labeled_sgm = mp_.label(segmentation, connectivity=1)
        cell_props = ms_.regionprops(labeled_sgm)
        self.cells = []
        uid = 0
        for props in cell_props:
            if min_area <= props.area <= max_area:
                coords = (props.coords[:, 0], props.coords[:, 1])
                cell = cell_t.WithProperties(
                    uid, self.time_point, props.centroid, coords
                )
                cell.features["area"] = props.area
                self.cells.append(cell)
                uid += 1

    def CellPositions(self, offset: Sequence[float] = None) -> np_.ndarray:
        #
        if self.cells is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        positions = np_.empty((self.cells.__len__(), 2), dtype=np_.float64)

        for idx, cell in enumerate(self.cells):
            positions[idx, :] = cell.position

        if offset is not None:
            offset = np_.array(offset, dtype=positions.dtype)
            positions += offset

        return positions

    def CellSegmentation(self, binary: bool = False) -> np_.ndarray:
        #
        if self.cells is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        segmentation = np_.zeros(self.size, dtype=np_.uint16)

        if binary:
            for cell in self.cells:
                segmentation[cell.pixels] = 1
        else:
            for cell in self.cells:
                segmentation[cell.pixels] = cell.uid + 1  # 0 reserved for background

        return segmentation

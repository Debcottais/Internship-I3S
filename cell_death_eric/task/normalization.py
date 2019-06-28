import numpy as np_
import scipy.interpolate as in_
import skimage as im_
from typing import Tuple


def ContrastNormalized(img: np_.ndarray, block_shape: Tuple[int, int]) -> np_.ndarray:
    #
    assert (block_shape[0] % 2 == 1) and (block_shape[1] % 2 == 1)

    res_img = ImageCroppedToEntireBlocks(img, block_shape)
    lmp_img = LocalMostPresent(res_img, block_shape)
    rescaled = RescaledImage(lmp_img, block_shape, res_img.shape)

    res_img = BlockBasedCroppedImage(res_img, block_shape)
    rescaled = BlockBasedCroppedImage(rescaled, block_shape)

    return res_img - rescaled


def ImageCroppedToEntireBlocks(
    img: np_.ndarray, block_shape: Tuple[int, int]
) -> np_.ndarray:
    #
    row_margin = img.shape[0] % block_shape[0]
    col_margin = img.shape[1] % block_shape[1]

    row_half_margin = row_margin // 2
    col_half_margin = col_margin // 2

    if (row_margin > 0) and (col_margin > 0):
        return np_.array(
            img[
                row_half_margin : (row_half_margin - row_margin),
                col_half_margin : (col_half_margin - col_margin),
            ]
        )
    elif row_margin > 0:
        return np_.array(img[row_half_margin : (row_half_margin - row_margin), :])
    elif col_margin > 0:
        return np_.array(img[:, col_half_margin : (col_half_margin - col_margin)])

    return np_.array(img)


def LocalMostPresent(img: np_.ndarray, block_shape: Tuple[int, int]) -> np_.ndarray:
    #
    view = im_.util.view_as_blocks(img, block_shape)
    local_most = np_.empty(view.shape[:2], dtype=np_.float64)

    for row in range(view.shape[0]):
        for col in range(view.shape[1]):
            block = view[row, col, :, :]

            hist, bin_edges = np_.histogram(
                np_.log(block + 1.0), bins=max(block.size // 100, 10), density=False
            )
            hist = im_.filters.gaussian(hist.astype(np_.float64), sigma=3)
            most_present = np_.argmax(hist)

            local_most[row, col] = (
                np_.mean(np_.exp(bin_edges[most_present : (most_present + 2)])) - 1.0
            )

    return local_most


def RescaledImage(
    img: np_.ndarray, block_shape: Tuple[int, int], full_size: Tuple[int, int]
) -> np_.ndarray:
    #
    block_half_shape = (block_shape[0] // 2, block_shape[1] // 2)
    new_size = (full_size[0] - block_shape[0] + 1, full_size[1] - block_shape[1] + 1)

    rescaled = np_.zeros((full_size[0], img.shape[1]), dtype=np_.float64)

    old_rows = range(img.shape[0])
    flt_rows = np_.linspace(0, old_rows[-1], new_size[0])
    new_rows = slice(block_half_shape[0], rescaled.shape[0] - block_half_shape[0])
    for col in range(img.shape[1]):
        rescaled[new_rows, col] = in_.pchip_interpolate(old_rows, img[:, col], flt_rows)

    img = rescaled
    rescaled = np_.zeros(full_size, dtype=np_.float64)

    old_cols = range(img.shape[1])
    flt_cols = np_.linspace(0, old_cols[-1], new_size[1])
    new_cols = slice(block_half_shape[1], rescaled.shape[1] - block_half_shape[1])
    for row in range(img.shape[0]):
        rescaled[row, new_cols] = in_.pchip_interpolate(old_cols, img[row, :], flt_cols)

    return im_.filters.gaussian(rescaled, sigma=9)

    # img = im_.filters.gaussian(rescaled, sigma=9)
    # return (img - img.min())/(img.max() - img.min())
    
    # from skimage import filters 
    
    # filt_real, filt_imag = filters.gabor(rescaled, frequency=0.9)
    # return filt_imag

def BlockBasedCroppedImage(
    img: np_.ndarray, block_shape: Tuple[int, int]
) -> np_.ndarray:
    #
    block_half_shape = (block_shape[0] // 2, block_shape[1] // 2)

    return np_.array(
        img[
            block_half_shape[0] : (img.shape[0] - block_half_shape[0]),
            block_half_shape[1] : (img.shape[1] - block_half_shape[1]),
        ]
    )


# basic test of normalisation 

# img_norm = (img - img.min())/(img.max() - img.min())

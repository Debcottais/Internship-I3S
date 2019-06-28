import matplotlib.pyplot as pl_
from matplotlib.widgets import Slider as mp_slider_t
import numpy as np_
import scipy.ndimage as im_
import skimage.morphology as mp_
import skimage.segmentation as sg 

from typing import Sequence, Union


def CellSegmentation_Original(frame: np_.ndarray) -> np_.ndarray:
    #
    # 1.2 = 0.5*hsize(12) / sigma(5)
    smooth_frm = im_.gaussian_filter(frame, 5, truncate=1.2)
    crest_lines_map = mp_.watershed(-smooth_frm, watershed_line=True) == 0

    # Threshold-based binary segmentation
    sgm_threshold = 3 * np_.percentile(np_.fabs(frame), 20)
    segmentation = frame > sgm_threshold
    segmentation[crest_lines_map] = 0
    segmentation = mp_.binary_erosion(segmentation, selem=mp_.disk(3))

    return segmentation

def CellSegmentation_SLIC(frame: np_.ndarray) ->np_.ndarray:

    """"""""
    "superpixelisation method for cells segmentation"

    """"""""
    frame+=frame.min()
    frame/=frame.max()
    smooth_frm = im_.gaussian_filter(frame, 3)


    segments = sg.slic(smooth_frm*255, n_segments = 850, sigma = 0, compactness=10, enforce_connectivity=True)
    
    segm_empty = np_.empty_like(segments, dtype=np_.float64)

    for label in range(segments.max()+1):
        region = segments==label
        segm_empty[region]=np_.mean(smooth_frm[region])


    segmentation = np_.zeros_like(segments)

    for label in range(segments.max() + 1):
        region = segments == label
        dilated_region = mp_.binary_dilation(region)
        ring_region = np_.logical_and(dilated_region, np_.logical_not(region))
        labeled_background = mp_.label(np_.logical_not(region), connectivity=1)
        if (np_.min(segm_empty[region]) > np_.max(segm_empty[ring_region])) and (labeled_background.max() == 1):
            segmentation[region] = label

    segmentation = sg.relabel_sequential(segmentation)[0]

    return segmentation 

def CellSegmentation_Contours(frame: np_.ndarray) -> np_.ndarray:
    from skimage import filters, measure, feature  
    from sklearn.cluster import KMeans

    im_cfp_filt = im_.gaussian_filter(frame, 3)

    sgm_threshold = filters.threshold_otsu(im_cfp_filt)
    segmentation = im_cfp_filt>sgm_threshold
    
    # h,w=segmentation.shape[:2]
    # im_small_long = segmentation.reshape((h * w, 1)) #1=nb of channel 
    # im_small_wide = im_small_long.reshape((h,w,1))
    # km = KMeans(n_clusters=5)
    # km.fit(im_small_long)
    
    # seg = np_.asarray([(1 if i == 1 else 0)
    #               for i in km.labels_]).reshape((h,w))
    # contours = measure.find_contours(seg, sgm_threshold, fully_connected="high")
    # simplified_contours = [measure.approximate_polygon(c, tolerance=5) 
    #                     for c in contours]
    # for n, contour in enumerate(simplified_contours):
    #     pl_.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    # segmentation=feature.canny(segmentation)
    
    return segmentation


def CellSegmentation_Sobel(frame: np_.ndarray) -> np_.ndarray: 
    
    from skimage import filters 
    
    frame+=frame.min()
    frame/=frame.max()
  
    smooth_frm = filters.sobel(frame)
    
    segments = sg.slic(smooth_frm*255, n_segments = 850, sigma = 0, compactness=10, enforce_connectivity=True)
    
    segm_empty = np_.empty_like(smooth_frm, dtype=np_.float64)

    for label in range(segments.max()+1):
        region = segments==label
        segm_empty[region]=np_.mean(smooth_frm[region])


    segmentation = np_.zeros_like(segments)

    for label in range(segments.max() + 1):
        region = segments == label
        dilated_region = mp_.binary_dilation(region)
        ring_region = np_.logical_and(dilated_region, np_.logical_not(region))
        labeled_background = mp_.label(np_.logical_not(region), connectivity=1)
        if (np_.min(segm_empty[region]) > np_.max(segm_empty[ring_region])) and (labeled_background.max() == 1):
            segmentation[region] = label

    segmentation = sg.relabel_sequential(segmentation)[0]

    return segmentation 

########################################################################
def PlotSegmentations(
    segmentations: Union[np_.ndarray, Sequence[np_.ndarray]], show_figure: bool = True
) -> None:
    #
    if isinstance(segmentations, np_.ndarray):
        pl_.matshow(segmentations)
    else:

        def __UpdateFigure__(Frame, figure_, plot_, segmentations_):
            idx = int(round(Frame))
            plot_.set_data(segmentations_[idx])
            figure_.canvas.draw_idle()

        figure = pl_.figure()
        plot_axes = figure.add_axes([0.1, 0.2, 0.8, 0.65])
        slider_axes = figure.add_axes([0.1, 0.05, 0.8, 0.05])
        sgm_plot = plot_axes.matshow(segmentations[0])
        slider = mp_slider_t(
            slider_axes, "Frame", 0, segmentations.__len__() - 1, valinit=0
        )

        update_figure_fct = lambda value: __UpdateFigure__(
            value, figure, sgm_plot, segmentations
        )
        slider.on_changed(update_figure_fct)

    if show_figure:
        pl_.show()

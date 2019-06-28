from task import normalization as nm_
from type.sequence import sequence_t

import matplotlib.pyplot as pl_


from run_parameters import *


print("Reading Sequence")
post_processing_fct = lambda img: nm_.ContrastNormalized(img, block_size)
sequence = sequence_t.FromTiffFile(
# sequence = sequence_t.FromImageFolders(
    sequence_path,
    ("CFP", "FRET", "MOMP", "___"),
    from_frame=from_frame,
    to_frame=to_frame,
    post_processing_fct=post_processing_fct,
)

print("Registration")
sequence.RegisterChannelsOn("CFP")

print("Segmentation")
sequence.SegmentCellsOnChannel("CFP",min_area=min_area) 
print(sequence)
sequence.PlotSegmentations()

print("Tracking")
sequence.TrackCells()
sequence.PlotTracking()

print("Tracking Feature Computation")
sequence.ComputeCellFeatures(("CFP", "FRET", "MOMP"))
root_cell_list = tuple(
    root_cell for c_idx, root_cell in enumerate(sequence.RootCells()) if c_idx % 50 == 0
)
for feature in sequence.CellFeatureNames():
    sequence.PlotCellFeatureEvolutions(root_cell_list, feature, show_figure=False)
pl_.show()

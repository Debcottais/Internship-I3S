# from __future__ import annotations

from type.cell import cell_t

import matplotlib.pyplot as pl_
from mpl_toolkits import mplot3d
import networkx as nx_
from typing import Optional


class tracks_t(nx_.DiGraph):
    def TrackContainingCell(self, cell: cell_t) -> Optional[nx_.DiGraph]:
        #
        for component in nx_.weakly_connected_components(self):
            if cell in component:
                return self.subgraph(component)

        return None

    def Plot(self, show_figure: bool = True) -> None:
        #
        figure = pl_.figure()
        axes = figure.add_subplot(projection=mplot3d.Axes3D.name)
        colors = "bgrcmyk"

        for c_idx, component in enumerate(nx_.weakly_connected_components(self)):
            color_idx = c_idx % colors.__len__()
            for from_cell, to_cell in self.subgraph(component).edges:
                time_points = (from_cell.time_point, to_cell.time_point)
                rows = (from_cell.position[0], to_cell.position[0])
                cols = (from_cell.position[1], to_cell.position[1])
                axes.plot3D(rows, cols, time_points, colors[color_idx])

        if show_figure:
            pl_.show()

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


def siddon_algo_2d(start: Tensor, end: Tensor, height: int = 48, width: int = 36):
    """Find intersection points between a grid and a line.

    An adaption of Siddon algorithm. The grid is defined as::

        x = (2k + 1) / 2, for k integer (-inf, +inf)
        y = (2k + 1) / 2, for k integer (-inf, +inf)

    x for vertical lines and y for horizontal line,
    in particular, grid cells are centered around (i, j),
    where i, j are integers (-inf, +inf)

    The line parameterization used to find points::

        line = start + t * (end - start)

    where, t is in [0, 1].

    Parameters
    ----------
    start : Tensor, shape (2,)
        The starting point of the line.

    end : Tensor, shape (2,)
        The ending point of the line.

    height : int, default=48
        The height of the grid.

    width : int, default=36
        The width of the grid.

    Return
    ------
    intersection_points, dist, grid : Tuple[Tensor, Tensor, Tensor]
        The intersection points [start, *intersection_points, end]
        the distance between the points
        the grid where each grid cell has value equal to the distance of the intersection

    Notes
    -----
    - The start and end are also part of the returned points.
    """
    # line is invariant to start/end (swapping them doesn't change anything)
    # fixing the convention of starting from the point closest to the origin over the x-axis
    start, end = sorted([start, end], key=lambda x: x[0])

    t_params = []
    for s, e in zip(start, end):

        diff = e - s
        if diff == 0:
            continue

        # get integers in between the interval
        # NOTE since arange requires start < end, the order is manually handled
        # and then reverted it back
        # NOTE left/right are mapped then remapped again to account for the grid not begin on integer, in particular,
        #     left <= (2*k + 1) / 2 <= right   <----> (2*left-1) / 2 <= k <= (2*right-1) / 2
        left, right = (s, e) if diff > 0 else (e, s)

        left, right = (2 * left - 1) / 2, (2 * right - 1) / 2
        interval = torch.arange(left.ceil(), right.floor() + 1)
        if diff < 0:
            interval = interval.flip(0)
        interval = (2 * interval + 1) / 2

        interval = torch.cat((s[None], interval, e[None]))
        interval = interval.unique()

        # deduce the line params
        t = (interval - s) / diff
        t_params.append(t)

    t_params = torch.cat(t_params).unique()
    intersection_points = start + t_params[:, None] * (end - start)

    # --- generate grid
    # compute distance between points (NOTE they should be sorted first)
    # for that select on point and compute with respect to it
    arg_sorted = torch.argsort(
        (intersection_points - intersection_points[[0]]).norm(dim=-1)
    )
    intersection_points = intersection_points[arg_sorted]
    dist = intersection_points.diff(dim=0).norm(dim=-1)

    # NOTE deciding on the grid cell depends on whether start is above or below end
    grid = torch.zeros((height, width))
    if end[1] - start[1] >= 0:
        indices = (intersection_points[:-1] + 1 / 2).floor().to(torch.long)
        indices_x, indices_y = indices[:, 0], indices[:, 1]
    else:
        indices_x = (intersection_points[:-1, 0] + 1 / 2).floor().to(torch.long)
        indices_y = (intersection_points[:-1, 1] - 1 / 2).ceil().to(torch.long)

    # NOTE xy are inverted since columns represent x-axis and rows y-axis
    grid[indices_y, indices_x] = dist
    # ---

    return intersection_points, dist, grid


# --- this for test only
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # grid_size = 10
    height, width = 48, 36

    start = torch.tensor([1.5, 2.1])
    # end = torch.tensor([4.7, 7.2])
    end = torch.tensor([32.4, 45.7])
    # end = torch.tensor([4.7, 3.2])

    # start = torch.tensor([1.5, 5.1])
    # end = torch.tensor([4.7, 1.2])
    # start = torch.tensor([1.0, 5.0])
    # end = torch.tensor([3.0, 1.0])

    line = torch.cat((start[None], end[None]))
    points, dist, grid = siddon_algo_2d(start, end, height=height, width=width)

    plt.plot(line[:, 0], line[:, 1], marker="o")
    plt.scatter(points[:, 0], points[:, 1], color="red", marker="x")

    plt.imshow(grid, cmap="Grays", origin="lower")

    plt.xticks(torch.arange(width) - 1 / 2)
    plt.yticks(torch.arange(height) - 1 / 2)

    plt.colorbar(
        label="length of the segments",
        # orientation="horizontal",
    )

    plt.grid()
    plt.gca().set_aspect(1)
"""provides tools for selecting region-of-interest (roi) on PIL images"""
__version__ = "0.1.0"

from typing import Any, Callable, Iterable, Literal, NamedTuple, NewType, ParamSpec, Self, Tuple, TypeAlias, TypeVarTuple, Annotated, Dict, TypeVar
from numpy.typing import NDArray
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.draw import line, polygon_perimeter
# from skimage.segmentation import flood


def flood(
	image: Annotated[NDArray, "N-dim"],
	seed_point: Annotated[tuple[int | None, ...], "N-point"],
	*, footprint: Annotated[NDArray, "N-dim"] | None = None,
	connectivity: float | None = None,
	tolerance: float | None = None
) -> NDArray[bool]:
	pass


ValueAtPoint: TypeAlias = int | NDArray
PointIndex: TypeAlias = tuple[int, ...]  # N-dimesional point
PlantingCondition: TypeAlias = Callable[[PointIndex, ValueAtPoint], bool]
FloodFunc: TypeAlias = Callable[[Annotated[NDArray, "image ndarray"], Annotated[PointIndex, "point-index or slice of image"]], NDArray[np.uint8] | NDArray[np.bool8]]


def flood_along_points(
	image: Annotated[NDArray, "N-dim"],
	seed_points: Annotated[tuple[tuple[int, ...] | None, ...], "list of P-points of ith-coordinates"],
	planting_condition: PlantingCondition,
	flood_func: FloodFunc,
	**kwargs: Annotated[Any, "flood_func.params.kwargs"]
):
	"""_summary_

	Parameters
	----------
	image : NDArray of dimensions `N`
		an `N` dimensional np array. if you're working with `RGB`, `RGBA`, etc... image arrays,
		then you'd typically want `N = 3`, with the coodinate ordering as `[y, x, c]` (where `c` stands for channel)

	seed_points : tuple of length `N` of ( (`P` point-indexes of ith-coordinate) or `None` )
		supposing you have `P = 7` points, and `N = 3` dimisional `image` of coordinates `[y, x, c]`.
		and you want to place `seed_points` at `x` and `y` coordinates, but not the color `c` coordinates,
		you'd define `seed_points` as:

		`seed_points = [[y0, y1, y2, y3, y4, y5, y6], [x0, x1, x2, x3, x4, x5, x6], None]`

	planting_condition : ndarray, optional
		The footprint (structuring element) used to determine the neighborhood
		of each evaluated pixel. It must contain only 1's and 0's, have the
		same number of dimensions as `image`. If not given, all adjacent pixels
		are considered as part of the neighborhood (fully connected).

	**kwargs : flood_func.params.kwargs, optional
		see `kwargs` of your `flood_func`

	Returns
	-------
	mask : NDArray[bool] of dimensions `N`, with `K` empty dimensions, where `K =` number of `None` in `seed_points`
		a boolean ndarray with the same shape as `image` is returned,
		with True values for areas connected to the seed point,
		based off the `planting_condition`, and `connectivity_condition`.
		all other values are False.

	"""
	dims = len(seed_points)
	assert dims == img.ndim
	n_points = np.min([len(coords) for coords in seed_points if coords is not None])
	# seed_points2: list[tuple[int, ...]] = [seed_points[d][0: n_points] for d in range(dims) if seed_points[d] is not None]
	point_indexes: Iterable[tuple[int | slice, ...]] = zip(*[
		seed_points[d][0: n_points] if seed_points[d] is not None
		else [slice(0, None)] * n_points
		for d in range(dims)
	])
	selection_point_indexes: Iterable[tuple[int, ...]] = zip(*[
		seed_points[d][0: n_points]
		for d in range(dims)
		if seed_points[d] is not None
	])
	selection_shape: list[int] = [
		img.shape[d]
		for d in range(dims)
		if seed_points[d] is not None
	]
	selection = np.zeros(selection_shape, np.ubyte)
	for img_p, sel_p in zip(point_indexes, selection_point_indexes):
		if planting_condition(img_p, img[*img_p]) and selection[*sel_p] == 0:
			selection |= flood_func(img, img_p, **kwargs)


polygon_coords = [180, 3100, 1130, 3950, 1740, 3920, 1900, 3670]
upper_color = [200, 200, 200]
lower_color = [180, 180, 180]


def seeding_condition(y: int, x: int, color: tuple[int, int, int]):
	if np.all(np.greater(color, lower_color)) and np.all(np.greater(upper_color, color)):
		return True
	return False


img = imread("./q2,n20-242.jpg")

xc = polygon_coords[::2]
yc = polygon_coords[1::2]
rrs: list[NDArray] = []
ccs: list[NDArray] = []
for i in range(1, len(xc)):
	rs, cs = line(yc[i - 1], xc[i - 1], yc[i], xc[i])
	rrs.append(rs)
	ccs.append(cs)
rr: NDArray = np.concatenate(rrs)
cc: NDArray = np.concatenate(ccs)
del rrs, ccs
selection = np.zeros((img.shape[0], img.shape[1]), np.ubyte,)
count = 0
for y, x in zip(rr, cc):
	if seeding_condition(y, x, img[y, x]) and selection[y, x] == 0:
		count += 1
		selection |= flood(img[:, :, 0], (y, x), tolerance=5) & flood(img[:, :, 1], (y, x), tolerance=5) & flood(img[:, :, 2], (y, x), tolerance=5)
		# selection[y - 5:y + 5, x - 5:x + 5] = 100

imshow(selection)
imsave("./roi.png", selection)

"""provides tools for selecting region-of-interest (roi) on PIL images"""
__version__ = "0.1.0"

from typing import Any, ParamSpecKwargs, Callable, Concatenate, Iterable, Literal, NamedTuple, NewType, ParamSpec, Self, Tuple, TypeAlias, TypeVarTuple, Annotated, Dict, TypeVar
from numpy.typing import NDArray
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.draw import line_nd
from skimage.segmentation import flood as skflood


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
CoordIndex: TypeAlias = list[tuple[int, ...]]  # N-dimesional list of P-point coordinates in bundled in tuples (coordinate wise)
PlantingCondition: TypeAlias = Callable[[PointIndex, ValueAtPoint], bool]
FloodFunc_kwargs = ParamSpec("FloodFunc_kwargs")
FloodFunc: TypeAlias = Callable[Concatenate[
	Annotated[NDArray, "image ndarray"],
	Annotated[PointIndex, "point-index or slice of image"],
	FloodFunc_kwargs
], NDArray[np.uint8] | NDArray[np.bool8]]


def flood_along_points(
	image: Annotated[NDArray, "N-dim"],
	seed_points: Annotated[tuple[tuple[int, ...] | None, ...], "list of P-points of ith-coordinates"],
	planting_condition: PlantingCondition,
	flood_func: FloodFunc,
	**kwargs: Annotated[FloodFunc_kwargs, "flood_func.params.kwargs"]
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
	assert dims == image.ndim
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
		image.shape[d]
		for d in range(dims)
		if seed_points[d] is not None
	]
	selection = np.zeros(selection_shape, np.ubyte)
	for img_p, sel_p in zip(point_indexes, selection_point_indexes):
		if planting_condition(img_p, image[*img_p]) and selection[*sel_p] == 0:
			selection |= flood_func(image, img_p, **kwargs)
	return selection


def linepath_coords(
	flat_points:
		Annotated[list[int], "[p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, ...]"]
		| Annotated[list[list[int]], "[(p0_x, p0_y, p0_z), (p1_x, p1_y, p1_z), (p2_x, p2_y, p2_z), ... ]"],
	dimensions: int
) -> Annotated[list[NDArray[np.int32]], "[(p0_x, p0.1_x, p0.2_x, ..., p1_x, p1.1_x, ..., p2_x, p2.1_x, ...), (p0_y, p0.1_y, ...), (p0_z, p0.1_z, ...)]"]:
	points: list[list[int]]
	if isinstance(flat_points[0], (int, float)):
		# we've got to split the flat array of concatenated coords into an array of tuples of point-coords
		points = [flat_points[i:i + dimensions] for i in range(0, len(flat_points) - dimensions + 1, dimensions)]
	else:
		# flat_points is not actually flat, rather it's of the kind:
		# [(p0_x, p0_y, p0_z, ...), (p1_x, p1_y, p1_z, ...), (p2_x, p2_y, p2_z, ...), ... ]
		# which is the exact format we wanted to convert our points to
		points = flat_points
	n_points = len(points)
	line_coords: list[list[NDArray]] = [[] for d in range(dimensions)]
	for p in range(1, n_points):
		this_line_coords: tuple[NDArray, ...] = line_nd(points[p - 1], points[p], endpoint=(p == n_points - 1))
		for d in range(dimensions):
			line_coords[d].append(this_line_coords[d])
	line_coords_joined: list[NDArray[np.int32]] = [np.concatenate(line_coords[d]) for d in range(dimensions)]
	return line_coords_joined


polygon_coords_yx = [3100, 180, 3950, 1130, 3920, 1740, 3670, 1900]
upper_color = [200, 200, 200]
lower_color = [180, 180, 180]


def seeding_condition(yx: tuple[int, int], color: tuple[int, int, int]):
	if np.all(np.greater(color, lower_color)) and np.all(np.greater(upper_color, color)):
		return True
	return False


def seeding_condition2(yx: tuple[int, int], color: int):
	if color > 180 and color < 200:
		return True
	return False


img = imread("./q2,n20-242.jpg")

# yxc_coords: list[NDArray[np.int32] | None] = linepath_coords(polygon_coords_yx, 2) + [None,]
# flood_along_points(img, yxc_coords, seeding_condition, skflood, tolerance=5)
yx_coords = linepath_coords(polygon_coords_yx, 2)
selection0 = flood_along_points(img[:, :, 0], yx_coords, seeding_condition2, skflood, tolerance=5)
selection1 = flood_along_points(img[:, :, 1], yx_coords, seeding_condition2, skflood, tolerance=5)
selection2 = flood_along_points(img[:, :, 2], yx_coords, seeding_condition2, skflood, tolerance=5)

for y, x in zip(rr, cc):
	if seeding_condition(y, x, img[y, x]) and selection[y, x] == 0:
		count += 1
		selection |= flood(img[:, :, 0], (y, x), tolerance=5) & flood(img[:, :, 1], (y, x), tolerance=5) & flood(img[:, :, 2], (y, x), tolerance=5)

imshow(selection0 & selection1 & selection2)
imsave("./roi.png", selection0 & selection1 & selection2)

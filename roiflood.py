"""provides tools for selecting region-of-interest (roi) on PIL images"""
__version__ = "0.1.0"

from typing import Any, Iterator, Optional, Generator, Generic, Callable, Concatenate, Iterable, Literal, NamedTuple, NewType, ParamSpec, Self, Tuple, TypeAlias, TypeVarTuple, Annotated, Dict, TypeVar
from numpy.typing import NDArray
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.draw import line_nd
from skimage.segmentation import flood as skflood


ValueAtPoint: TypeAlias = int | NDArray

NPoint: TypeAlias = Annotated[tuple[int, ...], "N-dim point"]
"""N-dimesional point
>>> x0, y0, z0 = 8, 1, 2
>>> p: NPoint = (x0, y0, z0) # N = 3
"""

PNPoints: TypeAlias = Annotated[list[NPoint], "P-points of N-dim"]
"""list of P points, each of N-dimesions
>>> points: PNPoints = [(0, 0,), (1, 0,), (1, 5,), (2, 3,)] # P = 4, N = 2
"""

NShape: TypeAlias = Annotated[tuple[int, ...], "N-dim shape"]
"""N-dimesional shape
>>> width, height, depth = 2000, 512, 3
>>> s: NShape = (width, height, depth) # N = 3
>>> arr: NDArray = np.zeros(s, dtype=np.float32)
"""

NSlice: TypeAlias = Annotated[tuple[int | slice, ...], "N-dim slice"]
"""N-dimesional slice
>>> arr: NDArray = np.zeros((2000, 512, 3), dtype=np.float32)
>>> sel: NSlice = (1000, 255, slice(1, None)) # N = 3, nd-slice for selecting `arr[1000, 255, 1:]`
>>> arr[*sel] = 123.4
>>> arr[1000, 255]
array([  0. , 123.4, 123.4], dtype=float32)
"""

NPlane: TypeAlias = Annotated[tuple[int | None, ...], "N-dim plane"]
"""N-dimesional hyper-plane

the `NPlane` specifies for each coordinate `i`, either a specific `int` index, or a `None`
to dictate `slice(0, None)` (which is an unbound plane) for that `i`th dimension.
use the `plane_to_slice` function convert from this from to a proper slice.
>>> arr: NDArray = np.zeros((2000, 512, 3), dtype=np.float32)
>>> plane: NPlane = (1000, None, 1) # N = 3, nd-plane that can select `arr[1000, :, 1]`
>>> sel: NSlice = plane_to_slice(plane) # = `(1000, slice(0, None None), 1)`
>>> arr[*sel] = 2.0
>>> arr[1000, 0, 1]
2.0
"""

PNCoords: TypeAlias = Annotated[tuple[
	tuple[int, ...]
	| list[int]
	| NDArray[np.int_]
], "N-coordinate indexes of P-points"]
"""N-coordinates indexes of P-points

this can be typically used for masking ndarray. and it is also the return type of `np.where`.
use the `points_to_coords` function to convert `PNPoints` to `PNCoords`, or use `coords_to_points` for vise versa.
>>> arr: NDArray = np.zeros((4, 4))
>>> mask: PNCoords = ([0, 0, 1, 1], [0, 3, 2, 3]) # P = 4, N = 2. the 4 points are: (0, 0), (0, 3), (1, 2), (1, 3)
>>> arr[mask] = 2
>>> arr
array([[2., 0., 0., 2.],
       [0., 0., 2., 2.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
"""

PNCoordsPlane: TypeAlias = Annotated[tuple[
	None
	| tuple[int, ...]
	| list[int]
	| NDArray[np.int_]
], "N-coordinate indexes of P-points on a hyper-plane"]
"""N-coordinates indexes of P-points on a hyper-plane

tuple of length `N`, each element consisting of either:
 - `P` number of point-indexes of ith-coordinate
 - or `None`, indicating a plane (unbound slice)

for example: suppose you have `P = 7` points, and `N = 3` dimisional `image` of coordinates `[y, x, c]`.
and you want to describe a collection of `PNCoordsPlane` at certain `x` and `y` coordinates, but not the
color `c` coordinates, you'd describe it as:
>>> important_pixel_coords: PNCoordsPlane = ([y0, y1, y2, y3, y4, y5, y6], [x0, x1, x2, x3, x4, x5, x6], None)
"""

PlantingCondition: TypeAlias = Callable[[NPoint, ValueAtPoint], bool]
FloodFunc_kwargs = ParamSpec("FloodFunc_kwargs")
FloodFunc = Callable[Concatenate[
	Annotated[NDArray, "image ndarray"],
	Annotated[NPoint, "point-index or slice of image"],
	FloodFunc_kwargs
], NDArray[np.uint8] | NDArray[np.bool_]]


def plane_to_slice(plane: NPlane) -> NSlice:
	"""convert an `NPlane` to an `NSlice`
	>>> plane_to_slice((1000, None, 0,))
	(1000, slice(0, None, None), 0)
	"""
	return tuple(p if p is not None else slice(0, None) for p in plane)


def points_to_coords(points: PNPoints) -> PNCoords:
	"""convert a `PNPoints` list of points to a `PNCoords` tuple of coordinates
	>>> points_to_coords([(0, 0), (0, 3), (1, 2), (1, 3)])
	((0, 0, 1, 1), (0, 3, 2, 3))
	"""
	return tuple(zip(*points))


def coords_to_points(coords: PNCoords) -> PNPoints:
	"""convert a `PNCoords` tuple of coordinates to a `PNPoints` list of points
	>>> coords_to_points(([0, 0, 1, 1], [0, 3, 2, 3]))
	[(0, 0), (0, 3), (1, 2), (1, 3)]
	"""
	return list(zip(*coords))


def coordsplane_to_slice_gen(coords: PNCoordsPlane) -> Iterator[NSlice]:
	"""iterator for generating `P` number of `N` dimensional `NSlice`s based off of the provided `PNCoordsPlane`
	`N-tuple` of `P-point` indexes, with the possibility of a certain dimension being `None`, indicating
	and unbound plane slice.
	>>> y0, y1, y2, y3, y4 = 1, 2, 3, 4, 5
	>>> x0, x1, x2, x3, x4 = 9, 8, 7, 6, 5
	>>> list(coordsplane_to_slice_gen((
	>>>     [y0, y1, y2, y3, y4],
	>>>     [x0, x1, x2, x3, x4],
	>>>     None
	>>> )))
	[(1, 9, slice(0, None, None)),
	 (2, 8, slice(0, None, None)),
	 (3, 7, slice(0, None, None)),
	 (4, 6, slice(0, None, None)),
	 (5, 5, slice(0, None, None))]
	"""
	dims = len(coords)
	number_of_points = min(*[len(coord_indexes) for coord_indexes in coords if coord_indexes is not None])
	plane_dims: tuple[bool, ...] = tuple(True if coord_indexes is None else False for coord_indexes in coords)
	plane_slice = slice(0, None)
	for p in range(number_of_points):
		yield tuple(
			plane_slice if plane_dims[d]
			else coords[d][p]
			for d in range(dims)
		)


def coordsplane_to_reducedpoints_gen(coords: PNCoordsPlane) -> Iterator[NPoint]:
	"""iterator for generating `P` number of `N - K` dimensional `NPoint`s based off of the provided `PNCoordsPlane`
	`N-tuple` of `P-point` indexes. where `K` is the number of `None` dimensional indexes in the provided `PNCoordsPlane`,
	which will get purged/reduced/nullified in the output `NPoint`.
	>>> y0, y1, y2, y3, y4 = 1, 2, 3, 4, 5
	>>> c0, c1, c2, c3, c4 = 0, 2, 2, 1, 2
	>>> list(coordsplane_to_slice_gen((
	>>>     [y0, y1, y2, y3, y4],
	>>>     None
	>>>     [c0, c1, c2, c3, c4],
	>>> )))
	[(1, 0),
	 (2, 2),
	 (3, 2),
	 (4, 1),
	 (5, 2)]
	"""
	dims = len(coords)
	number_of_points = min(*[len(coord_indexes) for coord_indexes in coords if coord_indexes is not None])
	null_dims: tuple[bool, ...] = tuple(True if coord_indexes is None else False for coord_indexes in coords)
	for p in range(number_of_points):
		yield tuple(
			coords[d][p]
			for d in range(dims)
			if null_dims[d] is not None
		)


def flood_in_chunks(
	image: Annotated[NDArray, "N-dim"],
	seed_point: Annotated[tuple[int | None, ...], "N-point"],
	condition: Callable[[NPoint, ValueAtPoint], bool],
	*,
	chunkshape: Optional[NShape] = None,
	chunkoffset: Optional[NPoint] = None
) -> NDArray[np.bool_]:
	pass


def flood(
	image: Annotated[NDArray, "N-dim"],
	seed_point: Annotated[tuple[int | None, ...], "N-point"],
	*, footprint: Annotated[NDArray, "N-dim"],
	bigfootprint: int,
	condition: Callable[[PointIndex, ValueAtPoint], bool]
) -> NDArray[np.bool_]:
	pass


def flood_along_points(
	image: Annotated[NDArray, "N-dim"],
	seed_points: Annotated[tuple[tuple[int, ...] | None, ...], "list of P-points of ith-coordinates"],
	planting_condition: PlantingCondition,
	flood_func: Annotated[Callable[Concatenate[
		Annotated[NDArray, "image ndarray"],
		Annotated[NPoint, "point-index or slice of image"],
		FloodFunc_kwargs
	], NDArray[np.uint8] | NDArray[np.bool_]], "FloodFunc"],
	*args: FloodFunc_kwargs.args,
	**kwargs: Annotated[FloodFunc_kwargs.kwargs, "flood_func.params.kwargs"]
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
	# =? coordsplane_to_slice_gen(seed_points)
	selection_point_indexes: Iterable[tuple[int, ...]] = zip(*[
		seed_points[d][0: n_points]
		for d in range(dims)
		if seed_points[d] is not None
	])
	# =? coordsplane_to_reducedpoints_gen(seed_points)
	selection_shape: list[int] = [
		image.shape[d]
		for d in range(dims)
		if seed_points[d] is not None
	]
	# =? coordsplane_to_reducedpoints_gen(image.shape)
	selection = np.zeros(selection_shape, np.ubyte)
	for img_p, sel_p in zip(point_indexes, selection_point_indexes):
		if planting_condition(img_p, image[*img_p]) and selection[*sel_p] == 0:
			# the `selection[*sel_p] == 0` condition insures that we are flood filling a region that has not already been touched yet
			# this virtually eliminates redundant flood fills
			selection |= flood_func(image, img_p, *args, **kwargs)
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
img = imread("./q2,n20-242.jpg")

# yxc_coords: list[NDArray[np.int32] | None] = linepath_coords(polygon_coords_yx, 2) + [None,]
# flood_along_points(img, yxc_coords, seeding_condition, skflood, tolerance=5)

yx_coords = linepath_coords(polygon_coords_yx, 2)
selection0 = flood_along_points(img[:, :, 0], yx_coords, lambda yx, c: c > lower_color[0] and c < upper_color[0], skflood, tolerance=5)
selection1 = flood_along_points(img[:, :, 1], yx_coords, lambda yx, c: c > lower_color[1] and c < upper_color[1], skflood, tolerance=5)
selection2 = flood_along_points(img[:, :, 2], yx_coords, lambda yx, c: c > lower_color[2] and c < upper_color[2], skflood, tolerance=5)

imshow(selection0 & selection1 & selection2)
imsave("./roi.png", (selection0 & selection1 & selection2) * 255)

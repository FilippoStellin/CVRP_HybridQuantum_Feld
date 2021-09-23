
import numpy as np

EARTH_RADIUS_M = 6371000  # earth radius in meters


def compute_distances(coords: np.array) -> np.ndarray:
    def tile(dims, repetitions):
        return (np.tile(dim, (repetitions, 1)) for dim in dims)

    radian_coords = np.radians(coords)

    max_points = len(coords)

    lat = radian_coords[:, 0]
    lng = radian_coords[:, 1]

    src_lat, src_lng = tile((lat, lng), max_points)
    dst_lat, dst_lng = (dim.T for dim  in tile((lat, lng), max_points))

    diff_lat = src_lat - dst_lat
    diff_lng = (src_lng - dst_lng) * np.cos((src_lat + dst_lat) / 2)

    return EARTH_RADIUS_M * np.hypot(diff_lng, diff_lat)

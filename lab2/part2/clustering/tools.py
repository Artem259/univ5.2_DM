import math
import numpy as np


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calc_zip_distances(set1, set2, distance_metric=euclidean_distance):
    return [distance_metric(p1, p2) for p1, p2 in zip(set1, set2)]


def calc_max_zip_distance(set1, set2, distance_metric=euclidean_distance):
    return max(calc_zip_distances(set1, set2, distance_metric))


def calc_min_zip_distance(set1, set2, distance_metric=euclidean_distance):
    return min(calc_zip_distances(set1, set2, distance_metric))


def calc_distance_matrix(points, targets, distance_metric=euclidean_distance):
    return np.array([
        [distance_metric(point, target) for target in targets]
        for point in points
    ])


def find_closest_point(points, target, distance_metric=euclidean_distance):
    distances = calc_zip_distances(points, [target] * len(points), distance_metric)
    closest_index = int(np.argmin(distances))
    return closest_index, points[closest_index]

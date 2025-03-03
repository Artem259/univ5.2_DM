import math


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calc_zip_distances(set1, set2, distance_metric):
    return [distance_metric(p1, p2) for p1, p2 in zip(set1, set2)]


def calc_max_zip_distance(set1, set2, distance_metric):
    return max(calc_zip_distances(set1, set2, distance_metric))


def calc_distance_matrix(points, centers, distance_metric):
    return [
        [distance_metric(point, center) for center in centers]
        for point in points
    ]

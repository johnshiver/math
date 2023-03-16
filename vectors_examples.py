from math import cos, pi
from vector_drawing import Points, Polygon, draw, green
from vectors import to_cartesian


dino_vectors = [
    (6, 4),
    (3, 1),
    (1, 2),
    (-1, 5),
    (-2, 5),
    (-3, 4),
    (-4, 4),
    (-5, 3),
    (-5, 2),
    (-2, 2),
    (-5, 1),
    (-4, 0),
    (-2, 1),
    (-1, 0),
    (0, -3),
    (-1, -4),
    (1, -4),
    (2, -3),
    (1, -2),
    (3, -1),
    (5, 1),
]


def draw_dino_vectors():
    # draw(Points(*dino_vectors))

    draw(Points(*dino_vectors), Polygon(*dino_vectors))


def draw_flower():
    polar_coords = [(cos(x * pi / 100.0), 2 * pi * x / 1000.0) for x in range(0, 1000)]
    vectors = [to_cartesian(p) for p in polar_coords]
    draw(Polygon(*vectors, color=green))

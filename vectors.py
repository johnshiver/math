from math import sqrt, sin, cos, atan2


def subtract(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1])


def add(*vectors):
    return (sum([v[0] for v in vectors]), sum([v[1] for v in vectors]))


# pythagorean theorem
def length(v):
    x, y = v
    return sqrt(x**2 + y**2)


def distance(v1, v2):
    return length(subtract(v1, v2))


def perimeter(vectors):
    distances = [
        distance(vectors[i], vectors[(i + 1) % len(vectors)])
        for i in range(0, len(vectors))
    ]
    return sum(distances)


def scale(scalar, v):
    return (scalar * v[0], scalar * v[1])


def translate(translation, vectors):
    return [add(translation, v) for v in vectors]


## trigonometry section functions ---------------------------------


def to_cartesian(polar_vector):
    """
    to_cartesian takes a polar_vector coordinate (length, angle)
    and translates it to caretesian coordinates (x, y)
    """
    length, angle = polar_vector
    return (length * cos(angle), length * sin(angle))


def rotate(angle, vectors):
    polars = [to_polar(v) for v in vectors]
    return [to_cartesian((l, a + angle)) for l, a in polars]


def to_polar(vector):
    """
    to polar takes a vector (cartesian coordinates x,y) and returns polar coordinates (length, angle)

    NOTE: angle is calculated using arctan. Inverse trigonomic functions are tricky, because trignomic functions
          can have multiple inputs produce the same output, inversing them is not necessarily deterministic.
          that is why arctan is required here because it does work correctly
    """
    x, y = vector
    angle = atan2(y, x)
    return (length(vector), angle)

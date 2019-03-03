
from numpy.linalg import norm


def euclidean(a, b):
    """Compute and return the euclidean distance between a and b."""
    return norm(a-b)

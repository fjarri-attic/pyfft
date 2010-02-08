from math import log10

from numpy import arange, dot, get_printoptions, set_printoptions, allclose, max, sum
from numpy.random import rand, randn, seed
from numpy.linalg import norm

set_printoptions(precision=5, threshold=20, suppress=True, linewidth=150)

def relative_error(result, reference):
    assert(result.shape == reference.shape)
    return norm(result - reference) / norm(reference)

def psnr(result, reference):
    # make sure that pixels take values in [0, 255] and not in [0,1]
    assert(1 < max(reference) <= 255)
    mse = sum((result - reference) ** 2) / reference.size
    return 10 * log10(255**2 / mse)

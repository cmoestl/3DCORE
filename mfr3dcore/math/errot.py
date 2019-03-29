# -*- coding: utf-8 -*-

"""errot.py

Implements vector rotation using the Euler-Rodrigues formula.
For a quick recap see (https://en.wikipedia.org/wiki/Euler–Rodrigues_formula).
"""

import numba as nb
import numpy as np


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def errot(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by the given Euler-Rodrigues coefficients.

    :param v: Vector
    :param w: Euler-Rodrigues coefficients
    :return: Rotated Vector
    """
    m = np.array([
        [(w[0] ** 2 + w[1] ** 2 - w[2] ** 2 - w[3] ** 2), 2 * (w[1] * w[2] - w[0] * w[3]),
         2 * (w[1] * w[3] + w[0] * w[2])],
        [2 * (w[1] * w[2] + w[0] * w[3]), (w[0] ** 2 + w[2] ** 2 - w[1] ** 2 - w[3] ** 2),
         2 * (w[2] * w[3] - w[0] * w[1])],
        [2 * (w[1] * w[3] - w[0] * w[2]), 2 * (w[2] * w[3] + w[0] * w[1]),
         (w[0] ** 2 + w[3] ** 2 - w[1] ** 2 - w[2] ** 2)]
    ])

    return np.dot(m, v)


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def errot_compose(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """
    Compose two rotations given by their Euler-Rodrigues coefficients.

    :param w1: Rotation A
    :param w2: Rotation B
    :return: Composition A * B
    """
    wa = w1[0] * w2[0] - w1[1] * w2[1] - w1[2] * w2[2] - w1[3] * w2[3]
    wb = w1[0] * w2[1] + w1[1] * w2[0] - w1[2] * w2[3] + w1[3] * w2[2]
    wc = w1[0] * w2[2] + w1[2] * w2[0] - w1[3] * w2[1] + w1[1] * w2[3]
    wd = w1[0] * w2[3] + w1[3] * w2[0] - w1[1] * w2[2] + w1[2] * w2[1]

    return np.array([wa, wb, wc, wd])


@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def errot_get(ang: np.float64, rot: np.ndarray) -> np.ndarray:
    """
    Calculate Euler-Rodrigues coefficients for a specific rotation (angle + vector).

    :param ang: Rotation angle
    :param rot: Rotation vector
    :return: Euler-Rodrigues coefficients
    """
    return np.array([
        np.cos(np.radians(ang / 2)),
        rot[0] * np.sin(np.radians(ang / 2)),
        rot[1] * np.sin(np.radians(ang / 2)),
        rot[2] * np.sin(np.radians(ang / 2)),
    ])

# -*- coding: utf-8 -*-

"""csys_polar.py

Implements polar coordinate system.
"""

import numba as nb
import numpy as np


@nb.njit(nb.float64[:](nb.float64[:]))
def csys_polar_to_xyz(v: np.ndarray) -> np.ndarray:
    """
    Transforms Polar coordinates (r, psi, phi) to Cartesian coordinates (x, y, z).

    :param v: Polar coordinates (r, psi, phi)
    :return: Cartesian coordinates (x, y, z)
    """
    return np.array([
        v[0] * np.cos(v[2]) * np.cos(v[1]),
        v[0] * np.cos(v[2]) * np.sin(v[1]),
        v[0] * np.sin(v[2])
    ])

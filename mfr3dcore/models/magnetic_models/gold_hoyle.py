# -*- coding: utf-8 -*-

"""gold_hoyle.py

Constant twist Gold-Hoyle magnetic flux rope model.

For the tapered torus is, each sections is locally approximated as a cylinder for the purposes of calculating the
magnetic field vectors (see Gold and Hoyle (1960) or Hu et al. (2014)).
"""

import numba as nb
import numpy as np

from ...math import csys_ttorus_to_xyz_jacobian

@nb.njit(nb.float64[:](nb.float64, nb.float64, nb.float64))
def gh_cylindrical(b, t, h):
    """
    Magnetic field vector according to the Gold & Hoyle model in cylindrical coordinates.

    :param b: Magnetic field strength
    :param t: Number of twists (z=0-2pi)
    :param h: Field handedness
    :return: Magnetic field vector
    """
    b_r = 0
    b_phi = b * t * h / np.sqrt(1 + t ** 2)
    b_z = b / np.sqrt(1 + t ** 2)

    return np.array([b_r, b_phi, b_z])

@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def gh_ttorus_to_xyz(v: np.ndarray, b: float, t: float, h: float, rho_0: float, rho_1: float, a: float) -> np.ndarray:
    """
    Magnetic field vector according to the Gold & Hoyle model in cartesian coordinates.

    :param v: Toroidal coordinates (r, psi, phi)
    :param b: Magnetic field strength
    :param t: Number of twists (psi=0-2pi)
    :param h: Field handedness
    :param rho_0: Torus Radius (major)
    :param rho_1: Torus Radius (minor)
    :param a: Cross section aspect ratio (a / b)
    :return: Magnetic field vector
    """
    b_r = 0
    b_psi = b / np.sqrt(1 + t ** 2)
    b_phi = h * b * t / np.sqrt(1 + t ** 2)

    jac = csys_ttorus_to_xyz_jacobian(np.array([v[0], v[1], v[2]]), rho_0, rho_1, a)
    b_xyz = np.dot(jac, np.array((b_r, b_psi, b_phi)))

    return b * b_xyz / np.linalg.norm(b_xyz)

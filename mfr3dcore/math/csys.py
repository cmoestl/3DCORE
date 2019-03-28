# -*- coding: utf-8 -*-

import numba as nb
import numpy as np


@nb.njit(nb.float64[:](nb.float64[:]))
def csys_polar_xyz(v):
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


@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64))
def csys_ttorus_to_xyz(v, rho_0, rho_1, aspect=1):
    """
    Transforms tapered Torus coordinates (r, psi, phi) to Cartesian coordinates (x, y, z).

    :param v: Tapered Torus coordinates (r, psi, phi)
    :param rho_0: Torus Radius
    :param rho_1: Tours Radius (minor)
    :param aspect: Cross section aspect ratio (a / b)
    :return: Cartesian coordinates (x, y, z)
    """
    a = v[0] * rho_1
    b = v[0] * rho_1 / aspect

    return np.array([
        -(rho_0 + a * np.sin(v[1] / 2) * np.cos(v[2])) * np.cos(v[1]) + rho_0,
        (rho_0 + a * np.sin(v[1] / 2) * np.cos(v[2])) * np.sin(v[1]),
        b * np.sin(v[1] / 2) * np.sin(v[2])
    ])


@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64))
def csys_xyz_to_ttorus(v, rho_0, rho_1, aspect=1):
    """
    Transforms Cartesian coordinates (x, y, z) to tapered Torus coordinates (r, psi, phi).

    :param v: Cartesian coordinates (x, y, z)
    :param rho_0: Torus Radius
    :param rho_1: Tours Radius (minor)
    :param aspect: Cross section aspect ratio (a / b)
    :return: Tapered Torus coordinates (r, psi, phi)
    """
    if np.sqrt((rho_0 - v[0]) ** 2 + v[1] ** 2) - rho_0 == 0:
        if v[2] >= 0:
            phi = np.pi / 2
        else:
            phi = 3 * np.pi / 2
    else:
        rd = np.sqrt((rho_0 - v[0]) ** 2 + v[1] ** 2) - rho_0
        if rd > 0:
            phi = np.arctan(aspect * v[2] / rd)

            if phi < 0:
                phi += 2 * np.pi
        else:
            phi = -np.pi + np.arctan(aspect * v[2] / rd)

    if v[0] == rho_0:
        if v[1] >= 0:
            psi = 3 * np.pi / 2
        else:
            psi = np.pi / 2
    else:
        psi = np.arctan2(-v[1], v[0] - rho_0) + np.pi

    if psi == 0 or psi == 2 * np.pi:
        r = np.inf
    elif phi == np.pi / 2 or phi == 3 * np.pi / 2:
        r = aspect * v[2] / rho_1 / np.sin(psi / 2) / np.sin(phi)
    else:
        r = np.abs((np.sqrt((rho_0 - v[0]) ** 2 + v[1] ** 2) - rho_0) / np.sin(psi / 2) / np.cos(phi) / rho_1)

    return np.array([
        r,
        psi,
        phi
    ])

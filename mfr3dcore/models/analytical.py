# -*- coding: utf-8 -*-

"""analytical.py

Base class for analytical flux rope models. Generalization of the 3DCORE model
presented in Möstl et al. (2018).

Description:
    * Models a magnetic flux rope propagating in a certain direction, given
      by the longitude, latitude and inclination. The inertial coordinate
      system is assumed to be J2000. In this case the longitude increases
      towards the solar east. The latitude lies within [-90°, +90°]. The
      inclination is defined so that an inclination of 0° corresponds to a flux
      rope that lies within the XY plane and where the magnetic field points
      towards the solar east.
    * The geometric flux rope shape is defined as an isosurface in an arbitrary
      curvilinear coordiante system of choice. This is implemented by creating
      a transformation function g: q_i -> x_i that maps the curvilinear
      coordinates to cartesian coordinates (such that the flux rope propagates
      along the X axis). The model also required the inverse transformation
      f: x_i -> q_i and the Jacobian J. In the case where it is not possible
      to construct f, it is numerically approximated using scipy.least_squares.
      This numerical approximation then requires a function f* that gives an
      initial estimate of q_i for any x_i
    * The magnetic field within the flux rope is estimated using an additional
      model for the magnetic field, which must be implemented for the specific
      curvilinear coordinate system. Ideally the curvilinear coordinate system
      locally resembles those that were used for deriving known solutions like
      the Gold-Hoyle or Lundquist model.
"""

import datetime
import itertools
import numpy as np
import numba as nb
import scipy as sp

from typing import Callable, Union


class AnalyticalFluxRopeModel(object):
    f, f_appr = None
    g = None, None
    J = None

    params = dict()

    def __init__(
            self,
            time: datetime.datetime,
            longitude: float,
            latitude: float,
            inclination: float,
            g: Callable,
            J: Callable,
            f: Callable = None,
            f_appr: Callable = None,
    ) -> None:
        """
        Initialize base class. If the inverse transformation f is not given, it
        is replaced with a function that approximates it using least_squares
        and f_appr.

        :param time: initial time
        :param longitude: propagation longitude
        :param latitude: propagation latitude
        :param inclination: propagation inclination
        :param g: coordinate transform g: q_i -> x_i
        :param J: jacobian of the transform (dx_i/dq_i)
        :param f: coordinate transform f: x_i -> q_i
        :param f_appr: approximation of f, if f is not given (None)
        """
        self.t0 = time
        self.lon = longitude
        self.lat = latitude
        self.inc = inclination

        self.g = g
        self.J = J

        if f is None and f_appr is None:
            raise ValueError("inverse transform f or its approximation f_appr \
                must be given")

        self.f = f
        self.f_appr = f_appr

        if f is None:
            pass

        self.update_er_coefficients()

    def update_er_coefficients(self) -> None:
        """
        Calculate the Rodrigues-coefficients for transforming coordinates from
        the intertial system into the rotated cartesian coordinates and back.
        """
        ux = np.array([1.0, 0, 0])
        uy = np.array([0, 1.0, 0])
        uz = np.array([0, 0, 1.0])

        c1 = er_get(self.lon, uz)
        c2 = er_get(-self.lat, er_rot(uy, c1))
        c3 = er_get(self.inc, er_rot(ux, er_compose(c1, c2)))

        self.er_coeff_from = er_compose(er_compose(c1, c2), c3)
        self.er_coeff_into = np.array(
            [
                self.er_coeff_from[0],
                -self.er_coeff_from[1],
                -self.er_coeff_from[2],
                -self.er_coeff_from[3]
            ])

    def transform_into(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transforms inertial into curvilinear coordinates.

        :param x: inertial coordinates
        :return: curvilinear coordinates
        """
        if isinstance(x, list) or (isinstance(x, np.ndarray) and x.ndim > 1):
            return np.array([self.transform_into(_v) for _v in x])
        else:
            return self.f(er_rot(x, self.er_coeff_into), **self.params)

    def transform_from(self, q: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transforms curvilinear into inertial coordinates.

        :param q: curvilinear coordinates
        :return: inertial coordinates
        """
        if isinstance(q, list) or (isinstance(q, np.ndarray) and q.ndim > 1):
            return np.array([self.transform_from(_v) for _v in q])
        else:
            return er_rot(self.g(q, **self.params), self.er_coeff_from)


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def er_compose(er1: np.ndarray, er2: np.ndarray) -> np.ndarray:
    """
    Compose two rotations given by their Euler-Rodrigues coefficients.

    :param er1: rotation A
    :param er2: rotation B
    :return: composition A * B
    """
    era = er1[0] * er2[0] - er1[1] * er2[1] - er1[2] * er2[2] - er1[3] * er2[3]
    erb = er1[0] * er2[1] + er1[1] * er2[0] - er1[2] * er2[3] + er1[3] * er2[2]
    erc = er1[0] * er2[2] + er1[2] * er2[0] - er1[3] * er2[1] + er1[1] * er2[3]
    erd = er1[0] * er2[3] + er1[3] * er2[0] - er1[1] * er2[2] + er1[2] * er2[1]

    return np.array([era, erb, erc, erd])


@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def er_get(a: float, v: np.ndarray) -> np.ndarray:
    """
    Calculate Euler-Rodrigues coefficients for a specific rotation.

    :param a: rotation angle
    :param v: rotation vector
    :return: Euler-Rodrigues coefficients
    """
    arg = np.radians(a / 2)

    return np.array([
        np.cos(arg),
        v[0] * np.sin(arg),
        v[1] * np.sin(arg),
        v[2] * np.sin(arg),
    ])


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def er_rot(v: np.ndarray, er: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by the given Euler-Rodrigues coefficients.

    :param v: vector
    :param er: Euler-Rodrigues coefficients
    :return: rotated Vector
    """
    m = np.array([
        [
            (er[0] ** 2 + er[1] ** 2 - er[2] ** 2 - er[3] ** 2),
            2 * (er[1] * er[2] - er[0] * er[3]),
            2 * (er[1] * er[3] + er[0] * er[2])
        ],
        [
            2 * (er[1] * er[2] + er[0] * er[3]),
            (er[0] ** 2 + er[2] ** 2 - er[1] ** 2 - er[3] ** 2),
            2 * (er[2] * er[3] - er[0] * er[1])
        ],
        [
            2 * (er[1] * er[3] - er[0] * er[2]),
            2 * (er[2] * er[3] + er[0] * er[1]),
            (er[0] ** 2 + er[3] ** 2 - er[1] ** 2 - er[2] ** 2)
        ]
    ])

    return np.dot(m, v)

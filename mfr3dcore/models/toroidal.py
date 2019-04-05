# -*- coding: utf-8 -*-

"""toroidal.py

Implementation of the basic CME MFR model as detailed in Möstl et al. (2018).

The flux rope is geometrically described using a tapered torus that is attached to the sun. The geometry is
constructed using circular or elliptical cross sections of varying size (being their smallest at the sun and their
largest at the apex of the torus).
"""

import ciso8601
import datetime
import itertools
import numpy as np

from typing import Union

from .magnetic_models import gh_ttorus_to_xyz
from .propagation_models import vk_drag_model

from ..math import csys_ttorus_to_xyz, csys_xyz_to_ttorus
from ..math import errot, errot_compose, errot_get


class ToroidalMFR(object):
    def __init__(self, radius: float, speed: float, time: datetime.datetime, aspect: float,
                 longitude: float, latitude: float, inclination: float, diameter: float,
                 handedness: int, strength: float, turns: float, background_drag: float,
                 background_speed: float, background_strength: float):
        """
        Initialize model parameters and calculate the Rodrigues-coefficients for rotating into and out of the
        coordinate system.

        :param radius: CME apex at t=0 (km)
        :param speed: CME speed at t=0 (km/s)
        :param time: CME t=0 (as datetime)
        :param aspect: MFR cross-section aspect ratio
        :param longitude: MFR longitude (deg)
        :param latitude: MFR latitude (deg)
        :param inclination: MFR inclination (deg)
        :param diameter: MFR diameter (at 1AU, AU)
        :param handedness: MFR handedness (+/-1)
        :param strength: MFR magnetic field strength (at 1AU, nT)
        :param turns: MFR magnetic field twists (total number)
        :param background_drag: solar wind drag coefficient
        :param background_speed: solar wind speed (km/s)
        :param background_strength: solar wind magnetic field strength (nT)
        """
        self._radius_t = float(radius)
        self._radius_0 = float(radius)

        self._speed_t = float(speed)
        self._speed_0 = float(speed)

        self._time_t = time
        self._time_0 = time

        self._aspect = float(aspect)
        self._longitude = float(longitude)
        self._latitude = float(latitude)
        self._inclination = float(inclination)
        self._diameter = float(diameter)

        self._handedness = int(handedness)

        if handedness == 1:
            self._helicity = "R"
        else:
            self._helicity = "L"

        self._strength = float(strength)
        self._turns = float(turns)

        self._bg_drag = float(background_drag)
        self._bg_sign = int(-1 + 2 * int(speed > background_speed))
        self._bg_speed = float(background_speed)
        self._bg_strength = float(background_strength)

        self._errot_from = None
        self._errot_into = None

        self.update_geometry()

    def __repr__(self):
        return "<MagneticFluxRope @ {0:.2f}AU with {1:.0f}km/s>".format(self.r, self.v)

    @classmethod
    def from_text(cls, filename: str):
        lines = open(filename).read().splitlines()

        return cls(
            float(lines[9]) * 695508,
            float(lines[7]),
            ciso8601.parse_datetime(lines[11]),

            float(1.0),
            float(lines[26]),
            float(lines[28]),
            float(lines[30]) - 90.0,
            float(lines[36]),

            int(lines[32]),
            float(lines[38]),
            float(lines[34]),

            float(lines[13]) * 1e-7,
            float(lines[15]),
            float(lines[17])
        )

    # ===== PROPERTIES =====

    @property
    def b(self):
        """
        :return: Axial magnetic field strength (see Leitner (2007)).
        """
        return float(self._strength * (2 * self.rho_0) ** (-1.64))

    @property
    def r(self):
        """
        :return: MFR distance (at apex)
        """
        return float(self._radius_t / 1.496e8)

    @property
    def v(self):
        """
        :return: MFR speed
        """
        return float(self._speed_t)

    @property
    def rho_0(self):
        """
        :return: Torus radius (major)
        """
        return float((self.r - self.rho_1) / 2)

    @property
    def rho_1(self):
        """
        :return: Torus radius (minor)
        """
        return float(self._diameter * (self.r ** 1.14) / 2)

    # ===== FUNCTIONS =====

    def propagate(self, simulation_time: datetime.datetime):
        """
        Propagate NFR.

        :param simulation_time:  simulation time (datetime).
        """
        if simulation_time < self._time_0:
            raise ValueError("given simulation time is before initial")

        dt = int((simulation_time - self._time_0).total_seconds())

        (self._radius_t, self._speed_t) = vk_drag_model(self._radius_0, self._speed_0, dt, self._bg_speed,
                                                        self._bg_drag, self._bg_sign)

    def magnetic_field(self, v: np.ndarray) -> (np.ndarray, np.bool):
        """
        Extract magnetic field vector at the given Cartesian coordinates.

        :param v: Cartesian coordinates (x, y, z)
        :return: Magnetic field vector in Cartesian coordinates (x, y, z), Flag if v was inside MFR
        """
        v = self.transform_into(v)

        if v[0] < 1:
            b = gh_ttorus_to_xyz(v, self.b, self._turns, self._handedness, self.rho_0, self.rho_1, self._aspect)
            return errot(b, self._errot_from), True
        else:
            return np.array([0.0, 0.0, 0.0]), False

    # ===== VISUALIZATION =====

    def plot_fieldlines(self, x0: np.ndarray, h: float = None, **kwargs):
        """
        Plots magnetic field line starting at x0 (r, psi, phi) in Cartesian coordinates.

        :param x0: initial position (r, psi, phi)
        :param h: integration step size
        :kwargs: optional parameters (axes handle and plotting arguments)
        :return: magnetic field line (x, y, z)
        """
        psi0 = x0[1]
        x0 = self.transform_from(x0)
        xs = [x0]

        if not h:
            h = self.rho_0 / 100

        def iterate(xk):
            ub = gh_ttorus_to_xyz(self.transform_into(xk), self.b, self._turns, self._handedness, self.rho_0,
                                  self.rho_1,
                                  self._aspect)

            return errot(ub / np.linalg.norm(ub), self._errot_from)

        while self.transform_into(xs[-1])[1] < 2 * np.pi - psi0:
            # RK4
            k1 = h * iterate(xs[-1])
            k2 = h * iterate(xs[-1] + k1 / 2)
            k3 = h * iterate(xs[-1] + k2 / 2)
            k4 = h * iterate(xs[-1] + k3)
            xs.append(xs[-1] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

        xs = np.array(xs)

        if "handle" not in kwargs:
            from matplotlib import pyplot as plt

            ax = plt.figure().add_subplot(111, projection='3d')
        else:
            ax = kwargs.pop("handle")

        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], **kwargs)

    def plot_wireframe(self, r: float = 1.0, resolution: int = 0, **kwargs):
        """
        Plots MFR wireframe.

        :resolution: resolution parameter (higher = better)
        :kwargs: optional parameters (axes handle and plotting arguments)
        :return: wireframe array
        """
        r = [np.abs(r)]
        dc = [60, 30, 20, 15, 10, 5, 2]
        d = dc[np.min([resolution, len(dc) - 1])]
        c = int(360 / d) + 1
        u = np.radians(np.r_[0:360. + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        wf = self.transform_from(np.array(list(itertools.product(r, u, v))).reshape(c ** 2, 3))
        wf = wf.reshape((c, c, 3))

        if "handle" not in kwargs:
            from matplotlib import pyplot as plt

            ax = plt.figure().add_subplot(111, projection='3d')
        else:
            ax = kwargs.pop("handle")

        ax.plot_wireframe(wf[:, :, 0], wf[:, :, 1], wf[:, :, 2], **kwargs)

    # ===== INTERNAL FUNCTIONS =====

    def update_geometry(self):
        """
        Calculate the Rodrigues-coefficients for rotating into and out of the coordinate system. This function must be
        called after changing either the longitude, latitude or inclination of the MFR.
        """
        ux = np.array([1.0, 0, 0])
        uy = np.array([0, 1.0, 0])
        uz = np.array([0, 0, 1.0])

        c1 = errot_get(self._longitude, uz)
        c2 = errot_get(-self._latitude, errot(uy, c1))
        c3 = errot_get(self._inclination, errot(ux, errot_compose(c1, c2)))

        self._errot_from = errot_compose(errot_compose(c1, c2), c3)
        self._errot_into = np.array(
            [
                self._errot_from[0],
                -self._errot_from[1],
                -self._errot_from[2],
                -self._errot_from[3]
            ])

    def transform_into(self, v: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transforms cartesian coordinates into tapered torus coordinates (including rotation).

        :param v: Cartesian coordinates
        :return: Tapered Torus coordinates
        """
        if isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim > 1):
            return np.array([self.transform_into(_v) for _v in v])
        else:
            return csys_xyz_to_ttorus(errot(v, self._errot_into), self.rho_0, self.rho_1, self._aspect)

    def transform_from(self, v: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transforms tapered toroidal coordinates into cartesian coordinates (including rotation).

        :param v: Tapered Torus coordinates
        :return: Cartesian coordinates
        """
        if isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim > 1):
            return np.array([self.transform_from(_v) for _v in v])
        else:
            return errot(csys_ttorus_to_xyz(v, self.rho_0, self.rho_1, self._aspect), self._errot_from)

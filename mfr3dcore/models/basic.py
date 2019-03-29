# -*- coding: utf-8 -*-

import ciso8601
import datetime
import itertools
import numba as nb
import numpy as np

from typing import Union

from ..math import csys_ttorus_to_xyz, csys_xyz_to_ttorus
from ..math import errot, errot_compose, errot_get


@nb.njit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64,
                       nb.float64))
def gold_hoyle(r: np.float64, psi: np.float64, phi: np.float64, b: np.float64, rho_0: np.float64, rho_1: np.float64,
               a: np.float64, h: np.float64, t: np.float64) -> np.ndarray:
    """
    Gold & Hoyle Flux Rope Model (see Gold & Hoyle (1960)).

    Calculates the magnetic field vector in Cartesian coordinates from the tapered Torus coordinates.
    """
    t_ratio = t / (r * rho_1)

    b_r = 0
    b_psi = b / np.sqrt(1 + (t_ratio ** 2) * ((r * rho_1) ** 2))
    b_phi = h * b * t_ratio * (r * rho_1) / np.sqrt(1 + (t_ratio ** 2) * ((r * rho_1) ** 2))

    vr = np.array([
        -np.cos(phi) * np.cos(psi),
        np.cos(phi) * np.sin(psi),
        np.sin(phi) / a
    ])
    vr = vr / np.linalg.norm(vr)

    vpsi = np.array([
        np.sin(psi) * (rho_0 + r * rho_1 * np.cos(phi) + 0.5 * r * rho_1 * np.cos(phi) * np.cos(psi / 2)),
        np.cos(psi) * (rho_0 + r * rho_1 * np.cos(phi) + 0.5 * r * rho_1 * np.cos(phi) * np.cos(psi / 2)),
        r * rho_1 / 2 / a * np.cos(psi / 2) * np.sin(phi)
    ])
    vpsi = vpsi / np.linalg.norm(vpsi)

    vphi = np.array([
        np.sin(phi) * np.cos(psi),
        -np.sin(phi) * np.sin(psi),
        np.cos(phi) / a
    ])
    vphi = vphi / np.linalg.norm(vphi)

    m = np.array([
        [vr[0], vpsi[0], vphi[0]],
        [vr[1], vpsi[1], vphi[1]],
        [vr[2], vpsi[2], vphi[2]]
    ])

    f = np.dot(m, np.array((b_r, b_psi, b_phi)))

    return f


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def propagate_r(dv: np.float64, dt: np.float64, bg_speed: np.float64, bg_drag: np.float64, bg_sign: np.int32,
                r_0: np.float64) -> np.float64:
    return bg_sign / bg_drag * np.log1p(bg_sign * bg_drag * dv * dt) + bg_speed * dt + r_0


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def propagate_v(dv: np.float64, dt: np.float64, bg_speed: np.float64, bg_drag: np.float64,
                bg_sign: np.int32) -> np.float64:
    return dv / (1 + bg_sign * bg_drag * dv * dt) + bg_speed


class MagneticFluxRope(object):
    """
    Implementation of the basic CME MFR model as detailed in C. Möstl et al. (2018).

    The MFR is geometrically described using a tapered torus that is attached to the sun. The geometry is
    constructed using circular or elliptical cross sections of varying size (being their smallest at the sun and their
    largest at the apex of the torus).

    The magnetic field is described using the twisted Gold-Hoyle MFR model. At each point, the tapered torus is
    approximated as a cylinder for the purposes of calculating the magnetic field vectors (see Gold and Hoyle (1960) or
    Hu et al. (2014)).

    The propagation of the CME is described using a simple drag based model (see Vrsnak et al. 2013).
    """
    def __init__(self, radius: np.float64, speed: np.float64, time: datetime.datetime, aspect: np.float64,
                 longitude: np.float64, latitude: np.float64, inclination: np.float64, diameter: np.float64,
                 handedness: np.int32, strength: np.float64, turns: np.float64, background_drag: np.float64,
                 background_speed: np.float64, background_strength: np.float64):
        """
        Initialize model parameters and calculate the Rodrigues-coefficients for rotating into and out of the
        coordinate system.

        :param radius: CME apex at t=0 (km)
        :param speed: CME speed at t=0 (km/s)
        :param time: CME t=0 (as datetime)
        :param aspect: cross-section aspect ratio
        :param longitude: CME longitude (deg)
        :param latitude: CME latitude (deg)
        :param inclination: CME inclination (deg)
        :param diameter: CME diameter (at 1AU, AU)
        :param handedness: CME handedness (+/-1)
        :param strength: CME magnetic field strength (at 1AU, nT)
        :param turns: CME magnetic field twists (per AU)
        :param background_drag: solar wind drag coefficient
        :param background_speed: solar wind speed (km/s)
        :param background_strength: solar wind magnetic field strength (nT)
        """
        self._radius_t = np.float64(radius)
        self._radius_0 = np.float64(radius)

        self._speed_t = np.float64(speed)
        self._speed_0 = np.float64(speed)

        self._time_t = time
        self._time_0 = time

        self._aspect = np.float64(aspect)
        self._longitude = np.float64(longitude)
        self._latitude = np.float64(latitude)
        self._inclination = np.float64(inclination)
        self._diameter = np.float64(diameter)

        self._handedness = np.int32(handedness)

        if handedness == 1:
            self._helicity = "R"
        else:
            self._helicity = "L"

        self._strength = np.float64(strength)
        self._turns = np.float64(turns)

        self._bg_drag = np.float64(background_drag)
        self._bg_sign = np.int32(-1 + 2 * int(speed > background_speed))
        self._bg_speed = np.float64(background_speed)
        self._bg_strength = np.float64(background_strength)

        self._errot_from = None
        self._errot_into = None

        self.update_rodrigues_coefficients()

    def __repr__(self):
        return "<MagneticFluxRope at {0:.2f}AU with {1:.0f}km/s>".format(self.r_apex, self.v_apex)

    # ========== FACTORIES ==========

    @classmethod
    def from_text(cls, filename: str):
        lines = open(filename).read().splitlines()

        return cls(
            np.float64(np.float64(lines[9]) * 695508),
            np.float64(lines[7]),
            ciso8601.parse_datetime(lines[11]),

            np.float64(1.0),
            np.float64(lines[26]),
            np.float64(lines[28]),
            np.float64(np.float64(lines[30]) - 90.0),
            np.float64(lines[36]),

            np.int32(lines[32]),
            np.float64(lines[38]),
            np.float64(lines[34]),

            np.float64(np.float64(lines[13]) * 1e-7),
            np.float64(lines[15]),
            np.float64(lines[17])
        )

    # ========== PROPERTIES ==========

    @property
    def b_axial(self):
        """
        Axial magnetic field strength (see Leitner (2007)).
        """
        return np.float64(self._strength * (2 * self.rho_0) ** (-1.64))

    @property
    def r_apex(self):
        return np.float64(self._radius_t / 1.496e8)

    @property
    def v_apex(self):
        return np.float64(self._speed_t)

    @property
    def rho_0(self):
        return np.float64((self.r_apex - self.rho_1) / 2)

    @property
    def rho_1(self):
        """
        CME diameter as a function of distance (see Leitner (2007)).
        """
        return np.float64(self._diameter * (self.r_apex ** 1.14) / 2)

    # ========== KINEMATICS ==========

    def propagate(self, simulation_time: datetime.datetime):
        """
        Propagate CME to given time using a simple drag model (see Vrsnak et al. 2013).

        :param simulation_time:  simulation time (datetime).
        """
        if simulation_time < self._time_0:
            raise ValueError("given simulation time is before initial")

        dt = np.int32((simulation_time - self._time_0).total_seconds())
        dv = np.float64(self._speed_0 - self._bg_speed)

        self._radius_t = propagate_r(dv, dt, self._bg_speed, self._bg_drag, self._bg_sign, self._radius_0)
        self._speed_t = propagate_v(dv, dt, self._bg_speed, self._bg_drag, self._bg_sign)

    # ========== FLUX ROPE ==========

    def extract_field(self, v: np.ndarray) -> (np.ndarray, np.bool):
        """
        Extract magnetic field vector at the given Cartesian coordinates.

        :param v: Cartesian coordinates (x, y, z)
        :return: Magnetic field vector in Cartesian coordinates (x, y, z), Flag if v was inside MFR
        """
        (r, psi, phi) = self.transform_into(v)

        if r < 1:
            v = gold_hoyle(r, psi, phi, self.b_axial, self.rho_0, self.rho_1, self._aspect, self._handedness,
                           self._turns)
            return errot(v, self._errot_from), True
        else:
            return np.array([0.0, 0.0, 0.0]), False

    # ========== VISUALIZE ==========

    def wire_frame(self, s: int) -> np.ndarray:
        """
        Returns matrices for plotting the wire frame of the MFR

        :param s: size (wireframe resolution)
        :return: three s x s arrays
        """
        r = [1.0]
        d = int(360 / (s - 1))
        u = np.radians(np.r_[0:360. + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        p = self.transform_from(np.array(list(itertools.product(r, u, v))).reshape(s ** 2, 3))

        return p.reshape((s, s, 3))

    # ========== INTERNAL ==========

    def update_rodrigues_coefficients(self):
        """
        Calculate the Rodrigues-coefficients for rotating into and out of the coordinate system. This function must be
        called after changing either the longitude, latitude or inclination of the CME.
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

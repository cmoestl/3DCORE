# -*- coding: utf-8 -*-

"""vrsnak.py

Propagation drag model (see Vrsnak et al. 2013).
"""

import numba as nb
import numpy as np


@nb.njit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def vk_drag_model(radius: float, speed: float, dt: float, bg_speed: float, bg_drag: float, bg_sign: int) -> np.ndarray:
    dv = speed - bg_speed
    radius_dt =  bg_sign / bg_drag * np.log1p(bg_sign * bg_drag * dv * dt) + bg_speed * dt + radius
    speed_dt = dv / (1 + bg_sign * bg_drag * dv * dt) + bg_speed

    return np.array([radius_dt, speed_dt])

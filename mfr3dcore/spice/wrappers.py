# -*- coding: utf-8 -*-

"""wrappers.py

Contains various wrapper functions for handling SPICE kernels with spiceypy.
"""

import datetime
import numpy as np
import os
import spiceypy

from typing import Union


def load_kernels(name: str = "generic", external_path: str = None):
    """
    Load all SPICE kernels listed in the "kernellist.txt" file for a specific folder. Optionally a external folder can
    be specified which contains the required SPICE kernels.

    Note that not all SPICE kernels are part of the repository and may have to be downloaded from the official sources.
    The locations of the required kernels are listed in the respective "kernellist_url.txt" files.

    :param name: folder name
    :param external_path: external directory for SPICE kernels (optional)
    """
    path = os.path.join(os.path.dirname(__file__), "kernels", name.lower(), "kernellist.txt")

    if not os.path.isfile(path):
        raise FileNotFoundError("kernellist.txt file not found for: {}.".format(name))

    with open(path) as fh:
        lines = fh.read().splitlines()

    for line in lines:
        if external_path:
            spiceypy.furnsh(os.path.join(external_path, line))
        else:
            spiceypy.furnsh(os.path.join(os.path.dirname(__file__), "kernels", name.lower(), line))


def get_trajectory(body: str, times: Union[datetime.datetime, list], frame: str = "J2000", observer: str = "Sun",
                   units=None) -> np.ndarray:
    """
    Get body trajectory in a specific reference frame at a single or multiple epoch(s).

    :param body: body name
    :param times: datetime or list of datetime's
    :param frame: reference frame
    :param observer: observer name
    :param units: unit length (AU, km, m)
    :return:
    """
    if isinstance(times, list):
        times_et = [spiceypy.datetime2et(t.replace(tzinfo=None)) for t in times]
    elif isinstance(times, datetime.datetime):
        times_et = spiceypy.datetime2et(times.replace(tzinfo=None))
    else:
        raise TypeError("Function parameter \"times\" is an invalid type ({}).".format(type(times)))

    result = np.array(spiceypy.spkpos(body, times_et, frame, 'NONE', observer)[0])

    if not units or units == "km":
        return result
    elif units.upper() == "AU":
        return result / 1.496e8
    elif units.lower() == "m":
        return result * 1e3
    else:
        raise ValueError("{} is not a valid unit.".format(units))


def get_vector(frame_from: str, frame_to: str, vector: np.ndarray, time: datetime.datetime) -> np.ndarray:
    """
    Transform vector from one reference frame to another.

    :param frame_from: source frame
    :param frame_to: target frame
    :param vector: vector
    :param time: datetime
    :return:
    """
    if frame_from == frame_to:
        return vector

    times_et = spiceypy.datetime2et(time.replace(tzinfo=None))

    return spiceypy.mxv(spiceypy.pxform(frame_from, frame_to, times_et), vector)

# -*- coding: utf-8 -*-

import numpy as np
import os
import spiceypy


def get_trajectory(spacecraft, times, frame, observer, units=None):
    """
    Fetch spacecraft trajectory from SPICE kernels

    :param spacecraft: target body name
    :param times: datetime or list of datetimes
    :param frame: reference frame
    :param observer: observer body name
    :return:
    """
    if isinstance(times, list):
        times_et = [spiceypy.datetime2et(t.replace(tzinfo=None)) for t in times]
    else:
        times_et = spiceypy.datetime2et(times.replace(tzinfo=None))

    result = np.array(spiceypy.spkpos(spacecraft, times_et, frame, 'NONE', observer)[0])

    if not units or units == "km":
        return result
    elif units.upper() == "AU":
        return result / 1.496e8


def load_kernels(spacecraft, external_path=None):
    """
    Load all SPICE kernels listed in the "kernellist.txt" file for a specific spacecraft.

    Note that not all SPICE kernels are part of the repository and may have to be downloaded from the official sources.
    The locations of the required kernels may be listed in the respective "kernellist_url.txt" files.

    :param spacecraft: spacecraft name
    :param external_path: external directory for SPICE kernels
    """
    path = os.path.join(os.path.dirname(__file__), "kernels", spacecraft.lower(), "kernellist.txt")

    with open(path) as fh:
        lines = fh.read().splitlines()

    for line in lines:
        if external_path:
            spiceypy.furnsh(os.path.join(external_path, line))
        else:
            spiceypy.furnsh(os.path.join(os.path.dirname(__file__), "kernels", spacecraft.lower(), line))


def transform_vector(frame_from, frame_to, vector, time):
    times_et = spiceypy.datetime2et(time.replace(tzinfo=None))

    return spiceypy.mxv(spiceypy.pxform(frame_from, frame_to, times_et), vector)

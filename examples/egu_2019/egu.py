# -*- coding: utf-8 -*-

"""egu.py

This script generates the figures for the Möstl et al. EGU 2019 poster "PREDICTED STATISTICS OF CORONAL MASS EJECTIONS
OBSERVED BY PARKER SOLAR PROBE AND FORWARD MODELING OF THEIR IN SITU MAGNETIC FIELD".
"""

import ciso8601
import datetime
import matplotlib
import mfr3dcore
import numpy as np
import spiceypy

matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    t0 = ciso8601.parse_datetime("2022-06-01 20:00+00:00")

    event = mfr3dcore.models.TorusV1(
        5 * 695508, # R_0
        600,        # V_0
        t0,
        1.0,        # Aspect ratio
        145,        # Longitude
        2.5,        # Latitude
        90-90,      # Inclination
        0.24,       # Diameter (1AU)
        -1,         # Handedness
        12,         # Strength (1AU)
        10,         # Turns
        1.5e-7,     # Drag parameter
        400,        # Solar wind speed
        5           # Solar wind strength
    )

    # load spice kernels
    mfr3dcore.spice.load_kernels("generic")
    mfr3dcore.spice.load_kernels("SPP")

    # required because of two separate names for PSP
    spiceypy.boddef("SPP_SPACECRAFT", spiceypy.bodn2c("SPP"))

    # crossing times (hours after t0, these values were found manually)
    t1 = t0 + datetime.timedelta(hours=3)
    t2 = t0 + datetime.timedelta(hours=28)

    # ========== 3D PLOTS ==========

    # define time series
    ts = [t0 + datetime.timedelta(hours=i) for i in range(-100, 100)]
    ts_step1 = [t0 + datetime.timedelta(hours=i) for i in range(-1, 1 + 1)]
    ts_step2 = [t1 + datetime.timedelta(hours=i) for i in range(-1, 1 + 1)]
    ts_step3 = [t2 + datetime.timedelta(hours=i) for i in range(-1, 1 + 1)]

    # get trajectories
    psp_trajectory = mfr3dcore.spice.get_trajectory("SPP", ts, "ECLIPJ2000", "Sun", units="AU")
    psp_trajectory_step1 = mfr3dcore.spice.get_trajectory("SPP", ts_step1, "ECLIPJ2000", "Sun", units="AU")
    psp_trajectory_step2 = mfr3dcore.spice.get_trajectory("SPP", ts_step2, "ECLIPJ2000", "Sun", units="AU")
    psp_trajectory_step3 = mfr3dcore.spice.get_trajectory("SPP", ts_step3, "ECLIPJ2000", "Sun", units="AU")

    # FIRST PLOT
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.scatter(0, 0, 0, color="y", s=100)

    # plot full psp trajectory
    ax.plot(psp_trajectory[:, 0], psp_trajectory[:, 1], psp_trajectory[:, 2], color="k", lw=1, ls="--")

    # plot trajectories at steps +/- 1h
    ax.plot(psp_trajectory_step1[:, 0], psp_trajectory_step1[:, 1], psp_trajectory_step1[:, 2], color="r", lw=2)
    ax.plot(psp_trajectory_step2[:, 0], psp_trajectory_step2[:, 1], psp_trajectory_step2[:, 2], color="g", lw=2)

    # plot psp positions at steps
    ax.scatter(psp_trajectory_step1[1, 0], psp_trajectory_step1[1, 1], psp_trajectory_step1[1, 2], color="r", s=25)
    ax.scatter(psp_trajectory_step2[1, 0], psp_trajectory_step2[1, 1], psp_trajectory_step2[1, 2], color="g", s=25)

    # plot wireframe step 1
    event.propagate(t0)
    event.plot_wireframe(1, 4, alpha=0.1, color="r", handle=ax)

    # plot wireframe & field line step 2
    event.propagate(t1)
    event.plot_wireframe(1, 4, alpha=0.1, color="g", handle=ax)
    event.plot_fieldlines(np.array([1, 0.01, 0]), alpha=0.3, color="g", handle=ax)

    ax.set_xlim([-0.05, 0.05])
    ax.set_ylim([-0.05, 0.05])
    ax.set_zlim([-0.05, 0.05])

    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_zlabel("Z [AU]")

    ax.view_init(elev=25, azim=160)

    ax.legend(handles=[
        Line2D([0], [0], color='k', lw=1, label='PSP Trajectory', ls="--"),
        Line2D([0], [0], color='r', lw=1, label="CME + {}h".format(0), ls="-"),
        Line2D([0], [0], color='g', lw=1, label="CME + {}h".format(3), ls="-"),
    ], loc='lower right')

    # SECONT PLOT
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.scatter(0, 0, 0, color="y", s=100)

    # plot full psp trajectory
    ax.plot(psp_trajectory[:, 0], psp_trajectory[:, 1], psp_trajectory[:, 2], color="k", lw=1, ls="--")

    # plot trajectories at steps +/- 1h
    ax.plot(psp_trajectory_step3[:, 0], psp_trajectory_step3[:, 1], psp_trajectory_step3[:, 2], color="b", lw=2)

    # plot psp positions at step 3
    ax.scatter(psp_trajectory_step3[1, 0], psp_trajectory_step3[1, 1], psp_trajectory_step3[1, 2], color="b", s=25)

    # plot wireframe & field line step 3
    event.propagate(t2)
    event.plot_wireframe(1, 4, alpha=0.1, color="b", handle=ax)
    event.plot_fieldlines(np.array([1, 0.01, 0]), alpha=0.3, color="b", handle=ax)

    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.25, 0.25])
    ax.set_zlim([-0.25, 0.25])

    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_zlabel("Z [AU]")

    ax.view_init(elev=25, azim=160)

    ax.legend(handles=[
        Line2D([0], [0], color='k', lw=1, label='PSP Trajectory', ls="--"),
        Line2D([0], [0], color='b', lw=1, label="CME + {}h".format(28), ls="-")
    ], loc='lower right')

    # ========== SYNTHETIC MAGNETIC FIELD MEASUREMENTS ==========

    # define time series for measurement
    ts_a = [t0 + datetime.timedelta(hours=2.5, minutes=i) for i in range(0, 120)]
    ts_b = [t0 + datetime.timedelta(hours=24, minutes=i) for i in range(0, 840)]

    b_a = []
    b_b = []
    b_c = []

    # measure for time series a
    for t in ts_a:
        event.propagate(t)
        b_a.append(
            mfr3dcore.spice.get_vector("ECLIPJ2000", "PSP_RTN", event.magnetic_field(
                mfr3dcore.spice.get_trajectory("SPP_SPACECRAFT", t, "ECLIPJ2000", "Sun",
                                               units="AU"))[0], t)
        )

    # measure for time series b
    for t in ts_b:
        event.propagate(t)
        b_b.append(
            mfr3dcore.spice.get_vector("ECLIPJ2000", "PSP_RTN", event.magnetic_field(
                mfr3dcore.spice.get_trajectory("SPP_SPACECRAFT", t, "ECLIPJ2000", "Sun",
                                               units="AU"))[0], t)
        )

    # measure for time series a (but fixed position)
    t_c = t0 + datetime.timedelta(hours=0)

    for t in ts_a:
        event.propagate(t)
        b_c.append(
            mfr3dcore.spice.get_vector("ECLIPJ2000", "PSP_RTN", event.magnetic_field(
                mfr3dcore.spice.get_trajectory("SPP_SPACECRAFT", t_c, "ECLIPJ2000", "Sun",
                                               units="AU"))[0], t_c)
        )

    b_a = np.array(b_a)
    b_b = np.array(b_b)
    b_c = np.array(b_c)

    # plot stuff
    fig_a, ax_a = plt.subplots(1, 1, figsize=(10, 5))
    fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 5))
    fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 5))

    kwargs = {
        "lw": 2,
        "ls": "-",
        "marker": "",
        "alpha": 0.8
    }

    ax_a.plot_date(date2num(ts_a), np.sqrt(np.sum(b_a ** 2, axis=1)), color="k", **kwargs, label="B Total")
    ax_a.plot_date(date2num(ts_a), b_a[:, 0], color="r", **kwargs, label="B Radial")
    ax_a.plot_date(date2num(ts_a), b_a[:, 1], color="g", **kwargs, label="B Tangential")
    ax_a.plot_date(date2num(ts_a), b_a[:, 2], color="b", **kwargs, label="B Normal")

    ax_b.plot_date(date2num(ts_b), np.sqrt(np.sum(b_b ** 2, axis=1)), color="k", **kwargs, label="B Total")
    ax_b.plot_date(date2num(ts_b), b_b[:, 0], color="r", **kwargs, label="B Radial")
    ax_b.plot_date(date2num(ts_b), b_b[:, 1], color="g", **kwargs, label="B Tangential")
    ax_b.plot_date(date2num(ts_b), b_b[:, 2], color="b", **kwargs, label="B Normal")

    ax_c.plot_date(date2num(ts_a), np.sqrt(np.sum(b_c ** 2, axis=1)), color="k", **kwargs, label="B Total")
    ax_c.plot_date(date2num(ts_a), b_c[:, 0], color="r", **kwargs, label="B Radial")
    ax_c.plot_date(date2num(ts_a), b_c[:, 1], color="g", **kwargs, label="B Tangential")
    ax_c.plot_date(date2num(ts_a), b_c[:, 2], color="b", **kwargs, label="B Normal")

    fig_a.autofmt_xdate()
    fig_b.autofmt_xdate()
    fig_c.autofmt_xdate()

    ax_a.set_xlabel("")
    ax_a.set_ylabel("B [nT]")

    ax_b.set_xlabel("")
    ax_b.set_ylabel("B [nT]")

    ax_c.set_xlabel("")
    ax_c.set_ylabel("B [nT]")

    ax_a.grid()
    ax_b.grid()
    ax_c.grid()

    ax_a.legend(loc="upper right")
    ax_b.legend(loc="upper right")
    ax_c.legend(loc="upper right")

    ax_a.xaxis.set_major_formatter(DateFormatter('%b %d %H:%M'))
    ax_b.xaxis.set_major_formatter(DateFormatter('%b %d %H:%M'))
    ax_c.xaxis.set_major_formatter(DateFormatter('%b %d %H:%M'))

    plt.tight_layout()
    plt.show()
    exit()

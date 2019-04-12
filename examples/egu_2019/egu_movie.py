# -*- coding: utf-8 -*-

"""egu_movie.py

This script generates an accompanying movie for the Möstl et al. EGU 2019 poster "PREDICTED STATISTICS OF CORONAL MASS EJECTIONS
OBSERVED BY PARKER SOLAR PROBE AND FORWARD MODELING OF THEIR IN SITU MAGNETIC FIELD".
"""

import ciso8601
import datetime
import matplotlib
import mfr3dcore
import numpy as np
import spiceypy
import sys
import os

#matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':


    plt.close('all')
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

    # define time series for simulation
    ts = [t0 + datetime.timedelta(minutes=i) for i in range(-1*60, 50*60,5)] 
    #ts_step1 = [t0 + datetime.timedelta(hours=i) for i in range(-1, 1 + 1)]
    #ts_step2 = [t1 + datetime.timedelta(hours=i) for i in range(-1, 1 + 1)]
    #ts_step3 = [t2 + datetime.timedelta(hours=i) for i in range(-1, 1 + 1)]

    # get trajectories
    psp_trajectory = mfr3dcore.spice.get_trajectory("SPP", ts, "ECLIPJ2000", "Sun", units="AU")
    #psp_trajectory_step1 = mfr3dcore.spice.get_trajectory("SPP", ts_step1, "ECLIPJ2000", "Sun", units="AU")
    #psp_trajectory_step2 = mfr3dcore.spice.get_trajectory("SPP", ts_step2, "ECLIPJ2000", "Sun", units="AU")
    #psp_trajectory_step3 = mfr3dcore.spice.get_trajectory("SPP", ts_step3, "ECLIPJ2000", "Sun", units="AU")


    #for plotting full trajectory
    tf = [t0 + datetime.timedelta(hours=i) for i in range(-100, 100,1)] 
    psp_trajectory_full = mfr3dcore.spice.get_trajectory("SPP", tf, "ECLIPJ2000", "Sun", units="AU")

    #Sun 
    scale=695510/149597870.700 #Rs in km, AU in km
    # sphere with radius Rs in AU
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v)*scale
    y = np.sin(u)*np.sin(v)*scale
    z = np.cos(v)*scale

    # Movie
    animdirectory='mfr_animation'
    if os.path.isdir(animdirectory) == False: os.mkdir(animdirectory)
    directory='mfr_plots'
    if os.path.isdir(directory) == False: os.mkdir(directory)

    fig = plt.figure(figsize=(12, 12))
    
    boxsize=0.1

    # for k in np.arange(100,np.size(psp_trajectory,0),1):
    for k in np.arange(0,len(ts),1):
    #for k in np.arange(0,540,1):
     
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        ax.scatter(0, 0, 0, color="y", s=100)
     
        # draw Sun sphere
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow', linewidth=0, antialiased=False)

        # plot full psp trajectory
        ax.plot(psp_trajectory_full[:, 0], psp_trajectory_full[:, 1], psp_trajectory_full[:, 2], color="k", lw=1, ls="--")

         
        ax.view_init(elev=30, azim=151+k/8)
        #ax.view_init(elev=90, azim=171)

        ax.set_xlim([-boxsize, boxsize])
        ax.set_ylim([-boxsize, boxsize])
        ax.set_zlim([-boxsize, boxsize])

        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_zlabel("Z [AU]")
        plt.figtext(0.1, 0.95,'t0 + '+str(np.round(k/12,1))+' h', fontsize=20, ha='left',color='black')	


        # plot trajectories at steps (time of trajectory is ts)
        ax.scatter(psp_trajectory[k, 0], psp_trajectory[k, 1], psp_trajectory[k, 2], color="k", marker='o', alpha=0.8,s=20)
        plt.tight_layout()
        
        #if time after cme launch time
        if ts[k] > t0:  
        
          event.propagate(ts[k])
          event.plot_wireframe(1, 4, alpha=0.1, color="g", handle=ax)
          event.plot_fieldlines(np.array([1, 0.01, 0]), alpha=0.5, color="r", handle=ax)
          event.plot_fieldlines(np.array([0.3, 0.01, 0.1]), alpha=0.5, color="blue", handle=ax)
 
        #save figure
        framestr = '%05i' % (k)  
        filename=animdirectory+'/mfr_anim_'+framestr+'.jpg'    
        plt.savefig(filename,dpi=100,facecolor=fig.get_facecolor(), edgecolor='none')
        print(k)

    os.system('ffmpeg -r 30 -i mfr_animation/mfr_anim_%05d.jpg -b 5000k -r 30 mfr_plots/egu19_anim_top.mp4 -y -loglevel quiet')
 
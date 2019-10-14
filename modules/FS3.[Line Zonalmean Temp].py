# -*- coding: utf-8 -*-
"""
This code generates Fig. S3

Zonal mean climatological surface air temperature from With-Aerosol scenario (a) and changes induced by anthropogenic aerosols (b)

by Yixuan Zheng (yxzheng@carnegiescience.edu)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset

#import geopandas as gp
from mpl_toolkits.basemap import Basemap,maskoceans
import matplotlib.cm as cm
import _env
import matplotlib.colors as mcolors
import cmocean
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)
of_plot = odir_plot + 'FS3.Line_Zonalmean_Temp.png'

year = _env.year
p_scen = 'No-Aerosol' #aerosol removal scenario
ds = 'ERA-Interim'


if_temp = _env.odir_root + '/sim_temperature/Simulated_Global_Gridded_TREFHT.nc'
lat = Dataset(if_temp)['lat'][:]
lon = Dataset(if_temp)['lon'][:]


itbl_temp = pd.DataFrame(index=np.arange(0,len(lat)),columns = ['Lat','Climatological','Diff_No-Aerosol','Diff_No-Sufalte'])
itbl_temp['Lat'] = lat.data

arr_btemp = Dataset(if_temp)['TREFHT_With-Aerosol'][:]
itbl_temp['Climatological'] = arr_btemp.mean(axis=1)

for scen in _env.scenarios[1::]:
    arr_dtemp = Dataset(if_temp)['TREFHT_With-Aerosol'][:] - Dataset(if_temp)['TREFHT_' + scen][:]
    itbl_temp['Diff_'+scen] = arr_dtemp.mean(axis=1)

#=============Line plot: zonal mean climatological temperature and changes============#
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(121)
cp = sns.color_palette("Blues",10)[8]
p = plt.plot(itbl_temp['Climatological'],itbl_temp['Lat'],lw=3,alpha=0.7,color =  cp)
plt.ylim([-92,92])
plt.xlim([-10,30])
xticks = [-20,-10,0,10,20,30]
plt.xticks(xticks)
plt.ylabel('Latitude',size=14)
plt.xlabel('Surface T (\N{DEGREE SIGN}C)',size=14)

for xt in xticks[::]:
    plt.vlines(xt,-92,92,lw=1,linestyles='dashed',colors='grey')

####latitudinal temperature changes####
ax = fig.add_subplot(122)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
cp = sns.color_palette("Purples",10)[8]
p = plt.plot(itbl_temp['Diff_No-Aerosol'],itbl_temp['Lat'],lw=3,alpha=0.7,color = cp)
cp = sns.color_palette("Greys",10)[8]
p = plt.plot(itbl_temp['Diff_No-Sulfate'],itbl_temp['Lat'],lw=3,alpha=0.7,color = cp)

plt.ylim([-92,92])
#plt.xlim([-3.5,0])
#plt.xticks([-3,-2,-1,0])
plt.xlim([-2.5,0])
plt.xticks([-2,-1,0])
plt.ylabel('Latitude',size=14,rotation=270)
#plt.xlabel('Change in T (\N{DEGREE SIGN}C)',size=14)
plt.xlabel('Change in T (\N{DEGREE SIGN}C)',size=14)
plt.vlines(-2,-92,92,lw=1,linestyles='dashed',colors='grey')
plt.vlines(-1,-92,92,lw=1,linestyles='dashed',colors='grey')
plt.savefig(of_plot, dpi=300,bbox_inches='tight')   

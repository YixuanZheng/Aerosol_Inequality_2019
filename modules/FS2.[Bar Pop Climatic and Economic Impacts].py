# -*- coding: utf-8 -*-
"""
This code generates Fig. S2

Population distribution of climatic and economic impacts of anthropogenic aerosol emissions

by Yixuan Zheng (yxzheng@carnegiescience.edu)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import _env
import seaborn.apionly as sns
import matplotlib
import colorcet
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)
of_plot = odir_plot + 'FS2.Bar_Pop_Climatic_and_Economic_Impacts.png'

year = _env.year
p_scen = 'No-Aerosol' #aerosol removal scenario
ds = 'ERA-Interim'

#=============================================================================
#================================vertical plot (4x1)=====================================
if_pop = _env.idir_root + '/pop/' + 'GPW_POP_25x19deg_2010.nc'
P_grid = Dataset(if_pop)['pop'][:]

if_aod = _env.odir_root + '/sim_aodvis/Simulated_Global_Gridded_AODVIS.nc'
arr_aod = -Dataset(if_aod)['AODVIS_d_' + p_scen + '_With-Aerosol'][:]

if_temp = _env.odir_root + '/sim_temperature/Simulated_Global_Gridded_TREFHT.nc'
arr_dtemp = Dataset(if_temp)['TREFHT_With-Aerosol'][:] - Dataset(if_temp)['TREFHT_' + p_scen ][:]

if_dyr = _env.odir_root + '/sim_temperature_running_mean/Year-Delayed_RunningAvg_' + p_scen + '.nc'
arr_year = Dataset(if_dyr)['TREFHT'][:]

if_rgdp = _env.odir_root + '/gdp_'+ds+'/GDP_Changes_Burke_country-lag0_'+str(year)+'_' + ds + '_' + p_scen + '_gridded.nc'
arr_gdp = Dataset(if_rgdp)['GDP_Ratio_Median'][:]*100 # to percent

fig = plt.figure(figsize=(5,20))
#===============================Bar: changes in sulfate burden==============================
otbl_t_b = pd.DataFrame(index=np.arange(0,0.17,0.01),columns=['aod','color'])
#otbl_t_b['color'] = sns.color_palette('PuBuGn_d',17).as_hex()[::-1]
otbl_t_b['color'] = sns.color_palette('magma',17).as_hex()[::-1]


for itick,tick in enumerate(otbl_t_b.index):
    if itick == (len(otbl_t_b)-1):
        ind  = np.where(arr_aod >= 0.17)
        otbl_t_b.loc[tick,'aod'] = (P_grid[ind]).sum()
    else:
        ind  = np.where((arr_aod >= itick*0.01) & (arr_aod < (itick+1)*0.01))
        otbl_t_b.loc[tick,'aod'] = (P_grid[ind]).sum()


ax = fig.add_subplot(411)
(otbl_t_b['aod']/1e9).plot(kind='bar',width=1,colors=otbl_t_b['color'])
print(plt.xlim())
plt.xlim([-0.5,16.5])
plt.xticks(np.arange(18)[::2]-0.5,[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16],size=14,rotation=0)
plt.yticks(size=14)
plt.ylabel('Billion People',size=14)

plt.text(0.02,0.92,'a',size=16, transform=ax.transAxes)  #fontweight='bold',

#===============================Bar: temp change==============================

otbl_t_p = pd.DataFrame(index=-2.2+0.2*np.arange(0,13),columns=['pop','color'])

my_cmap = []
for ind in [0,20,40,60,80,100,120,140,160,200,240]:
    my_cmap.append(colorcet.kbc[ind])    
my_cmap.append(colorcet.fire[250])
my_cmap.append(colorcet.fire[230])
otbl_t_p['color'] = my_cmap

for itick,tick in enumerate(otbl_t_p.index):
    if itick == 0:
        ind  = np.where(arr_dtemp.data < -2)
        otbl_t_p.loc[tick,'pop'] = (P_grid[ind]).sum()
    elif itick == (len(otbl_t_p)-1):
        ind  = np.where(arr_dtemp >= 0.2)
        otbl_t_p.loc[tick,'pop'] = (P_grid[ind]).sum()
    else:
        ind  = np.where((arr_dtemp >= -2.2+itick*0.2) & (arr_dtemp < -2.0+itick*0.2))
        otbl_t_p.loc[tick,'pop'] = (P_grid[ind]).sum()


ax = fig.add_subplot(412) 
#ax.yaxis.tick_right()
#ax.yaxis.set_label_position("right")
(otbl_t_p['pop']/1e9).plot(kind='bar',width=1,colors=otbl_t_p['color'])
#ax.set_yscale("log")
plt.xticks(np.arange(22)[::2]+0.5,[-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2],size=14,rotation=0)
plt.yticks(size=14)
plt.xlim([-0.5,12.5])
plt.xlabel('\N{DEGREE SIGN}C',size=14)
plt.ylabel('Billion People',size=14)

plt.text(0.02,0.92,'b',size=16, transform=ax.transAxes)  #fontweight='bold',
#==============================Bar: Year change================================
otbl_t_y = pd.DataFrame(index=np.arange(0,80,10),columns=['year','color'])
#otbl_t_y['color'] = sns.cubehelix_palette(26).as_hex()[0:18:3]
otbl_t_y['color'] = sns.cubehelix_palette(24).as_hex()[0:24:3]

for itick,tick in enumerate(otbl_t_y.index):
    if itick == 0:
        ind  = np.where(arr_year.data < 10)
        otbl_t_y.loc[tick,'year'] = (P_grid[ind]).sum()
    elif itick == (len(otbl_t_y)-1):
        ind  = np.where(arr_year >= 70)
        otbl_t_y.loc[tick,'year'] = (P_grid[ind]).sum()
    else:
        ind  = np.where((arr_year >= itick*10) & (arr_year < (itick+1)*10))
        otbl_t_y.loc[tick,'year'] = (P_grid[ind]).sum()

ax = fig.add_subplot(413) 
(otbl_t_y['year']/1e9).plot(kind='bar',width=1,colors=otbl_t_y['color'])
plt.xlim([-0.5,7.5])
plt.xticks(np.arange(7)+0.5,[10,20,30,40,50,60,70],size=14,rotation=0)
plt.yticks(size=14)
plt.xlabel('Year',size=14)
plt.ylabel('Billion People',size=14)

plt.text(0.02,0.92,'c',size=16, transform=ax.transAxes)  #fontweight='bold',
#==============================Bar: GDP change rate============================
otbl_t_g = pd.DataFrame(index=-1.6+np.arange(0,16)*0.2,columns=['gdp','color'])
otbl_t_g['color'] = sns.color_palette('RdBu_r',20).as_hex()[2:18]
arr_gdp = -arr_gdp
for itick,tick in enumerate(otbl_t_g.index):
    if itick == 0:
        ind  = np.where(arr_gdp.data < -1.4)
        otbl_t_g.loc[tick,'gdp'] = (P_grid[ind]).sum()
    elif itick == (len(otbl_t_g)-1):
        ind  = np.where(arr_gdp >= 1.4)
        otbl_t_g.loc[tick,'gdp'] = (P_grid[ind]).sum()
    else:
        ind  = np.where((arr_gdp >= -1.6 + itick*0.2) & (arr_gdp < -1.6+(itick+1)*0.2))
        otbl_t_g.loc[tick,'gdp'] = (P_grid[ind]).sum()


ax = fig.add_subplot(414)
(otbl_t_g['gdp']/1e9).plot(kind='bar',width=1,colors=otbl_t_g['color'])
plt.xlim([-0.5,15.5])
plt.xticks(np.arange(15)[::2]+0.5,[-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4][::2],size=14,rotation=0)
plt.yticks(size=14)
plt.xlabel('%',size=14)
plt.ylabel('Billion People',size=14)
plt.text(0.02,0.92,'d',size=16, transform=ax.transAxes)#, fontname = 'Helvetica',fontweight = 'bold')

plt.savefig(of_plot, dpi=300,bbox_inches='tight') 

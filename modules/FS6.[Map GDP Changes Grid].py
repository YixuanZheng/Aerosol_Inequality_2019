# -*- coding: utf-8 -*-
'''
This code generates Fig. S6

Spatial distribution of economic impacts introduced by anthropogenic aerosol-induced cooling.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from mpl_toolkits.basemap import Basemap,maskoceans
import _env
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

def set_latlon_ticks(ax,m):
    ax.set_xticks(np.arange(-160,161,40))
    ax.set_xticklabels('')
    ax.set_yticks(np.arange(-90,91,45))
    ax.set_yticklabels('')
    
    parallels = np.arange(-90.,91,45.)
    m.drawparallels(parallels,labels=[True,False,False,False],dashes=[3,3],xoffset=5,linewidth = 0)
    meridians = np.arange(-160,161,40.)
    m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[3,3],yoffset=5,linewidth = 0)


p_scen = 'No-Aerosol' #aerosol removal scenario
ds = 'ERA-Interim'

odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)
of_plot = odir_plot + 'FS6.Map_GDP_Changes_Grid.png'

if_mask = _env.idir_root + '/regioncode/CESM_19x26_Land-Ocean_Mask.nc'
iarr_land = Dataset(if_mask)['landmask'][:]

fig = plt.figure(figsize=(21,5))

if_rgdp = _env.odir_root + '/gdp_' + ds + '/GDP_Changes_Burke_country-lag0_2010_' + ds + '_' + p_scen + '_gridded.nc'

arr_gdp = Dataset(if_rgdp)['GDP_Ratio_Median'][:]*100 # to percent

if_temp = _env.odir_root + '/sim_temperature/Simulated_Global_Gridded_TREFHT.nc'
arr_t_pval = Dataset(if_temp)['TREFHT_P_No-Aerosol_With-Aerosol'][:]


ax = fig.add_subplot(111) 
m = Basemap(ellps = 'WGS84',
    llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
    suppress_ticks=True)#resolution='i',

m.drawmapboundary()

lat = Dataset(if_temp)['lat'][:]
lon = Dataset(if_temp)['lon'][:]

#rearrange matrix for plot
lon_tmp = lon.copy()

lon[72:144] = lon[0:72]
lon[0:72] = lon_tmp[72:144]
lon[lon>=180] = lon[lon>=180] - 360

arr_gdp_tmp = arr_gdp.copy()
arr_gdp[:,72:144] = arr_gdp[:,0:72]
arr_gdp[:,0:72] = arr_gdp_tmp[:,72:144]

#mask Atlantic
arr_gdp[0:21,:] = np.nan

arr_t_pval_tmp = arr_t_pval.copy()
arr_t_pval[:,72:144] = arr_t_pval[:,0:72]
arr_t_pval[:,0:72] = arr_t_pval_tmp[:,72:144]

x,y = np.meshgrid(lon,lat)

lat_os = 180/95.
lon_os = 360/144.
lon_ = lon+2.5/2
lat_ = lat+lat_os/2
lon_arr = np.repeat(lon_[np.newaxis,:],96,axis=0)
lat_arr = np.repeat(lat_[:,np.newaxis],144,axis=1)

arr_gdp_ocean_masked = maskoceans(np.repeat(lon_[np.newaxis,:],96,axis=0),np.repeat(lat_[:,np.newaxis],144,axis=1),arr_gdp)


my_cmap = ListedColormap(sns.color_palette('RdBu_r',20).as_hex()[2:18])
cs = m.pcolormesh(lon,lat,-np.squeeze(arr_gdp_ocean_masked),cmap=my_cmap, vmin=-1.6, vmax=1.6) #,legend_ticks)

# add colorbar.
cbar = m.colorbar(cs,location='right',pad="5%",ticks=[-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4][::2])
cbar.set_label('(%)',fontsize = 14,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4][::2],size=14)

m.readshapefile(_env.idir_root + '/shape/kx-world-coastline-110-million-SHP/world-coastline-110-million',
        'coastline',drawbounds=True,linewidth=0.8,color='k',
        zorder=2) 

arr_t_pval_ocean_masked = maskoceans(np.repeat(lon_[np.newaxis,:],96,axis=0),np.repeat(lat_[:,np.newaxis],144,axis=1),arr_t_pval)
arr_t_pval_ocean_masked = arr_t_pval_ocean_masked.filled(-1)
lon_arr_m = lon_arr[np.where(arr_t_pval_ocean_masked>0.05)]
lat_arr_m = lat_arr[np.where(arr_t_pval_ocean_masked>0.05)]

x,y = m(lon_arr_m, lat_arr_m)
m.scatter(x, y, marker='+', color='black', zorder=5,s=0.1,alpha=0.4)

set_latlon_ticks(ax,m)

plt.savefig(of_plot, dpi=300,bbox_inches='tight') 

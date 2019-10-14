# -*- coding: utf-8 -*-

'''
This code generates Fig. S4

Spatial distribution of climatic and economic impacts introduced by anthropogenic SULFATE aerosol emissions.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import _env
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns
import matplotlib
import colorcet
import geopandas as gp
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def shift_lon(arr):
    ndim = arr.ndim
    
    if ndim == 1: #lon
        arr_tmp = arr.copy()
        arr[72:144] = arr[0:72]
        arr[0:72] = arr_tmp[72:144]
        arr[arr>=180] = arr[arr>=180] - 360
    elif ndim == 2:
        arr_tmp = arr.copy()
        arr[:,72:144] = arr[:,0:72]
        arr[:,0:72] = arr_tmp[:,72:144]
    return arr
    
def set_latlon_ticks(ax,m):
    ax.set_xticks(np.arange(-160,161,40))
    ax.set_xticklabels('')
    ax.set_yticks(np.arange(-90,91,45))
    ax.set_yticklabels('')
    
    parallels = np.arange(-90.,91,45.)
    m.drawparallels(parallels,labels=[True,False,False,False],dashes=[3,3],xoffset=5,linewidth = 0)
    meridians = np.arange(-160,161,40.)
    m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[3,3],yoffset=5,linewidth = 0)



matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

odir_plot = _env.odir_root + '/plot/'

_env.mkdirs(odir_plot)

of_plot = odir_plot + 'FS4.Map_Climatic_Economic_Impacts_Sulfate.png'

p_scen = 'No-Sulfate' #aerosol removal scenario
ds = 'ERA-Interim'

fig = plt.figure(figsize=(21,20))

##=========================MAP: changes in AOD=====================#        
if_aod = _env.odir_root + '/sim_aodvis/Simulated_Global_Gridded_AODVIS.nc'

arr_aod = -Dataset(if_aod)['AODVIS_d_' + p_scen + '_With-Aerosol'][:]

ax = fig.add_subplot(411)    

m = Basemap(ellps = 'WGS84',
    llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
    suppress_ticks=False)

m.drawmapboundary()

lat = Dataset(if_aod)['lat'][:]
lon = Dataset(if_aod)['lon'][:]

#rearrange matrix for plot
lon = shift_lon(lon)
arr_aod = shift_lon(arr_aod)

x,y = np.meshgrid(lon,lat)
x,y = m(x,y)    
    
my_cmap = ListedColormap(sns.color_palette('magma',17).as_hex()[::-1]) 
cs = m.pcolormesh(lon,lat,np.squeeze(arr_aod),cmap=my_cmap, vmin=0, vmax=0.17)
# add colorbar.
cbar = m.colorbar(cs,location='right',pad="5%",ticks=[0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16])
cbar.set_label(' ',fontsize = 14,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16],size=14)

m.readshapefile(_env.idir_root + '/shape/kx-world-coastline-110-million-SHP/world-coastline-110-million',
        'coastline',drawbounds=True,linewidth=0.8,color='k',
        zorder=2) 

plt.title('Sulfate-induced aerosol optical depth (at 550 nm) changes',size=18)
set_latlon_ticks(ax,m)
plt.text(0.03,0.98, (chr(ord('a') + 0)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')

#=========================MAP: global temperature changes=====================#    
if_temp = _env.odir_root + '/sim_temperature/Simulated_Global_Gridded_TREFHT.nc'
arr_dtemp = Dataset(if_temp)['TREFHT_With-Aerosol'][:] - Dataset(if_temp)['TREFHT_' + p_scen ][:]
arr_t_pval = Dataset(if_temp)['TREFHT_P_' + p_scen + '_With-Aerosol'][:]

    
ax = fig.add_subplot(412) 

m = Basemap(ellps = 'WGS84',
    llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
    suppress_ticks=False)

m.drawmapboundary()

lat = Dataset(if_temp)['lat'][:]
lon = Dataset(if_temp)['lon'][:]

#rearrange matrix for plot
lon = shift_lon(lon)
arr_dtemp = shift_lon(arr_dtemp)

x,y = np.meshgrid(lon,lat)
x,y = m(x,y)    

    
my_cmap = []
for ind in [0,20,40,60,80,100,120,140,160,200,240]:
    my_cmap.append(colorcet.linear_blue_5_95_c73[ind])
    
my_cmap.append(colorcet.linear_kryw_0_100_c71[250])
my_cmap.append(colorcet.linear_kryw_0_100_c71[230])

my_cmap = ListedColormap(my_cmap)    
cs = m.pcolormesh(lon,lat,np.squeeze(arr_dtemp),cmap=my_cmap, vmin=-2.2, vmax=0.4) #,legend_ticks)

cbar = m.colorbar(cs,location='right',pad="5%",ticks=[-2,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2])
cbar.set_label('\N{DEGREE SIGN}C',fontsize = 14,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([-2.0,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2.0],size=14)


m.readshapefile(_env.idir_root + '/shape/kx-world-coastline-110-million-SHP/world-coastline-110-million',
        'coastline',drawbounds=True,linewidth=0.8,color='k',
        zorder=2) 

lat_os = 180/95.
lon_os = 360/144.

lon_ = lon+2.5/2
lat_ = lat+lat_os/2

lon_arr = np.repeat(lon_[np.newaxis,:],96,axis=0)
lat_arr = np.repeat(lat_[:,np.newaxis],144,axis=1)

arr_t_pval = shift_lon(arr_t_pval)

lon_arr_m = lon_arr[np.where(arr_t_pval>0.05)]
lat_arr_m = lat_arr[np.where(arr_t_pval>0.05)]

x,y = m(lon_arr_m, lat_arr_m)
m.scatter(x, y, marker='+', color='black', zorder=5,s=0.1,alpha=0.4)

set_latlon_ticks(ax,m)

plt.title('Sulfate-induced temperature changes',size=18)
plt.text(0.03,0.98, (chr(ord('a') + 1)),size=16, color='white', horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')
#=====================MAP: years of delayed warming from sulfate===============#     
if_dyr = _env.odir_root + '/sim_temperature_running_mean/Year-Delayed_RunningAvg_' + p_scen + '.nc'

arr_year = Dataset(if_dyr)['TREFHT'][:]
   
ax = fig.add_subplot(413) 

m = Basemap(ellps = 'WGS84',
    llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
    suppress_ticks=False)

m.drawmapboundary()

lat = Dataset(if_temp)['lat'][:]
lon = Dataset(if_temp)['lon'][:]

#rearrange matrix for plot
lon = shift_lon(lon)
arr_year = shift_lon(arr_year)

arr_year[np.isnan(arr_year)] = 150

x,y = np.meshgrid(lon,lat)

my_cmap = ListedColormap(sns.cubehelix_palette(24).as_hex()[0:24:3]) 
cs = m.pcolormesh(lon,lat,np.squeeze(arr_year),cmap=my_cmap, vmin=0, vmax=80) #,legend_ticks)


cbar = m.colorbar(cs,location='right',pad="5%",ticks=[10,20,30,40,50,60,70])
cbar.set_label('Year',fontsize = 14,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([10,20,30,40,50,60,70],size=14)

m.readshapefile(_env.idir_root + '/shape/kx-world-coastline-110-million-SHP/world-coastline-110-million',
        'coastline',drawbounds=True,linewidth=0.8,color='k',
        zorder=2) 

lon_arr_m = lon_arr[np.where(arr_t_pval>0.05)]
lat_arr_m = lat_arr[np.where(arr_t_pval>0.05)]

x,y = m(lon_arr_m, lat_arr_m)
m.scatter(x, y, marker='+', color='black', zorder=5,s=0.1,alpha=0.4)

plt.title('Sulfate-induced delay in warming',size=18)

set_latlon_ticks(ax,m)
plt.text(0.03,0.98, (chr(ord('a') + 2)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')
#=====================MAP: percent of gdp changes from sulfate (country)===============#    

if_gdp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_GDP_' + ds + '_' + p_scen + '_Burke.xls'

if_ctrylist = _env.idir_root + '/regioncode/Country_List.xls'
if_ctryshp = (_env.idir_root + '/shape/country/country1.shp')

itbl_gdp = pd.read_excel(if_gdp,'country-lag0')
itbl_gdp.set_index('iso',inplace = True)
ishp_ctry = gp.read_file(if_ctryshp)

#correct country code
ishp_ctry.loc[ishp_ctry['GMI_CNTRY'] == 'ROM','GMI_CNTRY'] = 'ROU'
ishp_ctry.loc[ishp_ctry['GMI_CNTRY'] == 'ZAR','GMI_CNTRY'] = 'COD'
ishp_ctry.set_index('GMI_CNTRY',inplace = True)

ishp_ctry['GDP_median'] = itbl_gdp['GDP_median_benefit_ratio']
ishp_ctry.loc[pd.isna(ishp_ctry['GDP_median']),'GDP_median'] = -999

_env.mkdirs(_env.odir_root+'gdp_map_' + ds)
ishp_ctry.to_file(_env.odir_root+'gdp_map_' + ds  + '/gdp_country_'+p_scen+'.shp')
ishp_ctry.drop('geometry',axis=1).to_csv(_env.odir_root+'gdp_map_' + ds + '/country_gdp_ratio_median_' + ds + '.csv')
    
ax = fig.add_subplot(414) 
m = Basemap(ellps = 'WGS84',
    llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
    suppress_ticks=False)

m.drawmapboundary()
m.readshapefile(_env.odir_root+'gdp_map_' + ds  + '/gdp_country_'+p_scen,
        'country',drawbounds=True,linewidth=0.8,color='k',
        zorder=2) 


my_cmap = sns.color_palette('RdBu_r',20).as_hex()[2:18]
ind_ctry = 0
for info, shape in zip(m.country_info, m.country):
    gdp_r = info['GDP_median']
    
    if gdp_r == -999:
        color = '#E0E0E0'
    else:
        if gdp_r<-1.4:
            ind_color = 0
        elif  gdp_r>1.4:
            ind_color = len(my_cmap) - 1
        else:
            ind_color = int((gdp_r+1.6)/0.2)
        
        color = my_cmap[ind_color]

    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)
    
    ind_ctry = ind_ctry+1

# add colorbar.
my_cmap = ListedColormap(sns.color_palette('RdBu_r',20).as_hex()[2:18])
arr_nan = np.zeros([_env.nlat,_env.nlon])
arr_nan[:] = np.nan
cs = m.pcolormesh(lon,lat,arr_nan,cmap=my_cmap, vmin=-1.6, vmax=1.6) 

cbar = m.colorbar(cs,location='right',pad="5%",ticks=[-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4][::2])
cbar.set_label('(%)',fontsize = 14,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4][::2],size=14)

    
plt.title('Changes in GDP associated with sulfate-induced cooling',size=18)

set_latlon_ticks(ax,m)
plt.text(0.03,0.98, (chr(ord('a') + 3)),size=16, horizontalalignment='center',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')

plt.savefig(of_plot, dpi=300,bbox_inches='tight')    
plt.savefig(of_plot+'.eps', dpi=300,bbox_inches='tight') 
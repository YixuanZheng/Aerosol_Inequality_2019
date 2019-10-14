# -*- coding: utf-8 -*-
'''
This code generates Fig. S14

The probability that aerosol-induced cooling has resulted in economic benefits calculated based on different climatological temperature datasets.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import _env
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns
import matplotlib
import geopandas as gp
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


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

datasets = ['CESM','Reanalysis-1','ERA-Interim']
p_scen = 'No-Aerosol'

if_ctryshp = (_env.idir_root + '/shape/country/country1.shp')
odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)
of_plot = odir_plot + 'FS14.Map_Ctry_GDP_Lost_Possibility_3Datasets.png'

fig = plt.figure(figsize=(21,15))
    
for ids,ds in enumerate(datasets):
    
    if_gdp = _env.odir_root + '/summary_'+ds+'/country_specific_statistics_GDP_' + ds + '_' + p_scen + '_Burke.xls'
    if_ctrylist = _env.idir_root + '/regioncode/Country_List.xls'
    itbl_gdp = pd.read_excel(if_gdp,'country-lag0')
    itbl_gdp.set_index('iso',inplace = True)
    ishp_ctry = gp.read_file(if_ctryshp)
    
    ishp_ctry.loc[ishp_ctry['GMI_CNTRY'] == 'ROM','GMI_CNTRY'] = 'ROU'
    ishp_ctry.loc[ishp_ctry['GMI_CNTRY'] == 'ZAR','GMI_CNTRY'] = 'COD'
    ishp_ctry.set_index('GMI_CNTRY',inplace = True)
    
    ishp_ctry['prob_damg'] = 1-itbl_gdp['probability_damage']
    ishp_ctry.loc[pd.isna(ishp_ctry['prob_damg']),'prob_damg'] = -999
    _env.mkdirs(_env.odir_root+'gdp_map_' + ds )
    ishp_ctry.to_file(_env.odir_root+'gdp_map_' + ds + '/country_gdp_lost_spossibility_' + ds + '_' + p_scen + '.shp')
        
    #=========================MAP: percent of gdp changes from sulfate=====================#        
    ax = fig.add_subplot(311+ids) 
    
    m = Basemap(ellps = 'WGS84',
        llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
        suppress_ticks=True)#resolution='i',
    
    m.drawmapboundary()
    m.readshapefile(_env.odir_root+'gdp_map_' + ds + '/country_gdp_lost_spossibility_' + ds + '_' + p_scen,
            'country',drawbounds=True,linewidth=0.8,color='k',
            zorder=2) 
    
    my_cmap = sns.color_palette('RdBu_r',10).as_hex()#[2:18]
    ind_ctry = 0
    for info, shape in zip(m.country_info, m.country):
        gdp_r = info['prob_damg']
        
        if gdp_r == -999:
            color = '#D3D3D3' #'#000000' 
        else:
            ind_color = int(gdp_r*10)
            
            if ind_color >=10:
                ind_color=9
            color = my_cmap[ind_color]
    
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
        
        ind_ctry = ind_ctry+1
    
    plt.text(0.03,0.17, (chr(ord('a') + ids)) + ' ' + ds,size=16, horizontalalignment='left',#fontweight = 'bold',
                 verticalalignment='top',transform=ax.transAxes,fontweight='bold')

    set_latlon_ticks(ax,m)
    
cax = fig.add_axes([0.42, 0.09, 0.15, 0.017]) # posititon
cb = matplotlib.colorbar.ColorbarBase(cax,cmap = ListedColormap(my_cmap), orientation='horizontal')
cb.ax.tick_params(labelsize=14)
cb.set_label('Probability of economic benefits',size=16)
plt.savefig(of_plot, dpi=300,bbox_inches='tight')    
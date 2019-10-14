# -*- coding: utf-8 -*-
'''
This code generates Fig. S5

The probability that cooling associated with anthropogenic aerosols has resulted in economic benefits at the country-level. 

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


ds = 'ERA-Interim'

if_ctryshp = (_env.idir_root + '/shape/country/country1.shp')
odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)
of_plot = odir_plot + 'FS5.Map_Ctry_GDP_Lost_Possibility.png'

fig = plt.figure(figsize=(21,10))
    
for iscen,scen in enumerate(_env.scenarios[1::]):
    
    if_gdp = _env.odir_root + '/summary_'+ds+'/country_specific_statistics_GDP_' + ds + '_' + scen + '_Burke.xls'
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
    ishp_ctry.to_file(_env.odir_root+'gdp_map_' + ds + '/country_gdp_lost_spossibility_' + ds + '_' + scen + '.shp')
    #ishp_ctry.drop('geometry',axis=1).to_csv(_env.odir_root+'gdp_map/country_gdp_lost_spossibility.csv')
        
    #=========================MAP: percent of gdp changes from sulfate=====================#        
    ax = fig.add_subplot(211+iscen) 
    
    m = Basemap(ellps = 'WGS84',
        llcrnrlon=-180,llcrnrlat=-90, urcrnrlon=177.5,urcrnrlat=90.,
        suppress_ticks=True)#resolution='i',
    
    m.drawmapboundary()
    m.readshapefile(_env.odir_root+'gdp_map_' + ds + '/country_gdp_lost_spossibility_' + ds + '_' + scen,
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
    
    if iscen == 0:
        plt.text(0.03,0.17, (chr(ord('a') + iscen)) + ' All Aerosols',size=16, horizontalalignment='left',#fontweight = 'bold',
                 verticalalignment='top',transform=ax.transAxes,fontweight='bold')
    else:
        plt.text(0.03,0.17, (chr(ord('a') + iscen)) + ' Sulfate-Only',size=16, horizontalalignment='left',#fontweight = 'bold',
                 verticalalignment='top',transform=ax.transAxes,fontweight='bold')

    set_latlon_ticks(ax,m)
    
cax = fig.add_axes([0.42, 0.06, 0.2, 0.025]) # posititon
cb = matplotlib.colorbar.ColorbarBase(cax,cmap = ListedColormap(my_cmap), orientation='horizontal')
cb.ax.tick_params(labelsize=14)
cb.set_label('Probability of economic benefits',size=16)
plt.savefig(of_plot, dpi=300,bbox_inches='tight')    
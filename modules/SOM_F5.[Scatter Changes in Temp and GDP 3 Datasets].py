# -*- coding: utf-8 -*-
'''
This code generates Fig. S15

Country-level temperature changes and GDP increment induced by anthropogenic aerosols calculated based on different climatological temperature datasets.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import _env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn.apionly as sns
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

gdp_year = _env.year
sgdp_year = str(gdp_year)

p_scen = 'No-Aerosol' #aerosol removal scenario

odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)

of_plot = odir_plot + 'SOM_F5.Scatter_Changes_in_Temp_and_GDP.png'

datasets = ['CESM','Reanalysis-1','ERA-interim']

fig=plt.figure(figsize=(9,19.5))
my_cmap = ListedColormap(sns.color_palette('viridis',11).as_hex())  #[::-1]) #modified by yz mar21,2019, suggested by SJD
for ids, ds in enumerate(datasets):

    if_temp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_Temp_' + ds + '_' + p_scen + '.csv'
    if_gdp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_GDP_' + ds + '_' + p_scen + '_Burke.xls'

    itbl_temp = pd.read_csv(if_temp,index_col = 0)
    itbl_gdp = pd.read_excel(if_gdp,'country-lag0')
    
    
    mtbl_tg = itbl_temp[['Temp_mean_climatological', 'Temp_mean_noaero', 'Temp_Changes']].copy()
    for gc in ['iso', sgdp_year + '_gdp', sgdp_year + '_pop', 'GDP_median_benefit_ratio']:
        mtbl_tg[gc] = itbl_gdp[gc].copy()
    
    ax=plt.subplot(311+ids)
    
    ##global mean results
    g_pop = mtbl_tg['2010_pop'].sum()
    g_gdp = (mtbl_tg['2010_gdp'].sum())
    g_t_change = 0.9960902 
    g_gdp_change = 120668850000.0/g_gdp*100 
    g_gdp_size = 3*1e10 
    
    ##global mean results
    g_pop = mtbl_tg['2010_pop'].sum()
    g_gdp = (mtbl_tg['2010_gdp'].sum())
    inc_temp = Dataset(_env.odir_root + '/sim_temperature/'  + 'Simulated_Global_and_Country_TREFHT_20yravg.nc')
    g_t_change = np.mean(inc_temp['TREFHT_Global_PW'][:,1]-inc_temp['TREFHT_Global_PW'][:,0])
    g_gdp_change = ((mtbl_tg['GDP_median_benefit_ratio']*mtbl_tg['2010_gdp']).sum())/(mtbl_tg['2010_gdp'].sum())
    g_gdp_size = 3*1e10 
    
    plt.scatter(-g_t_change,g_gdp_change,c = 'black',vmin=-3,vmax=30,s=np.sqrt(g_gdp_size/1e7),alpha=0.8)
    
    pscat = plt.scatter(-mtbl_tg['Temp_Changes'],mtbl_tg['GDP_median_benefit_ratio'],c = mtbl_tg['Temp_mean_climatological'],vmin=-3,vmax=30,cmap=my_cmap,s=np.sqrt(mtbl_tg['2010_gdp']/1e7),
                        edgecolors='none',alpha=0.8,label=None) 
    
    #generate legend to indicate GDP size
    if ids == 0:
        gdp_labels = ['10$^9$','10$^{10}$','10$^{11}$','10$^{12}$','10$^{13}$']
        for igdp,gdp in enumerate([1e9, 1e10, 1e11,1e12,1e13]):
            plt.scatter([], [], c='',edgecolor='k', s=np.sqrt(gdp/1e7),
                        label=gdp_labels[igdp])
        plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title='GDP ($)',loc=6)
    
    
    ax.set_ylim(-2,2.2)
    ax.set_xlim(-1.8,-0.2)
    #set font size
    plt.xticks(size=14)
    plt.yticks(size=14)
    ax.set_xlabel('Aerosol-induced temperature changes (\N{DEGREE SIGN}C)',fontsize=16)
    ax.set_ylabel('GDP changes linked to aerosol-induced cooling (%)',fontsize=16) 
    
    cbar = plt.colorbar(pscat,ticks=[0,3,6,9,12,15,18,21,24,27])
    cbar.set_label('2010 temperature (\N{DEGREE SIGN}C)',fontsize = 16,rotation=270,labelpad=18)
    cbar.ax.set_yticklabels([0,3,6,9,12,15,18,21,24,27],size=14)
    plt.text(0.03,0.08, (chr(ord('a') + ids) + ' '+ds),size=16, horizontalalignment='left',#fontweight = 'bold',
                 verticalalignment='top',transform=ax.transAxes,fontweight='bold')

plt.savefig(of_plot, dpi=300,bbox_inches='tight')  
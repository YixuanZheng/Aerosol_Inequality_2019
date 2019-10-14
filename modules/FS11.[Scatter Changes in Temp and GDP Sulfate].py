# -*- coding: utf-8 -*-
'''
This code generates Fig. S11

Country-level temperature changes and the associated GDP increment induced by anthropogenic SULFATE aerosols

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

p_scen = 'No-Sulfate' #aerosol removal scenario
ds = 'ERA-Interim'

if_temp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_Temp_' + ds + '_' + p_scen + '.csv'
if_gdp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_GDP_' + ds + '_' + p_scen + '_Burke.xls'

odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)

of_plot = odir_plot + 'FS11.Scatter_Changes_in_Temp_and_GDP_Sulfate.png'

itbl_temp = pd.read_csv(if_temp,index_col = 0)
itbl_gdp = pd.read_excel(if_gdp,'country-lag0')

mtbl_tg = itbl_temp[['Temp_mean_climatological', 'Temp_mean_noaero', 'Temp_Changes']].copy()
for gc in ['iso', sgdp_year + '_gdp', sgdp_year + '_pop', 'GDP_median_benefit_ratio']:
    mtbl_tg[gc] = itbl_gdp[gc].copy()
 
    
##############################################################################
##########################Panel a###########################

fig=plt.figure(figsize=(9,13))
ax=plt.subplot(211)

my_cmap = ListedColormap(sns.color_palette('viridis',11).as_hex())  #[::-1]) #modified by yz mar21,2019, suggested by SJD


pscat = plt.scatter(-mtbl_tg['Temp_Changes'],mtbl_tg['GDP_median_benefit_ratio'],c = mtbl_tg['Temp_mean_climatological'],vmin=-3,vmax=30,cmap=my_cmap,s=np.sqrt(mtbl_tg['2010_gdp']/1e7),
                    edgecolors='none',alpha=0.8,label=None) 

##global mean results
g_pop = mtbl_tg['2010_pop'].sum()
g_gdp = (mtbl_tg['2010_gdp'].sum())

inc_temp = Dataset(_env.odir_root + '/sim_temperature/'  + 'Simulated_Global_and_Country_TREFHT_20yravg.nc')
g_t_change = np.mean(inc_temp['TREFHT_Global_PW'][:,2]-inc_temp['TREFHT_Global_PW'][:,0])
g_gdp_change = ((mtbl_tg['GDP_median_benefit_ratio']*mtbl_tg['2010_gdp']).sum())/(mtbl_tg['2010_gdp'].sum())
g_gdp_size = 3*1e10 

plt.scatter(-g_t_change,g_gdp_change,c = 'black',vmin=-3,vmax=30,s=np.sqrt(g_gdp_size/1e7),alpha=0.8)


#generate legend to indicate GDP size
gdp_labels = ['10$^9$','10$^{10}$','10$^{11}$','10$^{12}$','10$^{13}$']
for igdp,gdp in enumerate([1e9, 1e10, 1e11,1e12,1e13]):
    plt.scatter([], [], c='',edgecolor='k', s=np.sqrt(gdp/1e7),
                label=gdp_labels[igdp])
plt.legend(scatterpoints=1, frameon=False, labelspacing=0.5, title='GDP ($)',loc=6)

def plot_rec(llx,lly,urx,ury):
    plt.vlines(llx,lly,ury,colors='grey',linestyle = 'dashed',lw=1)
    plt.hlines(lly,llx,urx,colors='grey',linestyle = 'dashed',lw=1)

    plt.vlines(urx,lly,ury,colors='grey',linestyle = 'dashed',lw=1)
    plt.hlines(ury,llx,urx,colors='grey',linestyle = 'dashed',lw=1)

ollx = -1.2
olly = -0.5
ourx = -0.4
oury = 1.5

plot_rec(ollx,olly,ourx,oury)


ax.set_ylim(-2,2.2)
ax.set_xlim(-1.8,-0.2)
#set font size
plt.xticks(size=14)
plt.yticks(size=14)
ax.set_xlabel('Sulfate-induced temperature changes (\N{DEGREE SIGN}C)',fontsize=16)
ax.set_ylabel('GDP changes due to sulfate-induced cooling (%)',fontsize=16) 

cbar = plt.colorbar(pscat,ticks=[0,3,6,9,12,15,18,21,24,27])
cbar.set_label('2010 temperature (\N{DEGREE SIGN}C)',fontsize = 16,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([0,3,6,9,12,15,18,21,24,27],size=14)
plt.text(0.03,0.98, (chr(ord('a') + 0)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')
##############################################################################
##########################Panel b###########################
#
ax=plt.subplot(212)

pscat = plt.scatter(-mtbl_tg['Temp_Changes'],mtbl_tg['GDP_median_benefit_ratio'],c = mtbl_tg['Temp_mean_climatological'],vmin=-3,vmax=30,cmap=my_cmap,s=np.sqrt(mtbl_tg['2010_gdp']/1e7),
                    edgecolors='none',alpha=0.8)
plt.scatter(-g_t_change,g_gdp_change,c = 'black',vmin=-3,vmax=30,s=np.sqrt(g_gdp_size/1e7),alpha=0.8)

#set font size
plt.xticks(size=14)
plt.yticks(size=14)

ax.set_ylim([olly,oury])
ax.set_xlim([ollx,ourx])

ax.set_xlabel('Sulfate-induced temperature changes (\N{DEGREE SIGN}C)',fontsize=16)
ax.set_ylabel('GDP changes due to sulfate-induced cooling (%)',fontsize=16) 

cbar = plt.colorbar(pscat,ticks=[0,3,6,9,12,15,18,21,24,27])
cbar.set_label('2010 temperature (\N{DEGREE SIGN}C)',fontsize = 16,rotation=270,labelpad=18)
cbar.ax.set_yticklabels([0,3,6,9,12,15,18,21,24,27],size=14)
plt.text(0.03,0.98, (chr(ord('a') + 1)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')
plt.savefig(of_plot, dpi=300,bbox_inches='tight')  

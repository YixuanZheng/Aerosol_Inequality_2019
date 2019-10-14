# -*- coding: utf-8 -*-
'''
This code generates Fig. S8

GDP increment introduced by aerosol-induced cooling among the G20 nations.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn.apionly as sns

import _env

gdp_year = _env.year
sgdp_year = str(gdp_year)


p_scen = 'No-Aerosol' #aerosol removal scenario
ds = 'ERA-Interim'

if_temp = _env.odir_root + '/summary_'+ds+'/country_specific_statistics_Temp_'+ds+'_'+p_scen+'.csv'
if_gdp = _env.odir_root + '/summary_'+ds+'/country_specific_statistics_GDP_'+ds+'_'+p_scen+'_Burke.xls'
if_ctrylist = _env.idir_root + '/regioncode/Country_List.xls'
odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)

of_plot = odir_plot + 'FS8.Bar_GDP_Impacts_G20.png'

ctry_g20 = ['United States','Australia','Canada','Saudi Arabia','India','Russia','South Africa','Turkey','Argentina','Brazil','Mexico','France','Germany','Italy','United Kingdom','China','Indonesia','Japan','South Korea']

cmap_virdis = (sns.color_palette('viridis',11).as_hex())

itbl_temp = pd.read_csv(if_temp,index_col=0)
itbl_gdp = pd.read_excel(if_gdp,'country-lag0')
itbl_ctrylist = pd.read_excel(if_ctrylist)

itbl_ctrylist.set_index('ISO',inplace = True)

mtbl_tg = itbl_temp[['Temp_mean_climatological']].copy()
for gc in ['iso', sgdp_year + '_gdpcap',sgdp_year + '_gdp', sgdp_year + '_pop', 'GDP_median_benefit','GDP_median_benefit_ratio']:
    mtbl_tg[gc] = itbl_gdp[gc].copy()
 

mtbl_tg.set_index('iso',inplace = True)
mtbl_tg['Ctry_Name'] = itbl_ctrylist['NAME_ENGLI']

mtbl_tg.set_index(['Ctry_Name'],inplace=True)
mtbl_tg['Ctry_Name'] = mtbl_tg.index

mtbl_tg = mtbl_tg.loc[ctry_g20]

cmap_virdis_2 = np.array(cmap_virdis)
mtbl_tg['color'] = (cmap_virdis_2[((mtbl_tg['Temp_mean_climatological']/3.).astype(int)+1).values.tolist()]).tolist()

mtbls_4plot = {}

itbl_gdp_quan = mtbl_tg[['Ctry_Name','GDP_median_benefit','color']]
itbl_gdp_rat = mtbl_tg[['Ctry_Name','GDP_median_benefit_ratio','color']]



mtbls_4plot[1] = itbl_gdp_quan.sort_values('GDP_median_benefit',ascending=False)

mtbls_4plot[2] = itbl_gdp_rat.sort_values('GDP_median_benefit_ratio',ascending=False)

mtitle_4plot = ['GDP impacts','Percent GDP impacts'] #'Top 10 GDP costs',,'Top 10 percent GDP costs']

#sort by actual GDP benefits and ratio
fig = plt.figure(figsize=(8,8))

def to_str1(i):
    return '%.1f' % i 

def to_str2(i):
    return '%.2f' % i + ' %' 


fig = plt.figure(figsize=(10,10))
for subp in np.arange(1,3):
    ax = fig.add_subplot(120+subp)
    mtbl_4plot = mtbls_4plot[subp].copy()
    if subp == 1:
        mtbl_4plot['GDP_median_benefit'] = mtbl_4plot['GDP_median_benefit']/1e9
        mtbl_4plot['GDP_str'] = mtbl_4plot['GDP_median_benefit'].apply(to_str1)
    else:
        mtbl_4plot['GDP_str'] = mtbl_4plot['GDP_median_benefit_ratio'].apply(to_str2)
    
    mtbl_4plot.set_index('GDP_str',inplace=True)
    mtbl_4plot.index.name = None
    
    mtbl_4plot = mtbl_4plot.iloc[::-1]
    
    if subp == 1:
        p_gdp = mtbl_4plot['GDP_median_benefit'].plot(ax = ax,kind='barh', width = 0.8, colors= mtbl_4plot['color'], legend=False,alpha=0.8) #colors= , 
    else:
        mtbl_4plot['GDP_median_benefit_ratio'].plot(ax = ax,kind='barh', width = 0.8, colors= mtbl_4plot['color'], legend=False,alpha=0.8) #colors= , 
    
    title_text = mtitle_4plot[subp-1]
    
    if subp == 1:
        title_text = title_text + ' (billion US$)'
    else:
        title_text = title_text + ' (%)'
    #plt.xlabel(title_text,size=16)
    plt.title(title_text,size=16)
    ax.set_ylabel('')
    
    plt.xticks(size=14)
    #plt.yticks(size=14)
    plt.yticks([])
    
    print(mtbl_4plot)
    
    for rid in range(0,19):
        xlim = plt.xlim()[1]
        xlim_2 = plt.xlim()[0]
        plt.text(xlim/20.,rid,mtbl_4plot['Ctry_Name'].iloc[rid],alpha=1,rotation=0,
         horizontalalignment='left',
         verticalalignment='center',fontsize=14,
         multialignment='center')
        
        if rid > 0:
            plt.text(xlim_2*0.88,rid,mtbl_4plot.index[rid],alpha=1,rotation=0,
                     horizontalalignment='left',
                     verticalalignment='center',fontsize=14,
                     multialignment='center',color = '#21918c')
        else:
            plt.text(xlim_2*0.88,rid,mtbl_4plot.index[rid],alpha=1,rotation=0,
                     horizontalalignment='left',
                     verticalalignment='center',fontsize=14,
                     multialignment='center',color = '#5ec962')

#plot colorbar
ax_cb = fig.add_axes([0.92,0.25,0.03,0.5], frame_on=False)

colors_4cb = matplotlib.colors.ListedColormap(cmap_virdis_2)
norm = matplotlib.colors.BoundaryNorm(np.arange(-3,31,3), len(np.arange(-3,31,3)))
cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=colors_4cb, norm=norm,ticks=np.arange(0,30,3),alpha=0.8)
cb.set_label('2010 temperature (\N{DEGREE SIGN}C)',fontsize = 16,rotation=270,labelpad=18)
cb.ax.set_yticklabels([0,3,6,9,12,15,18,21,24,27],size=14)

plt.savefig(of_plot, dpi=300,bbox_inches='tight',pad=0.1)   


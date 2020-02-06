# -*- coding: utf-8 -*-

'''
This code generates Fig. S12

Country-level percentage changes in GDP associated with anthropogenic SULFATE-induced cooling

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import _env

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

gdp_year = _env.year
sgdp_year = str(gdp_year)


p_scen = 'No-Sulfate' #aerosol removal scenario
ds = 'ERA-Interim'

if_temp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_Temp_' + ds + '_' + p_scen + '.csv'
if_gdp = _env.odir_root + '/summary_' + ds + '/country_specific_statistics_GDP_' + ds + '_' + p_scen + '_Burke.xls'


if_ctrylist = _env.idir_root + '/regioncode/Country_List.xls'
odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)

of_plot = odir_plot + 'SOM_F2.Bar_GDP_Impacts_Errorbar_Sulfate.png'

itbl_gdp = pd.read_excel(if_gdp,'country-lag0')
itbl_ctrylist = pd.read_excel(if_ctrylist)

itbl_ctrylist.set_index('ISO',inplace = True)

mtbl_tg = itbl_gdp[['iso',  sgdp_year + '_gdpcap', sgdp_year + '_gdp', sgdp_year + '_pop', 'GDP_median_benefit_ratio','GDP_95_benefit','GDP_5_benefit']].copy()
    
mtbl_tg['GDP_95_benefit_ratio'] =  mtbl_tg['GDP_95_benefit']/mtbl_tg[sgdp_year + '_gdp']*100
mtbl_tg['GDP_5_benefit_ratio'] =  mtbl_tg['GDP_5_benefit']/mtbl_tg[sgdp_year + '_gdp']*100

mtbl_tg.set_index('iso',inplace = True)
mtbl_tg['Ctry_Name'] = itbl_ctrylist['NAME_ENGLI']

mtbl_tg.set_index(['Ctry_Name'],inplace=True)
mtbl_tg['Ctry_Name'] = mtbl_tg.index

mtbl_tg['gdp_share'] = mtbl_tg['2010_gdp']/mtbl_tg['2010_gdp'].sum()*100

colors = sns.color_palette('RdYlBu',5).as_hex()
colors[2] = '#A9A9A9'

gdp_bin = [5000,10000,20000,40000,200000]
for ig,gdpcap in enumerate(gdp_bin[::-1]):
    mtbl_tg.loc[mtbl_tg['2010_gdpcap'] < gdpcap,'color'] = colors[ig]


mtbl_tg.sort_values(['GDP_median_benefit_ratio'], inplace=True)

mtbl_tg['cum_gdp_share'] = mtbl_tg['gdp_share'].copy()

for i in np.arange(1,len(mtbl_tg)):
    mtbl_tg['cum_gdp_share'].iloc[i] =  mtbl_tg['cum_gdp_share'].iloc[i] + mtbl_tg['cum_gdp_share'].iloc[i-1]

bins = np.array([0] + mtbl_tg['cum_gdp_share'].values.tolist())
widths = bins[1:] - bins[:-1]
heights = mtbl_tg['GDP_median_benefit_ratio']

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
plt.bar(bins[:-1], heights, width= widths,color = mtbl_tg['color'],align='edge')

eb_x = bins[:-1] + widths/2
eb_y = heights

y_err1 = (mtbl_tg['GDP_95_benefit_ratio']-mtbl_tg['GDP_median_benefit_ratio']).values
y_err2 = (mtbl_tg['GDP_5_benefit_ratio']-mtbl_tg['GDP_median_benefit_ratio']).values

#First row contains the lower errors, the second row contains the upper errors. (positive value)
y_err = np.array([-y_err1,y_err2])

eb_ind = (np.where(mtbl_tg['gdp_share']>0.5))[0]
plt.errorbar(eb_x[eb_ind], eb_y[eb_ind], y_err[:,eb_ind], elinewidth=0.2, ls='none', c='Black',capsize=2,capthick=0.2)

plt.xlim([-0.1,100.1])
plt.ylim([-4,4.5])

plt.xticks(size=16) 
plt.yticks(size=16)
plt.ylabel('GDP changes due to sulfate-induced cooling (%)',size=16)
plt.xlabel('Cumulative share of global GDP (%)',size=16)

#plot colorbar
ax_cb = fig.add_axes([0.74,0.15,0.03,0.28], frame_on=False)

colors_4cb = matplotlib.colors.ListedColormap(colors[::-1])
norm = matplotlib.colors.BoundaryNorm([0]+gdp_bin, 1+colors_4cb.N)
cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=colors_4cb, norm=norm,ticks=[0,5000,10000,20000,40000],alpha=0.8)
cb.set_label('GDP per capita in 2010\n(2010 US$)',rotation = 270,size=14,labelpad=35)
cb.ax.tick_params(labelsize=14)

plt.savefig(of_plot, dpi=300,bbox_inches='tight',pad=0.1,alpha=0.4)  
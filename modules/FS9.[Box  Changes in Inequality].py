# -*- coding: utf-8 -*-

'''
This code generates Fig. S9

Impact of aerosol-induced cooling on country-level economic inequality.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   


import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#import rasterstats
import _env
import matplotlib
import seaborn.apionly as sns
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
###main

ds = 'ERA-Interim'
p_scen = 'No-Aerosol'
b_m = 'country-lag0'

gdp_year = _env.year

sgdp_year = str(gdp_year)
idir_temp = _env.odir_root + '/sim_temperature/'              

idir_gdp = _env.odir_root + '/gdp_'+ds+'/'        
odir_summary = _env.odir_root + '/summary_'+ds+'/'  
odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_summary)

of_plot = odir_plot + 'FS9.Box_Changes_in_Inequality.png'

itbl_gdp_baseline = pd.read_csv(_env.odir_root  + 'basic_stats' + '/Country_Basic_Stats.csv')
itbl_gdp_baseline.sort_values([sgdp_year + '_gdpcap'],inplace=True)
tot_pop = itbl_gdp_baseline[sgdp_year + '_pop'].sum()

itbl_gdp_baseline[sgdp_year + '_gdpsum'] = 0
itbl_gdp_baseline[sgdp_year + '_popsum'] = 0

for irow, row in enumerate(itbl_gdp_baseline.index):
    if irow == 0:
        itbl_gdp_baseline.loc[row,sgdp_year + '_gdpsum'] = itbl_gdp_baseline.loc[row,sgdp_year + '_gdp']
        itbl_gdp_baseline.loc[row, sgdp_year + '_popsum'] = itbl_gdp_baseline.loc[row,sgdp_year + '_pop']
    else:
        itbl_gdp_baseline.loc[row,sgdp_year + '_gdpsum'] = itbl_gdp_baseline[sgdp_year + '_gdpsum'].iloc[irow-1] + itbl_gdp_baseline.loc[row,sgdp_year + '_gdp']
        itbl_gdp_baseline.loc[row, sgdp_year + '_popsum'] = itbl_gdp_baseline[sgdp_year + '_popsum'].iloc[irow-1] + itbl_gdp_baseline.loc[row,sgdp_year + '_pop'] 

itbl_gdp_baseline[sgdp_year + '_pop_ratio_sum'] = itbl_gdp_baseline[sgdp_year + '_popsum']/tot_pop

#deciles (<=10% and >=90%)

deciles = {}

ind10 = np.where(itbl_gdp_baseline[sgdp_year + '_pop_ratio_sum']<=0.1)[0]
deciles[10] =  itbl_gdp_baseline.iloc[ind10].copy()


ind90 = np.where(itbl_gdp_baseline[sgdp_year + '_pop_ratio_sum']>=0.9)[0]
deciles[90] = itbl_gdp_baseline.iloc[ind90].copy()

#quintiles  (<=20% and >=80%)

ind20 = np.where(itbl_gdp_baseline[sgdp_year + '_pop_ratio_sum']<=0.2)[0]
deciles[20] = itbl_gdp_baseline.iloc[ind20].copy()

ind80 = np.where(itbl_gdp_baseline[sgdp_year + '_pop_ratio_sum']>=0.8)[0]
deciles[80] = itbl_gdp_baseline.iloc[ind80].copy()


inc_gdp = Dataset(idir_gdp + 'GDP_Changes_Burke_' + b_m + '_' + str(gdp_year) +  '_' + ds + '_' + p_scen + '.nc')
imtrx_gdp = inc_gdp['GDP'][:]

dec_var = {}
dec_base = {}

for perc in [10,20,80,90]:
    dec = deciles[perc].copy()
    dec_pop_tot = dec[sgdp_year + '_pop'].sum()
    dec_gdp_tot = dec[sgdp_year + '_gdp'].sum()
    dec_base[perc] = dec_gdp_tot/dec_pop_tot
    ind_ctry = dec.index
    imtrx_dec = imtrx_gdp[:,ind_ctry,:]
    imtrx_dec_sum = dec_gdp_tot-(imtrx_dec.data).sum(axis=1) #+ dec_gdp_tot
    
    dec_gdpcap = imtrx_dec_sum/dec_pop_tot
    dec_var[perc] = dec_gdpcap.copy()


dec_diff = (dec_var[90]/dec_var[10]-dec_base[90]/dec_base[10])/(dec_base[90]/dec_base[10])*100
quin_diff = (dec_var[80]/dec_var[20] - dec_base[80]/dec_base[20])/(dec_base[80]/dec_base[20])*100

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
itbl_samples = pd.DataFrame((np.array([np.reshape(dec_diff,8008),np.reshape(quin_diff,8008)])).T,index = np.arange(0,8008), columns = ['90:10\nratio','80:20\nratio'])

bp = itbl_samples.plot(ax = ax, kind='box', color={'medians': sns.xkcd_rgb['windows blue'],'boxes':
sns.xkcd_rgb['windows blue'],'whiskers':sns.xkcd_rgb['greyish'],'caps':sns.xkcd_rgb['greyish']},
boxprops = {'linewidth': 2.5}, whiskerprops = {'linestyle': '--','linewidth': 2.5}, capprops = {'linewidth': 2.5},        
     medianprops={'linestyle': '-', 'linewidth': 2.5},showfliers=False,whis=[5, 95])    
    
plt.scatter([1,2],[np.percentile(itbl_samples.iloc[:,0],1),np.percentile(itbl_samples.iloc[:,1],1)],color = sns.xkcd_rgb['pale red'],s=60,
        alpha=0.6,edgecolors='none',label= '1%',marker = 'o')
    
plt.scatter([1,2],[np.percentile(itbl_samples.iloc[:,0],99),np.percentile(itbl_samples.iloc[:,1],99)],color = sns.xkcd_rgb['pale red'],s=60,
        alpha=0.6,edgecolors='none',label= '99%',marker = 'o')

#plot annotation 
for pctl in [1,5,25,50,75,95,99]:
    plt.text(1.15,np.percentile(itbl_samples['90:10\nratio'],100-pctl),str(pctl) + '%',size=14,horizontalalignment='left',verticalalignment='center',transform=ax.transData)

plt.xticks(size=14)
plt.yticks(size=14)

plt.ylim([-2,0])

plt.ylabel('Percent change in ratio of population-\nweighted percentiles of GDP per capita',size=14)
           

plt.savefig(of_plot, dpi=300,bbox_inches='tight')    

# -*- coding: utf-8 -*-
"""
This code generates Fig. S1

Trend of global mean surface temperature in different forcing scenarios

by Yixuan Zheng (yxzheng@carnegiescience.edu)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import geopandas as gp
import _env
import seaborn.apionly as sns

import matplotlib
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.stattools import acf
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'


#import matplotlib
nens = _env.nens
parameters_info = _env.parameters_info
scenarios = _env.scenarios
par = 'TREFHT'



if_temp = _env.odir_root + '/' + parameters_info[par]['dir']+ '/Global_Mean_' + par + '_1850-2019_ensembles' + '.xls'
if_temp_pi = _env.odir_root + '/' + parameters_info[par]['dir']+ '/Global_Mean_Temperature_pre-industrial_110yrs.xls'

odir_plot = _env.odir_root + '/plot/' 
_env.mkdirs(odir_plot)

of_plot = odir_plot + 'ED_F1.Trend_Temp_forcing_scenarios.png'



itbl_temp = pd.read_excel(if_temp,index_col=0)
itbl_temp_pi = pd.read_excel(if_temp_pi,index_col=0)

itbl_temp = itbl_temp - float(itbl_temp_pi.mean()) #itbl_temp.iloc[0]#float(itbl_temp_pi.mean()) #273.15

itbl_temp_all = {}
itbl_temp_20_avg_all = {}
itbl_temp_20_ste_all = {}
for scen in scenarios[0:2]:
    scen_sulf = scenarios[2]
    scen_base = scen
    
    #add results for differences in temeprature between two scenarios
    for ens in np.arange(1,nens+1):
        if scen == scenarios[1]:
            itbl_temp['Diff%d' % ens] = -itbl_temp[scen_base + '%d' % ens] + itbl_temp[scen_sulf + '%d' % ens]
        else:
            itbl_temp['Diff%d' % ens] = itbl_temp[scen_base + '%d' % ens] - itbl_temp[scen_sulf + '%d' % ens]
    
    
    #statistics
    itbls_temp_20_sam = {}
    itbl_temp_20_avg = pd.DataFrame(index=itbl_temp.index,columns=[scen_base,scen_sulf,'Diff']) 
    itbl_temp_20_ste = pd.DataFrame(index=itbl_temp.index,columns=[scen_base,scen_sulf,'Diff'])         
    
    for year in np.arange(1850,2001):
        ens_names = {}
        for cols in [scen_base,scen_sulf,'Diff']:
            
            if not cols in itbls_temp_20_sam:
                itbls_temp_20_sam[cols] = {}
            
            ens_name = [cols+'%d' % ens for ens in np.arange(1,nens+1)]
            ens_names[cols] = ens_name.copy()
             
            ylist = np.arange(year,year+20)
            arr_sam1 = itbl_temp.loc[ylist[ylist < 1950],ens_name[0]].values.tolist()
            arr_sam2 = itbl_temp.loc[ylist[ylist >= 1950],ens_name].values
            arr_sam2 = (np.reshape(arr_sam2,np.shape(arr_sam2)[0]*np.shape(arr_sam2)[1])).tolist()
            
            itbls_temp_20_sam[cols][year+10] = arr_sam1 + arr_sam2
            
            itbl_temp_20_avg.loc[year,cols] = np.mean(itbls_temp_20_sam[cols][year+10])
            #calculate effective sample size to account for autocorrelation
            adf = ADF(itbls_temp_20_sam[cols][year+10])
            
            if (acf(itbls_temp_20_sam[cols][year+10])[1]>0):
                cf = (1-acf(itbls_temp_20_sam[cols][year+10])[1])/(1+acf(itbls_temp_20_sam[cols][year+10])[1])
            else:
                cf = 1
            itbl_temp_20_ste.loc[year,cols] = stats.sem(itbls_temp_20_sam[cols][year+10])/np.sqrt(cf) 
        
        
    itbl_temp_20_avg.drop(np.arange(2001,2020),inplace=True)
    itbl_temp_20_avg.index = itbl_temp_20_avg.index+10
    
    itbl_temp_20_ste.drop(np.arange(2001,2020),inplace=True)
    itbl_temp_20_ste.index = itbl_temp_20_ste.index+10


    itbl_temp_all[scen] = itbl_temp.copy()
    itbl_temp_20_avg_all[scen] = itbl_temp_20_avg.copy()
    itbl_temp_20_ste_all[scen] = itbl_temp_20_ste.copy()
    
    
    
fig = plt.figure(figsize=(14,6.5))

colors = {scenarios[0]:sns.color_palette("Blues",10),
          scenarios[1]:sns.color_palette("Reds",10),
          scenarios[2]:sns.color_palette("Greys",10),
          'Diff_' + scenarios[0]:sns.color_palette("PuRd",10),
          'Diff_' + scenarios[1]:sns.color_palette("Purples",10)}


for iscen, bscen in enumerate(scenarios[0:2]):
    ax = fig.add_subplot(1,2,iscen + 1)
    
    plt.hlines(0,1850,2020,lw=1)
    
    for scen in [bscen,scenarios[2],'Diff']:
        if scen == 'Diff':
            cp = colors[scen + '_' + bscen]
        else:
            cp = colors[scen ]
            
        for ens in np.arange(1,nens+1):
            
            plt.scatter(np.arange(1950,2020),itbl_temp_all[bscen].loc[np.arange(1950,2020),scen+'%d' % ens],marker='o',c=[cp[6]],s=4,alpha = 0.7)
            
        plt.fill_between(itbl_temp_20_avg_all[bscen][scen].index.tolist(), (itbl_temp_20_avg_all[bscen][scen]-itbl_temp_20_ste_all[bscen][scen]).values.tolist(), 
                         (itbl_temp_20_avg_all[bscen][scen]+itbl_temp_20_ste_all[bscen][scen]).values.tolist(),color=[cp[5]],alpha=0.6)
        
        itbl_temp_20_avg_all[bscen][scen].plot(kind='Line',color=[cp[8]],lw=2,alpha=0.8)   
        plt.scatter(np.arange(1850,1950),itbl_temp_all[bscen].loc[np.arange(1850,1950),scen+'1'],marker='o',c=[cp[6]],s=2,alpha = 0.7)
    
    plt.ylim([-1.2,2])
    plt.xlabel('Years',size=16)
    #plt.ylabel('Temperature changes (\N{DEGREE SIGN}C)',size=16)
    plt.ylabel('Temperature anomaly (\N{DEGREE SIGN}C)',size=16)
    #plt.xticks(np.arange(1950,2020,10),size=14)
    plt.xlim([1850,2020])
    plt.xticks(np.arange(1850,2020,20),size=14)
    plt.yticks(size=14)
    
    plt.text(0.04,0.98, (chr(ord('a') + iscen)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')

plt.savefig(of_plot, dpi=300,bbox_inches='tight')  



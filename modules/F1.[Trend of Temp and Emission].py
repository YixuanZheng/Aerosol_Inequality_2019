# -*- coding: utf-8 -*-

'''
This code generates Fig. 1

Trend of global mean surface temperature and anthropogenic aerosol emissions

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _env
import seaborn.apionly as sns

import matplotlib
from scipy import stats
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.stattools import acf
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

#import matplotlib
nens = _env.nens
parameters_info = _env.parameters_info
scenarios = _env.scenarios
par = 'TREFHT'

scen_aero = scenarios[1]
scen_base = scenarios[0]

if_temp = _env.odir_root + '/' + parameters_info[par]['dir']+ '/Global_Mean_' + par + '_1850-2019_ensembles' + '.xls'
if_temp_pi = _env.odir_root + '/' + parameters_info[par]['dir']+ '/Global_Mean_Temperature_pre-industrial_110yrs.xls'

odir_plot = _env.odir_root + '/plot/' 
_env.mkdirs(odir_plot)

of_plot = odir_plot + 'F1.Trend_Temp_Emission.png'

itbl_temp = pd.read_excel(if_temp,index_col=0)
itbl_temp_pi = pd.read_excel(if_temp_pi,index_col=0)

itbl_temp = itbl_temp - float(itbl_temp_pi.mean()) #itbl_temp.iloc[0]#float(itbl_temp_pi.mean()) #273.15

#add results for differences in temeprature between two scenarios
for ens in np.arange(1,nens+1):
    itbl_temp['Diff%d' % ens] = itbl_temp[scen_base + '%d' % ens] - itbl_temp[scen_aero + '%d' % ens]

#statistics
itbls_temp_20_sam = {}
itbl_temp_20_avg = pd.DataFrame(index=itbl_temp.index,columns=[scen_base,scen_aero,'Diff']) 
itbl_temp_20_ste = pd.DataFrame(index=itbl_temp.index,columns=[scen_base,scen_aero,'Diff']) 

for year in np.arange(1850,2001):
    
    ens_names = {}
    for cols in [scen_base,scen_aero,'Diff']:
        
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


fig = plt.figure(figsize=(6.5,14))
ax = fig.add_subplot(2,1,1)
plt.hlines(0,1850,2020,lw=1)

colors = {scen_base:sns.color_palette("Blues",10),
          scen_aero:sns.color_palette("Reds",10),
          'Diff':sns.color_palette("Purples",10)}

for scen in [scen_base,scen_aero,'Diff']:
    cp = colors[scen]
        
    for ens in np.arange(1,nens+1):
        plt.scatter(np.arange(1950,2020),itbl_temp.loc[np.arange(1950,2020),scen+'%d' % ens],marker='o',c=[cp[6]],s=4,alpha = 0.7)
        
        
    plt.fill_between(itbl_temp_20_avg.index.tolist(), (itbl_temp_20_avg[scen]-itbl_temp_20_ste[scen]).values.tolist(), 
                     (itbl_temp_20_avg[scen]+itbl_temp_20_ste[scen]).values.tolist(),color=[cp[5]],alpha=0.6)
    
    itbl_temp_20_avg[scen].plot(kind='Line',color=[cp[8]],lw=2,alpha=0.8)   
    
    plt.scatter(np.arange(1850,1950),itbl_temp.loc[np.arange(1850,1950),scen+'1'],marker='o',c=[cp[6]],s=2,alpha = 0.7)



plt.ylim([-1.2,2])
plt.xlabel('Years',size=16)
plt.ylabel('Temperature anomaly (\N{DEGREE SIGN}C)',size=16)
plt.xlim([1850,2020])
plt.xticks(np.arange(1850,2020,20),size=14)
plt.yticks(size=14)

plt.text(0.04,0.98, (chr(ord('a') + 0)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax.transAxes,fontweight='bold')

#============================plot emissions===================================
ax2 = fig.add_subplot(2,1,2)
if_emis = _env.idir_root + 'emis/AERO_Emission_His-RCP8.5.xlsx'
itbl_emis = pd.read_excel(if_emis,index_col=0)

itbl_emis.loc[1850:2020,'SO2'].plot(kind='Line',color = 'Yellow',lw=4,alpha=0.6,ax=ax2,legend=False)
plt.ylim([0,130])
plt.xlim([1850,2020])

plt.xlabel('Years',size=16)
plt.ylabel('Global SO$_{2}$ emissions (Tg)',size=16)
plt.xticks(np.arange(1850,2020,20),size=14)
plt.yticks(size=14)

ax3 = ax2.twinx()
itbl_emis.loc[1850:2020,['BC','OC']].plot(kind='Line',colors = ['Brown','Darkorange'],lw=4,alpha=0.6,ax=ax3,legend=False)
plt.ylim([0,16])
plt.yticks(size=14)
plt.ylabel('Global carbonaceous aerosol emissions (Tg)',size=16,rotation=270,labelpad=20)

plt.text(0.04,0.98, (chr(ord('a') + 1)),size=16, horizontalalignment='center',#fontweight = 'bold',
             verticalalignment='top',transform=ax2.transAxes,fontweight='bold')

plt.savefig(of_plot, dpi=300,bbox_inches='tight')  



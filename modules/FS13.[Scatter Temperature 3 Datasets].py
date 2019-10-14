# -*- coding: utf-8 -*-

'''
This code generates Fig. S13

Comparison of climatological surface air temperature (in °C) at the country-level across different datasets.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   
import _env
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


odir_plot = _env.odir_root + '/plot/'
_env.mkdirs(odir_plot)
of_plot = odir_plot + 'FS13.Scatter_compare_Temp_3datasets.png'

if_temp = _env.odir_root + 'sim_temperature/Climatological_Temp_Ctry_3ds.csv'
itbl_temp = pd.read_csv(if_temp,index_col=0)

#scatter plot:CESM simution vs reanalysis data over countries
fig = plt.figure(figsize=(16,4))

combs = [['CESM','Reanalysis-1'],
         ['CESM','ERA-Interim'],
         ['Reanalysis-1','ERA-Interim']]



for ic, comb in enumerate(combs):
    print(comb)

    ax = fig.add_subplot(131+ic,aspect='equal')
    
    itbl_2cols = itbl_temp[comb]
    
    temp_x = itbl_2cols[comb[0]]
    temp_y = itbl_2cols[comb[1]]
    
    f_means = plt.scatter(x=temp_x, y=temp_y,alpha = 0.6)
    
    z = np.polyfit(temp_x, temp_y, 1)
    r = np.corrcoef(temp_x, temp_y)[0, 1]
    nmb = 100*np.sum(np.sum(temp_y-temp_x))/np.sum(np.sum(temp_x))
    rmse = np.sqrt(np.mean((temp_y-temp_x)**2))

    if ic <=1:    
        eq = " Y = %.2f*X - %.2f "%(z[0],np.abs(z[1])) + '\n R = %.2f' %r + '\n NMB = %.2f' % nmb + '%'
    else:
        eq = " Y = %.2f*X + %.2f"%(z[0],np.abs(z[1])) + '\n R = %.2f' %r + '\n NMB = %.2f' % nmb + '%'
    
    plt.text(0.04, 0.95,eq, fontweight='bold',horizontalalignment='left',
     verticalalignment='top',transform=ax.transAxes,size = 12)
    
    plt.text(0.9, 0.1,  (chr(ord('a') + ic))  , fontweight='bold',horizontalalignment='left',
     verticalalignment='top',transform=ax.transAxes,size = 16)
    
    
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel(comb[0],size=16)
    plt.ylabel(comb[1],size=16,labelpad=-1)
    
    plt.xlim([-15,35])
    plt.ylim([-15,35])
    
    x = np.linspace(-15,35,2)
    y = x
    plt.plot(x, y, '-k',alpha=0.6)


plt.savefig(of_plot, dpi=300,bbox_inches='tight')   



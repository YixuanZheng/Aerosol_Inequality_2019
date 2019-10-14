# -*- coding: utf-8 -*-

'''
This code calculates changes in the ratio between different population-weighted GDP deciles and quintiles

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import pandas as pd
import numpy as np
from netCDF4 import Dataset
import _env
    
datasets = _env.datasets   
scenarios = _env.scenarios
  
gdp_year = 2010
sgdp_year = str(gdp_year)
idir_temp = _env.odir_root + '/sim_temperature/'      

####summarize global and regional GDP changes####
gdp_year = 2010
sgdp_year = str(gdp_year)
boot_methods = ['country-lag0','country-lag1','country-lag5','year','year-blocks']
itbl_gdp_baseline = pd.read_csv(_env.odir_root  + 'basic_stats' + '/Country_Basic_Stats.csv')

itbl_gdp_baseline.sort_values([sgdp_year + '_gdpcap'],inplace=True)
tot_pop = itbl_gdp_baseline[sgdp_year + '_pop'].sum()
#itbl_gdp_baseline['2010_pop_ratio'] = itbl_gdp_baseline['2010_pop']/tot_pop

itbl_gdp_baseline[sgdp_year + '_gdpsum'] = 0
#itbl_gdp_baseline['2010_popw_gdp'] = 0
itbl_gdp_baseline[sgdp_year + '_popsum'] = 0
#itbl_gdp_baseline['2010_pop_ratio_sum'] = 0

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


for ds in datasets:
    
    scens = ['No-Aerosol']
    if ds == 'ERA-Interim':
        scens = ['No-Aerosol','No-Sulfate']
    
    idir_gdp = _env.odir_root + '/gdp_' + ds + '/'        
    odir_summary = _env.odir_root + '/summary_' + ds + '/'  
    _env.mkdirs(odir_summary)
    
    for scen in scens: 
        writer = pd.ExcelWriter(odir_summary + 'Deciles_and_Quintile_ratio_changes_'+ds+'_'+scen+'_Burke.xls')
        otbls_ctry_GDP_stat = {}
        
        otbls = {}
        otbl_ineq = pd.DataFrame(index = boot_methods,columns = ['median_ratio','5_ratio','95_ratio','10_ratio','90_ratio','probability_reduced'])
        
        otbls['deciles'] = otbl_ineq.copy()
        otbls['quintiles'] = otbl_ineq.copy()
        
        for b_m in boot_methods:
            
            inc_gdp = Dataset(idir_gdp + 'GDP_Changes_Burke_' + b_m + '_' + str(gdp_year) +  '_'+ds+'_'+scen+'.nc')
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
                imtrx_dec_sum = dec_gdp_tot-(imtrx_dec.data).sum(axis=1) 
#                print(perc, np.median(imtrx_dec_sum),dec_gdp_tot,np.median(imtrx_dec_sum)/dec_gdp_tot)
                
                dec_gdpcap = imtrx_dec_sum/dec_pop_tot
                dec_var[perc] = dec_gdpcap.copy()
            
            dec_diff = (dec_var[90]/dec_var[10]-dec_base[90]/dec_base[10])/(dec_base[90]/dec_base[10])*100
            quin_diff = (dec_var[80]/dec_var[20] - dec_base[80]/dec_base[20])/(dec_base[80]/dec_base[20])*100
            
            
            
            otbls['deciles'].loc[b_m,'median_ratio'] = np.median(dec_diff)
            otbls['deciles'].loc[b_m,'5_ratio'] = np.percentile(dec_diff,5)
            otbls['deciles'].loc[b_m,'95_ratio'] = np.percentile(dec_diff,95)
            
            otbls['deciles'].loc[b_m,'10_ratio'] = np.percentile(dec_diff,10)
            otbls['deciles'].loc[b_m,'90_ratio'] = np.percentile(dec_diff,90)
            otbls['deciles'].loc[b_m,'probability_reduced'] = len(dec_diff[dec_diff<0])/np.size(dec_diff)
            
            otbls['quintiles'].loc[b_m,'median_ratio'] = np.median(quin_diff)
            otbls['quintiles'].loc[b_m,'5_ratio'] = np.percentile(quin_diff,5)
            otbls['quintiles'].loc[b_m,'95_ratio'] = np.percentile(quin_diff,95)
            
            otbls['quintiles'].loc[b_m,'10_ratio'] = np.percentile(quin_diff,10)
            otbls['quintiles'].loc[b_m,'90_ratio'] = np.percentile(quin_diff,90)
            otbls['quintiles'].loc[b_m,'probability_reduced'] = len(quin_diff[quin_diff<0])/np.size(quin_diff)
            
        otbls['deciles'].to_excel(writer,'deciles')
        otbls['quintiles'].to_excel(writer,'quintiles')
        
        writer.save()



    


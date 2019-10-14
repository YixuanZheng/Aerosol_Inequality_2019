# -*- coding: utf-8 -*-

'''
This code calculates impacts of temperature changes induced by aerosols on GDP

positive results mean benefits from aerosol-induced cooling

damage function developed in Burke, Matthew Davis, and Diffenbaugh (2018) 

assume results of each year is independent for the CESM repeating cycle simulations

climatological temeprature obtained from three datasets (ERA-Interim, CESM, Reanalysis-1)

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import _env
import datetime
from scipy import stats

nens  = _env.nens
datasets = _env.datasets

year = _env.year
syr = str(year)
gdp_year = year
sgdp_year = str(gdp_year)

par = 'TREFHT' 

if_temp = _env.odir_root + '/sim_temperature/Simulated_Global_and_Country_' + par + '_20yravg.nc'
if_boot_par = _env.idir_root + 'Burke2018_Replication/data/output/bootstrap.csv' #bootstrapped parameters from Burke et al. 2018
if_ctry_list = _env.idir_root + '/regioncode/Country_List.xls'
if_ctry_pr = _env.idir_root + '/historical_stat/Ctry_Poor_Rich_from_Burke.csv' #adopt country list from Burke et al. 2018

#climatological temperature from three datasets
if_clim_temp = _env.odir_root + 'sim_temperature/Climatological_Temp_Ctry_3ds.csv'

#bootstrap methods
itbl_boots_par = pd.read_csv(if_boot_par,index_col = 0)
boot_methods = (pd.unique(itbl_boots_par['spec'])).tolist()  #['country-lag0','country-lag1','country-lag5','year','year-blocks']


#full country list
itbl_ctry_list = pd.read_excel(if_ctry_list,index_col = 0)
#list of countries analyzed in this study (adopted from Burke et al. 2018)
itbl_ctry = pd.read_csv(if_ctry_pr,index_col = 0)
#map index of analyzed countries in full country list 
itbl_ctry['ind_in_full_list'] = (np.zeros(np.shape(itbl_ctry)[0])-1)
for ictry,ctry in enumerate(itbl_ctry['iso']):
    itbl_ctry.loc[ictry+1,'ind_in_full_list'] = (itbl_ctry_list['ISO'].tolist()).index(ctry)

#gdp/cap and pop from WB dataset
#constant 2010 us$
if_ctry_gdpcap = _env.idir_root + '/historical_stat/API_NY.GDP.PCAP.KD_DS2_en_csv_v2.csv'
if_ctry_pop = _env.idir_root + '/historical_stat/API_SP.POP.TOTL_DS2_en_csv_v2.csv'
itbl_ctry_gdpcap = pd.read_csv(if_ctry_gdpcap,skiprows=4)
itbl_ctry_pop = pd.read_csv(if_ctry_pop,skiprows=4)

#keep 167 countries analyszed in Burke et al. 2018
itbl_ctry_gdpcap.set_index(['Country Code'],inplace=True)
itbl_ctry_pop.set_index(['Country Code'],inplace=True)

itbl_ctry_info = itbl_ctry.copy()
itbl_ctry_info.set_index(['iso'],inplace = True)

for year in range(gdp_year,gdp_year+1):
    itbl_ctry_info[str(year)+'_gdpcap'] = itbl_ctry_gdpcap[str(year)]
    itbl_ctry_info[str(year)+'_pop'] = itbl_ctry_pop[str(year)]
    itbl_ctry_info[str(year)+'_gdp'] = itbl_ctry_info[str(year)+'_gdpcap']*itbl_ctry_info[str(year)+'_pop']

itbl_ctry_info.drop(['meantemp','gdp','growth','gdpcap','pop'],axis=1, inplace=True)
_env.mkdirs(_env.odir_root + '/basic_stats/')
itbl_ctry_info.to_csv(_env.odir_root + '/basic_stats/' + 'Country_Basic_Stats.csv')

omtrxs_gdp_all = {}

#calculate changes in gdp and gdp growth rate from aerosol-induced cooling
for ds in datasets:
    omtrxs_gdp_all[ds] = {}
    
    itbl_clim_temp = pd.read_csv(if_clim_temp,index_col = 0)[['iso',ds]]

    odir_gdp = _env.odir_root + '/gdp_' + ds + '/'
    _env.mkdirs(odir_gdp)
    
    scenarios = _env.scenarios[0:2] #['With-Aerosol', 'No-Aerosol']
    
    if ds == 'ERA-Interim':
        scenarios = _env.scenarios #['With-Aerosol', 'No-Aerosol', 'No-Sulfate']
    
    for iscen,scen in enumerate(scenarios[1::]):
    
        #read global and country-level temperature
        T_glob = Dataset(if_temp)['TREFHT_Global'][:,[0,iscen+1]]
        T_ctry_full = Dataset(if_temp)['TREFHT_Country'][:,:,[0,iscen+1]]
        
        #extract temperature for analyzed countries
        T_ctry = T_ctry_full[((itbl_ctry['ind_in_full_list'].astype(int)).tolist()),:,:]
        
        if not (ds == 'CESM'):
        #apply changes in temperature derived from CESM simulations to the reanalysis-based climatological data
            T_diff = T_ctry[:,:,1]-T_ctry[:,:,0]
            T_ctry = T_ctry.copy()
            T_ctry[:,:,0] = np.repeat(np.array(itbl_clim_temp[ds].values)[:,np.newaxis],8,axis=1)
            T_ctry[:,:,1] = T_ctry[:,:,0] + T_diff
        
        imtrx_ctry_gdp = np.repeat((np.array(itbl_ctry_info[str(gdp_year) + '_gdp'].values)[:,np.newaxis]),nens,axis=1)
        
        ####country-level changes in GDP/cap growth rate####
        omtrxs_dratio = {}
        omtrxs_dgdp = {}
        
        for b_m in boot_methods: #loop of bootstarpping methods 
            print(ds,scen,b_m)
            mtbl_boot_par = itbl_boots_par[itbl_boots_par['spec'] == b_m]
        
            n_boot_sample = len(mtbl_boot_par.index)
            n_ctry = len(itbl_ctry.index)
            
            gr = np.einsum('l,ijk->lijk',mtbl_boot_par['b1'],T_ctry) + np.einsum('l,ijk->lijk',mtbl_boot_par['b2'],T_ctry**2)
            diff_gr = gr[:,:,:,1] - gr[:,:,:,0]
            diff_gdp = np.einsum('lij,ij->lij',diff_gr,imtrx_ctry_gdp)
            
            omtrxs_dratio[b_m] = diff_gr.copy() #omtrx_ratio_spec.copy()
            omtrxs_dgdp[b_m] = diff_gdp.copy() #omtrx_gdp_spec.copy()
        
            _env.rmfile(odir_gdp + 'GDP_Changes_Burke_' + b_m + '_' + str(gdp_year) +  '_' + ds + '_' + scen +'.nc')
            onc = Dataset(odir_gdp + 'GDP_Changes_Burke_' + b_m + '_' + str(gdp_year) +  '_' + ds + '_' + scen +'.nc', 'w', format='NETCDF4')
            
            d_ctry = onc.createDimension('boots',n_boot_sample)
            d_ctry = onc.createDimension('countries',n_ctry)
            d_ens = onc.createDimension('ensembles',nens)
            
            v_ratio = onc.createVariable('GDP_Ratio','f4',('boots','countries','ensembles'))
            v_ratio.desc = 'Impacts of aerosol-induced cooling on annual GDP growth rate'
            v_ratio[:] = diff_gr
            
            v_gdp = onc.createVariable('GDP','f4',('boots','countries','ensembles'))
            v_gdp.desc = 'Impacts of aerosol-induced cooling on country-level annual GDP'
            v_gdp[:] = diff_gdp.copy()
            
            #write global attribute
            onc.by = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
            onc.desc = 'Impacts of aerosol-induced cooling on annual GDP and GDP growth rate (based on damage functions developed by Burke et al. 2018)'
            onc.creattime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            onc.close()
        
        omtrxs_gdp_all[ds][scen] = omtrxs_dgdp.copy()
        
        
        ####summarize global and regional temperature changes####
        if ds == 'ERA-Interim':
            print('T-test of global temperature changes (between ' + scenarios[1] + ' and ' + scenarios[0] +')', stats.ttest_ind(T_glob[:,1],T_glob[:,0]))
            print('Changes in global mean temperature (between ' + scenarios[1] + ' and ' + scenarios[0] +')',np.mean(T_glob[:,1]-T_glob[:,0]))
            print('Stderr of global temperature changes (between ' + scenarios[1] + ' and ' + scenarios[0] +')',stats.sem(T_glob[:,1]-T_glob[:,0]))
        
        #t-test for each country
        otbl_ctry_T_stat = pd.DataFrame(index=np.arange(0,np.shape(T_ctry)[0]),columns = ['Temp_mean_climatological','Temp_mean_noaero','Temp_Changes','STD_Changes','STE_Changes','T-test_Changes','P-value_Changes'])
        
        for ctry in np.arange(0,np.shape(T_ctry)[0]):
            otbl_ctry_T_stat.loc[ctry,'Temp_mean_climatological'] = T_ctry[ctry,:,0].mean()
            otbl_ctry_T_stat.loc[ctry,'Temp_mean_noaero'] = T_ctry[ctry,:,1].mean()
            
            t_val = _env.cal_ttest_1sam_autocor(T_ctry[ctry,:,1] - T_ctry[ctry,:,0])
            
            otbl_ctry_T_stat.loc[ctry,'T-test_Changes'] = float(t_val[0])
            otbl_ctry_T_stat.loc[ctry,'P-value_Changes'] = float(t_val[1])
            
            T_diff = T_ctry[ctry,:,1]-T_ctry[ctry,:,0]
            otbl_ctry_T_stat.loc[ctry,'STD_Changes'] = np.std(T_diff)
            otbl_ctry_T_stat.loc[ctry,'STE_Changes'] = stats.sem(T_diff)
            
        otbl_ctry_T_stat['Temp_Changes'] = otbl_ctry_T_stat['Temp_mean_noaero'] - otbl_ctry_T_stat['Temp_mean_climatological']
          
        pvalues = otbl_ctry_T_stat['P-value_Changes'].astype('float')
        
        print(pvalues[pvalues>0.05])
        print(' ')
        print(str(len(pvalues[pvalues>0.05])) + ' countries failed to pass the T-test (aerosol-induced temperature changes)')
        
        ind_fail = np.where(pvalues>0.05)[0]
        
        if len(pvalues[pvalues>0.05]) > 0:
            print(otbl_ctry_T_stat.iloc[ind_fail])
        
        odir_summary = _env.odir_root + 'summary_' + ds
        _env.mkdirs(odir_summary)
        otbl_ctry_T_stat.to_csv(odir_summary + '/country_specific_statistics_Temp_' + ds + '_' +scen +  '.csv')
        
        ####summarize global and regional GDP changes####
        itbl_gdp_baseline = itbl_ctry_info.copy() 
        
        writer = pd.ExcelWriter(odir_summary + '/country_specific_statistics_GDP_'+ds+'_'+scen+'_Burke.xls')
        otbls_ctry_GDP_stat = {}
        
        gdp_tot = itbl_gdp_baseline[sgdp_year + '_gdp'].sum()
        
        otbl_median = pd.DataFrame(index=boot_methods,columns = ['median','median_ratio','5','5_ratio','95','95_ratio','10','10_ratio','90','90_ratio','prob_benefit'])
        
        for b_m in boot_methods:
            
            imtrx_gdp = omtrxs_dgdp[b_m].copy()
            ##global total
            imtrx_gdp_glob = (imtrx_gdp).sum(axis=1)
            
            otbl_median.loc[b_m] = np.median(imtrx_gdp_glob)/1e9,np.median(imtrx_gdp_glob)/gdp_tot*100,np.percentile(imtrx_gdp_glob,95)/1e9,np.percentile(imtrx_gdp_glob,95)/gdp_tot*100,np.percentile(imtrx_gdp_glob,5)/1e9,np.percentile(imtrx_gdp_glob,5)/gdp_tot*100,    np.percentile(imtrx_gdp_glob,90)/1e9,np.percentile(imtrx_gdp_glob,90)/gdp_tot*100,np.percentile(imtrx_gdp_glob,10)/1e9,np.percentile(imtrx_gdp_glob,10)/gdp_tot*100,len(np.where(imtrx_gdp_glob<0)[0])/np.size(imtrx_gdp_glob)
            
            otbl_ctry_GDP_stat = itbl_gdp_baseline.copy()
            otbl_ctry_GDP_stat['GDP_mean_benefit'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_median_benefit'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_mean_benefit_ratio'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_median_benefit_ratio'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_90_benefit'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_10_benefit'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_95_benefit'] = np.zeros(len(otbl_ctry_GDP_stat.index))
            otbl_ctry_GDP_stat['GDP_5_benefit'] = np.zeros(len(otbl_ctry_GDP_stat.index))    
            otbl_ctry_GDP_stat['probability_damage'] = np.zeros(len(otbl_ctry_GDP_stat.index)) #add  by yz 20190719
            
            for ictry,ctry in enumerate(itbl_ctry_info.index):
                imtrx_country = (imtrx_gdp)[:,ictry,:]
                
                otbl_ctry_GDP_stat.loc[ctry,'GDP_mean_benefit'] = -np.mean(imtrx_country)
                otbl_ctry_GDP_stat.loc[ctry,'GDP_median_benefit'] = -np.median(imtrx_country)
                otbl_ctry_GDP_stat.loc[ctry,'GDP_90_benefit'] = -np.percentile(imtrx_country,90)
                otbl_ctry_GDP_stat.loc[ctry,'GDP_10_benefit'] = -np.percentile(imtrx_country,10)
                
                otbl_ctry_GDP_stat.loc[ctry,'GDP_95_benefit'] = -np.percentile(imtrx_country,95)
                otbl_ctry_GDP_stat.loc[ctry,'GDP_5_benefit'] = -np.percentile(imtrx_country,5)
                
                otbl_ctry_GDP_stat.loc[ctry,'probability_damage'] = len(imtrx_country[imtrx_country>0])/np.size(imtrx_country)
                
            otbl_ctry_GDP_stat['GDP_mean_benefit_ratio'] = otbl_ctry_GDP_stat['GDP_mean_benefit']/otbl_ctry_GDP_stat[sgdp_year+'_gdp']*100
            otbl_ctry_GDP_stat['GDP_median_benefit_ratio'] = otbl_ctry_GDP_stat['GDP_median_benefit']/otbl_ctry_GDP_stat[sgdp_year+'_gdp']*100
            otbl_ctry_GDP_stat.to_excel(writer,b_m)
            otbls_ctry_GDP_stat[b_m] = otbl_ctry_GDP_stat.copy()
        
        otbl_median = -otbl_median
        
        otbl_median.to_excel(writer,'median_summary')
            
        writer.save()

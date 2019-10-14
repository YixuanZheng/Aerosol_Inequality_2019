# -*- coding: utf-8 -*-

'''
This code summaries the percentage of country, gdp and population that are statistically 
influenced (benefit or damage) by aerosol-induced coolings.

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn.apionly as sns
#import geopandas as gp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from colormap import rgb2hex
import _env

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'


gdp_year = _env.year
sgdp_year = str(gdp_year)

datasets = _env.datasets

for ds in datasets:
    scens = ['No-Aerosol']
    if ds == 'ERA-Interim':
        scens = ['No-Aerosol','No-Sulfate']
    for scen in scens:
        if_temp = _env.odir_root + '/summary_'+ds+'/country_specific_statistics_Temp_'+ds+'_'+scen+'.csv'
        if_gdp = _env.odir_root + '/summary_'+ds+'/country_specific_statistics_GDP_'+ds+'_'+scen+'_Burke.xls'
        if_ctrylist = _env.idir_root + '/regioncode/Country_List.xls'
        odir_plot = _env.odir_root + '/plot/'
        _env.mkdirs(odir_plot)
        
        
        itbl_temp = pd.read_csv(if_temp)
        itbl_gdp = pd.read_excel(if_gdp,'country-lag0')
        itbl_ctrylist = pd.read_excel(if_ctrylist)
        
        itbl_temp = itbl_temp.loc[itbl_gdp.ind_in_full_list]
        itbl_temp.index = itbl_gdp.index
        
        itbl_ctrylist.set_index('ISO',inplace = True)
        
        itbl_gdp['GDP_90_benefit_ratio'] =  itbl_gdp['GDP_90_benefit']/itbl_gdp[sgdp_year + '_gdp']
        itbl_gdp['GDP_10_benefit_ratio'] =  itbl_gdp['GDP_10_benefit']/itbl_gdp[sgdp_year + '_gdp']
        
        itbl_gdp.set_index('iso',inplace = True)
        itbl_gdp['Ctry_Name'] = itbl_ctrylist['NAME_ENGLI']
        
        itbl_gdp.set_index(['Ctry_Name'],inplace=True)
        itbl_gdp['Ctry_Name'] = itbl_gdp.index
        
        
        ##########Summarize
        
        itbl_gdp['gdp_share'] = itbl_gdp['2010_gdp']/itbl_gdp['2010_gdp'].sum()*100
        
        otbl_sum = pd.DataFrame(index = [0.9,0.8], columns = ['ctry_dam','ctry_ben','gdp_dam','gdp_ben','pop_dam','pop_ben'])
        otbl_sum_r = pd.DataFrame(index = [0.9,0.8], columns = ['ctry_dam','ctry_ben','gdp_dam','gdp_ben','pop_dam','pop_ben'])
        
        writer = pd.ExcelWriter(_env.odir_root + '/summary_'+ds+'/summary_robustness_'+ds+'_'+scen+'.xlsx')
        
        prob_levels = [0.9,0.8,0.66]
        
        for prob in prob_levels:
            mtbl_dam = itbl_gdp[itbl_gdp['probability_damage'] > prob]
            mtbl_ben = itbl_gdp[itbl_gdp['probability_damage'] < (1-prob)]
            
            otbl_sum.loc[prob,'ctry_dam'] = len(mtbl_dam)
            otbl_sum.loc[prob,'ctry_ben'] = len(mtbl_ben)
        
            otbl_sum_r.loc[prob,'ctry_dam'] = len(mtbl_dam)/len(itbl_gdp)*100
            otbl_sum_r.loc[prob,'ctry_ben'] = len(mtbl_ben)/len(itbl_gdp)*100
            
            otbl_sum.loc[prob,'gdp_dam'] = mtbl_dam[sgdp_year + '_gdp'].sum()
            otbl_sum.loc[prob,'gdp_ben'] = mtbl_ben[sgdp_year + '_gdp'].sum()
            
            otbl_sum_r.loc[prob,'gdp_dam'] = mtbl_dam[sgdp_year + '_gdp'].sum()/itbl_gdp[sgdp_year + '_gdp'].sum()*100
            otbl_sum_r.loc[prob,'gdp_ben'] = mtbl_ben[sgdp_year + '_gdp'].sum()/itbl_gdp[sgdp_year + '_gdp'].sum()*100
        
            otbl_sum.loc[prob,'pop_dam'] = mtbl_dam[sgdp_year + '_pop'].sum()
            otbl_sum.loc[prob,'pop_ben'] = mtbl_ben[sgdp_year + '_pop'].sum()
                
            otbl_sum_r.loc[prob,'pop_dam'] = mtbl_dam[sgdp_year + '_pop'].sum()/itbl_gdp[sgdp_year + '_pop'].sum()*100
            otbl_sum_r.loc[prob,'pop_ben'] = mtbl_ben[sgdp_year + '_pop'].sum()/itbl_gdp[sgdp_year + '_pop'].sum()*100
                
            otbl_sum.to_excel(writer,'Sum_Quan')
            otbl_sum_r.to_excel(writer,'Sum_Rate')
            
        for prob in prob_levels:
            mtbl_dam = itbl_gdp[itbl_gdp['probability_damage'] > prob]
            mtbl_ben = itbl_gdp[itbl_gdp['probability_damage'] < (1-prob)]
            
            mtbl_dam.to_excel(writer,'ctry_list_damage_prob' + str(prob))
            mtbl_ben.to_excel(writer,'ctry_list_benefit_prob' + str(prob))
        
        writer.save()

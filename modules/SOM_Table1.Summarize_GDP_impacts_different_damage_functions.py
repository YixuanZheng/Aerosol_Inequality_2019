# -*- coding: utf-8 -*-

'''
This code calculates aerosol-induced GDP changes based on different damage functions
and summaries results shown in Table S1:
    Global economic impacts of aerosol-induced cooling derived from various forms of damage functions. 
    
by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import _env

gdp_year = _env.year
sgdp_year = str(gdp_year)

ds = 'ERA-Interim'
p_scen = 'No-Aerosol'
cols_otbl = ['Model','Speciations','Apply to','Global GDP impacts (billion $)','Global GDP impact ratio (%)','Percent changes in 90:10 ratio','Percent changes in 80:20 ratio','Probability of reduced inequality']

models = ['Burke et al.','Dell et al.','Pretis et al.', 'DICE']
apply2s = ['GDP growth','GDP growth','GDP growth','GDP levels']

speciations = {'Burke et al.':['country-lag0','country-lag1','country-lag5','year','year-blocks'],
                     'Dell et al.':['Dell'],
                     'Pretis et al.':['M1','M2','M3'],
                     'DICE':['DICE-2013R','DICE-2016R']}

odir_tbl = _env.odir_root + '/table/'
_env.mkdirs(odir_tbl)

of_tbl = odir_tbl + 'SOM_Table1.xls'
writer = pd.ExcelWriter(of_tbl)

#===================applied functions====================
def text_fmt(median, p5, p95,digits = 2):
    #formating text for output table cells
    
    tp_min = np.min([p5,p95])
    tp_max = np.max([p5,p95])
    
    return str(np.around(median,digits)) + ' (' + str(np.around(tp_min,digits)) + ', ' + str(np.around(tp_max,digits)) + ')'
    
def probability_scale(prob_deci,prob_quin):
    #return probabolity scale of reduced inequality
    #based on the likelihood scale presented in the IPCC AR5 Guidance Note
    
    likelihood_scale = ['Virtually certain','Very likely','Likely','About as likely as not','Unlikely','Very unlikely','Exceptionally unlikely']
    likelihood_scale_lb = np.array([0.99,0.9,0.66,0.33,0,0,0])
    likelihood_scale_ub = np.array([1,1,1,0.66,0.33,0.1,0.01])
    
    prob = np.min([prob_deci,prob_quin])
    
    if prob > 0.66:
        prob_scale = likelihood_scale[(np.where((prob >= likelihood_scale_lb) & (prob <= likelihood_scale_ub)))[0][0]]
    elif prob < 0.33:
        prob_scale = likelihood_scale[(np.where((prob >= likelihood_scale_lb) & (prob <= likelihood_scale_ub)))[0][-1]]
    else:
        prob_scale = 'About as likely as not'
    
    
    return prob_scale
  
def gen_otbl_model(model_ind,model_speciations):
    #generate model specific output table
    otbl_model = pd.DataFrame(index=np.arange(0,len(model_speciations)),columns = cols_otbl)
    otbl_model['Model'] = models[model_ind]
    otbl_model['Apply to'] = apply2s[model_ind]

    model_author = models[model_ind].split(' et al.')[0]
    if_gdp_ipct = _env.odir_root + '/summary_' + ds + '/' + 'country_specific_statistics_GDP_' + ds + '_' + p_scen + '_' + model_author + '.xls'
    if_ineq = _env.odir_root + '/summary_' + ds + '/' + 'Deciles_and_Quintile_ratio_changes_' + ds + '_' + p_scen + '_' + model_author + '.xls'
    
    itbl_gdp_ipct = pd.read_excel(if_gdp_ipct,'median_summary',index_col = 0)    
    itbl_ineq_deci = pd.read_excel(if_ineq,'deciles',index_col = 0)    
    itbl_ineq_quin = pd.read_excel(if_ineq,'quintiles',index_col = 0)    
    
    for ispe,spe in enumerate(model_speciations):
        otbl_model.loc[ispe,'Speciations'] = model_speciations[ispe]
        
        itbl_gdp_ipct_spe = itbl_gdp_ipct.loc[spe]
        otbl_model.loc[ispe,'Global GDP impacts (billion $)'] = text_fmt(itbl_gdp_ipct_spe['median'], itbl_gdp_ipct_spe['5'], itbl_gdp_ipct_spe['95'],digits=1)
        otbl_model.loc[ispe,'Global GDP impact ratio (%)'] = text_fmt(itbl_gdp_ipct_spe['median_ratio'], itbl_gdp_ipct_spe['5_ratio'], itbl_gdp_ipct_spe['95_ratio'])
        
        itbl_ineq_deci_spe = itbl_ineq_deci.loc[spe]
        itbl_ineq_quin_spe = itbl_ineq_quin.loc[spe]
        otbl_model.loc[ispe,'Percent changes in 90:10 ratio'] = text_fmt(itbl_ineq_deci_spe['median_ratio'], itbl_ineq_deci_spe['5_ratio'], itbl_ineq_deci_spe['95_ratio'],digits=2)
        otbl_model.loc[ispe,'Percent changes in 80:20 ratio'] = text_fmt(itbl_ineq_quin_spe['median_ratio'], itbl_ineq_quin_spe['5_ratio'], itbl_ineq_quin_spe['95_ratio'],digits=2)
        otbl_model.loc[ispe,'Probability of reduced inequality'] = probability_scale(itbl_ineq_deci_spe['probability_reduced'],itbl_ineq_quin_spe['probability_reduced'])
     
    return otbl_model



otbls = {}
#===================summarize results based on Burke et al., Dell et al., and Pretis et al.====================

for model_ind in np.arange(0,3):
    model_name = models[model_ind]
    model_speciations = speciations[model_name]
    otbls[model_name] = (gen_otbl_model(model_ind,model_speciations)).copy()


#===================calculate GDP impacts based on DICE model====================
model_ind = 3
model_name = models[model_ind]
model_speciations = speciations[model_name]

otbl_model = pd.DataFrame(index=np.arange(0,len(model_speciations)),columns = cols_otbl)
otbl_model['Model'] = models[model_ind]
otbl_model['Apply to'] = apply2s[model_ind]

#DICE parameters
parameters = {'DICE-2013R':{'a1':0,'a2':0.00267,'a3':2},
              'DICE-2016R':{'a1':0,'a2':0.00236,'a3':2}}

par = 'TREFHT' 
if_temp = _env.odir_root + '/sim_temperature/Simulated_Global_and_Country_' + par + '_20yravg.nc'

#country list 
itbl_ctry_info = pd.read_csv(_env.odir_root + '/basic_stats/' + 'Country_Basic_Stats.csv')

#read global and country-level temperature
T_glob = Dataset(if_temp)['TREFHT_Global'][:,[0,1]]
diff_T_glob = T_glob[:,1] - T_glob[:,0] 

for ispe,spe in enumerate(speciations['DICE']):
    otbl_model.loc[ispe,'Speciations'] = model_speciations[ispe]
    
    parameters_spe = parameters[spe]
    diff_gr = (parameters_spe['a1'] * diff_T_glob) + (parameters_spe['a2'] * diff_T_glob**parameters_spe['a3'])

    diff_gdp = itbl_ctry_info['2010_gdp'].sum() * diff_gr
    
    
    otbl_model.loc[ispe,'Global GDP impacts (billion $)'] = str(np.around(np.median(diff_gdp)/1e9,1)) 
    otbl_model.loc[ispe,'Global GDP impact ratio (%)'] = str(np.around(np.median(diff_gr)*100,2)) 
        
otbls[model_name] = otbl_model.copy()


for model in models:
    if model == models[0]:
        otbl_all =  otbls[model].copy()
      
    else:
        otbl_all = otbl_all.append(otbls[model],ignore_index=False)

otbl_all.to_excel(writer)
writer.save()

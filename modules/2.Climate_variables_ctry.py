# -*- coding: utf-8 -*-

'''
This code calculates global and country-level annual mean climate variables from different scenarios (i.e., With-Aerosol, No-Aerosol, and No-Sulfate) 

based on 2000-2019 transient simulations from 8-ensemble member of each scenario

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import json
import _env
import datetime


nscen = _env.nscen ##number of scenarios
nens = _env.nens   #number of ensemble members

nyr = _env.nyr   #total year numbers of one ensemble
nyr_ra = _env.nyr_ra
nyr_app = _env.nyr_app #n-year applied for further analysis

year = _env.year
syr = str(year)

nlat = _env.nlat
nlon = _env.nlon

idir_reg = _env.idir_root + '/regioncode/'
idir_cesm = _env.idir_root + '/simulated_variables_annual/'

if_grid = idir_reg + '/Country_Grid_Index.json'
if_ctry = idir_reg + '/Country_List.xls'

scenarios = _env.scenarios #scenarios considered in this study
parameters = ['TREFHT','AODVIS'] 
parameters_c = ['latitude','longitude']

parameters_info = _env.parameters_info

####main####

with open(if_grid, 'r') as fp:
    i_ctry_grid = json.load(fp)
itbl_ctry = pd.read_excel(if_ctry)
arr_domain = np.zeros([nlat,nlon])


####read and calcualte 20-year mean of climate variables of each ensemble member in each scenario####
i_pars = {}
for ipar,par in enumerate(parameters):
    i_pars[par] = {}
    for scen in scenarios:
        #averaged over 2000-2019
        
        i_par_avg = np.zeros([nens, nlat, nlon])
        
        for iens in np.arange(1,nens+1): #1-8
            if_nc = idir_cesm +  par + '_' + scen + '-' + str(iens) + '_1950-2019.nc'
            
            inc = Dataset(if_nc)
            par_val = (inc[par][-nyr_ra::,:,:] + parameters_info[par]['delta']) * parameters_info[par]['scale']
            lat = inc['lat'][:]
            lon = inc['lon'][:]
            inc.close()
            
            i_par_avg[iens-1,:,:] = par_val.mean(axis=0)
        
        i_pars[par][scen] = i_par_avg.copy() 

        if not 'lat' in i_pars:
            i_pars['lat'] = lat.copy()
            i_pars['lon'] = lon.copy()
        
#### Global temperature changes ####        
o_global_pars = {}

lat_r = _env.cal_lat_weight(i_pars['lat'])
for par in parameters :
    o_global_pars[par] = np.zeros([nens,len(scenarios)])  
    for iscen, scen in enumerate(scenarios):
        i_par = i_pars[par][scen].copy()
        o_global_pars[par][:,iscen] = _env.cal_global_mean(i_par,lat_r)
 
#### Global population-weighted mean temperature changes ####        
#load population data (GPW)
inc_pop = Dataset(_env.idir_root + '/pop/GPW_POP_25x19deg_'+syr+'.nc')
pop = np.squeeze(inc_pop['pop'][:])
inc_pop.close()

o_g_pw_pars = {}

for par in parameters:
    o_g_pw_pars[par] = np.zeros([nens,nscen]) 
    for iscen, scen in enumerate(scenarios):
        i_par = i_pars[par][scen].copy()
        o_g_pw_pars[par][:,iscen] = _env.cal_cty_mean(i_par,pop,'global')
        
      
#### Country-level climate changes #### 

#calculate grid indices for each country
ind_ctry = {}
for ctry in i_ctry_grid:
    ind_ctry[int(ctry)] = _env.grid_ind_2d(i_ctry_grid[ctry])

o_ctry_pars = {}
for par in parameters :
    o_ctry_pars[par] = np.zeros([len(itbl_ctry.index),nens,nscen])
    for iscen,scen in enumerate(scenarios):
        i_par = i_pars[par][scen].copy()
        for ctry in itbl_ctry.index:
            ipar_ctry = i_par[:,ind_ctry[ctry][0,:],ind_ctry[ctry][1,:]]
            pop_ctry = pop[ind_ctry[ctry][0,:],ind_ctry[ctry][1,:]]
            o_ctry_pars[par][ctry,:,iscen] = _env.cal_cty_mean(ipar_ctry,pop_ctry,ctry)
    
    
    #output global and country-level results# 
    _env.mkdirs(_env.odir_root + '/' + parameters_info[par]['dir'] + '/')
    of_nc =  _env.odir_root + '/' + parameters_info[par]['dir'] + '/Simulated_Global_and_Country_' + par + '_20yravg.nc'
    _env.rmfile(of_nc)

    
    onc = Dataset(of_nc, 'w', format='NETCDF4')
    
    d_ctry = onc.createDimension('countries',len(itbl_ctry.index))
    d_year = onc.createDimension('ensembles',nens)
    d_ens = onc.createDimension('scenarios',nscen)
    
    v_par_global = onc.createVariable(par + '_Global','f4',('ensembles','scenarios'))
    v_par_global_pw = onc.createVariable(par + '_Global_PW','f4',('ensembles','scenarios'))
    v_par_ctry = onc.createVariable(par + '_Country','f4',('countries','ensembles','scenarios'))
    
    v_par_global.unit = parameters_info[par]['unit_out']
    v_par_global_pw.unit = parameters_info[par]['unit_out']
    v_par_ctry.unit = parameters_info[par]['unit_out']

    
    v_par_global.desc = 'Global mean ' + parameters_info[par]['longname'] + ' (area-weighted)'
    v_par_global_pw.desc = 'Global mean ' + parameters_info[par]['longname'] + ' (population-weighted)'
    v_par_ctry.desc = 'Population-weighted mean country level ' + parameters_info[par]['longname']
    
    v_par_global[:] = o_global_pars[par]
    v_par_global_pw[:] = o_g_pw_pars[par]
    v_par_ctry[:] = o_ctry_pars[par]
        
    #write global attribute
    onc.by = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
    onc.desc = 'Global mean and country level population-weighted mean ' + parameters_info[par]['longname'] + ' simulation from different CESM ensembles (results averaged over 2000-2019)'
    onc.creattime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    onc.close()



####output gridded global temperature changes based on 160 years simulations

for par in parameters:
    
    _env.mkdirs(_env.odir_root + '/' + parameters_info[par]['dir'] + '/')
    of_nc_g =  _env.odir_root + '/' + parameters_info[par]['dir'] + '/Simulated_Global_Gridded_' + par + '.nc'
    _env.rmfile(of_nc_g)
    
    onc_g = Dataset(of_nc_g, 'w', format='NETCDF4')
    

    d_ens = onc_g.createDimension('ensembles',nens)
    d_lat = onc_g.createDimension('lat',nlat)
    d_lon = onc_g.createDimension('lon',nlon)
    
    v_d_lat = onc_g.createVariable('lat','f4',('lat'))
    v_d_lat[:] = lat.data
    v_d_lon = onc_g.createVariable('lon','f4',('lon'))
    v_d_lon[:] = lon.data
    
    v_d_ens = onc_g.createVariable('ensembles','f4',('ensembles'))
    v_d_ens[:] = np.arange(1,nens+1)
    
    for scen in scenarios:
        v_par = onc_g.createVariable(par + '_' + scen,'f4',('lat','lon'))
        v_par.unit = parameters_info[par]['unit_out']
        v_par.desc = 'Mean ' + parameters_info[par]['longname']
        v_par[:] = i_pars[par][scen].mean(axis=0)
        
        v_par_ens = onc_g.createVariable(par + '_' + scen + '_ensemble','f4',('ensembles','lat','lon'))
        v_par_ens.unit = parameters_info[par]['unit_out']
        v_par_ens.desc = 'Mean ' + parameters_info[par]['longname']
        v_par_ens[:] = i_pars[par][scen]
        
        
    for scen in scenarios[1::]:
        d_par = (i_pars[par][scen][:]-i_pars[par][scenarios[0]][:])
    
        #one sample t-test with the consideration of autocorrelation
        rd_par = np.reshape(d_par,[nens,nlat*nlon])
        arr_ttest = np.apply_along_axis(_env.cal_ttest_1sam_autocor,axis=0,arr = rd_par)
        arr_tval = np.reshape(arr_ttest[0],[nlat,nlon])
        arr_pval = np.reshape(arr_ttest[1],[nlat,nlon])
            
        v_par = onc_g.createVariable(par + '_d_' + scen + '_' + scenarios[0],'f4',('lat','lon'))
        v_par.unit = parameters_info[par]['unit_out']
        v_par.desc = 'Difference in ' + par + ' between ' + scen + ' and ' + scenarios[0]
        v_par[:] = d_par.mean(axis=0)
        
        #store results for t-test
        v_tval = onc_g.createVariable(par + '_T_' + scen + '_' + scenarios[0],'f4',('lat','lon'))
        v_tval.desc = 'T statistics from the Student T-test (' + par + ' between ' + scen + ' and ' + scenarios[0] + ')'

        v_tval[:] = arr_tval
        
        v_pval = onc_g.createVariable(par + '_P_' + scen + '_' + scenarios[0],'f4',('lat','lon'))
        v_pval.desc = 'P-value from the Student T-test (' + par + ' between ' + scen + ' and ' + scenarios[0] + ')'
        v_pval[:] = arr_pval
        
    #write global attribute
    onc_g.by = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
    onc_g.desc = 'Gridded global multi-year mean ' + parameters_info[par]['longname'] + '  simulation from different CESM ensembles'
    onc_g.creattime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    onc_g.close()

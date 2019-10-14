# -*- coding: utf-8 -*-

'''
This code calculates country level population-weighted climatological 
surface air temperature (centered at Jan 1 2010) based on three datasets:
    1. CESM simulated results of With-Aerosol scenario (average over 2000-2019)
    2. NCEP Reanalysis-1 reanalysis dataset (average over 2001-2018)
    3. ECMWF ERA-Interim reanalysis dataset (average over 2001-2018)

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import json
import _env

nens = _env.nens 
nyr = _env.nyr
nyr_app = _env.nyr_app

year = _env.year
syr = str(year)

nlat = _env.nlat
nlon = _env.nlon

parameters_info = _env.parameters_info

rean_ttag = '2001-2018' 

idir_reg = _env.idir_root + '/regioncode/'
idir_cesm = _env.idir_root + '/cam5_annual/'

if_grid = idir_reg + '/Country_Grid_Index.json'
if_ctry = idir_reg + '/Country_List.xls'



idir_reg = _env.idir_root + '/regioncode/'
idir_cesm = _env.idir_root + '/cam5_annual/'

if_grid = idir_reg + '/Country_Grid_Index.json'
if_ctry = idir_reg + '/Country_List.xls'

par = 'TREFHT' 

#calculate grid indices for each country
#load population data (GPW)
inc_pop = Dataset(_env.idir_root + '/pop/GPW_POP_25x19deg_'+syr+'.nc')
pop = np.squeeze(inc_pop['pop'][:])
inc_pop.close()



def grid_ind_2d(ind_1d):
    #ind_1d started from 1
    ind_1d = np.array(ind_1d)-1
    ind_2d = np.array([np.array(ind_1d)//nlon, np.mod(ind_1d,nlon)])
    return(ind_2d)  
  
with open(if_grid, 'r') as fp:
    i_ctry_grid = json.load(fp)
itbl_ctry = pd.read_excel(if_ctry)
o_ctry_pars = np.zeros([len(itbl_ctry.index),3])

i_pars = {}
#########################CESM simulation


def temp_grid2ctry(if_nc,var_name,ds_name):
    
    inc = Dataset(if_nc)
    par_val = inc[var_name][:].data
    if ds_name == 'ERA-Interim':
        par_val = par_val-273.15
        
    lat = inc['lat'][:]
    inc.close()
    
    i_par = par_val.copy() #par_val.copy()
    
    lat_r = _env.cal_lat_weight(lat)    
    print('Global mean temperature based on '+ ds_name + ' %2f' % _env.cal_global_mean_1yr(par_val,lat_r))
        
    #calculate grid indices for each country
    ind_ctry = {}
    for ctry in i_ctry_grid:
        ind_ctry[int(ctry)] = grid_ind_2d(i_ctry_grid[ctry])
    
    o_ctry = []
    for ctry in itbl_ctry.index:
        ipar_ctry = i_par[ind_ctry[ctry][0,:],ind_ctry[ctry][1,:]]
        pop_ctry = pop[ind_ctry[ctry][0,:],ind_ctry[ctry][1,:]]
        o_ctry.append(_env.cal_cty_mean_1yr(ipar_ctry,pop_ctry,ctry))
        
    return o_ctry

####    CESM    ####
if_nc = _env.odir_root + '/' + parameters_info[par]['dir'] + '/Simulated_Global_Gridded_' + par + '.nc'
var_name = 'TREFHT_With-Aerosol'
ds_name='CESM'
o_ctry_pars[:,0] = temp_grid2ctry(if_nc,var_name,ds_name)

####    NCEP Reanalysis-1    ####
if_nc = _env.idir_root+ '/reanalysis/Reanalysis-1_Surface_Temp_' + rean_ttag + '_Regridded.nc'
var_name = 'air_mean'
ds_name = 'Reanalysis-1'
o_ctry_pars[:,1] = temp_grid2ctry(if_nc,var_name,ds_name)


####    ECMWF ERA-Interim    ####
if_nc = _env.idir_root+ '/reanalysis/ERA-Interim_Surface_Temp_' + rean_ttag + '_Regridded.nc'
var_name = 't2m_mean'
ds_name = 'ERA-Interim'
o_ctry_pars[:,2] = temp_grid2ctry(if_nc,var_name,ds_name)

#Calcualye results for analyzed countries
###adopt country list from Burke et al. 2018
if_ctry_pr = _env.idir_root + '/historical_stat/' + '/Ctry_Poor_Rich_from_Burke.csv' 
otbl_ctry = pd.read_csv(if_ctry_pr,index_col = 0)

##full country list
if_ctry_list = idir_reg + '/Country_List.xls'
itbl_ctry_list = pd.read_excel(if_ctry_list,index_col = 0)

##map index of analyzed countries in full country list 
otbl_ctry['ind_in_full_list'] = (np.zeros(np.shape(otbl_ctry)[0])-1)
for ictry,ctry in enumerate(otbl_ctry['iso']):
    otbl_ctry.loc[ictry+1,'ind_in_full_list'] = (itbl_ctry_list['ISO'].tolist()).index(ctry)
    
#extract temperature for analyzed countries
otbl_ctry['CESM'] = o_ctry_pars[((otbl_ctry['ind_in_full_list'].astype(int)).tolist()),0]
otbl_ctry['Reanalysis-1'] = o_ctry_pars[((otbl_ctry['ind_in_full_list'].astype(int)).tolist()),1]
otbl_ctry['ERA-Interim'] = o_ctry_pars[((otbl_ctry['ind_in_full_list'].astype(int)).tolist()),2]

otbl_ctry.to_csv(_env.odir_root + 'sim_temperature/Climatological_Temp_Ctry_3ds.csv')



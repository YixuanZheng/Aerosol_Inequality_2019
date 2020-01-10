# -*- coding: utf-8 -*-

'''
This code calculates impacts of temperature changes induced by aerosols on GDP

positive results mean benefits from aerosol-induced cooling

damage function developed in Burke, Matthew Davis, and Diffenbaugh (2018) 

assume results of each year is independent for the CESM repeating cycle simulations

calculate for each grid

climatological temeprature obtained from the ERA-Interim reanalysis dataset

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''   
 

import pandas as pd
import numpy as np
import _env
import datetime
import xarray as xr

nens  = _env.nens

year = _env.year
syr = str(year)
gdp_year = year
sgdp_year = str(gdp_year)

par = 'TREFHT' 
ds = 'ERA-Interim'

rean_ttag = '2001-2018'

if_base = _env.idir_root+ '/reanalysis/ERA-Interim_Surface_Temp_' + rean_ttag + '_Regridded.nc'
if_sim = _env.odir_root + '/sim_temperature/Simulated_Global_Gridded_' + par + '.nc'

if_boot_par = _env.idir_root + 'Burke2018_Replication/data/output/bootstrap.csv'
if_pop = _env.idir_root + '/pop/GPW_POP_25x19deg_2000.nc'


odir_gdp = _env.odir_root + '/gdp_' + ds + '/'
_env.mkdirs(odir_gdp)

scenarios = ['With-Aerosol','No-Aerosol']  
#read gridded temperature

##base temperature from reanalysis data (ERA-Interim)
i_base = xr.open_dataset(if_base)
i_base = i_base.expand_dims({'ensembles':8})

i_base.transpose('ensembles', 'lat', 'lon')
T_grid_base = i_base['t2m_mean'] - 273.15 #K to C


i_sim = xr.open_dataset(if_sim)

T_grid_wa = i_sim[par + '_' +scenarios[0] + '_ensemble'] #With-Aerosol
T_grid_na = i_sim[par + '_' +scenarios[1] + '_ensemble'] #No-Aerosol

T_grid_na = T_grid_base - T_grid_wa + T_grid_na
T_grid_wa = T_grid_base.copy()

#bootstrap methods
itbl_boots_par = pd.read_csv(if_boot_par,index_col=0)
boot_methods = (pd.unique(itbl_boots_par['spec'])).tolist()  #['country-lag0','country-lag1','country-lag5','year','year-blocks']

#columns of output gdp ratio tables

#dict stores all output tables
otbls_boot = {}
omtrxs_ratio = {}
omtrxs_gdp = {}

for b_m in boot_methods: #loop of bootstarpping methods 
    
    mtbl_boot_par_b1 = xr.DataArray(itbl_boots_par.loc[itbl_boots_par['spec'] == b_m,'b1'],
                                 dims=('boots')
                                 )
    
    mtbl_boot_par_b2 = xr.DataArray(itbl_boots_par.loc[itbl_boots_par['spec'] == b_m,'b2'],
                                 dims=('boots')
                                 )
    
    o_dgr = xr.DataArray(np.zeros([len(mtbl_boot_par_b1),i_sim.ensembles.size,i_sim.lat.size,i_sim.lon.size]),
                        dims=('boots', 'ensembles','lat','lon'),
                        coords={'lat': i_sim.lat,
                                'lon': i_sim.lon})
    
    o_dgr = (T_grid_na*mtbl_boot_par_b1 + T_grid_na**2*mtbl_boot_par_b2) - (T_grid_wa*mtbl_boot_par_b1 + T_grid_wa**2*mtbl_boot_par_b2)
    
    o_dgr_m  = o_dgr.median(dim=['boots','ensembles'])
    
    o_dgr.attrs['desc'] = 'Median of impacts of aerosol-induced cooling on annual GDP growth rate'
    
    ods = xr.Dataset({'GDP_Ratio_Median': o_dgr_m})
    
    ods.attrs['by'] = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
    ods.attrs['desc'] = 'Impacts of aerosol-induced cooling on GDP growth rate (based on damage functions developed by Burke et al. 2018)'
    ods.attrs['creattime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    ods.to_netcdf(odir_gdp + 'GDP_Changes_Burke_' + b_m + '_' + sgdp_year +'_' + ds + '_'  + scenarios[1] + '_gridded.nc')

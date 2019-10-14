# -*- coding: utf-8 -*-

'''
This code calculate running average of temperature (global mean and gridded results)
based on 1850-2019 transient simulation

Preindustrial global mean temperature derived from a 110-year 1850 repeating cycle simulation is also calcualted

by Yixuan Zheng (yxzheng@carnegiescience.edu)

'''   

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import _env
import datetime


nlat = _env.nlat
nlon = _env.nlon
nens = _env.nens
scenarios = _env.scenarios
parameters_info = _env.parameters_info

run_avg_year = np.arange(1850,2001) #starting year of each averaging period.
par = 'TREFHT'


idir_cesm = _env.idir_root + '/simulated_variables_annual/'


otbl_gm = pd.DataFrame(index = np.arange(1850,2020),columns = [scenarios[0] + '%d' % i for i in np.arange(1,nens+1)]+[scenarios[1] + '%d' % i for i in np.arange(1,nens+1)] + [scenarios[2] + '%d' % i for i in np.arange(1,nens+1)])

o_pars = {}
for scen in scenarios:
    i_pars = {}
    
    #Period1: 1850-1949 (1 ensemble member for each scenario)
    nyr = 100    #1850-1949
    year = 1850
    syr = str(year)
    ttag = '1850-1949'
    if_nc = idir_cesm + par + '_' + scen + '_' + ttag + '.nc'

    print(if_nc)
    inc = Dataset(if_nc)
    par_val = (inc[par][:].data + parameters_info[par]['delta']) * parameters_info[par]['scale']
    
    lat = inc['lat'][:]
    lon = inc['lon'][:]
    inc.close()
            
            
    for ens in np.arange(1,nens+1):
        if not scen in i_pars:
            i_pars[ens] = par_val.copy()
        else:
            i_pars[ens] = np.append(i_pars[ens], par_val, axis=0) 
    
    if not 'lat' in i_pars:
        i_pars['lat'] = lat.copy()
        i_pars['lon'] = lon.copy()
           
        
    #Period1: 1850-1949 (8 ensemble member for each scenario)   
    nyr = 70    #total year numbers
    year = 1950
    syr = str(year)
    ttag = '1950-2019'
    
    for ens in np.arange(1,nens+1):
        if_nc = idir_cesm + par + '_' + scen + '-' + str(ens) +'_' + ttag + '.nc'
        
        print(if_nc)
        inc = Dataset(if_nc)
        par_val = (inc[par][:].data + parameters_info[par]['delta']) * parameters_info[par]['scale']
        inc.close()
        
        i_pars[ens] = np.append(i_pars[ens], par_val, axis=0) 

    ####output concatenated gridded parameter 
    _env.mkdirs(_env.odir_root + '/' + parameters_info[par]['dir']+ '/')
    of_nc =  _env.odir_root + '/' + parameters_info[par]['dir']+ '/' + par + '_' + scen + '_1850-2019_ensemble' + '.nc'
    _env.rmfile(of_nc)
    
    onc = Dataset(of_nc, 'w', format='NETCDF4')
    
    d_yr = onc.createDimension('years',(2020-1850))
    d_lat = onc.createDimension('lat',len(lat))
    d_lon = onc.createDimension('lon',len(lon))
    
    v_d_yr = onc.createVariable('years','f4',('years'))
    v_d_yr[:] = np.arange(1850,2020)
    
    v_d_lat = onc.createVariable('lat','f4',('lat'))
    v_d_lat[:] = lat.data
    v_d_lon = onc.createVariable('lon','f4',('lon'))
    v_d_lon[:] = lon.data
    
    
    for ens in np.arange(1,nens+1):
        v_par = onc.createVariable(par+'_'+'%d' % ens,'f4',('years','lat','lon'))
        v_par.unit = parameters_info[par]['unit_out']
        v_par.desc = parameters_info[par]['longname']
        v_par[:] = i_pars[ens]
        
    #write global attribute
    onc.by = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
    onc.desc = 'Gridded global mean ' + parameters_info[par]['longname']  + ' (1850-2019)'
    onc.creattime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    onc.close()
    
    o_pars[scen] = i_pars
    
    #calculate global mean
    lat_r = _env.cal_lat_weight(lat)
    #calculate global mean
    for ens in np.arange(1,nens+1):
        list_gm = _env.cal_global_mean(i_pars[ens],lat_r)
        otbl_gm[scen + '%d' % ens][:] = list_gm

writer = pd.ExcelWriter(_env.odir_root + '/' + parameters_info[par]['dir']+ '/Global_Mean_' + par + '_1850-2019_ensembles' + '.xls')
otbl_gm.to_excel(writer)
writer.save()


#=====================calculate preindustrial average=====================
scen = 'Pi'
par = 'TREFHT'
            
nyr = 110    #total year numbers
year = 1
syr = str(year)
ttag = '1850'

if_nc = idir_cesm + par + '_' + scen + '_' + ttag + '.nc'


inc = Dataset(if_nc)

par_val = (inc[par][:].data + parameters_info[par]['delta']) * parameters_info[par]['scale']
lat = inc['lat'][:]
lon = inc['lon'][:]
inc.close()
       
#calculate global mean
lat_r = _env.cal_lat_weight(lat)
#calculate global mean
list_gm = _env.cal_global_mean(par_val,lat_r)
otbl_pi = pd.DataFrame(list_gm,columns=['pre_industrial'])

writer = pd.ExcelWriter(_env.odir_root + '/' + parameters_info[par]['dir']+ '/Global_Mean_Temperature_pre-industrial_110yrs.xls')
otbl_pi.to_excel(writer)
writer.save()

#=====================calculate running average=====================
o_pars_rm = {}
for scen in scenarios:
    lat  = o_pars[scen]['lat']
    lon  = o_pars[scen]['lon']
    
    o_par_ens = o_pars[scen][1].copy()
    
    for ens in range(2,nens+1): 
        o_par_ens = o_par_ens + o_pars[scen][ens]
    o_par_ens = o_par_ens/nens
    
    o_par_rm = np.zeros([151,nlat,nlon])
    for year in run_avg_year:
        ind_year = year-1850
        o_par_rm[ind_year,:,:] = o_par_ens[ind_year:(ind_year+20),:,:].mean(axis=0)
    
    o_pars_rm[scen] = o_par_rm.copy()
    
    #output running averages
    
    _env.mkdirs(_env.odir_root + '/' + parameters_info[par]['dir'] + '_running_mean/')
    of_nc =  _env.odir_root + '/' + parameters_info[par]['dir'] + '_running_mean/' + scen + '_' + par + '_1850-2000_running_avg.nc'
    _env.rmfile(of_nc)
    
    onc = Dataset(of_nc, 'w', format='NETCDF4')
    
    d_yr = onc.createDimension('years',(2011-1860))
    d_lat = onc.createDimension('lat',len(lat))
    d_lon = onc.createDimension('lon',len(lon))
    
    v_d_yr = onc.createVariable('years','f4',('years'))
    v_d_yr[:] = np.arange(1860,2011) #center years of 20 year moving window
    v_d_yr.desc = 'Center year of each 20 year moving window'
    
    
    v_d_lat = onc.createVariable('lat','f4',('lat'))
    v_d_lat[:] = lat.data
    v_d_lon = onc.createVariable('lon','f4',('lon'))
    v_d_lon[:] = lon.data
    
    v_par = onc.createVariable(par,'f4',('years','lat','lon'))
    v_par.unit = parameters_info[par]['unit_out']
    v_par.desc = parameters_info[par]['longname']
    v_par[:] = o_par_rm
        
    #write global attribute
    onc.by = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
    onc.desc = 'Gridded global mean ' + parameters_info[par]['longname'] + ' (20-yr running average 1860-2010)'
    onc.creattime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    onc.close()

    

#=====================calculate difference in Temperature=====================
    
#difference in temperature between the last 20 years simulation in the with sulfate ensemble 
#and time series of 20 years running average of the simulation in the no aerosol ensemble
    
for scen in scenarios[1::]:
        
    #difference in temperature between temperature of nominal 2010 and time series of temperature in No-Aerosol or No-Sulfate scenarios (20-year moving average)
    t_diff_aero = o_pars_rm[scen][:,:,:] - o_pars_rm[scenarios[0]][-1,:,:] 
    
    #=====================calculate delayed years=====================
    
    arr_delyr = np.zeros([nlat,nlon])
    arr_delyr[:,:] = np.nan
    for ilat in np.arange(0,nlat):
        for ilon in np.arange(0,nlon):
            ts_grid = t_diff_aero[:,ilat,ilon]
            
            #mask grids with warming impacts from aerosols
            if ts_grid[-1] > 0:
                sign_change = (np.diff(np.sign(ts_grid)) == 2)*1
                if len(np.where(sign_change)[0]) >=1:
                    arr_delyr[ilat,ilon] = 150-np.where(sign_change)[0][-1] + 1 #+1 to indicate the latest years that temperature in No-Aerosol/No-Sulfate scenario is lower than the nominal 2010 temperature in the control scenario
            else:
                arr_delyr[ilat,ilon] = 0
    
    _env.mkdirs(_env.odir_root + '/' + parameters_info[par]['dir'] + '_running_mean/')
    of_nc =  _env.odir_root + '/' + parameters_info[par]['dir'] + '_running_mean/Year-Delayed_RunningAvg_' + scen + '.nc'
    _env.rmfile(of_nc)
    
    onc = Dataset(of_nc, 'w', format='NETCDF4')
    
    d_lat = onc.createDimension('lat',len(lat))
    d_lon = onc.createDimension('lon',len(lon))
    
    v_d_lat = onc.createVariable('lat','f4',('lat'))
    v_d_lat[:] = lat.data
    v_d_lon = onc.createVariable('lon','f4',('lon'))
    v_d_lon[:] = lon.data
    
    v_par = onc.createVariable(par,'f4',('lat','lon'))
    v_par.desc = 'year'
    v_par[:] = arr_delyr
        
    #write global attribute
    onc.by = 'Yixuan Zheng (yxzheng@carnegiescience.edu)'
    onc.desc = 'Delayed years of warming induced by aerosols (20-year moving average)'
    onc.creattime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    onc.close()        

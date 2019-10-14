# -*- coding: utf-8 -*-

'''
enviromental functions and variables

by Yixuan Zheng (yxzheng@carnegiescience.edu)
'''    

#for revision round 1
#yz,20190725

import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import acf
from scipy import stats

#useful functions
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
  
def rmfile(file):      
    if os.path.isfile(file):
        os.remove(file)


def cal_ttest_1sam_autocor(sample,hvalue=0):
    #conduct 1 sample t-test with an effective sample size adjusted for autocorrelation
    sam_mean = np.mean(sample)
    
    g_acf = acf(sample)[1]
    N = np.size(sample)
    
    if g_acf > 0:
        cf = N * (1-g_acf) / (1+g_acf)
    else:
        cf = N*1
    
#    say = stats.sem(sample)
    say = stats.sem(sample)*np.sqrt(N)/np.sqrt(cf)
    
    that = (sam_mean-hvalue)/say
    
    ## Compare with the critical t-value
    #Degrees of freedom
    df = N - 1

    #p-value after comparison with the t 
    pvalue = 2* stats.t.cdf(-np.abs(that),df=df)
    
    return that,pvalue  

def grid_ind_2d(ind_1d):
    #ind_1d started from 1
    ind_1d = np.array(ind_1d)-1
    ind_2d = np.array([np.array(ind_1d)//nlon, np.mod(ind_1d,nlon)])
    return(ind_2d)
    
def cal_lat_weight(i_lat):
    #calculate area weights of grids with different latitude
    ##change  latitude of centers of two polar grids
    i_lat[0] = (i_lat[0] + i_lat[1])/2
    i_lat[-1] = (i_lat[-1] + i_lat[-2])/2
    lat_r = np.cos(i_lat*np.pi/180)##cosine of latitude, converted to radians, as weight
    lat_r = np.repeat(lat_r[:,np.newaxis],nlon,axis=1)
    #lat_r = lat_r/lat_r.sum()
    return(lat_r)

def cal_global_mean(par,lat_r):
    #grid weighted
    par = np.squeeze(par)
    mean_val=[]
    for yr in np.arange(0,(np.shape(par))[0]):
        par_yr = np.squeeze(par[yr,:,:])
        mean_val.append((np.sum(par_yr*lat_r))/(np.sum(lat_r)))
    return(mean_val)

def cal_cty_mean(par,pop,ctry):
    #population weighted
    #par = np.squeeze(par)
    par = par 
    mean_val=[]
    
    for yr in np.arange(0,(np.shape(par))[0]):
        par_yr = np.squeeze(par[yr,:])
        pop = np.squeeze(pop)
        #some country may have zero population (Antarctica or other small islands)
        if np.sum(pop) == 0:
            mean_val.append((np.mean(par_yr)))
        else:
            mean_val.append((np.sum(par_yr*pop))/(np.sum(pop)))
        
    return(mean_val)


def cal_global_mean_1yr(par,lat_r):
    #grid weighted
    par_yr = np.squeeze(par[:,:])
    mean_val = ((np.sum(par_yr*lat_r))/(np.sum(lat_r)))
    return(mean_val)
    
def cal_cty_mean_1yr(par,pop,ctry):
    #population weighted

    par_yr = np.squeeze(par)
    pop = np.squeeze(pop)
        #some country may have zero population (Antarctica or other small islands)
    if np.sum(pop) == 0:
        mean_val = ((np.mean(par_yr)))
    else:
        mean_val = ((np.sum(par_yr*pop))/(np.sum(pop)))
        
    return(mean_val)


#repeatly used variables
year = 2010
gdpyear = 2010

nscen = 3         #number of scenarios
nens = 8          #number of ensemble members
nyr_ra = 20       #running average window
nyr = nens * nyr_ra   #total year numbers
nyr_app = nyr #n-year applied for further analysis

nlat = 96         
nlon = 144

dir_proj = '../'
idir_root = dir_proj + 'input/'
odir_root = dir_proj + 'output/'
dir_mod = dir_proj + 'modules/'

scenarios = ['With-Aerosol','No-Aerosol','No-Sulfate']
datasets = ['ERA-Interim','CESM','Reanalysis-1']
parameters_info = {'TREFHT':{'pars':['TREFHT'],'expression':[1],'unit_org':'K','unit_out':'Celsius Degree','dir':'sim_temperature','longname':'temperature','delta':-273.15,'scale':1},
              'AODVIS':{'pars':['AODVIS'],'expression':[1],'unit_org':'','unit_out':'','dir':'sim_aodvis','longname':'AOD 550nm','delta':0,'scale':1}}
 

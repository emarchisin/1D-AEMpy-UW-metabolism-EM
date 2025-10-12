import pandas as pd
#new
import numpy as np
import pandas as pd
import os
#import scipy
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

#os.chdir("/home/robert/Projects/1D-AEMpy/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/1D-AEMpy/src")
#os.chdir("D:/bensd/Documents/Python_Workspace/1D-AEMpy/src")
#os.chdir("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/src")
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens,atmospheric_module, get_secview, get_lake_config, get_model_params, get_run_config, get_ice_and_snow , get_num_data_columns#, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
lake_config = get_lake_config("../input/lake_config_test.csv", 1)
model_params = get_model_params("../input/model_params_test.csv", 1)
run_config = get_run_config("../input/run_config_test.csv", 1)
ice_and_snow = get_ice_and_snow("../input/ice_and_snow_test.csv", 1)

windfactor = float(lake_config["WindSpeed"])
zmax = 25 # maximum lake depth
nx = 25 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min to convert s to day
dx = zmax/nx # spatial step

## area and depth values of our lake 
area, depth, volume = get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx, nx = nx)

                
## atmospheric boundary conditions
#secview = get_secview(secchifile = "../input/secchifile.csv")  
meteo_all = provide_meteorology(meteofile = '../input/ME_nldas-16-24.csv', 
                    windfactor = windfactor)

pd.DataFrame(meteo_all).to_csv("../input/NLDAS-ME-meteo16-24.csv", index = False)
                     
## time step discretization 

#get start time from input file
desired_start = pd.Timestamp(run_config["start_time"])  
#find the matching index in the meteo file
startTime = meteo_all.index[meteo_all['date'] == desired_start][0]
#get the date from that index
startingDate = meteo_all.loc[startTime, 'date']
n_years = (8.5) #7
hydrodynamic_timestep = 24 * dt
total_runtime =  (365 * n_years) * hydrodynamic_timestep/dt  
#startTime1 =   (182) * hydrodynamic_timestep/dt # DOY in 2016 * 24 hours 138
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1
#endTime1 =  (startTime1 + total_runtime)
#startingDate = meteo_all['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

print ("starting date", startingDate)
print ("starting time", startTime)

def provide_carbon(docfile, startingDate, startTime):
    # Read DOC input file
    doc = pd.read_csv(docfile)
    doc['date'] = pd.to_datetime(doc['datetime'])

    # Filter data starting from model start date
    daily_doc = doc.loc[
    (doc['date'] >= startingDate) & (doc['depth'] == 4)].copy()
    daily_doc['ditt'] = abs(daily_doc['date'] - startingDate)

    # Rename/standardize columns for model consistency
    daily_doc = daily_doc.rename(columns={
        'lake': 'site',
        'unit': 'units',
        'variable': 'var',
        'observation': 'doc_mgl'
    })

    # If startingDate precedes data, insert an initial row
    if startingDate < daily_doc['date'].min():
        first_row = {
            'id': -1,
            'datetime': startingDate,
            'site': doc['lake'].iloc[0],
            'depth': doc['depth'].iloc[0],
            'var': doc['variable'].iloc[0],
            'units': doc['unit'].iloc[0],
            'doc_mgl': doc['observation'].iloc[0],
            'date': startingDate,
            'ditt': (daily_doc['date'].iloc[0] - startingDate)
        }
        daily_doc = pd.concat([pd.DataFrame([first_row]), daily_doc], ignore_index=True)

    # Compute time offset from simulation start
    print("Unique datetimes in daily_doc:")
    print(daily_doc['date'].unique())
    print("Shape:", daily_doc.shape)

    daily_doc['dt'] = (daily_doc['date'] - daily_doc['date'].iloc[0]).dt.total_seconds() + startTime

    return daily_doc



carbon = provide_carbon(docfile =  '../input/peter_doc.csv', 
                                 startingDate=pd.to_datetime("2022-05-30 09:00:00"),
                                 startTime = startTime)

carbon = carbon.dropna(subset=['doc_mgl'])

TP_fillvals = tuple(carbon.doc_mgl.values[[0,-1]])
TP = interp1d(carbon['dt'].values, carbon['doc_mgl'].values,
              kind="linear", fill_value="extrapolate", bounds_error=False)



for i in range(10):
    print(TP(i))
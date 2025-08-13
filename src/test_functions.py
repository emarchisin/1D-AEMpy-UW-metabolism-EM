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
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens,atmospheric_module, get_secview, get_lake_config, get_lake_params


lake_config = get_lake_config("../input/lake_config.csv", 1)
lake_params = get_lake_params("../input/lake_params.csv", 1)


zmax = lake_config['Zmax']
print(f"zmax:  {zmax}")
windfactor = lake_config["WindSpeed"]
print(f"windfactor:  {windfactor}")
oc_load_input = lake_config["OCLoad"]
print(f"oc_load_input:  {oc_load_input}")

light_doc = lake_params["LECDOCR"]
print(f"light_doc:  {light_doc}")
light_poc = lake_params["LECPOCR"]
print(f"light_poc:  {light_poc}")
albedo = lake_config["Albedo"]
print(f"albed:  {albedo}")

resp_docr = lake_params["RDOCR"]# 0.001 0.0001 s-1
print(f"resp_docr:  {resp_docr}")
resp_docl = lake_params["RDOCL"] # 0.01 0.05 s-1
print(f"resp_docl:  {resp_docl}")
resp_poc = lake_params["RPOCR"] # 0.1 0.001 0.0001 s-1
print(f"resp_poc:  {resp_poc}")
theta_r = lake_params["RTheta"]
print(f"theta_r:  {theta_r}")

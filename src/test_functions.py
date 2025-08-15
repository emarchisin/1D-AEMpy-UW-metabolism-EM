import numpy as np
import pandas as pd
import datetime
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens,atmospheric_module, get_secview, get_lake_config, get_model_params, get_run_config, get_ice_and_snow #, heating_module, diffusion_module, mixing_module, convection_module, ice_module

# === Load configuration data ===
lake_config = get_lake_config("../input/lake_config_test.csv", 1)
lake_params = get_model_params("../input/model_params_test.csv", 1)
run_config = get_run_config("../input/run_config_test.csv", 1)
ice_and_snow = get_ice_and_snow("../input/ice_and_snow_test.csv", 1)

# === Print lake_config values ===
print(f"Zmax: {lake_config['Zmax']}")
print(f"WindSpeed (windfactor): {lake_config['WindSpeed']}")
print(f"OCLoad: {lake_config['OCLoad']}")
print(f"Albedo: {lake_config['Albedo']}")
print(f"Outflow depth: {lake_config.get('outflow_depth', 'N/A')}")

# === Print lake_params (model) values ===
print(f"Light DOC (LECDOCR): {lake_params['LECDOCR']}")
print(f"Light POC (LECPOCR): {lake_params['LECPOCR']}")
print(f"Respiration DOCR: {lake_params['RDOCR']}")
print(f"Respiration DOCL: {lake_params['RDOCL']}")
print(f"Respiration POCR: {lake_params['RPOCR']}")
print(f"Theta_r: {lake_params['RTheta']}")
print(f"Sedimentation rate: {lake_params['sediment_rate']}")
print(f"Settling rate: {lake_params['settling_rate']}")
print(f"Sed sink: {lake_params['sed_sink']}")
print(f"Theta NPP: {lake_params['theta_npp']}")
print(f"Conversion constant: {lake_params['conversion_constant']}")
print(f"K_half: {lake_params['k_half']}")
print(f"P_max: {lake_params['p_max']}")
print(f"IP: {lake_params['IP']}")
print(f"prop_oc_docr: {lake_params['prop_oc_docr']}")
print(f"prop_oc_docl: {lake_params['prop_oc_docl']}")
print(f"prop_oc_pocr: {lake_params['prop_oc_pocr']}")
print(f"prop_oc_pocl: {lake_params['prop_oc_pocl']}")
print(f"Physical constant p2: {lake_params['p2']}")
print(f"B: {lake_params['B']}")
print(f"g: {lake_params['g']}")
print(f"meltP: {lake_params['meltP']}")

# === Print run_config values ===
print(f"Start time: {run_config['start_time']}")
print(f"End time: {run_config['end_time']}")
print(f"nx: {run_config['nx']}")
print(f"dt: {run_config['dt']}")
print(f"dx: {run_config['dx']}")
print(f"PGDL mode: {run_config['pgdl_mode']}")
print(f"Training data path: {run_config['training_data_path']}")
print(f"Diffusion method: {run_config['diffusion_method']}")
print(f"Scheme: {run_config['scheme']}")

# === Print ice_and_snow values ===
print(f"Ice: {ice_and_snow['ice']}")
print(f"Hi: {ice_and_snow['Hi']}")
print(f"Hs: {ice_and_snow['Hs']}")
print(f"Hsi: {ice_and_snow['Hsi']}")
print(f"IceT: {ice_and_snow['iceT']}")
print(f"Supercooled: {ice_and_snow['supercooled']}")
print(f"dt_iceon_avg: {ice_and_snow['dt_iceon_avg']}")
print(f"Ice_min: {ice_and_snow['Ice_min']}")
print(f"KEice: {ice_and_snow['KEice']}")
print(f"rho_snow: {ice_and_snow['rho_snow']}")

# === Mixing & transport ===
print(f"km: {lake_params['km']}")
print(f"k0: {lake_params['k0']}")
print(f"weight_kz: {lake_params['weight_kz']}")
print(f"piston_velocity: {lake_params['piston_velocity']}")
print(f"Cd (wind coeff): {lake_params['Cd']}")
print(f"Hydro residence time: {lake_params['hydro_res_time_hr']}")
print(f"W_str: {lake_params['W_str']}")
print(f"Density threshold: {lake_params['denThresh']}")

# === Light & heat fluxes ===
print(f"kd_light: {lake_params['kd_light']}")
print(f"Light water: {lake_params['light_water']}")
print(f"eps: {lake_params['eps']}")
print(f"Emissivity: {lake_params['emissivity']}")
print(f"Sigma: {lake_params['sigma']}")
print(f"SW factor: {lake_params['sw_factor']}")
print(f"Wind factor: {lake_params['wind_factor']}")
print(f"AT factor: {lake_params['at_factor']}")
print(f"Turb factor: {lake_params['turb_factor']}")
print(f"Geothermal heat flux (Hgeo): {lake_params['Hgeo']}")


import numpy as np
import pandas as pd
import datetime
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens, atmospheric_module, get_secview, get_lake_config, get_model_params, get_run_config, get_ice_and_snow

def print_with_type(label, value):
    print(f"{label}: {value} (type: {type(value).__name__})")

# === Load configuration data ===

lake_config = get_lake_config("../input/lake_config_test.csv", 1)
lake_params = get_model_params("../input/model_params_test.csv", 1)

run_config = get_run_config("../input/run_config_test.csv", 1)
ice_and_snow = get_ice_and_snow("../input/ice_and_snow_test.csv", 1)

# === Print lake_config values ===
print_with_type("Zmax", lake_config["Zmax"])
print_with_type("WindSpeed (windfactor)", lake_config["WindSpeed"])
print_with_type("OCLoad", lake_config["OCLoad"])
print_with_type("Albedo", lake_config["Albedo"])
print_with_type("Outflow depth", lake_config.get("outflow_depth", "N/A"))

# === Print lake_params (model) values ===
print_with_type("Light DOC (LECDOCR)", lake_params["LECDOCR"])
print_with_type("Light POC (LECPOCR)", lake_params["LECPOCR"])
print_with_type("Respiration DOCR", lake_params["RDOCR"])
print_with_type("Respiration DOCL", lake_params["RDOCL"])
print_with_type("Respiration POCR", lake_params["RPOCR"])
print_with_type("Theta_r", lake_params["RTheta"])
print_with_type("Sedimentation rate", lake_params["sediment_rate"])
print_with_type("Settling rate", lake_params["settling_rate"])
print_with_type("Sed sink", lake_params["sed_sink"])
print_with_type("Theta NPP", lake_params["theta_npp"])
print_with_type("Conversion constant", lake_params["conversion_constant"])
print_with_type("K_half", lake_params["k_half"])
print_with_type("P_max", lake_params["p_max"])
print_with_type("IP", lake_params["IP"])
print_with_type("prop_oc_docr", lake_params["prop_oc_docr"])
print_with_type("prop_oc_docl", lake_params["prop_oc_docl"])
print_with_type("prop_oc_pocr", lake_params["prop_oc_pocr"])
print_with_type("prop_oc_pocl", lake_params["prop_oc_pocl"])
print_with_type("Physical constant p2", lake_params["p2"])
print_with_type("B", lake_params["B"])
print_with_type("g", lake_params["g"])
print_with_type("meltP", lake_params["meltP"])

# === Print run_config values ===
print_with_type("Start time", run_config["start_time"])
print_with_type("End time", run_config["end_time"])
print_with_type("nx", run_config["nx"])
print_with_type("dt", run_config["dt"])
print_with_type("dx", run_config["dx"])
print_with_type("PGDL mode", run_config["pgdl_mode"])
print_with_type("Training data path", run_config["training_data_path"])
print_with_type("Diffusion method", run_config["diffusion_method"])
print_with_type("Scheme", run_config["scheme"])

# === Print ice_and_snow values ===
print_with_type("Ice", ice_and_snow["ice"])
print_with_type("Hi", ice_and_snow["Hi"])
print_with_type("Hs", ice_and_snow["Hs"])
print_with_type("Hsi", ice_and_snow["Hsi"])
print_with_type("IceT", ice_and_snow["iceT"])
print_with_type("Supercooled", ice_and_snow["supercooled"])
print_with_type("dt_iceon_avg", ice_and_snow["dt_iceon_avg"])
print_with_type("Ice_min", ice_and_snow["Ice_min"])
print_with_type("KEice", ice_and_snow["KEice"])
print_with_type("rho_snow", ice_and_snow["rho_snow"])

# === Mixing & transport ===
print_with_type("km", lake_params["km"])
print_with_type("k0", lake_params["k0"])
print_with_type("weight_kz", lake_params["weight_kz"])
print_with_type("piston_velocity", lake_params["piston_velocity"])
print_with_type("Cd (wind coeff)", lake_params["Cd"])
print_with_type("Hydro residence time", lake_params["hydro_res_time_hr"])
print_with_type("W_str",  None if pd.isna(lake_params["W_str"]) else lake_params["W_str"],)
print_with_type("Density threshold", lake_params["denThresh"])

# === Light & heat fluxes ===
print_with_type("kd_light", lake_params["kd_light"])
print_with_type("Light water", lake_params["light_water"])
print_with_type("eps", lake_params["eps"])
print_with_type("Emissivity", lake_params["emissivity"])
print_with_type("Sigma", lake_params["sigma"])
print_with_type("SW factor", lake_params["sw_factor"])
print_with_type("Wind factor", lake_params["wind_factor"])
print_with_type("AT factor", lake_params["at_factor"])
print_with_type("Turb factor", lake_params["turb_factor"])
print_with_type("Geothermal heat flux (Hgeo)", lake_params["Hgeo"])


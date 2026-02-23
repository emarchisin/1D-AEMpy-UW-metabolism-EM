#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:25:44 2026

@author: emmamarchisin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def is_on(config, lake_key, key):
    return str(config.loc[key, lake_key]).lower() == "yes"

def get_value(config, lake_key, key):
    return config.loc[key, lake_key]


def depth_to_index(depth_m, dx):
    return int(float(depth_m) / dx)


def save_fig(fig, output_dir, lake_key, name):
    fig.tight_layout()
    fig.savefig(output_dir / f"{lake_key}_{name}.png", dpi=300)
    plt.close(fig)
    
def load_observations(driver_dir, config, lake_key, startDate, endDate):
    obs_file = str(get_value(config, lake_key, "obs_file"))

    if obs_file.lower() == "no":
        return None

    df = pd.read_csv(driver_dir / obs_file, parse_dates=["datetime"])
    df = df[(df["datetime"] >= startDate) & (df["datetime"] <= endDate)]
    return df
    
def post_process(
    res,
    times,
    volume,
    dx,
    lake_output_dir,
    postprocess_config,
    lake_key,
    driver_dir,
    startDate,
    endDate,
    meteo_all):
    
    temp = res["temp"]
    o2 = res["o2"]
    docl = res["docl"]
    docr = res["docr"]
    pocl = res["pocl"]
    pocr = res["pocr"]
    npp = res["npp"]
    atm_flux = res["atm_flux_output"]
    docl_resp = res["docl_respiration"]
    docr_resp = res["docr_respiration"]
    poc_resp = res["poc_respiration"]
    secchi = res["secchi"]

    doc_total = docl + docr
    poc_total = pocl + pocr
    
    surf_depth = float(get_value(postprocess_config, lake_key, "surf_depth")) ***
    deep_depth = float(get_value(postprocess_config, lake_key, "deep_depth"))


    surf_ix = depth_to_index(depth, surf_depth)
    deep_ix = depth_to_index(depth, deep_depth)

    df_obs = load_observations(
        driver_dir, postprocess_config, lake_key,
        startDate, endDate
    )
    
    def heatmap_plot(data, name, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(data, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    ax.set_ylabel("Depth index")
    ax.set_xlabel("Time index")
    save_fig(fig, lake_output_dir, lake_key, name)

#O2

   if is_on(postprocess_config, lake_key, "o2_heat"):
        heatmap_plot(o2/volume[:,None], "o2_heat", 0, 20)

    if is_on(postprocess_config, lake_key, "o2_line"):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, o2[surf_ix,:]/volume[surf_ix])
        ax.plot(times, o2[deep_ix,:]/volume[deep_ix], linestyle="--")

        if df_obs is not None:
            df_s = df_obs[(df_obs["variable"]=="do") &
                          (df_obs["depth"]==surf_depth)]
            df_d = df_obs[(df_obs["variable"]=="do") &
                          (df_obs["depth"]==deep_depth)]
            ax.scatter(df_s["datetime"], df_s["observation"], color="red")
            ax.scatter(df_d["datetime"], df_d["observation"], color="darkred")

        ax.set_ylabel("DO (mg/L)")
        save_fig(fig, lake_output_dir, lake_key, "o2_line")

#Water Temp

   if is_on(postprocess_config, lake_key, "wtemp_heat"):
        heatmap_plot(temp, "wtemp_heat", 0, 30)

    if is_on(postprocess_config, lake_key, "wtemp_line"):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, temp[surf_ix,:])
        ax.plot(times, temp[deep_ix,:], linestyle="--")
        ax.set_ylabel("Temp (°C)")
        save_fig(fig, lake_output_dir, lake_key, "wtemp_line")

#DOC and POC

   variables = {
        "docr": docr,
        "docl": docl,
        "doctot": doc_total,
        "pocr": pocr,
        "pocl": pocl,
        "poctot": poc_total}

  for varname, var in variables.items():

        if is_on(postprocess_config, lake_key, f"{varname}_heat"):
            heatmap_plot(var/volume[:,None], f"{varname}_heat")

        if is_on(postprocess_config, lake_key, f"{varname}_line"):
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(times, var[surf_ix,:]/volume[surf_ix])
            ax.plot(times, var[deep_ix,:]/volume[deep_ix], linestyle="--")
            ax.set_ylabel(f"{varname} (mg/L)")
            save_fig(fig, lake_output_dir, lake_key, f"{varname}_line")

#Secchi

  if is_on(postprocess_config, lake_key, "secchi"):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, secchi.T)

        if df_obs is not None:
            df_sec = df_obs[df_obs["variable"]=="secchi"]
            ax.scatter(df_sec["datetime"], df_sec["observation"], color="red")

        ax.set_ylabel("Secchi (m)")
        save_fig(fig, lake_output_dir, lake_key, "secchi")

#Met

   if is_on(postprocess_config, lake_key, "met_line"):
        fig, axis = plt.subplots(4,1, figsize=(12,6), sharex=True)

        axis[0].plot(times, meteo_all['Shortwave_Radiation_Downwelling_wattPerMeterSquared'])
        axis[1].plot(times, meteo_all['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'])
        axis[2].plot(times, meteo_all['Precipitation_millimeterPerDay'])
        axis[3].plot(times, meteo_all['Air_Temperature_celsius'])

        save_fig(fig, lake_output_dir, lake_key, "met_panel")

#GPP, R, AtmEx Rates

 if is_on(postprocess_config, lake_key, "rates_panel"):

        r_all = (
            (docl * docl_resp) +
            (docr * docr_resp) +
            (pocl * poc_resp) +
            (pocr * poc_resp)
        ) / volume[:,None]

        gpp_all = npp/volume[:,None] + r_all

        r = r_all[surf_ix,:]
        gpp = gpp_all[surf_ix,:]
        atm = atm_flux[0,:] / volume[0]

        fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
        ax[0].plot(times, gpp)
        ax[0].set_ylabel("GPP (g/m3/d)")
        ax[1].plot(times, r)
        ax[1].set_ylabel("R (g/m3/d)")
        ax[2].plot(times, atm)
        ax[2].set_ylabel("AtmEx (g/m3/d)")

        save_fig(fig, lake_output_dir, lake_key, "rates_panel")
        
#INtregated GPP and R

   if is_on(postprocess_config, lake_key, "integrated_gpp") \
       or is_on(postprocess_config, lake_key, "integrated_r"):

        r_all = (
            (docl * docl_resp) +
            (docr * docr_resp) +
            (pocl * poc_resp) +
            (pocr * poc_resp)
        ) / volume[:,None]

        gpp_all = npp/volume[:,None] + r_all

        integrated_gpp = np.sum(gpp_all * dx, axis=0)
        integrated_r = np.sum(r_all * dx, axis=0)

        if is_on(postprocess_config, lake_key, "integrated_gpp"):
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(times, integrated_gpp)
            ax.set_ylabel("Integrated GPP (g/m²/d)")
            save_fig(fig, lake_output_dir, lake_key, "integrated_gpp")

        if is_on(postprocess_config, lake_key, "integrated_r"):
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(times, integrated_r)
            ax.set_ylabel("Integrated R (g/m²/d)")
            save_fig(fig, lake_output_dir, lake_key, "integrated_r")


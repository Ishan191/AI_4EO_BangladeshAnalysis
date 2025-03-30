import os
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import glob
from datetime import datetime
from rasterio.mask import mask

## Files
def getRootFolder():
    return "D:/Python Test Codes/"

def getInputs():
    return f"{getRootFolder()}Inputs_UHIRescaled/"

def getBangShp():
    return f"{getRootFolder()}AI4EO/"

## Export
def fix_invalid_geometry(geometry):
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry

def Export_Shapefile(filtered, output_folder, output_filename):
    filtered["geometry"] = filtered["geometry"].apply(fix_invalid_geometry)
    os.makedirs(output_folder, exist_ok=True)
    output_shapefile_path = os.path.join(output_folder, output_filename + ".shp")
    filtered.to_file(output_shapefile_path, driver='ESRI Shapefile')


def clippies():
    # Load necessary datasets
    Input_folder = glob.glob(os.path.join(getInputs(), "*.tif"))

    # Read shapefile
    BangShp = gpd.read_file(f"{getBangShp()}Area2_RetainedData.shp").to_crs(4326)
    print(BangShp)

    # Loop through all TIFF files
    for Input_file in Input_folder:
        input_var = rasterio.open(Input_file)

        # Extract year and month from the filename
        basename = os.path.basename(Input_file)  # "Bangladesh_Albedo_2024_06.tif"
        split_name = basename.split('_')  # Split by underscore
        year_month = f"{split_name[3][:]}_{split_name[4].split('.')[0]}"  # Extract '24_06'
    
        for i, row in BangShp.iterrows():
            # Clip the raster with the polygon geometry
            clipped_raster, clipped_transform = mask(input_var, [row.geometry], crop=True)
          
            for band_idx in range(clipped_raster.shape[0]): 
                band_data = clipped_raster[band_idx, :, :]  
                band_mean = band_data.mean()                
                truncated_name = f"Bd{band_idx}_{year_month}"  # E.g., "Band0_24_06"
                BangShp.at[i, truncated_name] = band_mean
                
    output_filename = f"UHIDistrict_dissolved_band_data_with_tiffs_Area2Rescaled-Test"
    Export_Shapefile(BangShp, getBangShp(), output_filename)

clippies()



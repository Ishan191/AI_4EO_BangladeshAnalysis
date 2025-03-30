# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:43:09 2025

@author: Dell
"""

import os
import rasterio
from rasterio.enums import Resampling

def rescale_tiff(input_file, output_file, scale_factor):
    with rasterio.open(input_file) as src:
        # Calculate new dimensions
        width = int(src.width * scale_factor)
        height = int(src.height * scale_factor)

        # Resample the image
        data = src.read(
            out_shape=(src.count, height, width),
            resampling=Resampling.bilinear  # Change this to a different method if needed
        )

        # Copy the metadata and adjust for new size
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            height=height,
            width=width,
            crs=src.crs
        )

        # Write the resampled data to the output file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data)

def process_all_tiffs_in_folder(folder_path, scale_factor):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, f"{filename}_rescaled")
            print(f"Processing {input_file}...")
            rescale_tiff(input_file, output_file, scale_factor)
            print(f"Saved {output_file}")

# Example usage
folder_path = "D:/Python Test Codes/AI4EO/Inputs_NDVI"  # Replace this with your folder path
scale_factor = 9  # From 1km to 9km
process_all_tiffs_in_folder(folder_path, scale_factor)

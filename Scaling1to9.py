"""
Created on Thu Mar 27 16:43:09 2025

@author: Dell
"""

import os
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

def rescale_tiff(input_file, output_file, scale_factor):
    with rasterio.open(input_file) as src:
        width = int(src.width * scale_factor)
        height = int(src.height * scale_factor)

        data = src.read(
            out_shape=(src.count, height, width),
            resampling=Resampling.bilinear  
        )
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,  
            height=height,
            width=width,
            crs=src.crs
        )

        new_transform = src.transform * Affine.scale(src.width / width, src.height / height)
        profile.update(transform=new_transform)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data)

def tiffs_in_folder(folder_path, scale_factor):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, f"rescaled_{filename}")
            rescale_tiff(input_file, output_file, scale_factor)

# Example usage
folder_path = r"D:\Python Test Codes\Inputs_UHI"  
scale_factor = 1/9  # From 1km to 9km
tiffs_in_folder(folder_path, scale_factor)

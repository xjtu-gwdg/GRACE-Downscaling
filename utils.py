"""
utils.py

Utility scripts for spatial data processing in groundwater modeling.

Includes:
- Mask application to raster files.
- Upscaling low-resolution groundwater storage data using interpolation.
- Saving raster data to GeoTIFF format.

This module supports preprocessing steps like masking ET data and generating
upscaled GWS data consistent with modeling inputs.

Author: [Shuitao Guo]
Date: [2025-07-11]
"""

import os
import rasterio
import numpy as np
from Model.data_reading import read_tif_file, get_month_list
from scipy.ndimage import zoom


def apply_mask_to_raster(mask_path, raster_dir, nodata_value=-9999):
    """
    Applies a binary mask to all GeoTIFF files in a directory, setting masked-out pixels to nodata.
    
    Args:
        mask_path (str): Path to the mask GeoTIFF file. Mask pixels with 0 indicate exclusion.
        raster_dir (str): Directory containing raster files to be masked.
        nodata_value (int or float): Value to assign to masked pixels. Default is -9999.
    
    Raises:
        ValueError: If mask and raster file dimensions do not match.
    """
    ...


def upscale_with_interpolation(gws_data, target_height, target_width, method="bilinear"):
    """
    Upscales a 2D numpy array to a target resolution using specified interpolation.
    
    Args:
        gws_data (np.ndarray): Input low-resolution array.
        target_height (int): Target height for upscaled output.
        target_width (int): Target width for upscaled output.
        method (str): Interpolation method, one of 'bilinear', 'nearest', or 'cubic'.
    
    Returns:
        np.ndarray: Upscaled data array.
    
    Raises:
        ValueError: If an unsupported interpolation method is specified.
    """
    ...


def write_tif(output_path, data, transform, crs, nodata=-9999):
    """
    Writes a 2D numpy array to a GeoTIFF file with spatial metadata.
    
    Args:
        output_path (str): File path to save the GeoTIFF.
        data (np.ndarray): 2D array to save.
        transform (affine.Affine): Affine transform for spatial referencing.
        crs (rasterio.crs.CRS): Coordinate reference system.
        nodata (int or float): NoData value to assign in output file.
    """
    ...


def generate_upscaled_gws(features_dirs, GWSBuffer_dir, output_dir, selected_features, start_year=2005, start_month=1, end_year=2019, end_month=12):
    """
    Generates high-resolution groundwater storage (GWS) data by upscaling low-resolution GWS rasters
    using bilinear interpolation to match the spatial resolution of selected feature rasters.
    
    Args:
        features_dirs (dict): Mapping of feature names to their directory paths.
        GWSBuffer_dir (str): Directory containing low-resolution GWS raster files.
        output_dir (str): Directory to save upscaled GWS rasters.
        selected_features (list): List of feature names to define the target resolution.
        start_year (int): Start year of data period. Default is 2005.
        start_month (int): Start month of data period. Default is January (1).
        end_year (int): End year of data period. Default is 2019.
        end_month (int): End month of data period. Default is December (12).
    
    Raises:
        FileNotFoundError: If required feature files for resolution reference are missing.
    """
    ...


def main():
    """
    Example script usage demonstrating:
    - Applying mask to ET rasters.
    - Upscaling GWS rasters.
    
    Adjust paths and parameters as needed.
    """
    ...

"""
dataloader.py

Utility functions for reading and preprocessing spatial and tabular data for groundwater modeling.

Includes functions to:
- Read GeoTIFF raster files and Excel measurement data.
- Generate a list of monthly timestamps.
- Interpolate missing values in raster data.
- Upscale low-resolution raster data using interpolation.
- Load and aggregate data including features, groundwater storage, measurements, and related raster buffers.

Author: [Shuitao Guo]
Date: [Date]
"""

import rasterio
import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy.interpolate import griddata
from scipy.ndimage import zoom


def read_tif_file(file_path):
    """
    Reads a GeoTIFF file and returns the first band as a numpy array along with spatial metadata.
    
    Args:
        file_path (str): Path to the GeoTIFF file.
        
    Returns:
        tuple: (data array, affine transform, coordinate reference system)
    """
    ...


def read_excel_file(file_path):
    """
    Reads an Excel file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Excel file.
        
    Returns:
        pandas.DataFrame: Loaded tabular data.
    """
    ...


def get_month_list(start_year, start_month, end_year, end_month):
    """
    Generates a list of month strings between two dates in the format 'YYYY_MM'.
    
    Args:
        start_year (int): Start year.
        start_month (int): Start month.
        end_year (int): End year.
        end_month (int): End month.
        
    Returns:
        list of str: List of months formatted as 'YYYY_MM'.
    """
    ...


def interpolate_nans(data):
    """
    Interpolates NaN values in a 2D numpy array using linear interpolation,
    falling back to nearest neighbor interpolation for remaining NaNs.
    
    Args:
        data (np.ndarray): 2D array possibly containing NaNs.
        
    Returns:
        np.ndarray: Array with NaNs filled by interpolation.
    """
    ...


def upscale_with_interpolation(gws_data, target_height, target_width, method="bilinear"):
    """
    Upscales a 2D array to a target size using specified interpolation method.
    
    Args:
        gws_data (np.ndarray): Input low-resolution array.
        target_height (int): Desired height after upscaling.
        target_width (int): Desired width after upscaling.
        method (str): Interpolation method - 'bilinear', 'nearest', or 'cubic'.
        
    Returns:
        np.ndarray: Upscaled array.
    """
    ...


def load_data(features_dirs, GWSBuffer_dir, gwl_dir, mask_file_path, RB_dir, selected_features=None):
    """
    Loads and preprocesses groundwater storage, features, measurements, and related raster buffer data.
    
    For each month between 2005 and 2019:
    - Reads low-resolution groundwater storage raster and interpolates missing values.
    - Loads selected feature rasters, filtering missing data.
    - Upscales groundwater storage data to match feature resolution.
    - Reads groundwater level measurements from Excel files.
    - Loads raster buffer data.
    - Aggregates all loaded data into a dictionary for further use.
    
    Args:
        features_dirs (dict): Mapping of feature names to their directory paths.
        GWSBuffer_dir (str): Directory path containing groundwater storage low-resolution raster files.
        gwl_dir (str): Directory path containing groundwater level Excel measurement files.
        mask_file_path (str): Path to mask raster file (not used internally here).
        RB_dir (str): Directory path containing raster buffer files.
        selected_features (list, optional): List of features to load. Loads all if None.
        
    Returns:
        dict: Contains lists of loaded data arrays and metadata, including:
            - 'gws_lr': list of low-resolution groundwater storage arrays (upscaled).
            - 'features': list of stacked feature arrays.
            - 'measurements': list of pandas DataFrames with measurements.
            - 'rb': list of raster buffer arrays.
            - 'months': list of month strings.
            - 'transform_hr': affine transform of high-res data.
            - 'crs_hr': coordinate reference system of high-res data.
    """
    ...

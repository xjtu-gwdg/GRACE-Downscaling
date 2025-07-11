"""
test.py

This script performs inference using a pretrained ResNet-based downscaling model for groundwater storage data.
It reads low-resolution groundwater storage data and auxiliary features, performs upscaling via interpolation,
runs the model to generate high-resolution predictions, and saves the results as GeoTIFF files.

Author: [Shuitao Guo]
Date: [2025-07-11]
"""

import torch
import numpy as np
import os
from model import ResNetDownscalingModel
from data_reading import read_tif_file, get_month_list
import rasterio
from scipy.ndimage import zoom


def upscale_with_interpolation(gws_data, target_height, target_width, method="bilinear"):
    """
    Upscales low-resolution groundwater storage data to a higher resolution using interpolation.
    
    Args:
        gws_data (np.ndarray): Input low-resolution 2D data array.
        target_height (int): Desired output height.
        target_width (int): Desired output width.
        method (str): Interpolation method, one of ["bilinear", "nearest", "cubic"].
    
    Returns:
        np.ndarray: Upscaled high-resolution data array.
    """
    ...


def test():
    """
    Runs the inference pipeline:
    
    - Loads pretrained model weights.
    - Reads input low-resolution groundwater storage and auxiliary feature raster files.
    - Upscales groundwater storage data to target resolution.
    - Prepares feature tensors.
    - Performs model prediction.
    - Saves predicted high-resolution results as GeoTIFFs.
    
    Prints status messages and skips missing files gracefully.
    """
    ...


if __name__ == '__main__':
    test()

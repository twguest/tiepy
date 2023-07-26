# -*- coding: utf-8 -*-
"""
Module for generating various intensity distributions.

This module provides functions to generate various intensity distributions, which can be used for
simulation and testing purposes. The functions within this module offer a variety of intensity patterns
that can be applied to 2D arrays to create different synthetic images.


"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_speckle_pattern(shape, intensity_range, speckle_size = 1):
    """
    Generate a speckle pattern (gaussian filtered random noise) array.

    :param shape: A tuple (rows, columns) specifying the shape of the array. [tuple]
    :param intensity_range: A tuple (min_intensity, max_intensity) specifying the range of intensity values. [tuple]
    :param speckle_size: width of gaussian filter 
    
    :return: A 2D numpy array containing the speckle pattern. [numpy.ndarray]
    """
    min_intensity, max_intensity = intensity_range
    speckle = np.random.uniform(min_intensity, max_intensity, shape)
    speckle = gaussian_filter(speckle, speckle_size)
    return speckle



def generate_gaussian_2d(shape, center, sigma):
    """
    Generate a 2D Gaussian array.

    :param shape: A tuple (rows, columns) specifying the shape of the array. [tuple]
    :param center: A tuple (row_center, col_center) specifying the center of the Gaussian. [tuple]
    :param sigma: A float representing the standard deviation of the Gaussian. [float]

    :return: A 2D numpy array containing the Gaussian. [numpy.ndarray]
    """
    rows, cols = shape
    row_center, col_center = center
    y, x = np.ogrid[:rows, :cols]
    gauss = np.exp(-((x - col_center) ** 2 + (y - row_center) ** 2) / (2 * sigma ** 2))
    return gauss

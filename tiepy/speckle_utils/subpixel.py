# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt



def normalize_image(image):
    # Subtract the mean and divide by standard deviation
    return (image - np.mean(image)) / np.std(image)


def calculate_correlation(reference_image,  subset_images, subset_centers):
    # Calculate the shift and center coordinates for the reference image
 
 
    correlations = []

    # Calculate shifts, centers, and correlation maps for each subset image with respect to the reference image
    for subset_image, subset_center in zip(subset_images, subset_centers):
        correlation = scipy.signal.correlate2d(subset_image, reference_image, mode='valid', boundary='symm')
        correlations.append(correlation)

    return correlations

def calculate_normalized_correlation(reference_image, subset_images, normalize=True):
    # Calculate the shift and center coordinates for the reference image
 
    correlations = []

    # Normalize the reference image if needed
    if normalize:
        reference_image = normalize_image(reference_image)

    # Calculate shifts, centers, and correlation maps for each subset image with respect to the reference image
    for subset_image in zip(subset_images):
        
        
        if normalize:
            subset_image = normalize_image(subset_image[0])
 
        
        correlation =scipy.signal.correlate2d(reference_image,subset_image, mode='valid', boundary='symm')
        correlations.append(correlation)
 

    return correlations


def fit_quadratic_and_find_peak(correlation_map, window_shape, subset_center, plot=False):
    
    window_width, window_height = window_shape
    cx, cy = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)

    # Define a window around the corrected center
    D = 5
    window = correlation_map[cx - window_height // D: cx + window_height // D + 1,
                             cy - window_width // D: cy + window_width // D + 1]
    
        
    plt.imshow(window)
    plt.show()
    
    # Create a grid for fitting (relative to the corrected center)
    x_fit = np.arange(-window_width // D, window_width // D + 1)
    y_fit = np.arange(-window_height // D, window_height // D + 1)
    x_fit, y_fit = np.meshgrid(x_fit, y_fit)

    # Flatten the window and corresponding coordinates
    z_fit = window.flatten()
    x_fit = x_fit.flatten()
    y_fit = y_fit.flatten()

    # Fit a quadratic function to the window data
    def quadratic_func(coords, a, b, c):
        x, y = coords
        return a * x**2 + b * x + c * y**2

    fit_params, _ = scipy.optimize.curve_fit(quadratic_func, (x_fit, y_fit), z_fit)
   
    # Find the peak coordinates of the fitted quadratic function
    x_peak = -fit_params[1] / (2 * fit_params[0])
    y_peak = -fit_params[2] / (2 * fit_params[0])

    # Calculate the peak coordinates relative to the corrected center
    dx, dy = subset_center


    if plot:
        plt.figure(figsize=(8, 6))
        #plt.scatter(x_fit, y_fit, c=z_fit, cmap='viridis', marker='o', label='Data')
        plt.xlabel('x')
        plt.ylabel('y')

        # Generate fitted data and plot it
        fitted_data = quadratic_func((x_fit, y_fit), *fit_params)
        #plt.scatter(x_fit, y_fit, c=fitted_data, cmap='inferno', marker='x', label='Fitted Data')
        nx = ny = int(np.sqrt(fitted_data.shape[0]))
        fitted_data = np.reshape(fitted_data, (nx,ny))
                                          
        plt.imshow(fitted_data,cmap='inferno', label='Fitted Data')

        plt.colorbar()

        # Plot the peak position
        plt.plot(x_peak+ (window_height // D), y_peak+(window_width // D), 'ro', markersize=10, label='Peak')
        plt.legend()
        plt.title('Quadratic Fit and Peak Detection')
        plt.show()


    return x_peak, y_peak


def process_subset_images(reference_image,
                          subset_images,
                          subset_centers,
                          plot_graphs=False,
                         method = calculate_normalized_correlation):

    
    # Initialize a dictionary to store the results for each subset image
    results = {
        'subset_centers': [],
        'subpixel_shifts': [],
        'shifts': []
    }
 
    # Calculate shifts, centers, and correlation maps with respect to the reference image
    correlations = method(reference_image,  subset_images, subset_centers)

    # Plot individual graphs for each subset image (if plot_graphs is True)

    for i, (subset_image, center, correlation) in enumerate(zip(subset_images, subset_centers, correlations), start=0):
        

        correlation_magnitude = np.abs(correlation)       
        max_corr_position = np.unravel_index(np.argmax(correlation_magnitude), correlation_magnitude.shape)
    
        corr_peaks = (max_corr_position[0] - correlation_magnitude.shape[0]//2, 
                             max_corr_position[1] - correlation_magnitude.shape[1]//2)
        
        if plot_graphs:
            plt.figure(figsize=(15, 5))

            # Plot the subset image
            ax1 = plt.subplot(1, 3, 1)
            im1 = ax1.imshow(subset_image, cmap='viridis', origin='upper')
            plt.colorbar(im1, ax=ax1, label='Intensity')
            ax1.set_title(f'Subset Image {i}')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')

            # Plot the larger image with the center position and subset center
            ax2 = plt.subplot(1, 3, 2)
            im2 = ax2.imshow(reference_image, cmap='viridis', origin='upper')
            plt.colorbar(im2, ax=ax2, label='Intensity')
            ax2.set_title('Larger Image with Speckle')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            ax2.scatter(center[1], center[0], color='blue', marker='x', s=100, label='Subset Center')
            ax2.legend()

            # Plot the magnitude of the correlation map with the center position and maximum correlation location
            ax3 = plt.subplot(1, 3, 3)
            im3 = ax3.imshow(correlation_magnitude, cmap='viridis', origin='upper')
            plt.colorbar(im3, ax=ax3, label='Magnitude')
            ax3.set_title(f'Correlation Map {i}')
            ax3.set_xlabel('Column Shift')
            ax3.set_ylabel('Row Shift')

            # Plot the maximum correlation location
            ax3.scatter(max_corr_position[0], max_corr_position[1], color='green', marker='x', s=100, label='Max Correlation')
            ax3.legend()

            plt.tight_layout()
            plt.show()

        # Calculate the corrected  
 
 

        # Fit a quadratic function to the correlation map and find its peak within a specified window        
        peak_x, peak_y = fit_quadratic_and_find_peak(correlation_magnitude,
                                                                         subset_image.shape,
                                                                         subset_centers[i],
                                                                         plot = True)
        peaks = (peak_x, peak_y)
 
        # Append the values to the corresponding lists in the results dictionary
        results['subset_centers'].append(center)
 
        results['subpixel_shifts'].append(peaks)
 
        results['shifts'].append(corr_peaks)

    return results
# -*- coding: utf-8 -*-
"""
This Python file contains several utility functions for image processing and correlation analysis.

Functions:
1. reshape_to_2d(data)
   Utility: Reshapes a 1D data array into a 2D array.
   
2. normalize_image(image)
   Utility: Normalizes an image by subtracting the mean and dividing by the standard deviation.

3. calculate_correlation(reference_image, subset_images, subset_centers)
   Utility: Calculates correlation maps between a reference image and a list of subset images.
            The correlation maps represent the similarity between the reference image and each subset image.

4. calculate_normalized_correlation(reference_image, subset_images, normalize=True)
   Utility: Calculates normalized correlation maps between a reference image and a list of subset images.
            The correlation maps represent the similarity between the reference image and each subset image
            after normalization.

5. fit_quadratic_and_find_peak(correlation_map, window_shape, subset_center, plot=False)
   Utility: Fits a quadratic function to a window of the correlation map and finds the peak coordinates.
            This function is used for subpixel peak detection.

6. process_subset_images(reference_image, subset_images, subset_centers, plot=False, method=calculate_normalized_correlation)
   Utility: Processes a set of images (subsets) by calculating shifts and correlation maps with respect to a reference image.
            The function returns the results of the correlation analysis for each subset image.

7. process_single_image(reference_image, sample_image, window_size, step_size, padding=0, plot=False)
   Main Function: Process a single image by extracting subsets, computing correlation, and subpixel shifts.

Note:
- These functions are designed for grayscale images represented as 2D numpy arrays.
- The 'plot' parameter allows for optional visualization of the correlation analysis results.
- The 'method' parameter in 'process_subset_images' can be used to specify a custom correlation function.
- The script uses the NumPy, SciPy, and Matplotlib libraries for image processing and visualization.
- The main function 'process_single_image' combines various utility functions to perform the image processing and analysis.

@author: twguest (trey.guest@xfel.eu)
"""


import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cv2 as cv

from tiepy.speckle.utils import get_subsets, reshape_to_2d, construct_arrays, generate_gaussian_mask

from tiepy.speckle.phase_retrieval import kottler
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable


def normalize_image(image):
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.

    This function normalizes an input image by subtracting its mean value and dividing by its standard deviation.

    Parameters:
        image: The input image to be normalized. [numpy.ndarray]

    Returns:
        numpy.ndarray: A 2D numpy array representing the normalized image.

    Example:
        >>> image = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        >>> normalized_image = normalize_image(image)
    """
    # Subtract the mean and divide by standard deviation
    return (image - np.mean(image)) / np.std(image)

def match_template(reference_image, subset_images, subset_centers, method = 3):
    """
    Perform template matching using OpenCV's cv2.matchTemplate function on multiple subset images.

    :param reference_image: numpy.ndarray
        The reference image to be matched against the subset images. It should be a 2D array (grayscale image).

    :param subset_images: list of numpy.ndarray
        A list containing the subset images to which the reference template will be matched.
        Each subset image should be a 2D array (grayscale image).

    :param method: int, optional
        The method to be used for matching. Refer to cv2.matchTemplate documentation for method options.
        Defaults to 3 (cv2.TM_CCOEFF_NORMED).

    :return: list of tuple
        A list containing the locations (x, y) of the best match in each subset image.
        Each location corresponds to the top-left corner of the matched region.

    :raises AssertionError:
        - If the 'method' parameter is not of integer type or outside the range [0, 5].

    :Note:
        - The 'reference_image' and each 'subset_image' should be grayscale images of the same dtype and have proper dimensions.
        - The 'method' parameter determines the matching method to be used. It should be an integer index in the range [0, 5].
        - The output list of tuples represents the locations of the best matches in the subset images.
    """
    assert isinstance(method, int) and 0 <= method <= 5, "method should be an integer index in the range [0, 5]"
    
    max_corr_position = []
    
    w = subset_images[0].shape[0]
    
    for itr, (subset_image, subset_center) in enumerate(zip(subset_images,subset_centers)):

        mask = 1        
        # mask = generate_gaussian_mask(*reference_image.T.shape,
        #                               subset_center[0],
        #                               subset_center[1],
        #                               1,
        #                               10)
 
        res = cv.matchTemplate(templ=subset_image, image=reference_image*mask, method=method)
        _, _, min_loc, max_loc = cv.minMaxLoc(res)
        # plt.imshow(reference_image*mask)
        # plt.scatter(*max_loc)
        # plt.show()
        # print(max_loc)
        # print(subset_center)
 
        max_corr_position.append(max_loc)
        
    return max_corr_position


def calculate_correlation(reference_image, subset_images, subset_centers):
    """
    Calculate correlation maps between a reference image and a list of subset images.

    This function calculates the correlation maps between a reference image and a list of subset images.
    The correlation maps represent the similarity between the reference image and each subset image.

    Parameters:
        reference_image: The reference image to which the subset images will be compared. [numpy.ndarray]
                         Should be a 2D numpy array representing the grayscale image.
        subset_images: A list of subset images to be compared with the reference image. [List[numpy.ndarray]]
                       Each subset image should be a 2D numpy array of the same dtype and dimensions as the reference image.
        subset_centers: A list of tuples (row_center, col_center) specifying the centers of the subset images. [List[tuple]]
                        Each tuple represents the center position (row, column) of the corresponding subset image.

    Returns:
        List[numpy.ndarray]: A list of 2D numpy arrays representing the correlation maps for each subset image.
                             Each correlation map represents the similarity between the reference image and the respective subset image.

    Example:
        >>> reference_img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> subset_imgs = [np.array([[0.1, 0.2], [0.4, 0.5]]), np.array([[0.8, 0.9], [0.5, 0.6]])]
        >>> centers = [(0, 0), (1, 1)]
        >>> correlation_maps = calculate_correlation(reference_img, subset_imgs, centers)
    """
    # Calculate the shift and center coordinates for the reference image

    correlations = []

    # Calculate shifts, centers, and correlation maps for each subset image with respect to the reference image
    for subset_image, subset_center in zip(subset_images, subset_centers):
        correlation = scipy.signal.correlate2d(subset_image, reference_image, mode="valid", boundary="symm")
        correlations.append(correlation)

    return correlations


def calculate_normalized_correlation(reference_image, subset_images, normalize=True):
    """
    Calculate normalized correlation maps between a reference image and a list of subset images.

    This function calculates the normalized correlation maps between a reference image and a list of subset images.
    The correlation maps represent the similarity between the reference image and each subset image after normalization.

    Parameters:
        reference_image: The reference image to which the subset images will be compared. [numpy.ndarray]
                         Should be a 2D numpy array representing the grayscale image.
        subset_images: A list of subset images to be compared with the reference image. [List[numpy.ndarray]]
                       Each subset image should be a 2D numpy array of the same dtype and dimensions as the reference image.
        normalize: A boolean flag to determine whether to normalize the reference and subset images. [bool, optional]
                   If True, the images will be normalized before calculating the correlation. Defaults to True.

    Returns:
        List[numpy.ndarray]: A list of 2D numpy arrays representing the normalized correlation maps for each subset image.
                             Each correlation map represents the similarity between the reference image and the respective subset image.

    Example:
        >>> reference_img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> subset_imgs = [np.array([[0.1, 0.2], [0.4, 0.5]]), np.array([[0.8, 0.9], [0.5, 0.6]])]
        >>> normalized_correlation_maps = calculate_normalized_correlation(reference_img, subset_imgs, normalize=True)
    """
    # Calculate the shift and center coordinates for the reference image

    correlations = []

    # Normalize the reference image if needed
    if normalize:
        reference_image = normalize_image(reference_image)

    # Calculate shifts, centers, and correlation maps for each subset image with respect to the reference image
    for subset_image in tqdm(zip(subset_images)):
        if normalize:
            subset_image = normalize_image(subset_image[0])

        correlation = scipy.signal.correlate2d(reference_image, subset_image, mode="valid", boundary="symm")
        correlations.append(correlation)

    return correlations


def fit_gaussian_and_find_peak(correlation_map, window_shape, subset_center, plot=False):
    """
    Fit a gaussian function to a window of the correlation map and find the peak coordinates.

    This function fits a quadratic function to a window of the correlation map centered around the
    specified subset_center. It then identifies the peak coordinates within the window.

    Parameters:
        correlation_map: The correlation map used for fitting and peak detection. [numpy.ndarray]
                         Should be a 2D numpy array representing the correlation map.
        window_shape: A tuple (window_width, window_height) specifying the dimensions of the fitting window. [tuple]
                      The window is centered around the subset_center and should be smaller than the correlation_map.
        subset_center: A tuple (center_x, center_y) specifying the center of the subset for the window. [tuple]
                       The subset_center should correspond to the center of the region of interest within the correlation_map.
        plot: A boolean flag to determine whether to plot the fitting results. [bool, optional]
              If True, the function will generate a plot showing the measured and fitted intensity data.
              Defaults to False.

    Returns:
        tuple: A tuple (x_peak, y_peak) representing the peak coordinates in the fitted quadratic function. [tuple]
               The coordinates are relative to the corrected center of the subset.

    Notes:
        - The correlation_map should be a 2D numpy array with proper dimensions.
        - The window_shape should be smaller than the dimensions of the correlation_map for accurate fitting.
        - The subset_center should correspond to the center of the region of interest within the correlation_map.
        - The function uses a quadratic fitting method and a Gaussian function for fitting the data.
        - The peak coordinates returned are relative to the corrected center of the subset.
        - If plot is set to True, the function will generate a 3D plot displaying the measured and fitted intensity data.
    """
    window_width, window_height = window_shape
    cx, cy = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)

    # Define a window around the corrected center
    D = 5
    window = correlation_map[
        cx - window_height // D : cx + window_height // D + 1, cy - window_width // D : cy + window_width // D + 1
    ]

    # Create a grid for fitting (relative to the corrected center)
    x_fit = np.arange(-window_width // D, window_width // D + 1)
    y_fit = np.arange(-window_height // D, window_height // D + 1)
    x_fit, y_fit = np.meshgrid(x_fit, y_fit)

    # Flatten the window and corresponding coordinates
    z_fit = window.flatten()
    x_fit = x_fit.flatten()
    y_fit = y_fit.flatten()

    ### Note: This could equally be done with a quadratic fn, maybe work for
    ###       another day/person.

    # # Fit a quadratic function to the window data
    # def quadratic_func(coords, a, b, c):
    #     x, y = coords
    #     return a * x**2 + b * x + c * y**2

    # fit_params, _ = scipy.optimize.curve_fit(quadratic_func, (x_fit, y_fit), z_fit)

    # # Find the peak coordinates of the fitted quadratic function
    # x_peak = -fit_params[1] / (2 * fit_params[0])
    # y_peak = -fit_params[2] / (2 * fit_params[0])

    # Define the 2D Gaussian function
    def gaussian_func(coords, A, x0, y0, sigma_x, sigma_y):
        x, y = coords
        return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2)))

    # Initial guess for the parameters (you may want to provide better initial values)
    initial_guess = [1.0, 0.0, 0.0, 1.0, 1.0]

    try:
        # Use curve_fit to find the optimal parameters that fit the Gaussian function
        fit_params, _ = scipy.optimize.curve_fit(gaussian_func, (x_fit, y_fit), z_fit, p0=initial_guess)

        # Find the peak coordinates and peak intensities of the fitted Gaussian function
        A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit = fit_params

        # Calculate the peak coordinates
        x_peak = x0_fit
        y_peak = y0_fit

        # Calculate the peak coordinates relative to the corrected center
        dx, dy = subset_center

        if plot:
            # Generate fitted data and plot it
            fitted_data = gaussian_func((x_fit, y_fit), *fit_params)

            # Reshape the variables to 2D
            variables_list = [x_fit, y_fit, z_fit, fitted_data]  # Add all your variables here
            variables_list = [reshape_to_2d(variable) for i, variable in enumerate(variables_list)]
            [x_fit, y_fit, z_fit, fitted_data] = variables_list

            # Create a 1x2 figure
            fig = plt.figure(figsize=(16, 6))

            # First subplot: Measured Intensity
            ax1 = fig.add_subplot(121, projection="3d")
            surf1 = ax1.plot_surface(x_fit, y_fit, z_fit, cmap="bone")
            ax1.set_xlabel("x", fontsize=14)
            ax1.set_ylabel("y", fontsize=14)
            ax1.set_zlabel("z", fontsize=14)
            ax1.tick_params(axis="both", which="major", labelsize=12)
            ax1.set_title("Measured Intensity", fontsize=16)
            cbar1 = fig.colorbar(surf1, ax=ax1, pad=0.1, shrink=0.6)  # Adjust the shrink parameter here
            cbar1.ax.tick_params(labelsize=12)
            cbar1.set_label("Intensity", fontsize=14)

            # Second subplot: Fitted Intensity
            ax2 = fig.add_subplot(122, projection="3d")
            surf2 = ax2.plot_surface(x_fit, y_fit, fitted_data, cmap="inferno", alpha=0.7)
            ax2.set_xlabel("x", fontsize=14)
            ax2.set_ylabel("y", fontsize=14)
            ax2.set_zlabel("z", fontsize=14)
            ax2.tick_params(axis="both", which="major", labelsize=12)
            ax2.set_title("Fitted Intensity", fontsize=16)
            cbar2 = fig.colorbar(surf2, ax=ax2, pad=0.1, shrink=0.6)  # Adjust the shrink parameter here
            cbar2.ax.tick_params(labelsize=12)
            cbar2.set_label("Intensity", fontsize=14)

            # Adjust view angles for better visualization
            ax1.view_init(elev=20, azim=-40)
            ax2.view_init(elev=20, azim=-40)

            plt.suptitle("Quadratic Fit and Peak Detection", fontsize=18)
            plt.tight_layout()
            plt.show()

    except RuntimeError:
        x_peak = 0
        y_peak = 0

    return x_peak, y_peak


def process_subset_images(
    reference_image, subset_images, subset_centers, plot=False, method=calculate_normalized_correlation, subpixel = True,
):
    """
    Process a set images (nominally subsets of a larger image) by calculating shifts and correlation maps with respect to a reference image.

    This function processes a list of subset images by calculating the subpixel shifts and correlation maps
    with respect to a given reference image. It utilizes a specified correlation method to perform the
    calculations.

    :param reference_image: numpy.ndarray
        The reference image to which the subset images will be compared. Should be a 2D numpy array representing the grayscale image.

    :param subset_images: List[numpy.ndarray]
        A list of subset images to be processed. Each subset image should be a 2D numpy array of the same dtype and dimensions as the reference image.

    :param subset_centers: List[tuple]
        A list of tuples (row_center, col_center) specifying the centers of the subset images.
        Each tuple represents the center position (row, column) of the corresponding subset image.

    :param plot: bool, optional
        A boolean flag to determine whether to plot individual graphs for each subset image.
        If True, plots will be generated for each subset image. Defaults to False.

    :param method: function, optional
        The method used to calculate the correlation between the reference image and subset images.
        This function should take the reference image, subset images, and subset centers as inputs and return
        a list of correlation maps for each subset. Defaults to `calculate_normalized_correlation`, a predefined function.

    :param subpixel: bool, optional
        A boolean flag to determine whether to compute subpixel shifts. If True, the function will compute
        subpixel shifts between the reference image and each subset image. Default is False.

    :return: dict
        A dictionary containing the processed results for each subset image.
        The dictionary includes the following keys:

            - 'subset_centers': A list of tuples (row_center, col_center) representing the centers of the subset images.

            - 'subpixel_shifts': A list of tuples (subpixel_shift_x, subpixel_shift_y) indicating the subpixel shifts
              between the reference image and each subset image. This key will be present only if subpixel is True.

            - 'shifts': A list of tuples (shift_x, shift_y) indicating the shifts between the reference image and
              each subset image, computed as the peak positions in the correlation maps.

    :raises:
        AssertionError: If the method is 'match_template' and subpixel is set to True. Subpixel resolution is not compatible with the 'match_template' method.

    :Example:
        >>> reference_img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> subset_imgs = [np.array([[0.1, 0.2], [0.4, 0.5]]), np.array([[0.8, 0.9], [0.5, 0.6]])]
        >>> centers = [(0, 0), (1, 1)]
        >>> results = process_subset_images(reference_img, subset_imgs, centers, plot=True)

    :Note:
        - The reference image and subset images should be grayscale images of the same dtype and proper dimensions.
        - The subset_centers should correspond to the centers of the respective subset images.
        - The method parameter allows for custom correlation methods to be used, with a default method provided.
        - If plot is set to True, individual plots will be generated for each subset image, displaying the subset image,
          reference image, and magnitude of the correlation map.
        - Subpixel shifts will be calculated only if subpixel is True.
    """

    # Initialize a dictionary to store the results for each subset image
    results = {"subset_centers": [], "subpixel_shifts": [], "shifts": []}

    if method == match_template:
        assert subpixel is False, "subpixel resolution not compatible with method 'match_template' "
        
    if method in [calculate_normalized_correlation, calculate_correlation]:
        # Calculate shifts, centers, and correlation maps with respect to the reference image
        correlations = method(reference_image, subset_images, subset_centers)
    elif method == match_template:
        max_corr_position = method(reference_image, subset_images, subset_centers, method = 3)
        correlations = max_corr_position ### quick fix
        
    # Plot individual graphs for each subset image (if plot_graphs is True)

    for i, (subset_image, center, correlation) in enumerate(
        zip(subset_images, subset_centers, correlations), start=0
    ):
        
        w = subset_image.shape[0]
        
        if method in [calculate_normalized_correlation, calculate_correlation]:
            correlation_magnitude = np.abs(correlation)
            max_corr_position = np.unravel_index(np.argmax(correlation_magnitude), correlation_magnitude.shape)
        
            corr_peaks = (max_corr_position[0] + w // 2 - center[0], max_corr_position[1] + w // 2 - center[1])
            
        elif method == match_template:
            #corr_peaks = (max_corr_position[i][0],max_corr_position[i][1])
            corr_peaks = (max_corr_position[i][1] - center[0] + w//2, max_corr_position[i][0] - center[1] + + w//2)
            
        # Plot individual graphs for each subset image (if plot is True)
        if plot:
            
            assert method != match_template, "plotting not supported for method = match_template"

            plt.figure(figsize=(15, 5))

            # Plot the subset image
            ax1 = plt.subplot(1, 3, 1)
            im1 = ax1.imshow(subset_image, cmap="viridis", origin="upper")
            ax1.set_title(f"Subset Image {i}", fontsize=16)
            ax1.set_xlabel("x (pixels)", fontsize=16)
            ax1.set_ylabel("y (pixels)", fontsize=16)
            ax1.tick_params(axis="both", which="major", labelsize=14)
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.set_label("Intensity", fontsize=14)
            cbar1.ax.tick_params(labelsize=12)

            # Plot the larger image with the center position and subset center
            ax2 = plt.subplot(1, 3, 2)
            im2 = ax2.imshow(reference_image, cmap="viridis", origin="upper")
            ax2.set_title("Reference Speckle", fontsize=16)
            ax2.set_xlabel("x (pixels)", fontsize=16)
            ax2.set_ylabel("y (pixels)", fontsize=16)
            ax2.scatter(center[1], center[0], color="blue", marker="x", s=100, label="Subset Center")
            ax2.legend()
            ax2.tick_params(axis="both", which="major", labelsize=14)
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.1)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_label("Intensity", fontsize=14)
            cbar2.ax.tick_params(labelsize=12)

            # Plot the magnitude of the correlation map with the center position and maximum correlation location
            ax3 = plt.subplot(1, 3, 3)
            im3 = ax3.imshow(correlation_magnitude, cmap="viridis", origin="upper")
            ax3.set_title(f"Correlation Map {i}", fontsize=16)
            ax3.set_xlabel("$\Delta x$", fontsize=16)
            ax3.set_ylabel("$\Delta y$", fontsize=16)
            ax3.scatter(
                max_corr_position[1],
                max_corr_position[0],
                color="green",
                marker="x",
                s=100,
                label="Max Correlation",
            )
            ax3.legend()
            ax3.tick_params(axis="both", which="major", labelsize=14)
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="5%", pad=0.1)
            cbar3 = plt.colorbar(im3, cax=cax3)
            cbar3.set_label("Magnitude", fontsize=14)
            cbar3.ax.tick_params(labelsize=12)

            plt.tight_layout()
            plt.show()

        
        if subpixel:
            
            # Fit a quadratic function to the correlation map and find its peak within a specified window
            peak_x, peak_y = fit_gaussian_and_find_peak(
                correlation_magnitude, subset_image.shape, subset_centers[i], plot=plot
            )
            peaks = (peak_x + corr_peaks[0], peak_y + corr_peaks[1])
            results["subpixel_shifts"].append(peaks)

        # Append the values to the corresponding lists in the results dictionary
        results["subset_centers"].append((center[0], center[1]))
        results["shifts"].append(corr_peaks)

    return results


def process_single_image(reference_image, sample_image, window_size, step_size, padding=0, method=calculate_normalized_correlation, subpixel = False, plot=False, extra_metadata = {}):
    """
    Process a single image by extracting subsets, computing correlation, and subpixel shifts.

    :param reference_image: numpy.ndarray
        The reference image to which the subset images will be compared.

    :param sample_image: numpy.ndarray
        The input 2D image array to be processed.

    :param window_size: int
        The window size for cropping subsets. The window will be a square of size window_size x window_size.

    :param step_size: int
        The step size for moving the window in both the x and y directions.

    :param padding: int, optional
        The padding distance from the image boundaries. The window will not be drawn within this distance
        from the edges of the image. Default is 0.

    :param method: callable, optional
        The method to be used for computing the correlation between the reference and subset images.
        It should be a callable function that accepts two numpy.ndarray arguments and returns a 2D correlation map.
        Defaults to `calculate_normalized_correlation`.

    :param subpixel: bool, optional
        A boolean flag to determine whether to compute subpixel shifts. If True, the function will compute
        subpixel shifts between the reference image and each subset image. Default is False.

    :param plot: bool, optional
        A boolean flag to determine whether to plot individual graphs for each subset image. Default is False.

    :param extra_metadata: dict, optional
        Additional metadata to include in the results dictionary. This should be provided as a dictionary
        containing key-value pairs representing the metadata. Default is an empty dictionary.

    :return: dict
        A dictionary containing the processed results for each subset image.
        The dictionary includes the following keys:

            - 'coords_x': numpy.ndarray
                A 1D array containing the horizontal (x) coordinates of the centers of the subset images.

            - 'coords_y': numpy.ndarray
                A 1D array containing the vertical (y) coordinates of the centers of the subset images.

            - 'shifts_x': numpy.ndarray
                A 1D array containing the horizontal (x) shifts between the reference image and each subset image,
                computed as the peak positions in the correlation maps.

            - 'shifts_y': numpy.ndarray
                A 1D array containing the vertical (y) shifts between the reference image and each subset image,
                computed as the peak positions in the correlation maps.

            - 'subpixel_shifts_x': numpy.ndarray (optional)
                A 1D array containing the horizontal (x) subpixel shifts between the reference image and each subset image.
                This key will be present in the dictionary only if the 'subpixel' parameter is set to True.

            - 'subpixel_shifts_y': numpy.ndarray (optional)
                A 1D array containing the vertical (y) subpixel shifts between the reference image and each subset image.
                This key will be present in the dictionary only if the 'subpixel' parameter is set to True.

            - Additional Metadata (optional):
                This dictionary can include any additional metadata specified by the 'extra_metadata' parameter.

    :Example:
        >>> import numpy as np
        >>> reference_image = np.random.rand(100, 100)  # Example 2D reference image of size 100x100
        >>> sample_image = np.random.rand(300, 300)  # Example 2D sample image of size 300x300
        >>> window_size = 50
        >>> step_size = 10
        >>> padding = 20
        >>> results = process_single_image(reference_image, sample_image, window_size, step_size, padding, plot=True)

    :Note:
        - The 'reference_image' and 'sample_image' should be 2D numpy arrays with proper dimensions.
        - The 'window_size' and 'step_size' should be positive integers.
        - The 'padding' should be a non-negative integer.
        - The 'method' parameter determines the correlation method used and should be a callable function.
    """
    
    assert type(extra_metadata) == dict, "Metadata should be in dictionary format"
    
    if method == match_template:
        assert subpixel is False, "subpixel resolution not compatible with method 'match_template' "
        
    subset, centers = get_subsets(sample_image, window_size=window_size, step_size=step_size, padding=padding)
    shift_results = process_subset_images(reference_image, subset_images=subset, subpixel = subpixel, method = method, subset_centers=centers, plot=plot)

    results = {}
    
    # Add metadata to results
    results.update(extra_metadata)
    
    # Convert list of tuples to NumPy arrays for better structure
    coords_x, coords_y, shifts_x, shifts_y = construct_arrays(
        shift_results["subset_centers"], shift_results["shifts"]
    )
    
    if subpixel:
        _, _, subpixel_shifts_x, subpixel_shifts_y = construct_arrays(
            shift_results["subset_centers"], shift_results["subpixel_shifts"]
        )

    # Store NumPy arrays in the dictionary with descriptive keys
    results["coords_x"] = coords_x  # Horizontal (x) coordinates of the centers of the subset images.
    results["coords_y"] = coords_y  # Vertical (y) coordinates of the centers of the subset images.
    results["shifts_x"] = shifts_x  # Horizontal (x) shifts between the reference image and each subset image.
    results["shifts_y"] = shifts_y  # Vertical (y) shifts between the reference image and each subset image.
    
    if subpixel == True:
            
        results[
            "subpixel_shifts_x"
        ] = subpixel_shifts_x  # Horizontal (x) subpixel shifts between the reference image and each subset image.
        results[
            "subpixel_shifts_y"
        ] = subpixel_shifts_y  # Vertical (y) subpixel shifts between the reference image and each subset image.
    
    
    return results

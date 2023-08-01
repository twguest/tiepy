# -*- coding: utf-8 -*-

import numpy as np


def reshape_to_2d(data):
    """
    Reshape a 1D data array into a 2D array.

    This function takes a 1D data array and reshapes it into a 2D array of dimensions (nx, ny), where nx and ny
    are the square roots of the length of the input data.

    Parameters:
        data: The 1D data array to be reshaped. [numpy.ndarray]

    Returns:
        numpy.ndarray: A 2D numpy array of dimensions (nx, ny) representing the reshaped data.

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> reshaped_data = reshape_to_2d(data)
    """
    nx, ny = int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0]))
    return np.reshape(data, (nx, ny))


def get_subsets(image, window_size, step_size, padding=0):
    """
    Extract subsets of an input image centered around each pixel with a specified window size.

    :param image: numpy.ndarray
        The input 2D image array.

    :param window_size: int
        The window size for cropping subsets. The window will be a square of size window_size x window_size.

    :param step_size: int
        The step size for moving the window in both the x and y directions.

    :param padding: int, optional
        The padding distance from the image boundaries. The window will not be drawn within this distance
        from the edges of the image. Default is 0.

    :return:
        subsets : list of numpy.ndarray
            A list containing the cropped subsets of the input image.

        centers : list of tuple (int, int)
            A list containing tuples of (i, j) representing the center positions of each cropped window.

    :Example:
        >>> import numpy as np
        >>> image = np.random.rand(100, 100)  # Example 2D image of size 100x100
        >>> subsets, centers = get_subsets(image, window_size=50, step_size=10, padding=20)
        >>> print(len(subsets))  # Number of cropped subsets
        9
        >>> print(len(centers))  # Number of center positions
        9

    :Description:
        This function extracts subsets of the input image by moving a square window of specified size across the entire image.
        The function returns the cropped subsets and the corresponding center positions as a list of tuples.
        The 'image' parameter should be a 2D numpy array representing a grayscale image.
        The 'window_size' parameter determines the size of the square window used for cropping.
        The 'step_size' parameter specifies the distance to move the window in both the x and y directions.
        The 'padding' parameter specifies the distance from the image boundaries within which the window should not be drawn.
        The function generates subsets and centers by scanning the entire image with the window.
        For each position (i, j) in the image where the window can fit without going beyond the padding area,
        a cropped subset centered at (i, j) is obtained.
        The subsets are stored in a list, and the center positions are represented as tuples (i, j) and stored in another list.

    :Note:
        - The input 'image' should be a 2D numpy array with proper dimensions.
        - The 'window_size' and 'step_size' should be positive integers.
        - The 'padding' should be a non-negative integer.
        - The function does not handle padding or boundary cases.

    """
    nx, ny = image.shape
    subsets = []
    centers = []

    half_w = window_size // 2

    for i in range(half_w + padding, nx - half_w - padding, step_size):
        for j in range(half_w + padding, ny - half_w - padding, step_size):
            subset = image[i - half_w : i + half_w, j - half_w : j + half_w]
            subsets.append(subset)
            centers.append((i, j))

    return subsets, centers


def calc_subsets_size(image, window_size, step_size, padding=0):
    nx, ny = image.shape

    half_w = window_size // 2

    x = len([i for i in range(half_w + padding, nx - half_w - padding, step_size)])
    y = len([j for j in range(half_w + padding, ny - half_w - padding, step_size)])

    print("Number of Subsets: {}".format(x * y))


def construct_arrays(coordinates, measurements):
    """
    Construct numpy arrays from sorted scan coordinates and measurements.

    :param coordinates: List[tuple]
        A list of tuples representing the scan coordinates. Each tuple should be in the format (y, x),
        where y and x are the row and column indices, respectively.

    :param measurements: List[tuple]
        A list of tuples representing the measurements corresponding to the scan coordinates.
        Each tuple should contain the measurement values in the format (measurement1, measurement2).

    :return: tuple
        A tuple (array_x, array_y, array_m1, array_m2) representing the constructed numpy arrays.

        - array_x: numpy.ndarray
            A 2D numpy array containing the x-coordinates corresponding to the scan coordinates.
            The shape of this array is determined by the number of unique x values and rows in the scan.
            Each element (i, j) in the array corresponds to the x-coordinate of the measurement at position (i, j).

        - array_y: numpy.ndarray
            A 2D numpy array containing the y-coordinates corresponding to the scan coordinates.
            The shape of this array is determined by the number of unique y values and columns in the scan.
            Each element (i, j) in the array corresponds to the y-coordinate of the measurement at position (i, j).

        - array_m1: numpy.ndarray
            A 2D numpy array containing the measurements for parameter 1.
            The shape of this array is determined by the number of unique x values and rows in the scan.
            Each element (i, j) in the array represents the measurement for parameter 1 at position (i, j).

        - array_m2: numpy.ndarray
            A 2D numpy array containing the measurements for parameter 2.
            The shape of this array is determined by the number of unique x values and rows in the scan.
            Each element (i, j) in the array represents the measurement for parameter 2 at position (i, j).

    :Example:
        >>> coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]
        >>> measurements = [(10, 20), (30, 40), (50, 60), (70, 80)]
        >>> array_x, array_y, array_m1, array_m2 = construct_arrays(coordinates, measurements)
        >>> print(array_x)
        [[0 1]
         [0 1]]
        >>> print(array_y)
        [[0 0]
         [1 1]]
        >>> print(array_m1)
        [[10 30]
         [50 70]]
        >>> print(array_m2)
        [[20 40]
         [60 80]]

    :Notes:
        - The function assumes that the coordinates and measurements are provided as lists of tuples.
        - The input coordinates should be sorted in a row-major order for accurate construction of the numpy arrays.
        - The resulting arrays will have the same number of rows as unique y values and the same number of columns as unique x values.
        - The order of the elements in the measurements list should correspond to the order of coordinates.
    """
    unique_x = np.unique([coord[1] for coord in coordinates])
    unique_y = np.unique([coord[0] for coord in coordinates])

    n, m = len(unique_y), len(unique_x)

    array_x, array_y = np.meshgrid(unique_x, unique_y, indexing="ij")

    array_m1 = np.reshape([meas[1] for meas in measurements], (n, m))
    array_m2 = np.reshape([meas[0] for meas in measurements], (n, m))

    return array_x.T, array_y.T, array_m1, array_m2

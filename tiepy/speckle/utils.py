# -*- coding: utf-8 -*-

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
            subset = image[i - half_w:i + half_w, j - half_w:j + half_w]
            subsets.append(subset)
            centers.append((i, j))

    return subsets, centers

import numpy as np

def get_windows(arr, nx, s):
    """
    Extract subsets and their center positions from a square 2D array.

    :param arr: The input square 2D array from which subsets will be extracted.
                The array must have the same number of rows and columns (i.e., it must be square).
    :param nx: The size of the subsets to be extracted. Should be smaller than or equal to the array dimensions.
    :param s: The step size for extracting subsets. Determines the distance between consecutive subsets.

    :return: A list containing the subsets extracted from the input array.
             Each subset is a 2D numpy array of size (nx x nx).
    :return: A list containing the center positions of each subset in the input array.
             Each center position is represented as a tuple (center_x, center_y).

    :raises ValueError: If the input array is not square or if nx is greater than the array dimensions.
    """
    
    # Get the shape of the input array
    n, m = arr.shape

    # Check if the input array is square
    if n != m:
        raise ValueError("Input array must be square.")

    # Check if nx is smaller than or equal to the array dimensions
    if nx > n:
        raise ValueError("nx must be smaller than or equal to the array dimensions.")

    # Calculate the number of subsets in each dimension
    subsets_per_dim = (n - nx) // s + 1

    # Initialize empty lists to store the subsets and their center positions
    subsets = []
    centers = []

    # Generate the subsets and their center positions
    for i in range(subsets_per_dim):
        for j in range(subsets_per_dim):
            subset = arr[i * s : i * s + nx, j * s : j * s + nx]
            center_i = i * s 
            center_j = j * s 
            center = (center_j, center_i)
            subsets.append(subset)
            centers.append(center)

    return subsets, centers



def generate_gaussian_mask(nx, ny, x0, y0, sigma, w):
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    dist_squared = (X - x0)**2 + (Y - y0)**2
    mask = np.exp(-dist_squared / (2 * sigma**2))
    mask[dist_squared <= w**2] = 1
    mask /= np.max(mask)
    return mask
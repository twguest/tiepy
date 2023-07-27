# -*- coding: utf-8 -*-

import imageio
import numpy as np

import h5py as h5

def save_dict_to_hdf5(file_path, data_dict, key):
    """
    Save a dictionary to an HDF5 file.

    :param file_path: str
        The path to the HDF5 file to be created or overwritten.

    :param data_dict: dict
        The dictionary containing the data to be saved.

    :param key: str
        The key in the HDF5 file where the dictionary will be stored.

    :return: None

    :Example:
        >>> data_dict = {'subset_centers': [(50, 175), (50, 200)],
        ...              'subpixel_shifts': [(-0.2189317094326493, 3.912062042065753), (0, 3)],
        ...              'shifts': [(0, 4), (0, 3)]}
        >>> save_dict_to_hdf5('output.h5', data_dict, 'my_data')

    :Description:
        This function saves a dictionary to an HDF5 file at a specified key location.
        The function uses the 'h5py' library to create or open the HDF5 file.
        The 'file_path' parameter should be a string representing the path to the HDF5 file.
        The 'data_dict' parameter should be a dictionary containing the data to be saved.
        The 'key' parameter should be a string representing the key in the HDF5 file
        where the dictionary will be stored.

    :Note:
        - The 'data_dict' should contain only data that can be stored in an HDF5 file.
          This typically includes basic data types, lists, and arrays.
    """
    
    with h5.File(file_path, 'a') as h5_file:
        # Create or open the HDF5 file
        if key in h5_file:
            del h5_file[key]  # Delete the key if it already exists

        # Save the dictionary to the specified key
        h5_file.create_group(key)
        for k, v in data_dict.items():
            h5_file[key].create_dataset(k, data=v)


def load_tiff_as_npy(file_path):
    """
    Load a TIFF file as a NumPy array.

    Parameters:
        file_path (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    try:
        image = imageio.imread(file_path)
        return np.array(image)
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None


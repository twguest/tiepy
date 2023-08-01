# -*- coding: utf-8 -*-
import os

import imageio
import numpy as np

import h5py as h5


def print_h5_keys(filename):
    """
    Print the keys present in an HDF5 file.

    :param filename: str
        The name of the HDF5 file from which keys will be retrieved and printed.

    :return: None

    :Example:
        >>> print_h5_keys('data.h5')
        Keys in the HDF5 file:
        dataset1
        dataset2
        dataset3
    """
    with h5.File(filename, "r") as f:
        keys = list(f.keys())
    print("Keys in the HDF5 file:")
    for key in keys:
        print(key)


def get_keys(filename):
    """
    Get the keys present in an HDF5 file.

    :param filename: str
        The name of the HDF5 file from which keys will be retrieved.

    :return: list
        A list containing the keys present in the HDF5 file.

    :Example:
        >>> keys = get_keys('data.h5')
        >>> print(keys)
        ['dataset1', 'dataset2', 'dataset3']
    """
    with h5.File(filename, "r") as f:
        keys = list(f.keys())
    return keys


def load_key_from_virtual_h5(virtual_file, key):
    # Open the virtual HDF5 file
    with h5.File(virtual_file, "r") as f:
        # Check if the key exists in the file
        if key not in f:
            raise ValueError(f"Key '{key}' not found in the HDF5 file.")

        # Get the group corresponding to the key
        group = f[key]

        # Extract all datasets within the group into a dictionary
        data = {name: np.array(dataset) for name, dataset in group.items()}

    return data


def create_virtual_h5(directory, output_filename):
    """
    Create a virtual HDF5 file from multiple individual HDF5 files in a directory.

    :param directory: str
        The directory containing the HDF5 files to be combined.

    :param output_filename: str
        The name of the virtual HDF5 file to be created.

    :return: None

    :Example:
        >>> create_virtual_h5('data_directory', 'virtual_data.h5')
        Virtual dataset created in virtual_data.h5
    """
    # Get list of all .h5 files in the directory
    filepaths = [
        os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".h5")
    ]

    # Create a dictionary to hold the virtual layouts for each group
    virtual_layouts = {}

    # For each file, create a virtual source for each dataset and add it to the corresponding layout
    for filepath in filepaths:
        with h5.File(filepath, "r") as f:
            group_name = os.path.splitext(os.path.basename(filepath))[
                0
            ]  # Use the filename (without extension) as the group name
            virtual_layouts[group_name] = {}
            for key in f.keys():
                vsource = h5.VirtualSource(filepath, key, shape=f[key].shape, dtype=f[key].dtype)
                virtual_layouts[group_name][key] = h5.VirtualLayout(shape=vsource.shape, dtype=vsource.dtype)
                virtual_layouts[group_name][key][:] = vsource

    # Create a virtual dataset for each layout
    with h5.File(output_filename, "w", libver="latest") as f:
        for group_name, layouts in virtual_layouts.items():
            for key, layout in layouts.items():
                f.create_virtual_dataset(f"{group_name}/{key}", layout, fillvalue=0)

    print(f"Virtual dataset created in {output_filename}")


def save_dict_to_h5(dictionary, filename):
    """
    Save a dictionary to an HDF5 file.

    :param dictionary: dict
        The dictionary containing the data to be saved. The keys represent the dataset names, and the values represent
        the data to be stored.

    :param filename: str
        The name of the HDF5 file where the dictionary will be saved. If the file already exists, it will be overwritten.

    :return: None

    :Example:
        >>> data = {'dataset1': [1, 2, 3], 'dataset2': (10, 20, 30), 'dataset3': 42}
        >>> save_dict_to_h5(data, 'data.h5')
        # Creates an HDF5 file 'data.h5' containing three datasets: 'dataset1', 'dataset2', and 'dataset3'.
    """
    with h5.File(filename, "w") as h5file:
        for key, value in dictionary.items():
            if all(isinstance(i, tuple) for i in value):
                value = np.array(value)
            h5file.create_dataset(key, data=value)


def load_dict_from_h5(filename):
    """
    Load a dictionary from an HDF5 file.

    :param filename: str
        The name of the HDF5 file from which the dictionary will be loaded.

    :return: dict
        A dictionary containing the data loaded from the HDF5 file. The keys represent the dataset names, and the values
        represent the corresponding data arrays.

    :Example:
        >>> loaded_data = load_dict_from_h5('data.h5')
        >>> print(loaded_data)
        {'dataset1': array([1, 2, 3]), 'dataset2': array([10, 20, 30]), 'dataset3': array(42)}
    """
    dictionary = {}
    with h5.File(filename, "r") as h5file:
        for key in h5file.keys():
            dictionary[key] = np.array(h5file[key])
    return dictionary


def load_tiff_as_npy(filename):
    """
    Load a TIFF file as a NumPy array.

    Parameters:
        filename (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    import warnings

    warnings.filterwarnings("ignore")

    try:
        image = imageio.imread(filename)
        return np.array(image)
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

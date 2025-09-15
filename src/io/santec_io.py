import pandas as pd
import numpy as np
import os, sys

def save_hologram_csv(values, path, grayscale_range = 1024):
    values_fixed = values.astype(np.ushort) % grayscale_range

    # Reconstruct the dataframe with flipped contents
    indices = [i for i in range(values_fixed.shape[0])]
    columns = [i for i in range(values_fixed.shape[1])]
    df = pd.DataFrame(values_fixed, columns=columns, index=indices)

    # Reinsert the first column
    df.insert(0, 'Y/X', indices)

    # Get the proper path
    df.to_csv(path, index=False)

def get_backplane_correction_grayscale(wavelength, path = None):
    """
    Get the backplane correction values from a csv file and convert them to grayscale values.
    
    Args:
    - wavelength: The wavelength of the light in nm.
    - path: The path to the csv file containing the backplane correction values.
    """
    if path is None:
        path = os.path.join('data', 'backplane', 'backplane_correction.csv')

    grayscale_range = 1024
    in_df = pd.read_csv(path, header=None, index_col=None)

    # Select the contents of the file excluding the first column and row
    backplane_thickness = in_df.iloc[:, :].values

    num_wavelengths = backplane_thickness / wavelength * 1e3
    residual = num_wavelengths - np.floor(num_wavelengths)

    backplane_correction_values = residual * (grayscale_range - 1) # 0 to 1023
    backplane_correction_values = backplane_correction_values.astype(np.uint16) % grayscale_range

    return backplane_correction_values
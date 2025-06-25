"""
data manipulation
"""

import numpy as np
import json
import os

"""
Data Persistence Utilities
--------------------------
This module provides utilities for saving, loading, and managing numerical data in files.

The module offers functions to:
1. Save arrays (NumPy or Python lists) to files
2. Load arrays from files 
3. Append values to existing array files
4. Handle directory creation and path validation

These utilities are particularly useful for:
- Saving training results or metrics during ML model development
- Persisting numerical data between program runs
- Creating data logs for later analysis
- Managing experiment results

All functions handle path validation and directory creation automatically,
making it easy to save files without worrying about directory structure.
"""


def find_and_prep_file(filename):
    """
    Ensures the directory for a filename exists, creates it if needed,
    or defaults to the current directory if the path doesn't exist.

    Args:
        filename: Path to the file including directories

    Returns:
        Updated filename with a valid path
    """
    dir_path = os.path.dirname(filename)

    # If filename has no directory component or directory doesn't exist
    if not dir_path or not os.path.exists(dir_path):
        print(f"Cannot locate path '{dir_path}', creating file in current directory")
        filename = "./" + os.path.basename(filename)
    else:
        # Directory exists, no need to create it
        pass

    # Ensure the directory of the (possibly updated) filename exists
    new_dir_path = os.path.dirname(filename)
    if new_dir_path:  # Only create if there's a directory component
        os.makedirs(new_dir_path, exist_ok=True)

    return filename


def save_array_to_file(array, filename):
    """
    Save a numerical array to a JSON file

    :param array: The array to save
    :param filename: The path to save the file to
    """

    # ensure file is available
    filename = find_and_prep_file(filename)

    # Convert to list if it's a numpy array
    if isinstance(array, np.ndarray):
        array = array.tolist()

    # Save to file
    with open(filename, 'w') as f:
        json.dump(array, f)

    print(f"Array saved to {filename}")


def append_values_to_file(new_values, filename, create_if_missing=True):
    """
    Append new values to an existing array file or create one if it doesn't exist

    :param new_values: Single value or list of values to append
    :param filename: The path to the file
    :param create_if_missing: If True, create the file if it doesn't exist
    :return: True if successful, False otherwise
    """

    # ensure file is available
    filename = find_and_prep_file(filename)

    try:
        # Ensure new_values is a list (convert single value if needed)
        if not isinstance(new_values, (list, np.ndarray)):
            new_values = [new_values]

        # Convert numpy arrays to lists
        if isinstance(new_values, np.ndarray):
            new_values = new_values.tolist()

        new_values = [float(i) for i in new_values]

        print(f"new values to be appended {new_values}")


        # Load existing data or create new file
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"File {filename} is corrupted")
                    return False
        else:
            existing_data = []

        # Append new values and save
        existing_data.extend(new_values)

        with open(filename, 'w') as f:
            json.dump(existing_data, f)

        print(f"Successfully appended {len(new_values)} value(s) to {filename}")
        return True

    except Exception as e:
        print(f"Error appending values to {filename}: {e}")
        return False


def load_array_from_file(filename):
    """
    Load a numerical array from a file

    :param filename: The path to the file to load
    :return: The loaded array
    """

    # ensure file is available
    filename = find_and_prep_file(filename)

    try:
        with open(filename, 'r') as f:
            array = json.load(f)
        # print(f"Array loaded from {filename}")
        return array
    except Exception as e:
        print(f"Error loading array from {filename}: {e}")
        return []


if __name__ == "__main__":
    path = "./data/0/right_arm_positions.txt"
    arr = load_array_from_file(path)
    np_arr = np.array(arr).reshape(-1, 12)
    print(np_arr.shape)


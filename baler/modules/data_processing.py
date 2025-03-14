# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple

import numpy as np
import torch
from numpy import ndarray
from sklearn.model_selection import train_test_split

from ..modules import helper
from ..modules import models


def convert_to_blocks_util(blocks, data):
    print(
        "Converted Dataset to Blocks of Size - ", blocks, " from original ", data.shape
    )
    blocks = np.array(blocks)
    original_shape = np.array(data.shape)
    total_size = np.prod(original_shape)
    data = data.reshape((total_size // (blocks[1] * blocks[2])), blocks[1], blocks[2])
    return data


def save_model(model, model_path: str) -> None:
    """Saves the models state dictionary as a `.pt` file to the given path.

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        None: Saved model state dictionary as `.pt` file.
    """
    print("Saving model to: ", model_path)
    torch.save(model.state_dict(), model_path)


def encoder_saver(model, model_path: str) -> None:
    """Saves the Encoder state dictionary as a `.pt` file to the given path

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        None: Saved encoder state dictionary as `.pt` file.
    """
    torch.save(model.encoder.state_dict(), model_path)


def decoder_saver(model, model_path: str) -> None:
    """Saves the Decoder state dictionary as a `.pt` file to the given path

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        None: Saved decoder state dictionary as `.pt` file.
    """
    torch.save(model.decoder.state_dict(), model_path)


def initialise_model(model_name: str):
    """Initializing the models attributes to a model_object variable.

    Args:
        model_name (str): The name of the model you wish to initialize. This should correspond to what your Model name.

    Returns:
        class: Object with the models class attributes
    """
    model_object = getattr(models, model_name)
    return model_object


def load_model(model_object, model_path: str, n_features: int, z_dim: int):
    """Loads the state dictionary of the trained model into a model variable. This variable is then used for passing
    data through the encoding and decoding functions.

    Args:
        model_object (object): Object with the models attributes
        model_path (str): Path to model
        n_features (int): Input dimension size
        z_dim (int): Latent space size

    Returns: nn.Module: Returns a model object with the attributes of the model class, with the selected state
    dictionary loaded into it.
    """
    device = helper.get_device()
    print("Loading model from: ", model_path)
    print("n_features: ", n_features)
    print("z_dim: ", z_dim)
    model = model_object(n_features, z_dim)
    model.to(device)

    # Loading the state_dict into the model
    model.load_state_dict(
        torch.load(str(model_path), map_location=device), strict=False
    )
    return model


# def find_minmax(data):
#     """Obtains the minimum and maximum values for each column.

#     Args:
#         data (ndarray): Any dataset as a `ndarray` which one eventually wants to normalize using the Min-Max method.

#     Returns: ndarray: An array of two lists. One of the lists contains the minimum of each column, while the other
#     list contains `feature_range = max - min` for each column.
#     """
#     data = list(data)
#     true_max_list = np.apply_along_axis(np.max, axis=0, arr=data)
#     true_min_list = np.apply_along_axis(np.min, axis=0, arr=data)

#     # Computes the range
#     feature_range_list = true_max_list - true_min_list

#     normalization_features = np.array([true_min_list, feature_range_list])
#     return normalization_features
"""MYCODE"""
def find_global_max_len(data):
    """Find the maximum length of any array across all samples and features."""
    max_len = 0
    for sample in data:
        for feature in sample:
            max_len = max(max_len, len(feature))
    return max_len

def pad_arrays(data, max_len):
    """Pad all arrays in the dataset to the maximum length with zeros."""
    padded_data = []
    for sample in data:
        padded_sample = []
        for feature in sample:
            padded_feature = np.pad(feature, (0, max_len - len(feature)), 
                                  'constant', constant_values=0)
            padded_sample.append(padded_feature)
        padded_data.append(padded_sample)
    return padded_data

def compute_min_max(data, feature_indices):
    """Compute min and max for specified features using original (unpadded) values."""
    min_values = {}
    max_values = {}
    for idx in feature_indices:
        # Collect all values for this feature across all samples
        all_values = [val for sample in data for val in sample[idx]]
        min_values[idx] = np.min(all_values)
        max_values[idx] = np.max(all_values)
    return min_values, max_values

def find_minmax(data):
    """Normalize the dataset after padding all arrays to the global maximum length."""
    # Step 1: Find the global maximum length
    max_len = find_global_max_len(data)
    print(f"Global maximum length found: {max_len}")
    
    # Step 2: Pad all arrays to max_len
    padded_data = pad_arrays(data, max_len)
    
    # Convert to a 3D NumPy array: (n_samples, 6, max_len)
    padded_array = np.stack([np.stack(sample) for sample in padded_data], axis=0)
    
    # Step 3: Define numerical features to normalize (exclude boolean feature at index 4)
    numerical_features = [0, 1, 2, 3, 5]
    
    # Compute min and max for numerical features using original data
    min_values, max_values = compute_min_max(data, numerical_features)
    
    # Initialize the normalized array
    normalized_data = np.zeros_like(padded_array, dtype=np.float32)
    
    # Step 4: Normalize numerical features
    for idx in numerical_features:
        min_val = min_values[idx]
        max_val = max_values[idx]
        # Avoid division by zero
        range_val = max_val - min_val if max_val > min_val else 1
        normalized_data[:, idx, :] = (padded_array[:, idx, :] - min_val) / range_val
    
    # Copy the boolean feature (index 4) as is
    normalized_data[:, 4, :] = padded_array[:, 4, :].astype(np.float32)
    
    return normalized_data
def reshape_dataset(data):
    """Reshape the dataset by padding all arrays to the global maximum length without normalization."""
    # Step 1: Find the global maximum length
    max_len = find_global_max_len(data)
    print(f"Global maximum length found: {max_len}")
    
    # Step 2: Pad all arrays to max_len
    padded_data = pad_arrays(data, max_len)
    
    # Step 3: Convert to a 3D NumPy array: (n_samples, n_features, max_len)
    reshaped_data = np.stack([np.stack(sample) for sample in padded_data], axis=0)
    
    return reshaped_data

def normalize(data, custom_norm: bool):
    """This function scales the data to be in the range [0,1], based on the Min Max normalization method. It finds
    the minimum and maximum values of each column and computes the values according to: x_norm = (x - x_min) / (x_max
    - x_min).

    Args: data (ndarray): A dataset of type `ndarray`. custom_norm (boolean): If you want to do Min Max normalization
    or any custom normalization. Custom normalization is not supported at the moment.

    Returns: ndarray: If not custom_norm: Input data where every column is scaled to be in the range [0,
    1]. Otherwise, the input data is returned
    """
    data = np.array(data)
    if custom_norm:
        pass
    elif not custom_norm:
        true_min = np.min(data)
        true_max = np.max(data)
        feature_range = true_max - true_min
        data = [((i - true_min) / feature_range) for i in data]
        data = np.array(data)
    return data
"""UPTO HERE"""

def split(data, test_size: float, random_state: int) -> Tuple[ndarray, ndarray]:
    """Splits the given data according to a test size into two sets, train and test. This is a sklearn function,
    opted from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Args: data (ndarray): Dataset which you want to split test_size (float): Value between 0.0 and 1.0 determining
    how large the test size will be. Example: test_size = 0.15 randomly selects 15% of the original dataset and puts
    it into a new array. random_state (int): Shuffling control. Passing an int here shuffles the data the same way
    every time. Used for reproducibility.

    Returns:
        ndarray, ndarray: Two datasets split in size according to the `test_size` value.
    """
    return train_test_split(data, test_size=test_size, random_state=random_state)


def renormalize_std(
    input_data: ndarray, true_min: float, feature_range: float
) -> ndarray:
    """Computes the un-normalization of normalized arrays. This function is used in combination with
    `renormalize_func` to perform this operation on an entire dataset.

    Args:
        input_data (ndarray): Normalized data you want to un-normalize from the range [0,1]
        true_min (float): A columns minimum
        feature_range (float): A columns feature range, computed as `max - min`

    Returns:
        ndarray: Un-normalized data array
    """
    return np.array([((i * feature_range) + true_min) for i in list(input_data)])


def renormalize_func(norm_data: ndarray, min_list: List, range_list: List) -> ndarray:
    """Un-normalizes an entire dataset. Applies the `renormalize_std`function across an entire dataset.
        `min_list` and `range_list` are obtained from the `normalization_features.npy` file.

    Args:
        norm_data (ndarray): Normalized data which one wants to un-normalized.
        min_list (list): List of minimum values. Should have the same length as the amount of columns.
        range_list (list): List of feature range values. Should have the same length as the amount of columns.

    Returns:
        ndarray: Array with the un-normalized values.
    """
    norm_data = np.array(norm_data)
    min_list = np.array(min_list)
    range_list = np.array(range_list)
    return norm_data * range_list + min_list

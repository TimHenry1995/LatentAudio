import random
import numpy as np, os
from typing import Tuple, List

def load_latent_sample(data_folder: str, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Loads latent representation for a sample (without replacement) of instances from ``data_folder``.
    Thus assumes that ``sample_sze`` is at most the number of X files in the data folder.
    
    :param data_folder: The path to the folder that contains the X and Y .npy files for the latent representation of a given layer.
    :type data_folder: str
    :param sample_size: The number of instances to load.
    :type sample_size: int

    :return:
        - X (:class:`numpy.ndarray`) - The sample of loaded X data. Shape == [``sample_size``, ...] where ... is the shape of the latent representation of a single instance.
        - Y (:class:`numpy.ndarray`) - The sample of Y data, corresponding to ``X``. Shape == [``sample_size``, factor count].
    """

    # Load a sample of X and Y
    x_file_names = find_matching_strings(strings=os.listdir(data_folder), token='_X_')
    X = [None] * sample_size; Y = [None] * sample_size
    for i, j in enumerate(random.sample(range(0, len(x_file_names)), sample_size)):
        x_path = os.path.join(data_folder, str(x_file_names[j]))
        X[i] = np.load(x_path)[np.newaxis,:]; Y[i] = np.load(x_path.replace('_X_','_Y_'))[np.newaxis,:]
    X = np.concatenate(X, axis=0); Y = np.concatenate(Y, axis=0)
    
    # Outputs
    return X, Y

def find_matching_strings(strings: List[str], token: str) -> List[str]:
    """Browses all ``strings`` and return a list of those that contain the ``token``.
    
    :param strings: The list of strings to be filtered.
    :type strings: List[str]
    :param token: The token which needs to exists inside a string of ``strings`` in order for that string to be returned.
    :type token: str

    :return:
        - selection (List[str]) - The strings from ``strings`` that contain the ``token``.
    """

    # Find the file names of X files
    selection = [None] * len(strings)
    i = 0
    for string in strings:
        if token in string:
            selection[i] = string; i+= 1
    selection = selection[:i]

    #Output
    return selection
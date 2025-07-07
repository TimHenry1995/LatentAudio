import numpy as np
from typing import List
import tensorflow as tf
import copy as cp
import os
import random as rd
from typing import Tuple, Callable

def persistent_factorized_pair_iterator(data_path: str, x_file_names: List[str], y_file_names: List[str], similarity_function: Callable, batch_size: int = 32) -> Tuple[tf.Tensor, tf.Tensor]:
    """This infinite iterator takes instances x_a of the data and finds for each of them an arbitrarily selected x_b such that they 
    form a pair. The X outputs will be batches of such pairs while the Y outputs will be the corresponding batches of factor-wise 
    label similarity. **IMPORTANT:** The files of ``y_file_names`` can, but do not have to include the residual factor. It is up to 
    the ``similarity_function`` to ensure that the similarity ratings are of shape [instance count, factor count], where factor count 
    then includes the residual factor. That is, the ``similarity_function`` could for instance insert a column of zeros at the index 
    0 for the residual factor.

    :param data_path: Path to the folder that contains the inputs that shall be paired based on ``label``.
    :type data_path: str
    :param x_file_names: File names that identify input instances stored at ``data_path`` in .npy files (including file extension).
    :type x_file_names: List[str]
    :param y_file_names: File names to label vectors that correspond to the instances listed in ``x_file_names``. Each file shall
        contain the vector for a single instance. Such a vector shall have factor count many entries whose value indicates the 
        corresponding value along that factor. Here, the residual factor can but does not have to be included.
    :type y_file_names: List[str]
    :param similarity_function: A callable that takes as input Y_a (:class:`numpy.nparray`), Y_b (:class:`numpy.nparray`) which are 
        each Y representations of shape [``batch_size``, factor_count] where the residual factor may but does not have to be included.
        It then calculates the similarity for each factor and outputs a :class:`tensorflow.Tensor` of shape [``batch_size``, 
        factor count], where the **residual factor now has to be included**.
    :type similarity_function: :class:`Callable`
    :param batch_size: The number of pairs that shall be put inside a batch.
    :type batch_size: int, optional

    yield:
        - X_ab (:class:`tensorflow.Tensor`) - A batch of instance pairs of shape [batch_size, 2, ...], where 2 is due to the 
            concatenation of X_a and X_b and ... is the same instance-wise shape as for ``X``.
        - Y_ab (:class:`tensorflow.Tensor`) - The corresponding batch of similarities as obtained by feeding the instances of X_a
            and X_b into ``similarity_function``. It has shape [``batch_size``, factor count], including the residual factor.
    """

    # Input validity
    assert len(x_file_names) == len(y_file_names), f"The inputs X_file_names and y_file_names were expected to have the same number of instances, yet have length {len(x_file_names)} and {len(y_file_names)}, respectively."
    y_shape = np.load(os.path.join(data_path, y_file_names[0])).shape
    assert len(y_shape) == 1, f"The y instances are expected to each be a vector of factor count many entries, but found y shape to be {y_shape}." 
    
    # Convenience variables
    instance_count = len(x_file_names)
    x_shape = np.load(os.path.join(data_path, x_file_names[0])).shape
    factor_count = y_shape[0] # Depending on the user, this might in- or exclude the residual factor

    while True:

        # Initialization
        X_ab = np.empty((batch_size, 2, *x_shape), dtype=tf.keras.backend.floatx())
        Y_a = np.empty((batch_size, factor_count), dtype=float)
        Y_b = np.empty((batch_size, factor_count), dtype=float)

        # Select indices for instances a and b
        a = np.random.randint(low=0, high=instance_count, size=batch_size)
        b = np.random.randint(low=0, high=instance_count, size=batch_size)

        # Load data
        for i in range(batch_size):
            # X_i
            x_a = np.load(os.path.join(data_path, x_file_names[a[i]]))
            x_b = np.load(os.path.join(data_path, x_file_names[b[i]]))
            X_ab[i,:] = np.concatenate([x_a[np.newaxis,:], x_b[np.newaxis,:]], axis=0) # Concatenate along pair axis
            
            # Y_i
            y_a = np.load(os.path.join(data_path, y_file_names[a[i]]))
            y_b = np.load(os.path.join(data_path, y_file_names[b[i]]))
            
            assert len(y_a.shape) == 1 and len(y_b.shape) == 1 and len(y_a) == factor_count and len(y_b) == factor_count, f"The y instances are expected to each have factor count many entries, but found y_a shape to be {y_a.shape} and y_b shape to be {y_b.shape}." 
            Y_a[i,:] = y_a
            Y_b[i,:] = y_b

        Y_ab = similarity_function(Y_a, Y_b)
        assert len(Y_ab.shape) == 2 and Y_ab.shape[1] == factor_count or Y_ab.shape[1] == factor_count + 1, f"The similartity function is expected to provide an output of shape [instance count, factor count] (including the residual factor) but provided {Y_ab.shape}."

        # Ensure data type
        X_ab = tf.constant(X_ab, dtype=tf.keras.backend.floatx())
        Y_ab = tf.constant(Y_ab, dtype=tf.keras.backend.floatx())

        # Outputs
        yield X_ab, Y_ab 

def volatile_factorized_pair_iterator(X: np.ndarray, Y: np.ndarray, similarity_function: Callable, batch_size: int, minimum_similarity: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """This infinite iterator yields pairs of instances X_a and X_b along with their corresponding factorized similarity Y. Pairs are obtained 
    by shuffling X once for X_a and once for X_b. It is thus possible that pair i has the same instance in X_a and X_b, yet unlikely 
    for large instance counts in X. The iterator produces a instance count many pairs, split into batches. The last batch may be 
    smaller than ``batch_size`` to reach that pair count.  **IMPORTANT:** The ``Y`` data can, but does not have to include 
    the residual factor. It is up to the ``similarity_function`` to ensure that the similarity ratings are of shape [instance count, 
    factor count], where factor count includes the residual factor. That is, the ``similarity_function`` could for instance insert a 
    column of zeros at the index 0 for the residual factor.

    :param X: Input data of shape [instance count, ...], where ... is any shape convenient for the caller.
    :type X: :class:`numpy.ndarray`
    :param Y: Scores of factors of shape [instance count, factor count], may in or exclude the residual factor.
    :type Y: :class:`numpy.ndarray`
    :param similarity_function: A callable that takes as input Y_a (:class:`numpy.nparray`), Y_b (:class:`numpy.nparray`) which are 
        each Y representations of shape [``batch_size``, factor_count], which may but do not have to include the residual factor. It 
        then calculates the similarity for each factor and outputs a :class:`numpy.ndarray` of shape [``batch_size``, factor count],
        where factor count now has to include the residual factor.
    :type similarity_function: :class:`Callable`
    :param batch_size: Desired number of instances per batch
    :type batch_size: int

    :yield: 
        - X_ab (:class:`numpy.ndarray`) - A batch of instance pairs of shape [batch_size`, 2, ...], where 2 is due to the 
            concatenation of X_a and X_b and ... is the same instance-wise shape as for ``X``. 
        - Y_ab (:class:`numpy.ndarray`) - The corresponding batch of similarities as obtained by feeding the instances of X_a 
            and X_b into ``similarity_function``. It has shape [``batch_size``, factor count], including the residual factor.
    """

    # Input validity
    assert len(X.shape) > 0 and Y.shape[0] > 0 and X.shape[0] == Y.shape[0], f"The inputs X and Y were expected to have the same number of instances along the initial axis, yet X has shape {X.shape} and Y has shape {Y.shape}."
    assert len(Y.shape) == 2, f"The shape of Y should be [instance count, factor count], but found {Y.shape}."
    factor_count = Y.shape[1] # Might in or exclude the residual factor
    assert len(Y.shape) >= 2, f"The input Y needs to have at least two instances."
    Y_ab_tmp = similarity_function(Y[0:1,:], Y[1:2,:])
    assert Y_ab_tmp.shape[1] == factor_count or Y_ab_tmp.shape[1] == factor_count + 1, f"The similarity function is expected to provide factor count many outputs (including the residual factor) but this could not be verified when feeding the first two instances in."

    # Convenience variables
    instance_count = Y.shape[0]
    Y_ab_shape = similarity_function(Y[:batch_size,:], Y[batch_size,:]).shape

    # Loop over batches
    while True:
        
        # Choose random X_a instances
        a_indices = np.random.randint(low=0, high=instance_count, size=batch_size)
        X_a = tf.cast(X[a_indices,:], tf.keras.backend.floatx())[:, tf.newaxis, :]
        X_b = [None] * batch_size

        # Search for partners to X_a such that each pair has minimum requried similarity and consists of 2 distinct instances
        Y_ab = [None] * batch_size

        for i in range(batch_size):
            found_partner = False
            while not found_partner:
                b_index = np.random.randint(low=0, high=instance_count)
                X_b[i] = tf.cast(X[b_index,:], tf.keras.backend.floatx())[tf.newaxis, tf.newaxis, :]
                Y_ab[i] = tf.cast(similarity_function(Y[a_indices[i],tf.newaxis,:], Y[b_index][tf.newaxis,:]), tf.keras.backend.floatx())
                found_partner = minimum_similarity <= tf.reduce_sum(Y_ab[i]).numpy() and a_indices[i] != b_index

        X_b = tf.concat(X_b, axis=0) # Concatenate along instance axis
        X_ab = tf.concat([X_a, X_b], axis=1) # Concatenate along pair axis
        Y_ab = tf.concat(Y_ab, axis=0) # Concatenate along instance axis

        yield X_ab, Y_ab

import tensorflow as tf 
import numpy as np
from typing import List, Tuple
import copy as cp

class UnsupervisedFactorLoss():
    pass

class SupervisedFactorLoss(tf.keras.losses.Loss):
    r"""
    This loss can be used to incentivize the entries of the output vector of a flow network to be arranged according to semantic 
    factors of the data with multivariate normal distribution. It implements the following formula:

    .. math:: 
        \mathcal{L} = \sum_{F=1}^{K} \mathbb{E}_{(z^a,z^b) \sim p(z^a, z^b | F) } l(z^a, z^b | F)

    .. math:: 
        :nowrap:

        \begin{eqnarray}
            l(z^a,z^b | F) &= \frac{1}{2} \sum_{k=0}^{K} ||T(z^a)_k||^2 - log|T'(z^a)| \\
                &+ \frac{1}{2} \sum_{k \neq F} ||T(z^b)_k||^2 - log|T'(z^b)| \\
                &+ \frac{1}{2} \frac{||T'(z^b)_F - \sigma_{ab} T(z^a)_F||^2}{1-\sigma_{ab}^2},
        \end{eqnarray}
        
    where :math:`T(z)` is the model whose loss shall be computed, :math:`z^a`, :math:`z^b` are instances passed trough :math:`T`,
    :math:`T'(z^a)` is the Jacobian of :math:`T` and :math:`\sigma_{ab}` is the clustering strength of instances (see below). 
    The factors can be thought of as independent components. A factor :math:`k` spreading across :math:`N_k` entries of the 
    output vector is incentivised by this loss to represent the similarity of two inputs :math:`z^a` and :math:`z^b` along one and 
    only one concept. For instance, factors can represent color, roughness, size, animal species, or material. The loss expects 
    training instances to come in pairs :math:`z^a` and :math:`z^b` for each such factor. A pair should have strong positive 
    association :math:`\sigma` such that the corresponding factor can capture the underlying concept of similarity. Yet, the 
    association shall be (on average) close to zero for all other factors. See also :class:`UnsupervisedFactorLoss`.

    :param sigma: This hyperparameter refletcs the clustering strength of instances in general. It is chosen to be in the interval 
        (0,1) exclusive and should be close to 1 when instances supervised to be similar shall cluster tightly together or set to a
        value closer to 0 when clustering shall be more dispersed.
    :type sigma: float, optional
    :param dimensions_per_factor: A list of integers that enumerates the number of dimensions (entries in a vector) of the factors thought to underly
        the representation of :math:`z^{\sim}`. These shall include the residual factor at index 0 which collect all variation not captured by the 
        true factors. The sum of all entries is assumed to be equal to the number of dimensions in :math:`z^{\sim}`.
    :type dimensions_per_factor: List[int] 

    References:
        - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer <https://arxiv.org/abs/2004.13166>`_
    """

    def __init__(self, dimensions_per_factor: List[int], sigma: float = 0.975, *args):
        
        # Super
        super(SupervisedFactorLoss, self).__init__(*args)

        # Attributes
        factor_count = len(dimensions_per_factor)
        factor_masks = np.zeros(shape=[factor_count, np.sum(dimensions_per_factor)])
        total = 0
        for u, dimension_count in enumerate(dimensions_per_factor): 
            factor_masks[u, total:total+dimension_count] = 1
            total += dimension_count
        self.__factor_masks__ = tf.constant(factor_masks, dtype=tf.keras.backend.floatx()) 
        """Collects masks (one per factor) that are 1 for each factor's dimensions and zero elsewhere. Shape == [factor count, dimension count]"""

        self.__sigma__ = sigma
        """(int) - Hyperparameter in (0,1) indicating clustering strength between pairs of instances."""

        self.__dimensions_per_factor__ = cp.copy(dimensions_per_factor)
        """(:class:`List[int]`) - The number of dimensions per factor. Length equals factor count."""

    def call(self, y_true: tf.Tensor, y_pred: Tuple[tf.Tensor]) -> tf.Tensor:
        """Computes the loss.
        
        :param y_true: A matrix of shape [batch size, factor count], that indicates for each pair in the batch and each factor, to 
            what extent the two instances from ``z_tilde_a`` and ``z_tilde_b`` share this factor. Similarity is assumed to be in the 
            range [0,1]. If the factors are all categorical, it makes sense to set these similarities either to 1 or 0, indicating 
            same class or not, respectviely. E.g. if there are two factors and 3 pairs of z_tilde, then ``y_true`` could be 
            [[0,1],[0,0],[0,1]], indicating that the first and last pairs of z_tilde share the concept of factor at index 1 and 
            the second pair does not have anything in common. The residual factor (located at index 0) is typically not the same for
            any two instances and thus usually stores a zero in this array. The hyperparameter sigma (typically close to 1) should be 
            set to reflect the category resemblance of instances. 
        :type y_true: :class:`tensorflow.Tensor`
        :param y_pred: A tuple containing [z_tilde_a, z_tilde_b, j_a, j_b]. 
        :type y_pred: Tuple[:class:`tensorflow.Tensor`]

        - z_tilde_a (:class:`tensorflow.Tensor`) - The output of model T on the first input of the pair z^a, z^b. Shape == [batch size, dimension count] where dimension count is the number of dimensions in the flattened output of T.
        - z_tilde_b (:class:`tensorflow.Tensor`) - The output of model T on the second input of the pair z^a, z^b. Shape == [batch size, dimension count] where dimension count is the number of dimensions in the flattened output of T.
        - j_a (:class:`tensorflow.Tensor`) - The jacobian determinant on logarithmic scale of T at z^a. Shape == [batch size]
        - j_b (:class:`tensorflow.Tensor`) - The jacobian determinant on logarithmic scale of T at z^b. Shape == [batch size]

        :return: loss (tf.Tensor) - A single value indicating the amount of error the model makes in factoring its inputs.
        """

        # Input validity
        assert len(y_pred) == 4, f"The input y_pred is expected to be a tuple of the four tensorflow.Tensor objects z_tilde_a, z_tilde_b, j_a and j_b."
        z_tilde_a, z_tilde_b, j_a, j_b = y_pred
        assert len(z_tilde_a.shape) == 2, f"z_tilde_a has shape {z_tilde_a.shape} but was expected to have shape [batch size, dimension count]."
        assert len(z_tilde_b.shape) == 2, f"z_tilde_b has shape {z_tilde_b.shape} but was expected to have shape [batch size, dimension count]."
        assert z_tilde_a.shape == z_tilde_b.shape, f"The inputs z_tilde_a and z_tilde_b where expected to have the same shape [batch size, dimension count] but found {z_tilde_a.shape} and {z_tilde_b.shape}, respectively."
        assert (z_tilde_a.shape[1] == self.__factor_masks__.shape[1]), f"z_tilde_a was expected to have as many dimensions along axis 1 as the sum of dimensions in dimensions_per_factor specified during initialization ({self.__factor_masks__.shape[1]}) but it has {z_tilde_a.shape[1]}."
        assert (z_tilde_b.shape[1] == self.__factor_masks__.shape[1]), f"z_tilde_b was expected to have as many dimensions along axis 1 as the sum of dimensions in dimensions_per_factor specified during initialization ({self.__factor_masks__.shape[1]}) but it has {z_tilde_b.shape[1]}."
    
        assert len(j_a.shape) == 1, f"The input j_a was expected to have shape [batch size] but found {j_a.shape}."
        assert len(j_b.shape) == 1, f"The input j_b was expected to have shape [batch size] but found {j_b.shape}."
        assert j_a.shape == j_b.shape, f"The inputs j_a and j_b where expected to have the same shape [batch size] but have {j_a.shape} and {j_b.shape}, respectively."
        assert j_a.shape[0] == z_tilde_a.shape[0], f"The inputs z_tilde and j are expected to have the same number of instances along the batch axis (axis 0)."
        
        assert len(y_true.shape) == 2, f"The input y_true is expected to have shape [batch size, factor count], but has shape {y_true.shape}."
        assert y_true.shape[0] == z_tilde_a.shape[0], f"The inputs y_true and z_tilde are assumed to have the same number of instances in the batch. Found {y_true.shape[0]} and {z_tilde_a.shape[0]}, respectively."
        
        # Convenience variables
        batch_size, factor_count = y_true.shape
        dimension_count =  z_tilde_a.shape[1] 
        
        # Implement formula (10) of referenced paper
        # L = sum_{F=1}^K expected_value_{x^a,x^b ~ p(x^a, x^b | F)} l(E(x^a), E(x^b)| F)       (term 10)
        # l(z^a, z^b | F) = 0.5 * sum_{k=0}^K ||T(z^a)_k||^2 - log|T'(z^a)|                     (term 7) 
        #                 + 0.5 * sum_{k != F} ||T(z^b)_k||^2 - log|T'(z^b)|                    (term 8) 
        #                 + 0.5 * ( || T(z^b)_F - sigma_{ab} T(z^a)_F || ^2) / (1-sigma_{ab}^2) (term 9) 
        # NOTE: The authors forgot the multiplier 0.5 in front. Since it is not applied to each entire term, it does make a difference for the final result 

        # Iterate factors (according to term 10)
        loss = 0
        for f in range(1, factor_count): # Excludes residual factor
            # Mask out the instances that do not have the same class for the current factor
            # because they are not relevant for the current iteration across factors
            f_z_tilde_a = y_true[:,f:f+1] * z_tilde_a 
            f_z_tilde_b = y_true[:,f:f+1] * z_tilde_b

            # This one leads points f_z_tilde_a to be multivariate normal
            term_7 = 0.5 * tf.reduce_sum(tf.pow(f_z_tilde_a, 2), axis=1) - y_true[:,f:f+1]*j_a # Shape == [batch size]
            
            # This leads points b to be normal along all other factor than f
            term_8 = 0
            for f_other in range(0, factor_count): # Includes residual factor
                if f_other != f: 
                    factor_mask = tf.repeat(self.__factor_masks__[f_other,:][tf.newaxis,:], repeats=batch_size, axis=0) # shape == [batch_size, dimension_count]
                    term_8 += 0.5 * tf.reduce_sum(factor_mask * tf.pow(f_z_tilde_b, 2), axis=1)  # Shape == [batch size]
            term_8 -= y_true[:,f:f+1]*j_b
            
            # This leads points a and b (if they are labelled similar for current factor) to be close to each other
            factor_mask = tf.repeat(self.__factor_masks__[f,:][tf.newaxis,:], repeats=batch_size, axis=0) # shape == [batch_size, dimension_count]
            term_9 = 0.5 * tf.reduce_sum(factor_mask * tf.pow(f_z_tilde_b - self.__sigma__ * f_z_tilde_a, 2) / (1.0-(y_true[:,f:f+1] * self.__sigma__)**2 + 1e-5), axis=1) # Shape == [batch size], 1e-5 to prevent division by 0
        
            # Take mean across instances and add to the total loss (according to term 10)
            instance_weight = 1.0-tf.reduce_sum(y_true[:,f]).numpy()/tf.reduce_sum(y_true).numpy() # To pretend each factor got an equal number of pairs, we apply a weight based on pair count
            loss += instance_weight*tf.reduce_sum(term_7 + term_8 + term_9, axis=0)/tf.reduce_sum(y_true[:,f]) # Shape == [1]
            
            # For the instances that are not the same along the current factor dimension, just let them be normally distributed without correlation
            f_z_tilde_a = (1-y_true[:,f:f+1]) * z_tilde_a 
            f_z_tilde_b = (1-y_true[:,f:f+1]) * z_tilde_b
            term_7 = 0.5 * tf.reduce_sum(tf.pow(f_z_tilde_a, 2), axis=1) - (1-y_true[:,f:f+1])*j_a # Shape == [batch size]
            term_8 = 0.5 * tf.reduce_sum(tf.pow(f_z_tilde_b, 2), axis=1) - (1-y_true[:,f:f+1])*j_b # Shape == [batch size]
            instance_weight = 1.0-tf.reduce_sum((1-y_true[:,f])).numpy()/tf.reduce_sum(1-y_true).numpy() # To pretend each factor got an equal number of pairs, we apply a weight based on pair count
            loss += instance_weight*tf.reduce_sum(term_7 + term_8, axis=0)/tf.reduce_sum(1-y_true[:,f]) # Shape == [1]
           

        # Outputs
        return loss

"""This script creates a flow model and calibrates it on projected latent represenetations to disentangle underlying factors for material and action.
It is possible to remove some classes because they might not all be separable.

Requirements:
- The script latent_yamnet_to_calibration_data_set needs to be executed apriori

Steps:
- creates a flow model
- loads the calibration data set
- calibrates the model on train split and tests on test split
- saves a plot for the loss trajectory
"""

from typing import List, Any, OrderedDict, Callable, Generator, Tuple
from gyoza.modelling import data_iterators as gmd, flow_layers as mfl, standard_layers as msl, masks as gmm, losses as gml
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings, random
from sklearn.decomposition import PCA
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import tensorflow_probability as tfp

class SupervisedFactorLossMultiNormal(gml.SupervisedFactorLoss):
    """This network is a :class:`SupervisedFactorLoss` whose loss incentivizes the probability distribution of the network's output
    to the sum of normals along selected factors.
    
    :param factor_to_muhs: A list of length factor count providing for each respective factor the means of the normal distributions. The residual factor's muhs are listed at index 0, yet for most cases it makes sense to simple set it to [0.0]. 
    :type factor_to_muhs: List[tensorflow.constant]
    :param factor_to_sigmas: A list of length factor count providing for each respective factor the standard deviations of the normal distributions. The residual factor's muhs are listed at index 0, yet for most cases it makes sense to simple set it to [1.0]. The indexing is assumed to be synchronous with factor_to_muhs. The values in factor_to_sigmas are conceptually different from the instance-instance correlation parameter sigma that can also be specified as input to this function. 
    :type factor_to_sigmas: List[tensorflow.constant]
    
    """

    def __init__(self, dimensions_per_factor: List[int], factor_to_muhs: List[tf.constant], factor_to_sigmas: List[tf.constant], sigma: float = 0.975, *args):
        super().__init__(dimensions_per_factor=dimensions_per_factor, sigma=sigma, *args)
        
        # Assert input vailidity
        assert type(factor_to_muhs) == type([[]]), f"The input factor_to_muhs was expeced to be of type List[tensorflow.constant] but factor_to_muhs is {factor_to_muhs}."
        assert type(factor_to_sigmas) == type([[]]), f"The input factor_to_sigmas was expeced to be of type List[tensorflow.constant] but factor_to_muhs is {factor_to_sigmas}."
        assert len(factor_to_sigmas) == len(factor_to_muhs), f"The inputs factor_to_sigmas and factor_to_muhs were expected to be of same length, but were found to have lengths {len(factor_to_sigmas)} and {len(factor_to_muhs)}, respectively."
        for i, (muhs, sigmas) in enumerate(zip(factor_to_muhs, factor_to_sigmas)):
            assert tf.is_tensor(muhs) and tf.is_tensor(sigmas) and muhs.shape == sigmas.shape, f"The inputs factor_to_muhs and factor_to_sigmas were expected to have the synchronous length along the nested lists. Yet at index {i} there are {len(muhs)} muhs and {len(sigmas)} sigmas."
               
        # Copy fields
        self.__factor_to_muhs__=factor_to_muhs
        self.__factor_to_sigmas__=factor_to_sigmas


    def call(self, y_true: tf.Tensor, y_pred: Tuple[tf.Tensor]) -> tf.Tensor:

        # Unpack inputs
        z_tilde_a, z_tilde_b, j_a, j_b = y_pred

        # Adjust z (note that Jacobian determinants do not have to be updated since the gradient of the loss w.r.t model parameters will not change due to the below z transformation)
        z_tilde_a_new = tf.zeros_like(z_tilde_a)
        z_tilde_b_new = tf.zeros_like(z_tilde_b)
        
        for f in range(len(self.__dimensions_per_factor__)):
            factor_mask = tf.repeat(self.__factor_masks__[f,:][tf.newaxis,:], repeats=batch_size, axis=0) # shape == [batch_size, dimension_count]
            z_tilde_a_new += factor_mask * self.sum_of_normals_to_single_normal(X=z_tilde_a, muhs=self.__factor_to_muhs__[f], sigmas=self.__factor_to_sigmas__[f])
            z_tilde_b_new += factor_mask * self.sum_of_normals_to_single_normal(X=z_tilde_b, muhs=self.__factor_to_muhs__[f], sigmas=self.__factor_to_sigmas__[f])


        # Compute loss
        loss = super().call(y_true=y_true, y_pred=(z_tilde_a_new, z_tilde_b_new, j_a, j_b))
        print(loss)

        # Outputs
        return loss

    def cumulative_normal_density_function_for_sum_of_normals(self, X: tf.Tensor, muhs: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
        """Computes the cumulative normal density for the sum of k normal distributions element-wise. Since no closed form exists for this operation,
        an approximation is used with an error margin of :math:`\pm 3*10^{-7}`.
        All tensors are assumed to have the same precision.

        :param X: A tensor containing input values to the density function.
        :type X: :class:`tensorflow.Tensor`
        :param muhs: A tensor collecting the means of the k normal distributions. Shape == [k].
        :type muhs: :class:`tensorflow.Tensor`
        :param sigmas: A tensor collecting the standard deviations of the k normal distributions. Indexing is assumed to be synchronous with `muhs`.
        :type sigmas: :class:`tensorflow.Tensor`

        :return: P (:class:'tensorflow.Tensor') - The cumulative probability values coresponding to `X`.

        References:

        - "Handbook of Mathematical Functions" by Abramowitz and Stegran, p. 299. Retrieved April 12, 2024 from https://personal.math.ubc.ca/%7Ecbm/aands/page_299.htm 
        """

        # Assumptions
        assert tf.is_tensor(X) and tf.is_tensor(muhs) and tf.is_tensor(sigmas), f"The inputs X, muhs and sigmas are all expected to be tensorflow.Tensor objects, but were found {type(X), type(muhs), type(sigmas)}, respectively."
        assert X.dtype == muhs.dtype and muhs.dtype == sigmas.dtype, f"The inputs X, muhs and sigmas are expected to have the same data type but were found to have {X.dtype, muhs.dtype, sigmas.dtype}, respectively."

        # Normal CDF (aproximated)
        G = lambda X, m, s: 0.5*(1+self.erf((X-m)/(s*np.sqrt(2))))

        # Compute cumulative density
        P = tf.reduce_sum([tfp.distributions.Normal(loc=muhs[i], scale=sigmas[i]).cdf(X) for i in range(len(muhs))], axis=0) / len(muhs)

        # Outputs
        return P

    # Error function (approximated)
    def erf(self, X):
        Y = tf.zeros_like(X)
        X_l = np.zeros(X.shape); X_l[np.where(X<=0)] = 1; X_l = tf.constant(X_l, dtype=X.dtype) # Mask selecting the non-poitive X values
        X_u = np.zeros(X.shape); X_u[np.where(0< X)] = 1; X_u = tf.constant(X_u, dtype=X.dtype) # Mask selecting the positive X values
        
        Y += X_l* -tf.math.erf(-X)#(1/(1-0.0705230784*X + 0.0422820123*X**2 - 0.0092705272*X**3 + 0.0001520143*X**4 - 0.0002765672*X**5 + 0.0000430638*X**6)**16 - 1)
        Y += X_u* tf.math.erf(X)#(1-1/(1+0.0705230784*X + 0.0422820123*X**2 + 0.0092705272*X**3 + 0.0001520143*X**4 + 0.0002765672*X**5 + 0.0000430638*X**6)**16)
        
        return Y

    def sum_of_normals_to_single_normal(self, X: tf.Tensor, muhs: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
        
        # Apply universality of uniform theorem
        P = 1e-7 + (1-2*(1e-7))*self.cumulative_normal_density_function_for_sum_of_normals(X=X, muhs=muhs, sigmas=sigmas) # Transform distribution to standard uniform
        Z = tfp.distributions.Normal(loc=tf.constant(0.0), scale=tf.constant(1.0)).quantile(P)#self.inverse_normal_cumulative_density_function(P=P) # Transform distribution to standard normal

        # Outputs
        return Z

# Define some functions
def create_network(Z_sample: np.ndarray, stage_count: int, dimensions_per_factor: List[int]) -> mfl.SupervisedFactorNetwork:
    """This function creates a supervised factor network that has ``stage_count`` many consecutive processing stages. Each stage
    consists of a reflection layer with 8 reflections, followed by a full coupling layer (two additive coupling layers, each
    followed by a SquareWave permutation, the latter with a negative mask) and an activation normalization layer. During inference,
    the network takes input :math:`Z` and outputs :math:'\tilde{Z}' which each have shape [:math:`M`, :math:`N`], where :math:`M` is
    the instance count and :math:`N` is the dimensionality.

    :param Z_sample: A representative sample of the data to be fed through the model. This is needed to initialize some layer
        parameters. Shape is assumed to be ["math"`M`, :math:`N`], where :math:`M` is the sample size and :math:`N` is the
        dimensionality of an instance.
    :type Z_sample: :class:`numpy.ndarray`
    :param stage_count: The number of processing stages that the ``network`` shall use.
    :type stage_count: int
    :param dimensions_per_factor: The number of dimensions that each factor shall have. The calibrated network will enumerate factors
        along the dimension axis in order and each will have as many dimensions as specified here. The residual factor at index 0 is
        expected to be included.
    :type dimensions_per_factor: List[int]
    :return: network (:class:`SupervisedFactorNetwork`) - The network.
    """

    # Set up the coupling functions and masks for the coupling layers
    dimensionality = Z_sample.shape[-1]
    layers = [None] * (7*stage_count)
    for i in range(stage_count):
        layers[7*i] = mfl.Shuffle(axes=[1], shape=[dimensionality])

        compute_coupling_parameters_1 = msl.BasicFullyConnectedNet(latent_dimension_count=dimensionality, output_dimension_count=dimensionality, depth=3)
        mask_1 = gmm.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[7*i+1] = mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1)
        layers[7*i+2] = mfl.CheckerBoard(axes=[1], shape=[dimensionality])

        compute_coupling_parameters_2 = msl.BasicFullyConnectedNet(latent_dimension_count=dimensionality, output_dimension_count=dimensionality, depth=3)
        mask_2 = gmm.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[7*i+3] = mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2)
        layers[7*i+4] = mfl.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[7*i+5] = mfl.Reflection(axes=[1], shape=[dimensionality], reflection_count=8)

        layers[7*i+6] = mfl.ActivationNormalization(axes=[1], shape=[dimensionality])

    # Construct the network
    network = mfl.SupervisedFactorNetwork(sequence=layers, dimensions_per_factor=dimensions_per_factor, sigma=0.95)
    network(Z_sample) # Initialization of some layer parameters

    # Outputs
    return network

def load_iterators(data_path: str, materials_to_drop: List[int], actions_to_drop: List[int], batch_size: int) -> Tuple[Generator, Generator]:

    # Load the data
    Z = np.load(os.path.join(data_path,'X.npy')); Y=np.load(os.path.join(data_path,'Y.npy'))

    # Eliminate classes to drop
    for m in materials_to_drop:
        Z = Z[Y[:,0] != m,:]; Y = Y[Y[:,0] != m,:]
    for a in actions_to_drop:
        Z = Z[Y[:,1] != a,:]; Y = Y[Y[:,1] != a,:]

    # Train test split
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z,Y, test_size=0.33, random_state=53)

    # Create the iterators
    def similarity_function(Y_a: np.ndarray, Y_b: np.ndarray) -> np.ndarray:
        Y_ab = Y_a == Y_b
        residual_factor_zeros = np.zeros([Y_ab.shape[0],1], dtype=Y_ab.dtype) # Shape == [instance count, 1]. For simplicity, instances are assumed to be residually different
        Y_ab = np.concatenate([residual_factor_zeros, Y_ab], axis=1)
        return Y_ab

    train_iterator = gmd.volatile_factorized_pair_iterator(X=Z_train, Y=Y_train, similarity_function=similarity_function, batch_size=batch_size)
    test_iterator = gmd.volatile_factorized_pair_iterator(X=Z_test, Y=Y_test, similarity_function=similarity_function, batch_size=batch_size)
    batch_count = len(Z_train)//batch_size

    return train_iterator, test_iterator, batch_count, Z_train, Z_test, Y_train, Y_test

def plot_calibration_trajectory(means_train, stds_train, means_validate, stds_validate, batch_size, plot_save_path):
    # Plot train error
    plt.figure(figsize=(10,5)); plt.title("Calibration Trajectory")
    standard_error_train = np.array(stds_train)/np.sqrt(batch_size)
    plt.fill_between(range(epoch_count), np.array(means_train)+2*standard_error_train, np.array(means_train)-2*standard_error_train, alpha=0.2)

    # Plot validation error
    standard_error_validate = np.array(stds_validate)/np.sqrt(batch_size)
    plt.fill_between(range(epoch_count), np.array(means_validate)+2*standard_error_validate, np.array(means_validate)-2*standard_error_validate, alpha=0.2)

    # Lines
    plt.plot(range(epoch_count), means_train)
    plt.plot(range(epoch_count), means_validate)

    plt.legend([r"Train Loss $\pm$ 2 SE", r"Validation Loss $\pm$ 2 SE"])
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(plot_save_path)
    plt.show()

if __name__ == "__main__":

    # Configuration
    inspection_layer_index = 9
    batch_size = 512
    random_seed = 678
    tf.keras.utils.set_random_seed(random_seed) # Sets the seed of python's random, numpy's random and tensorflow's random modules
    
    stage_count = 8
    epoch_count = 20
    dimensions_per_factor = [62,1,1]
    materials_to_keep = [0,1,2,3,4,5]; actions_to_keep = [0,1,2,3]
    materials_to_drop = list(range(6))
    for m in reversed(materials_to_keep): materials_to_drop.remove(m)
    actions_to_drop = list(range(4))
    for a in reversed(actions_to_keep): actions_to_drop.remove(a)
    m_string = ",".join(str(m) for m in materials_to_keep)
    a_string = ",".join(str(a) for a in actions_to_keep)
    data_path = os.path.join('data','latent yamnet',f'{np.sum(dimensions_per_factor)} dimensions',f'Layer {inspection_layer_index}')
    model_save_path = os.path.join('models', 'flow models',f'Layer {inspection_layer_index}')
    plot_save_path = os.path.join('plots','flow models', f'Layer {inspection_layer_index}')
    if not os.path.exists(model_save_path): os.makedirs(model_save_path)
    if not os.path.exists(plot_save_path): os.makedirs(plot_save_path)
    model_save_path = os.path.join(model_save_path, f'Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} seed {random_seed}.h5')
    
    # Load data iterators
    train_iterator, test_iterator, batch_count,_,Z_test,_,Y_test = load_iterators(data_path=data_path, materials_to_drop=materials_to_drop, actions_to_drop=actions_to_drop, batch_size=batch_size)
    Z_ab_sample, Y_ab_sample = next(train_iterator) # Sample
    
    print("The data is fed to the model in batches of shape:\n","Z_ab_sample: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_ab_sample: (instance count, factor count): \t', Y_ab_sample.shape)

    # Create network
    flow_network = create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
    #if os.path.exists(model_save_path): flow_network.load_weights(model_save_path)
    
    # Calibrate
    flow_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    flow_network.loss = SupervisedFactorLossMultiNormal(dimensions_per_factor=dimensions_per_factor, factor_to_muhs=[tf.constant([0.0]),tf.constant([-6.0,-2.0,2.0,6.0]),tf.constant([-6.0,-2.0,2.0,6.0])], factor_to_sigmas=[tf.constant([1.0]),tf.constant([1.0,1.0,1.0,1.0]),tf.constant([1.0,1.0,1.0,1.0])], sigma=0.99); del flow_network.__sigma__
    
    means_train, stds_train, means_validate, stds_validate = flow_network.fit(epoch_count=epoch_count, batch_count=batch_count, iterator=train_iterator, iterator_validate=test_iterator)
    print(means_train); print(means_validate)
    plot_calibration_trajectory(means_train=means_train, stds_train=stds_train, means_validate=means_validate, stds_validate=stds_validate, batch_size=batch_size, plot_save_path=os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Loss.png"))

    # Save existing model
    flow_network.save_weights(model_save_path)

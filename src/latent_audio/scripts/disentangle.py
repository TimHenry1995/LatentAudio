"""This script creates a flow model and calibrates it to disentangle latent factors of layerwise yamnet. It assumes that the data
is preprocessed by pre_process.py."""

from typing import List, Any
from gyoza.modelling import data_iterators as gmd, flow_layers as mfl, standard_layers as msl, masks as gmm
import os, numpy as np

# Configuration
inspection_layer_index = 3
data_path = os.path.join('data','pre-processed',f"Layer {inspection_layer_index}")
batch_size = 32
file_names = os.listdir(data_path)
for file_name in reversed(file_names):
    if '.npy' not in file_name: file_names.remove(file_name)

## Create data iterator
# Load file names
x_file_names = [None] * (len(file_names) // 2)
y_file_names = [None] * (len(file_names) // 2)
i = 0; j=0
for file_name in file_names:
    if "_X_" in file_name: 
        x_file_names[i] = file_name; i += 1
    if "_Y_" in file_name:
        y_file_names[j] = file_name; j += 1

# Get shape of X
x_shape = np.load(os.path.join(data_path, x_file_names[0])).shape

# Create the iterator
def similarity_function(Y_a: np.ndarray, Y_b: np.ndarray) -> np.ndarray:
    Y_ab = Y_a == Y_b
    residual_factor_zeros = np.zeros([Y_ab.shape[0],1], dtype=Y_ab.dtype) # Shape == [instance count, 1]. For simplicity, instances are assumed to be residually different
    Y_ab = np.concatenate([residual_factor_zeros, Y_ab], axis=1)
    return Y_ab

iterator = gmd.persistent_factorized_pair_iterator(data_path=data_path, x_file_names=x_file_names, y_file_names=y_file_names, similarity_function=similarity_function, batch_size=batch_size)
Z_ab_sample, Y_ab_sample = next(iterator) # Sample
print("The data is fed to the model in batches of shape:\n","Z: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_sample: (instance count, factor count): \t', Y_ab_sample.shape)

# Create a flow model
def create_network(Z_sample: np.ndarray, stage_count: int, dimensions_per_factor: List[int]) -> mfl.SupervisedFactorNetwork:
    """This function creates a supervised factor network that has ``stage_count`` many consecutive processing stages. Each stage 
    consists of a reflection layer with 8 reflections, followed by a full coupling layer (two additive coupling layers, each
    followed by a SquareWave permutation, the latter with a nagative mask) and an activation normalization layer. During inference, 
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
    dimensionality = Z_sample.shape[1]
    layers = [None] * (6*stage_count)
    for i in range(stage_count):
        layers[6*i] = mfl.Reflection(axes=[1], shape=[dimensionality], reflection_count=8)

        compute_coupling_parameters_1 = msl.BasicFullyConnectedNet(latent_dimension_count=dimensionality, output_dimension_count=dimensionality, depth=3)
        mask_1 = gmm.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[6*i+1] = mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1)
        layers[6*i+2] = mfl.CheckerBoard(axes=[1], shape=[dimensionality])

        compute_coupling_parameters_2 = msl.BasicFullyConnectedNet(latent_dimension_count=dimensionality, output_dimension_count=dimensionality, depth=3)
        mask_2 = gmm.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[6*i+3] = mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2)
        layers[6*i+4] = mfl.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[6*i+5] = mfl.ActivationNormalization(axes=[1], shape=[dimensionality])

    # Construct the network
    network = mfl.SupervisedFactorNetwork(sequence=layers, dimensions_per_factor=dimensions_per_factor) 
    network(Z_sample) # Initialization of some layer parameters

    # Outputs
    return network

# Take a few batches, to ensure enough similar pairs are found for dimensionality estimation
sample_count = 10
Z_ab_sample_large = [None] * sample_count; Y_ab_sample_large = [None] * sample_count
for i in range(sample_count):
    Z_ab_sample_large[i], Y_ab_sample_large[i] = next(iterator)
Z_ab_sample_large = np.concatenate(Z_ab_sample_large, axis=0)
Y_ab_sample_large = np.concatenate(Y_ab_sample_large, axis=0)

# Estimate dimensionality per factor
dimensions_per_factor = mfl.SupervisedFactorNetwork.estimate_factor_dimensionalities(Z_ab=Z_ab_sample_large, Y_ab=Y_ab_sample_large)

flow_network = create_network(Z_sample=Z_ab_sample, stage_count=5, dimensions_per_factor=[4,5,6])

# Calibrate
flow_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=tf.keras.losses.MeanSquaredError())
epoch_loss_means, epoch_loss_standard_deviations = flow_network.fit(epoch_count=2, batch_count=3, iterator=iterator)
plt.figure(figsize=(10,3)); plt.title('Loss Trajectory'); plt.plot(epoch_loss_means); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()
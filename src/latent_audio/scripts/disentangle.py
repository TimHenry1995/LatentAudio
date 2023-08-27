"""This script creates a flow model and calibrates it to disentangle latent factors of layerwise yamnet. It assumes that the data
is preprocessed by pre_process.py."""

from typing import List, Any, OrderedDict, Callable
from gyoza.modelling import data_iterators as gmd, flow_layers as mfl, standard_layers as msl, masks as gmm
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
# Configuration
inspection_layer_index = 8
data_path = os.path.join('data','pre-processed','64 PCA dimensions all in 1 file',f"Layer {inspection_layer_index}")
batch_size = 1024
np.random.seed(42) # Answer to everything
stage_count = 5
epoch_count = 10
batch_count = 10
model_save_path = os.path.join('models', f'{stage_count} stages {epoch_count} epochs {batch_count} batches.h5')
plot_save_path = os.path.join('plots','disentangle',f'{stage_count} stages {epoch_count} epochs {batch_count} batches')
## Create data iterator

# Create the iterator
def similarity_function(Y_a: np.ndarray, Y_b: np.ndarray) -> np.ndarray:
    Y_ab = Y_a == Y_b
    residual_factor_zeros = np.zeros([Y_ab.shape[0],1], dtype=Y_ab.dtype) # Shape == [instance count, 1]. For simplicity, instances are assumed to be residually different
    Y_ab = np.concatenate([residual_factor_zeros, Y_ab], axis=1)
    return Y_ab

Z = np.load(os.path.join(data_path,'X.npy')); Y=np.load(os.path.join(data_path,'Y.npy'))
Z_train, Z_test, Y_train, Y_test = train_test_split(Z,Y, test_size=0.33, random_state=42)
iterator = gmd.volatile_factorized_pair_iterator(X=Z_train, Y=Y_train, similarity_function=similarity_function, batch_size=batch_size)
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
    dimensionality = Z_sample.shape[-1]
    layers = [None] * (1+6*stage_count)
    layers[0] = mfl.ActivationNormalization(axes=[1], shape=[dimensionality])
    for i in range(stage_count):
        layers[6*i+1] = mfl.Reflection(axes=[1], shape=[dimensionality], reflection_count=8)

        compute_coupling_parameters_1 = msl.BasicFullyConnectedNet(latent_dimension_count=dimensionality, output_dimension_count=dimensionality, depth=3)
        mask_1 = gmm.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[6*i+2] = mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1)
        layers[6*i+3] = mfl.CheckerBoard(axes=[1], shape=[dimensionality])

        compute_coupling_parameters_2 = msl.BasicFullyConnectedNet(latent_dimension_count=dimensionality, output_dimension_count=dimensionality, depth=3)
        mask_2 = gmm.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[6*i+4] = mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2)
        layers[6*i+5] = mfl.CheckerBoard(axes=[1], shape=[dimensionality])

        layers[6*i+6] = mfl.ActivationNormalization(axes=[1], shape=[dimensionality])

    # Construct the network
    network = mfl.SupervisedFactorNetwork(sequence=layers, dimensions_per_factor=dimensions_per_factor) 
    network(Z_sample) # Initialization of some layer parameters

    # Outputs
    return network

# Take a few batches, to ensure enough similar pairs are found for dimensionality estimation
'''sample_count = 10
Z_ab_sample_large = [None] * sample_count; Y_ab_sample_large = [None] * sample_count
for i in range(sample_count):
    Z_ab_sample_large[i], Y_ab_sample_large[i] = next(iterator)
Z_ab_sample_large = np.concatenate(Z_ab_sample_large, axis=0)
Y_ab_sample_large = np.concatenate(Y_ab_sample_large, axis=0)

# Estimate dimensionality per factor
dimensions_per_factor = mfl.SupervisedFactorNetwork.estimate_factor_dimensionalities(Z_ab=Z_ab_sample_large, Y_ab=Y_ab_sample_large)
'''
dimensions_per_factor = [32,16,16]
flow_network = create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)

# Calibrate
flow_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=tf.keras.losses.MeanSquaredError())
epoch_loss_means, epoch_loss_standard_deviations = flow_network.fit(epoch_count=epoch_count, batch_count=batch_count, iterator=iterator)
plt.figure(figsize=(10,3)); plt.title('Loss Trajectory'); plt.plot(epoch_loss_means); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.savefig(plot_save_path + ' Loss.png'); plt.show()

# Save existing model
path = "example_model.h5"
flow_network.save_weights(model_save_path)
#new_model = create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)

# Load weights
#new_model.load_weights(model_save_path); 

# Compare
#print("The saved and the loaded model weights produce a prediction difference equal to", tf.reduce_sum((flow_network(Z_ab_sample[:,0,:])-new_model(Z_ab_sample[:,0,:]))**2))
#del new_model

def evaluate_factor_sensitivity(flow_network: mfl.SupervisedFactorNetwork, iterator: Callable, sample_size: int, dimensions_per_factor: List[int]) -> None:

    # Compute distance between pairs 
    delta_ma = compute_paired_differences(flow_network=flow_network, iterator=iterator, equal_material=True, equal_action=True, sample_size=sample_size) # ma means equal material, equal action
    delta_m = compute_paired_differences(flow_network=flow_network, iterator=iterator, equal_material=True, equal_action=False, sample_size=sample_size) # m means equal material, different action
    delta_a = compute_paired_differences(flow_network=flow_network, iterator=iterator, equal_material=False, equal_action=True, sample_size=sample_size) # a means different material, equal action

    count = 0
    distance = lambda x: tf.sqrt(tf.reduce_sum(x**2, axis=1)) # Shape of output == [instance count]
    factor_names = ['Residual','Material','Action']
    for f, dimension_count in enumerate(dimensions_per_factor):
        # Plot
        plt.figure(); plt.title(f'Sensitivity Of {factor_names[f]} Dimensions '+ r'Of $\tilde{Z}$' + '\nTo Changes In Z')
        plt.violinplot([distance(delta_ma[:,count:count+dimension_count]), distance(delta_m[:,count:count+dimension_count]), distance(delta_a[:,count:count+dimension_count])], showmeans=True)
        plt.xticks(ticks=[1,2,3], labels=['Same Material\nSame Action', 'Same Material\nOther Action', 'Other Material\nSame Action'])
        plt.ylabel('Distance'); plt.xlabel("Input Pair")
        count += dimension_count
        plt.savefig(plot_save_path + f' Sensitivity {factor_names[f]}.png')
    plt.show()

def compute_paired_differences(flow_network: mfl.SupervisedFactorNetwork, iterator: Callable, equal_material: bool, equal_action: bool, sample_size: int) -> None:

    # Initialization
    Z_ab = []
    current_sample_size = 0
    
    # Accumulate instances to meet the sample size
    while current_sample_size < sample_size:
        Z_ab_batch, Y_ab_batch = next(iterator)
        Z_ab.append(Z_ab_batch.numpy()[np.logical_and(Y_ab_batch[:,1] == equal_material, Y_ab_batch[:,2] == equal_action),:])
        current_sample_size += Z_ab[-1].shape[0]

    # Obtain proper shape
    Z_ab = np.concatenate(Z_ab, axis=0) # Instance axis
    Z_ab = Z_ab[:sample_size,:] # Shape == [sample_size, 2, ...] where 2 is for pair and ... is for a single instance 
    
    # Distance vectors in disentangled space
    delta = tf.reshape(flow_network(Z_ab[:,0,:]) - flow_network(Z_ab[:,1,:]), [sample_size, -1]) # Shape == [instance count, dimension count]
    
    # Output
    return delta

evaluate_factor_sensitivity(flow_network=flow_network, iterator=iterator, sample_size=100, dimensions_per_factor=dimensions_per_factor)
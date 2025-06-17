"""This script creates a flow model and calibrates it on the projected latent represenetations to disentangle the underlying factors material and action and then creates several plots to illustrate the outcome.
Each factor shall be disentangled into their respective clases, independent of the other factor. It is possible to remove some classes because they might not all be separable.
It is assumed that latent_yamnet_to_calibration_data_set.run() was executed apriori.
"""
import sys
sys.path.append(".")

from typing import List, Any, OrderedDict, Callable, Generator, Tuple
from gyoza.modelling import data_iterators as gmd, flow_layers as mfl, standard_layers as msl, masks as gmm
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings, random
from sklearn.decomposition import PCA
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Define some functions
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

        layers[7*i+5] = mfl.ActivationNormalization(axes=[1], shape=[dimensionality])

        layers[7*i+6] = mfl.Reflection(axes=[1], shape=[dimensionality], reflection_count=8)

    # Construct the network
    network = mfl.SupervisedFactorNetwork(sequence=layers, dimensions_per_factor=dimensions_per_factor, sigma=0.9)
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
    
    # Load Configuration
    import json, os
    with open(os.path.join('LatentAudio','configuration.json'),'r') as f:
        configuration = json.load(f)

    batch_size = configuration['flow_model_batch_size']
    np.random.seed(configuration['random_seed'])
    tf.keras.utils.set_random_seed(configuration['random_seed'])
    random.seed(configuration['random_seed'])
    stage_count = configuration['flow_model_stage_count']
    epoch_count = configuration['flow_model_epoch_count']
    dimensions_per_factor = configuration['flow_model_dimensions_per_factor']
    materials_to_keep = configuration['flow_model_materials_to_keep']; actions_to_keep = configuration['flow_model_actions_to_keep']
    materials_to_drop = list(range(6))
    for m in reversed(materials_to_keep): materials_to_drop.remove(m)
    actions_to_drop = list(range(4))
    for a in reversed(actions_to_keep): actions_to_drop.remove(a)
    m_string = ",".join(str(m) for m in materials_to_keep)
    a_string = ",".join(str(a) for a in actions_to_keep)
    
    data_path = os.path.join(configuration['latent_yamnet_data_folder'],'projected', f'{np.sum(configuration['dimensions_per_factor'])} dimensions',f'Layer {configuration['layer_index_full_PCA']}')
    model_save_path = os.path.join(configuration['model_folder'], 'flow models',f'Layer {configuration['layer_index_full_PCA']}')
    plot_save_path = os.path.join(configuration['plots_folder'],'flow models', f'Layer {configuration['layer_index_full_PCA']}')
    if not os.path.exists(model_save_path): os.makedirs(model_save_path)
    if not os.path.exists(plot_save_path): os.makedirs(plot_save_path)
    model_save_path = os.path.join(model_save_path, f'Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count}.h5')

    # Load data iterators
    train_iterator, test_iterator, batch_count,_,Z_test,_,Y_test = load_iterators(data_path=data_path, materials_to_drop=materials_to_drop, actions_to_drop=actions_to_drop, batch_size=batch_size)
    Z_ab_sample, Y_ab_sample = next(train_iterator) # 1 Sample

    print("The data is fed to the model in batches of shape:\n","Z_ab_sample: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_ab_sample: (instance count, factor count): \t', Y_ab_sample.shape)

    # Create network
    flow_network = create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
    if os.path.exists(model_save_path): flow_network.load_weights(model_save_path)

    # Calibrate
    flow_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
    means_train, stds_train, means_validate, stds_validate = flow_network.fit(epoch_count=epoch_count, batch_count=batch_count, iterator=train_iterator, iterator_validate=test_iterator)

    plot_calibration_trajectory(means_train=means_train, stds_train=stds_train, means_validate=means_validate, stds_validate=stds_validate, batch_size=batch_size, plot_save_path=os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Loss.png"))

    # Save existing model
    flow_network.save_weights(model_save_path)
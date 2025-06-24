"""This script creates a flow model and calibrates it on the projected latent represenetations to disentangle the underlying factors material and action and then creates several plots to illustrate the outcome.
Each factor shall be disentangled into their respective clases, independent of the other factor. It is possible to remove some classes because they might not all be separable.
It is assumed that latent_yamnet_to_calibration_data_set.run() was executed apriori.
"""
import sys
sys.path.append(".")
from LatentAudio.configurations import loader as configuration_loader
from typing import List, Any, OrderedDict, Callable, Generator, Tuple
from gyoza.modelling import data_iterators as gmd, flow_layers as mfl, standard_layers as msl, masks as gmm
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import test_proportion
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

def load_iterators(data_path: str, factor_index_to_included_class_indices: Dict[List[int]], batch_size: int, validation_proportion: float, random_seed: int) -> Tuple[Generator, Generator]:

    # Load the data
    Z = np.load(os.path.join(data_path,'X.npy')); Y=np.load(os.path.join(data_path,'Y.npy'))

    # Eliminate classes to drop
    for factor_index, included_class_indices in factor_index_to_included_class_indices.items():
        classes_to_drop = set(Y[:,factor_index]) - set(included_class_indices)
        for c in classes_to_drop:
            Z = Z[Y[:,0] != c,:]; Y = Y[Y[:,0] != c,:]
    
    # Train validation split
    Z_train, Z_validation, Y_train, Y_validation = test_proportion(Z,Y, test_size=validation_proportion, random_state=random_seed)

    # Create the iterators
    def similarity_function(Y_a: np.ndarray, Y_b: np.ndarray) -> np.ndarray:
        Y_ab = Y_a == Y_b
        residual_factor_zeros = np.zeros([Y_ab.shape[0],1], dtype=Y_ab.dtype) # Shape == [instance count, 1]. For simplicity, instances are assumed to be residually different
        Y_ab = np.concatenate([residual_factor_zeros, Y_ab], axis=1)
        return Y_ab

    train_iterator = gmd.volatile_factorized_pair_iterator(X=Z_train, Y=Y_train, similarity_function=similarity_function, batch_size=batch_size)
    validation_iterator = gmd.volatile_factorized_pair_iterator(X=Z_validation, Y=Y_validation, similarity_function=similarity_function, batch_size=batch_size)
    batch_count = len(Z_train)//batch_size

    return train_iterator, validation_iterator, batch_count, Z_train, Z_validation, Y_train, Y_validation

def plot_calibration_trajectory(means_train, stds_train, means_validate, stds_validate, batch_size, plot_save_path, epoch_count):
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

    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="create_scalers_and_PCA_model_for_latent_yamnet",
        description='''This script loads the calibration data stored by latent_yamnet_to_calibration_data_set at `calibration_data_set_folder` and calibrates a flow model on it.
                    The flow model consits of multiple stages. Each stage uses a set of permutation, coupling, activation normalization and reflection layers. 
                    Calibration involves creating pairs of instances and evaluating their equality for each factor. The model is then optimized to cluster like-wise instances along their
                    corresponding factor dimensions in the output. The optimizer used for calibration is the Adam optimizer.
                    The calibrated model will be saved at `flow_model_folder' with the name 'Flow model <'stage_count`> stages, <`epoch_count`> epochs, <`dimensions_per_factor`> dimensions per factor and <`factor_index_to_included_class_indices`> included class indices'
                    and a line-plot showing the calibration trajectory will be saved at `figure_folder` with the name 'Flow model Loss <'stage_count`> stages, <`epoch_count`> epochs, <`dimensions_per_factor`> dimensions per factor and <`factor_index_to_included_class_indices`> included class indices'. 
                    Since these file names can get very long, the overall file-paths will be truncated after 260 characters. If files of same name already exist at their target folders, the existing ones will be renamed using the appendix ' (old) ' and a time-stamp. 
                                                
                    There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                    The second way is to manually pass all these other arguments while calling the script.
                    For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                    When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')
    
    parser.add_argument("--random_seed", help="An int (in form of a json string) that is used to set the random module or Python, numpy and keras to make the model parameter initialization and instance sampling reproducible", type=str)
    parser.add_argument("--stage_count", help="An int (in form of a json string) that is used to set the number of stages in the flow model. The more stages are used, the more complex the model will be.", type=str)
    parser.add_argument("--batch_size", help="An int (in form of a json string) that is used to control the number of instances per batch that are fed to the flow model during training.", type=str)
    parser.add_argument("--epoch_count", help="An int (in form of a json string) indicating how many iterations through the dataset are made during training. An iteration is defined by finding one partner instance for each instance in the given training set.", type=str)
    parser.add_argument("--learning_rate", help="A float (in form of a json string) indicating the learning rate passed to the optimizer", type=str)
    parser.add_argument("--validation_proportion", help="A float (in form of a json string) indicating the proportions of the calibration data that should be used for validating the model.", type=str)
    parser.add_argument("--dimensions_per_factor", help="A list of ints (in form of a json string) indicating how many dimensions each factor should be allocated in the output of the flow model. The zeroth entry is for the residual factor, the first entry is for the first factor, the second entry for the second factor, etc. The sum of dimensions has to be equal to the dimensionality of the input to the flow model.", type=str)
    parser.add_argument("--factor_index_to_included_class_indices", help="A dictionary (in form of a json string) that maps the index of a factor to the indices of its classes that should be included in the training. This is a way to exclude unwanted classes from the calibration data if needed. The indexing of factors has to be in line with that in the Y files of the calibration data, i.e. the residual factor is excluded.", type=str)
    parser.add_argument("--calibration_data_set_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the projections are stored.", type=str)
    parser.add_argument("--flow_model_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder where the flow model shall be stored.", type=str)
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plot of the calibration trajectory should be saved.")

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.random_seed != None and args.stage_count != None and args.batch_size != None and args.epoch_count != None and args.learning_rate != None and args.validation_proportion != None and args.dimensions_per_factor != None and args.factor_index_to_included_class_indices != None and args.calibration_data_set_folder != None and args.flow_model_folder != None and args.figure_folder != None, "If no configuration file is provided, then all other arguments must be provided."
    
        random_seed = json.loads(args.random_seed)
        stage_count = json.loads(args.stage_count)
        batch_size = json.loads(args.batch_size)
        epoch_count = json.loads(args.epoch_count)
        learning_rate = json.loads(learning_rate)
        validation_proportion = json.loads(args.validation_proportion)
        dimensions_per_factor = json.loads(args.dimensions_per_factor)
        factor_index_to_included_class_indices = json.loads(args.factor_index_to_included_class_indices)
        calibration_data_set_folder = json.loads(args.calibration_data_set_folder)
        calibration_data_set_folder_path = os.path.join(*calibration_data_set_folder)
        flow_model_folder = json.loads(args.flow_model_folder)
        flow_model_folder_path = os.path.join(*flow_model_folder)
        figure_folder = json.loads(args.figure_folder)
        figure_folder_path = os.path.join(*figure_folder)
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.random_seed == None and args.stage_count == None and args.batch_size == None and args.epoch_count == None and args.learning_rate == None and args.validation_proportion == None and args.dimensions_per_factor == None and args.factor_index_to_included_class_indices == None and args.calibration_data_set_folder == None and args.flow_model_folder == None and args.figure_folder == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'disentangle' or configuration['script'] == 'disentangle.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'disentangle'."
        
        random_seed = configuration['arguments']['random_seed']
        stage_count = configuration['arguments']['stage_count']
        batch_size = configuration['arguments']['batch_size']
        epoch_count = configuration['arguments']['epoch_count']
        learning_rate = configuration['arguments']['learning_rate']
        validation_proportion = configuration['arguments']['validation_proportion']
        dimensions_per_factor = configuration['arguments']['dimensions_per_factor']
        factor_index_to_included_class_indices = configuration['arguments']['factor_index_to_included_class_indices']
        calibration_data_set_folder_path = os.path.join(*configuration['arguments']['calibration_data_set_folder'])
        flow_model_folder_path = os.path.join(*configuration['arguments']['flow_model_folder'])
        figure_folder_path = os.path.join(*configuration['arguments']['figure_folder'])
        
    print("\n\n\tStarting script disentangle")
    print("\t\tThe script parsed the following arguments:")
    print("\t\trandom_seed: ", random_seed)
    print("\t\tstage_count: ", stage_count)
    print("\t\tbatch_size: ", batch_size)
    print("\t\tepoch_count: ", epoch_count)
    print("\t\tlearning_rate: ",learning_rate)
    print("\t\tvalidation_proportion: ", validation_proportion)
    print("\t\tdimensions_per_factor: ", dimensions_per_factor)
    print("\t\tfactor_index_to_included_class_indices: ", factor_index_to_included_class_indices)
    print("\t\tcalibration_data_set_folder path: ", calibration_data_set_folder_path)
    print("\t\tflow_model_folder path: ", flow_model_folder_path)
    print("\t\tfigure_folder path: ", figure_folder_path)
    print("\n\tStarting script now:\n")
    
    ### Start actual data processing

    # Set random
    np.random.seed(random_seed)
    tf.keras.utils.set_random_seed(random_seed)
    random.seed(random_seed)
    
    # File management
    if not os.path.exists(flow_model_folder_path): os.makedirs(flow_model_folder_path)
    if not os.path.exists(figure_folder path): os.makedirs(figure_folder path)
    flow_model_file_path = os.path.join(flow_model_folder_path, f'Flow model {stage_count} stages, {epoch_count} epochs, {dimensions_per_factor} dimensions per factor and {factor_index_to_included_class_indices} included class indices')
    flow_model_file_path = flow_model_file_path[:257] + '.h5' # Trim path if too long
    flow_model_calibration_figure_file_path = os.path.join(figure_folder_path, f'Flow model loss {stage_count} stages, {epoch_count} epochs, {dimensions_per_factor} dimensions per factor and {factor_index_to_included_class_indices} included class indices')
    flow_model_calibration_figure_file_path = flow_model_calibration_figure_file_path[:256] + '.png' # Trim path if too long

    # Load data iterators
    train_iterator, validation_iterator, batch_count,_,Z_validation,_,Y_validation = load_iterators(data_path=data_path, factor_index_to_included_class_indices=factor_index_to_included_class_indices, batch_size=batch_size, validation_proportion=validation_proportion, random_seed=random_seed)
    Z_ab_sample, Y_ab_sample = next(train_iterator) # 1 Sample

    print("\t\tThe data is fed to the model in batches of shape:\n","Z_ab_sample: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_ab_sample: (instance count, factor count): \t', Y_ab_sample.shape)

    # Create network
    flow_network = create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
    #if os.path.exists(flow_model_file_path): flow_network.load_weights(flow_model_file_path)

    # Calibrate
    flow_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    means_train, stds_train, means_validation, stds_validation = flow_network.fit(epoch_count=epoch_count, batch_count=batch_count, iterator=train_iterator, iterator_validate=validation_iterator)

    plot_calibration_trajectory(means_train=means_train, stds_train=stds_train, means_validate=means_validation, stds_validate=stds_validation, batch_size=batch_size, plot_save_path=flow_model_calibration_figure_file_path, epoch_count=epoch_count)

    # Save existing model
    flow_network.save_weights(flow_model_file_path)

    # Log
    print("\n\n\Completed script disentangle")
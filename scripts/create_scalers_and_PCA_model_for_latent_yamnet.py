import sys
sys.path.append(".")
import os, shutil
from LatentAudio import utilities as utl
from LatentAudio.configurations import loader as configuration_loader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
from typing import Tuple
import random
import matplotlib.pyplot as plt
from typing import Dict

def run(layer_index: int,
        X_folder_path: str, 
        PCA_and_standard_scaler_folder_path: str,
        target_dimensionality: int,
        random_seed: int) -> Tuple[int, np.ndarray]:
    """Creates a standard scaler to be used before principal component analysis (PCA), a PCA model and a standard scaler to be used after PCA. 
    The data is expected to be stored at ``X_folder_path`` in the same format as output by the ``audio_to_latent_yamnet.run`` function. The here 
    created models are not used to perform the projection of the data. Instead, the models are simply saved in ``PCA_and_standard_scaler_folder_path``. 
    Before saving the models into the specified folder, that folder (if exists) is deleted and any other files inside it will thus be lost.

    :param layer_index: The index of the Yamnet layer for which the latent Yamnet data shall be loaded.
    :type layer_index: int
    :param X_folder_paht: The path to the folder where the latent Yamnet data is located.
    :type X_folder_path: str
    :param PCA_and_standard_scaler_folder_path: The path to the folder where the standard scalers and PCA model shall be stored.
    :type PCA_and_standard_scaler_folder_path: str
    :param target_dimensionality: The dimensionality that PCA should have for its output. Since a complete PCA model with same output as input dimensionality is resource intensive (for early layers even prohibitively expensive), it is recommended to keep this value as small as possible. Most layers will only need the forward PCA model and hence as small model is sufficient, e.g. 64 dimensions (default), for latent space exploration. If latent space manipulation is planned for the current layer, then a full PCA model is required. The full model will automatically be created if target_dimensionality is set to None.
    :type target_dimensionality: int
    :param random_seed: The seed used to set the random module of python right before taking a sample of instances based on which the models will be calibrated.
    :type random_seed: int
    :return: dimensions (Tuple[int, numpy.ndarray]) - The int is the original dimensionality of the layer and the array has shape [`target_dimensionality`] and lists the proportion of variance explained by the first each of the first `targte_dimensionality` many dimensions of PCA.
    """

    print(f"Running script to create scalers and PCA model for latent yamnet layer {layer_index}.")

    random.seed(random_seed)
    X_layer_folder = os.path.join(X_folder_path, f'Layer {layer_index}')
    PCA_and_standard_scaler_folder_path = os.path.join(PCA_and_standard_scaler_folder_path, f'{target_dimensionality if target_dimensionality != None else "All"} dimensions')
    PCA_and_standard_scaler_folder_path = os.path.join(PCA_and_standard_scaler_folder_path, f"Layer {layer_index}")
    if os.path.exists(PCA_and_standard_scaler_folder_path): shutil.rmtree(PCA_and_standard_scaler_folder_path)
    os.makedirs(PCA_and_standard_scaler_folder_path)

    # If no target_dimensionality provided, prepare a full pca model (costs memory, disk storage and may lead to index overflow)
    # This makes sense for layers whose original dimensionality is small enough anyways to afford a complete pca model, e.g. Yamnet layer 9
    if target_dimensionality == None:
        # Load one instance to get shape
        X_tmp, _ = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=1) # Shape == [sample size = 1, dimensionality]
        target_dimensionality = X_tmp.shape[1] 
        sample_size = (int)(1.1 * target_dimensionality) # PCA needs dimensionality at least as many unique data points as target dimensions. Here we take a few more data points to hopefully have this many unique ones
        del X_tmp
    else: # Prepare small pca model, suited for any Yamnet layer
        sample_size = 10000 # For a small pca model, e.g. 64 target dimensions, this many instances should suffice

    # Load sample
    print("\tLoading sample of latent data", end='')
    X_sample, Y_sample = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=sample_size)
    print(f" completed. Shape == [instance count, dimensionality] == {X_sample.shape}")

    # Fit scaler
    print("\tFitting Pre-PCA Standard Scaler to sample", end='')
    pre_scaler = StandardScaler()
    X_sample = pre_scaler.fit_transform(X_sample)
    print(" completed")

    # Fit PCA
    print(f"\tFitting {target_dimensionality}-dimensional PCA to sample", end='')
    pca = PCA(n_components=target_dimensionality)
    pca.fit(X_sample)
    print(" completed")

    # Fit scaler
    print("\tFitting Post-PCA Standard Scaler to sample", end='')
    post_scaler = StandardScaler()
    post_scaler.fit(pca.transform(X_sample))
    print(" completed")

    # Save
    with open(os.path.join(PCA_and_standard_scaler_folder_path, "Pre PCA Standard Scaler.pkl"),"wb") as file_handle:
        pkl.dump(pre_scaler, file_handle)
        
    with open(os.path.join(PCA_and_standard_scaler_folder_path, f"PCA.pkl"),"wb") as file_handle:
        pkl.dump(pca, file_handle)

    with open(os.path.join(PCA_and_standard_scaler_folder_path, "Post PCA Standard Scaler.pkl"), "wb") as file_handle:
        pkl.dump(post_scaler, file_handle)
    print("\tRun Completed")

    # Outputs
    return X_sample.shape[1], pca.explained_variance_ratio_

def plot(figure_output_folder: str,
         layer_index_to_dimensionality: Dict[int, str],
         layer_index_to_explained_variances: Dict[int, np.ndarray]) -> None:
    """
    Creates two plots to illustrate the dimensionality distribution over Yamnet's layers (plot 1) and the distribution of explained variance over Yament's layers (plot 2).

    :param figure_output_folder: The path pointing to the folder where the plots shall be stored.
    :type figure_output_folder: str
    :param layer_index_to_dimensionality: A dictionary mapping each layer index of Yamnet to the number of dimensions that the corresponding layer has.
    :type layer_index_to_dimensionality: Dict[int, str]
    :param layer_index_to_explained_variances: A dictionary mapping each layer index of Yamnet to a numpy vector storing the proportion of variance explained by principal components of that layer's space.
    :type layer_index_to_explained_variances: Dict[int, numpy.ndarray]
    """

    # Ensure folder exists
    if not os.path.exists(figure_output_folder): os.makedirs(figure_output_folder)

    # Prepare plot for proportion of variance in the original data that is explained by the variance that is in the projection
    dimensionality = len(list(layer_index_to_explained_variances.values())[0])
    plt.figure(figsize=(10,5)); plt.title(f"Principal Component Analysis ({dimensionality} components)")

    # Iterate the layers
    for layer_index in layer_index_to_explained_variances.keys():
        
        # Plot the proportion of variance
        plt.gca().set_prop_cycle(None)
        R = 0
        for i, r in enumerate(layer_index_to_explained_variances[layer_index]):
            plt.bar([str(layer_index)],[r], bottom=R, color='white', edgecolor='black')
            R += r

        plt.ylim(0,1)
    plt.ylabel('Explained Variance'); plt.xlabel('Layer')
    plt.savefig(os.path.join(figure_output_folder, f"Principal Component Analysis"))
    plt.show()

    # Plot original dimensionalities of Yamnet
    plt.figure(figsize=(10,5)); plt.title("Original Dimensionalities")
    plt.bar([f'{key}' for key in layer_index_to_dimensionality.keys()], layer_index_to_dimensionality.values(), color='white', edgecolor='black')
    plt.ylabel("Dimensionality"); plt.xlabel('Layer')
    plt.savefig(os.path.join(figure_output_folder, "Original Dimensionalities"))
    plt.show()

if __name__ == "__main__": 

    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="audio_to_latent_yamnet",
        description='''This script loads a set of sound files from the given sounds_folder and passes each of them through the Yamnet convolutional neural network. 
                        For a given sound, Yamnet will create a spectrogram and split it into ca. 1-second long, partially ovelapping slices. 
                        Each slice is then passed through Yamnet's convolutional layers. 
                        At each of the here specified layer_indices, the latent representation of a given slice will be extracted and saved to disk.
                        Saving takes place in the provided latent_representations_folder by first creating a subfolder named as the current Yamnet layer (if it does not already exist). 
                        Inside that subfolder, each slice is saved with the file name <waveform file name>_X_<slice index>.npy.
                        For example, if the waveform file for Metal Tapping was called 'MT.wav', the slice files would be called 'MT_X_1.npy', 'MT_X_2.npy', etc.
                        
                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--sounds_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder containing the sound data in .wav format. This sound data needs to be sampled at 48 Khz with int16 bit rate.", type=str)
    parser.add_argument("--latent_representations_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the latent representations shall be stored. If the folder does not yet exist, it will be created. If it already exists, it will not be replaced.", type=str)
    parser.add_argument("--layer_indices", help="A list containing the indices of the Yamnet layers for which the latent vectors of sounds shall be computed.", type=str)
    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.sounds_folder != None and args.latent_representations_folder != None and args.layer_indices != None, "If no configuration file is provided, then all other arguments must be provided."
    
        sounds_folder = json.loads(args.sounds_folder)
        sounds_folder_path = os.path.join(*sounds_folder)
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        layer_indices = json.loads(args.layer_indices)

    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.sounds_folder == None and args.latent_representations_folder == None and args.layer_indices == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'audio_to_latent_yamnet' or configuration['script'] == 'audio_to_latent_yamnet.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equalt to 'audio_to_latent_yamnet'."
        
        sounds_folder_path = os.path.join(*configuration['arguments']['sounds_folder'])
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        layer_indices = configuration['arguments']['layer_indices']

    print("\n\nThe script audio_to_latent_yamnet parsed the following arguments:")
    print("\tsounds_folder path: ", sounds_folder_path)
    print("\tlatent_representations_folder path: ", latent_representations_folder_path)
    print("\tlayer_indices: ", layer_indices)
    print("Starting script now:\n")

    ### Start actual data processing

    # Modelling
    target_dimensionality = configuration['PCA_target_dimensionality'] # The number of dimensions of the projections
    layer_index_to_dimensionality = {} # Will hold the dimensionality of each Yamnet layer
    layer_index_to_explained_variances = {} # Will hold the proportion of variance explained by the projections for each layer
    
    # Create PCA model and standard scalers of small dimensionality for most layers
    for layer_index in configuration['layer_indices_small_PCA']:
        layer_index_to_dimensionality[layer_index], layer_index_to_explained_variances[layer_index] = run(layer_index=layer_index,
                                                                                                          X_folder_path = os.path.join(configuration['latent_yamnet_data_folder'], 'original'),
                                                                                                          PCA_and_standard_scaler_folder_path = os.path.join(configuration['model_folder'], 'PCA and Standard Scalers'),
                                                                                                          target_dimensionality = target_dimensionality,
                                                                                                          random_seed=configuration['random_seed'])
        
    # Create PCA model and standard scalers of complete dimensionality for the full dimensionality layer
    layer_index_to_dimensionality[layer_index], tmp = run(layer_index=configuration['layer_index_full_PCA'],
                                                          X_folder_path = os.path.join(configuration['latent_yamnet_data_folder'], 'original'),
                                                          PCA_and_standard_scaler_folder_path = os.path.join(configuration['model_folder'], 'PCA and Standard Scalers'),
                                                          target_dimensionality = None,
                                                          random_seed=configuration['random_seed'])
    layer_index_to_explained_variances[layer_index] = tmp[:target_dimensionality] # Only keep the explained variances for the first few dimensions for plotting purposes (see next code block).

    # Plotting
    plot(figure_output_folder = os.path.join(configuration['plots_folder'],'explore latent yamnet',f'{target_dimensionality} dimensions'),
        layer_index_to_dimensionality = layer_index_to_dimensionality,
        layer_index_to_explained_variances = layer_index_to_explained_variances)
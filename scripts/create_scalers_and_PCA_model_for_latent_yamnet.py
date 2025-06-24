import sys, argparse, json
sys.path.append(".")
import os, shutil
from LatentAudio import utilities as utl
from LatentAudio.configurations import loader as configuration_loader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
from typing import Tuple
import random, time
import matplotlib.pyplot as plt
from typing import Dict

if __name__ == "__main__": 
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="create_scalers_and_PCA_model_for_latent_yamnet",
        description='''This script creates a standard scaler to be used before principal component analysis (PCA), a PCA model and a standard scaler to be used after PCA. 
                        The data is expected to be stored at `latent_representations_folder` in the same format as produced by the `audio_to_latent_yamnet` script. 
                        A sample will be drawn from `latent_representations_folder` for each layer. The sample size is given by `sample_sizes` and the sampling happens
                        uniformly at random with random seed given by `random_seeds` and without replacement.
                        Note, the here created models are NOT used to perform the projection of the data. Instead, the models are only fitted and saved at `PCA_and_standard_scaler_folder`. 
                        Before saving the models into the specified folder, that folder (if exists) is deleted and any other files inside it will thus be lost.
                        
                        There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                        The second way is to manually pass all these other arguments while calling the script.
                        For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                        When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')

    parser.add_argument("--latent_representations_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the latent representations are stored.", type=str)
    parser.add_argument("--PCA_and_standard_scaler_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the models shall be stored.", type=str)
    parser.add_argument("--layer_indices", help="A list (in form of a json string) containing the indices of the Yamnet layers for which the standard scalers and PCA shall be computed.", type=str)
    parser.add_argument("--target_dimensionalities", help="A list of integers (or None entries) (in form of a json string) indicating for each layer in the layer_indices list which dimensionality the corresponding PCA model should have for its output. Since a complete PCA model with same output as input dimensionality is resource intensive (for early layers even prohibitively expensive), it is recommended to keep this value small (e.g. 64). Most layers will only need the forward PCA model and hence as small model is sufficient for latent space exploration. If latent space manipulation is planned for a layer, then a full PCA model is required. The full model will automatically be created if target_dimensionality is set to None.", type=str)
    parser.add_argument("--random_seeds", help="A list of integers (in form of a json string) indicating for each layer in the layer_indices list which seed shall be used to set the random module of Python right before taking a sample of instances based on which the models will be calibrated. This argument helps wth reproducibility.", type=str)
    parser.add_argument("--sample_sizes", help="A list of integers (or None entries) (in form of a json string) indicating for each layer in the layer_indices list which sample size shall be used. Keep in mind that in order to train a PCA mode, one needs at least as many unique instances as output dimensions. If a None is specified, then 1.1 * the target dimensionality of the current layer is used as the sample size. An eexception will be raised if the sample size is larger than the number of available instances.", type=str)
    
    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.latent_representations_folder != None and args.PCA_and_standard_scaler_folder != None and args.layer_indices != None and args.target_dimensionalities != None and args.random_seeds != None and args.sample_sizes != None, "If no configuration file is provided, then all other arguments must be provided."
    
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        PCA_and_standard_scaler_folder = json.loads(args.PCA_and_standard_scaler_folder)
        PCA_and_standard_scaler_folder_path = os.path.join(*PCA_and_standard_scaler_folder)
        layer_indices = json.loads(args.layer_indices)
        target_dimensionalities = json.loads(args.target_dimensionalities)
        random_seeds = json.loads(args.random_seeds)
        sample_sizes = json.loads(args.random_seed)

    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert  args.latent_representations_folder == None and args.PCA_and_standard_scaler_folder == None and args.layer_indices == None and args.target_dimensionalities == None and args.random_seeds == None and args.sample_sizes == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'create_scalers_and_PCA_model_for_latent_yamnet' or configuration['script'] == 'create_scalers_and_PCA_model_for_latent_yamnet.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'create_scalers_and_PCA_model_for_latent_yamnet'."
        
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        PCA_and_standard_scaler_folder_path = os.path.join(*configuration['arguments']['PCA_and_standard_scaler_folder'])
        layer_indices = configuration['arguments']['layer_indices']
        target_dimensionalities = configuration['arguments']['target_dimensionalities']
        random_seeds = configuration['arguments']['random_seeds']
        sample_sizes = configuration['arguments']['sample_sizes']

    print("\n\n\tStarting script create_scalers_and_PCA_model_for_latent_yamnet")
    print("\t\tThe script parsed the following arguments:")
    print("\t\tlatent_representations_folder path: ", latent_representations_folder_path)
    print("\t\tPCA_and_standard_scaler_folder path: ", PCA_and_standard_scaler_folder_path)
    print("\t\tlayer_indices: ", layer_indices)
    print("\t\ttarget_dimensionalities: ", target_dimensionalities)
    print("\t\trandom_seeds: ", random_seeds)
    print("\t\tsample_sizes: ", sample_sizes)
    print("\n\tStarting script now:\n")

    ### Start actual data processing
    
    # Remove any previous outputs
    if os.path.exists(PCA_and_standard_scaler_folder_path): 
        print(f"\t\tFound existing folder at {PCA_and_standard_scaler_folder_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(PCA_and_standard_scaler_folder_path,
                  PCA_and_standard_scaler_folder_path + " (old) " + str(time.time()))
        
    # Create PCA model and standard scalers
    for l, layer_index in enumerate(layer_indices):
        print(f"\n\t\tLayer {layer_index}.")

        # Set random seed for sampling
        random_seed = random_seeds[l]
        random.seed(random_seed)

        # Create new folder at path
        latent_representations_folder_path = os.path.join(latent_representations_folder_path, f'Layer {layer_index}')
        PCA_and_standard_scaler_folder_path = os.path.join(PCA_and_standard_scaler_folder_path, f"Layer {layer_index}")
        os.makedirs(PCA_and_standard_scaler_folder_path)

        # If no sample_size provided, prepare a full pca model (costs memory, disk storage and may lead to index overflow)
        # This makes sense for layers whose original dimensionality is small enough anyways to afford a complete pca model, e.g. Yamnet layer 9
        target_dimensionality = target_dimensionalities[l]
        if target_dimensionality == None:
            # Load one instance to get shape
            X_tmp, _ = utl.load_latent_sample(data_folder=latent_representations_folder_path, sample_size=1) # Shape == [sample size = 1, dimensionality]
            target_dimensionality = X_tmp.shape[1] 
            del X_tmp

        # Determine sample size
        sample_size = sample_sizes[l]
        if sample_size == None:
            sample_size = (int)(1.1 * target_dimensionality) # PCA needs dimensionality at least as many unique data points as target dimensions. Here we take a few more data points to hopefully have this many unique ones
                   
        # Load sample
        print("\t\t\tLoading sample of latent data", end='')
        X_sample, Y_sample = utl.load_latent_sample(data_folder=latent_representations_folder_path, sample_size=sample_size)
        print(f" completed. Shape == [instance count, dimensionality] == {X_sample.shape}")

        # Fit scaler
        print("\t\t\tFitting Pre-PCA Standard Scaler to sample", end='')
        pre_scaler = StandardScaler()
        X_sample = pre_scaler.fit_transform(X_sample)
        print(" completed")

        # Fit PCA
        print(f"\t\t\tFitting {target_dimensionality}-dimensional PCA to sample", end='')
        pca = PCA(n_components=target_dimensionality)
        pca.fit(X_sample)
        print(" completed")

        # Fit scaler
        print("\t\t\tFitting Post-PCA Standard Scaler to sample", end='')
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
    
    # Log
    print("\n\n\tCompleted script create_scaler_and_PCA_models_for_latent_yamnet.")
        
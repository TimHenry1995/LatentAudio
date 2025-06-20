import sys
sys.path.append(".")
from LatentAudio import utilities as utl
from LatentAudio.configurations import loader as configuration_loader
import os, pickle as pkl, numpy as np, shutil
def run(layer_index: int,
    dimensionality: int, 
    input_data_path: str, 
    pca_path : str, 
    output_data_path: str):
    """This function takes the individual files of latent Yamnet sound representations, projects them to a manageable dimensionality and
    combines them into a single file for calibration of the flow network. Note, the output files (if exist) will be deleted before the new files are created.
    It is assumed that audio_to_latent_yamnet.run() and create_scaler_and_PCA_model_for_latent_yamnet.run() were executed beforehand.

    :param layer_index: The index of the layer for which the data shall be converted.
    :type layer_index: int
    :param dimensionality: The number of dimensions that the projection shall have. Note that this number must not be greater than what the PCA model saved at `pca_path` supports.
    :type dimensionality: int 
    :param input_data_path: The data to the input data. These are assumed to be stored in the same format as done by `audio_to_latent_yamnet.run()`.
    :type input_data_path: str
    :param pca_path: The path to the standard scalers and PCA model. Models are assumed to be stored in the same format as done by `create_scalers_and_PCA_mode_for_latent_yamnet`.
    :type pca_path: str
    :param output_data_path: The path to the folder where the output data shall be stored. The output will be stored in a folder called 'Layer `layer_index`' as X.npy of shape [instance count, `dimensionality`] and corresponding Y.npy of shape [instance count, factor count]."""
    
    print(f"Running script to convert latent yamnet layer {layer_index} to calibration data set.")
    
    # Adjust configuration
    output_data_path = os.path.join(output_data_path, f"{dimensionality} dimensions")
    if not os.path.exists(output_data_path): os.makedirs(output_data_path)

    # Processing
    input_layer_path = os.path.join(input_data_path, f'Layer {layer_index}')
    output_layer_path = os.path.join(output_data_path, f'Layer {layer_index}')
    pca_layer_path = os.path.join(pca_path, f"Layer {layer_index}")
    if os.path.exists(output_layer_path): shutil.rmtree(output_layer_path)
    os.makedirs(output_layer_path)

    # Loads models
    with open(os.path.join(pca_layer_path, "Pre PCA Standard Scaler.pkl"),'rb') as fh:
        pre_scaler = pkl.load(fh)

    with open(os.path.join(pca_layer_path, "PCA.pkl"),'rb') as fh:
        pca = pkl.load(fh)

    with open(os.path.join(pca_layer_path, "Post PCA Standard Scaler.pkl"), 'rb') as fh:
        post_scaler = pkl.load(fh)

    print(f"\tThe top {dimensionality} dimensions explain {np.round(100 * np.sum(pca.explained_variance_ratio_[:dimensionality]),2)} % of variance.")

    # Transform each X
    x_file_names = utl.find_matching_strings(strings=os.listdir(input_layer_path), token='_X_')
    Xs = [None] * len(x_file_names); Ys = [None] * len(x_file_names)
    for i, x_file_name in enumerate(x_file_names):
        X = np.load(os.path.join(input_layer_path, x_file_name))[np.newaxis,:]
        Xs[i] = post_scaler.transform(pca.transform(pre_scaler.transform(X)))[:,:dimensionality] 
        Ys[i] = np.load(os.path.join(input_layer_path, x_file_name.replace('_X_','_Y_')))[np.newaxis,:]
      
        print(f"\r\t{np.round(100*(i+1)/len(x_file_names))} % completed", end='')
    # Save
    np.save(os.path.join(output_layer_path, "X"), np.concatenate(Xs, axis=0))
    np.save(os.path.join(output_layer_path, "Y"), np.concatenate(Ys, axis=0))
        
    print("\n\tRun Completed")

if __name__ == "__main__":
    # Load Configuration
    configuration = configuration_loader.load()
    parser.add_argument("--sound_name_to_Y_labels", help="A dictionary mapping each sound file name to a corresponding list of numeric y-labels. The list should be as long as the number of non-residual factors that shall be part of the modelling. For example, with 4 materials and 2 actions, a file name 'MT', abbreviating metal tapping, could be mapped to [3,1] if metal is the material with index 3 and tapping the action with index 1. Indices are assumed to be start at 0.", type=str)
    
    # Describe mapping from audio file name to factor-wise class indices
    sound_name_to_Y_labels = {f"{m}{a}" : [material_to_index[m], action_to_index[a]] for m in material_to_index.keys() for a in action_to_index.keys()} # File names are of the form MA, where M is the material abbreviation and A the action abbreviation.
    
    # Create y
    name = '.'.join(raw_file_name.split('.')[:-1]) # removes the file extension
    y = np.array(sound_name_to_Y_labels[name])
    
    # Modelling
    target_dimensionality = configuration['PCA_target_dimensionality'] # The number of dimensions of the projections
    full_dimensionality_layer_index = configuration['layer_index_full_PCA'] # For this layer, the full PCA model will be created (costly)
    
    # Use the small PCA model for most layers
    for layer_index in configuration['layer_indices_small_PCA']:     
        
        run(layer_index=layer_index,
            dimensionality=target_dimensionality,
            input_data_path = os.path.join(*configuration['latent_yamnet_data_folder']+['original']),
            pca_path = os.path.join(*configuration['model_folder']+['PCA and Standard Scalers',f'{target_dimensionality} dimensions']),
            output_data_path = os.path.join(*configuration['latent_yamnet_data_folder']+['projected']))

    # Use the full PCA model for the layer for which disentanglement shall be performed later on
    run(layer_index=configuration['layer_index_full_PCA'],
        dimensionality=target_dimensionality,
        input_data_path = os.path.join(*configuration['latent_yamnet_data_folder']+['original']),
        pca_path = os.path.join(*configuration['model_folder']+['PCA and Standard Scalers', 'All dimensions']),
        output_data_path = os.path.join(*configuration['latent_yamnet_data_folder']+['projected']))

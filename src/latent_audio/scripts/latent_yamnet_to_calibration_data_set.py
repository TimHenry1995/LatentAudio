from latent_audio import utilities as utl
import os, pickle as pkl, numpy as np, shutil
def run(layer_index: int,
        dimensionality: int = 64, 
        input_data_path: str = os.path.join('data','latent yamnet','original'), 
        pca_path : str = os.path.join('models','Scaler and PCA','64 dimensions'), 
        output_data_path: str = os.path.join('data','latent yamnet')):
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
    for layer_index in range(14):
        run(layer_index=layer_index)

import os, shutil
from LatentAudio import utilities as utl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
from typing import Tuple
import random
import matplotlib.pyplot as plt
from typing import Dict

def run(layer_index: int,
        X_folder_path: str = os.path.join('LatentAudio','data','latent yamnet','original'), 
        PCA_folder_path: str = os.path.join('LatentAudio','models','Scaler and PCA'),
        target_dimensionality: int = 64) -> Tuple[int, np.ndarray]:
    """Creates a standard scaler to be used before principal component analysis (PCA), a PCA model and a standard scaler to be used after PCA. 
    The data is expected to be stored at ``X_folder_path`` in the same format as output by the ``audio_to_latent_yamnet.run`` function. The here 
    created models are not used to perform the projection of the data. Instead, the models are simply saved in ``PCA_folder_path``. 
    Before saving the models into the specified folder, that folder (if exists) is deleted and any other files inside it will thus be lost.

    :param layer_index: The index of the Yamnet layer for which the latent Yamnet data shall be loaded.
    :type layer_index: int
    :param X_folder_paht: The path to the folder where the latent Yamnet data is located.
    :type X_folder_path: str
    :param PCA_folder_path: The path to the folder where the standard scalers and PCA model shall be stored.
    :type PCA_folder_path: str
    :param target_dimensionality: The dimensionality that PCA should have for its output. Since a complete PCA model with same output as input dimensionality is resource intensive (for early layers even prohibitively expensive), it is recommended to keep this value as small as possible. Most layers will only need the forward PCA model and hence as small model is sufficient, e.g. 64 dimensions (default), for latent space exploration. If latent space manipulation is planned for the current layer, then a full PCA model is required. The full model will automatically be created if target_dimensionality is set to None.
    :type target_dimensionality: int
    :return: dimensions (Tuple[int, numpy.ndarray]) - The int is the original dimensionality of the layer and the array has shape [`target_dimensionality`] and lists the proportion of variance explained by the first each of the first `targte_dimensionality` many dimensions of PCA.
    """

    print(f"Running script to create scalers and PCA model for latent yamnet layer {layer_index}.")

    random.seed(42)
    PCA_folder_path = os.path.join(PCA_folder_path, f'{target_dimensionality if target_dimensionality != None else "All"} dimensions')
    X_layer_folder = os.path.join(X_folder_path, f'Layer {layer_index}')
    PCA_layer_folder = os.path.join(PCA_folder_path, f"Layer {layer_index}")
    if os.path.exists(PCA_layer_folder): shutil.rmtree(PCA_layer_folder)
    os.makedirs(PCA_layer_folder)

    # If no target_dimensionality provided, prepare a full pca model (costs memory, disk storage and may lead to index overflow)
    # This makes sense for layers whose original dimensionality is small enough anyways to afford a complete pca model, e.g. Yamnet layer 9
    if target_dimensionality == None:
        # Load one instance to get shape
        X_tmp, _ = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=1) # Shape == [sample size = 1, dimensionality]
        target_dimensionality = X_tmp.shape[1] 
        sample_size = (int)(1.1 * target_dimensionality) # PCA needs dimensionality many UNIQUE data points. Here we take a few more data points to hopefully have this many unique ones
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
    with open(os.path.join(PCA_layer_folder, "Pre PCA Standard Scaler.pkl"),"wb") as file_handle:
        pkl.dump(pre_scaler, file_handle)
        
    with open(os.path.join(PCA_layer_folder, f"PCA.pkl"),"wb") as file_handle:
        pkl.dump(pca, file_handle)

    with open(os.path.join(PCA_layer_folder, "Post PCA Standard Scaler.pkl"), "wb") as file_handle:
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
    layer_index_to_dimensionality = {}
    layer_index_to_explained_variances = {}
    
    # Modelling
    for layer_index in range(14):
        layer_index_to_dimensionality[layer_index], layer_index_to_explained_variances[layer_index] = run(layer_index=layer_index, 
                                                                                                          X_folder_path = os.path.join('data','latent yamnet','original'), 
                                                                                                          PCA_folder_path = os.path.join('models','Scaler and PCA'),
                                                                                                          target_dimensionality = 64)

    # Plotting
    plot(figure_output_folder = os.path.join('plots','explore latent yamnet','64 dimensions'),
         layer_index_to_dimensionality = layer_index_to_dimensionality,
         layer_index_to_explained_variances = layer_index_to_explained_variances)
    

import os, shutil
from latent_audio import utilities as utl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
from typing import Tuple
import random
import matplotlib.pyplot as plt

def run(layer_index: int,
        X_folder: str = os.path.join('data','latent yamnet','original'), 
        PCA_folder: str = os.path.join('models','Scaler and PCA'),
        target_dimensionality: int = 64) -> Tuple[int, np.ndarray]:
    """Creates a standard scaler to be used before principal component analysis (PCA), a PCA model and a standard scaler to be used after PCA. 
    The data is expected to be stored in ``X_folder`` in the same format as output by the ``audio_to_latent_yamnet.run`` function. The here 
    created models are not used to perform the projection of the data, instead the models are simply saved in ``PCA_folder``. 

    Requirements:
    - requires the audio_to_latent_yamnet script to be executed apriori

    Steps:
    - for a given layer it loads a sample of latent yamnet representations
    - fits a standard scaler to the sample
    - performs a full principal component analysis (for invertibility)
    - saves the scaler and PCA models as .pkl files

    Side effects
    - before saving the models into the specified folder, that folder (if exists) is deleted and any other files inside of it will thus be lost.

    :param layer_index: The index of the Yamnet layer for which the latent Yamnet data shall be loaded.
    :type layer_index: int
    :param X_folder: The path to the folder where the latent Yamnet data is located.
    :type X_folder: str
    :param PCA_folder: The path to the folder where the standard scalers and PCA model shall be stored.
    :type PCA_folder: str
    :param target_dimensionality: The dimensionality that PCA should have for its output. Since a complete PCA model with same output as input dimensionality is resource intensive (for early layers even prohibitively expensive), it is recommended to keep this value as small as possible. Most layers will only need the forward PCA model and hence as small model is sufficient, e.g. 64 dimensions (default), for latent space exploration. If latent space manipulation is planned for the current layer, then a full PCA model is required. The full model will automatically be created if target_dimensionality is set to None.
    :type target_dimensionality: int
    :return: dimensions (Tuple[int, numpy.ndarray]) - The int is the original dimensionality of the layer and the array has shape [`target_dimensionality`] and lists the proportion of variance explained by the first each of the first `targte_dimensionality` many dimensions of PCA.

    """
     
    print("Running script to create scalers and PCA model for latent yamnet")

    random.seed(42)
    X_layer_folder = os.path.join(X_folder, f'Layer {layer_index}')
    PCA_layer_folder = os.path.join(PCA_folder, f"Layer {layer_index}")
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
    print(f" Completed. Shape == [instance count, dimensionality] == {X_sample.shape}")

    # Fit scaler
    print("\tFitting Pre-PCA Standard Scaler to sample", end='')
    pre_scaler = StandardScaler()
    X_sample = pre_scaler.fit_transform(X_sample)
    print(" Completed")

    # Fit PCA
    print(f"\tFitting {target_dimensionality}-dimensional PCA to sample", end='')
    pca = PCA(n_components=target_dimensionality)
    pca.fit(X_sample)
    print(" Completed")

    # Fit scaler
    print("\tFitting Post-PCA Standard Scaler to sample", end='')
    post_scaler = StandardScaler()
    post_scaler.fit(pca.transform(X_sample))
    print(" Completed")

    # Save
    with open(os.path.join(PCA_layer_folder, "Pre PCA Standard Scaler.pkl"),"wb") as file_handle:
        pkl.dump(pre_scaler, file_handle)
        
    with open(os.path.join(PCA_layer_folder, f"Complete PCA.pkl"),"wb") as file_handle:
        pkl.dump(pca, file_handle)

    with open(os.path.join(PCA_layer_folder, "Post PCA Standard Scaler.pkl"), "wb") as file_handle:
        pkl.dump(post_scaler, file_handle)
    print("\tRun Completed")

    # Outputs
    return X_sample.shape[1], pca.explained_variance_ratio_

if __name__ == "__main__": 
    figure_output_folder = os.path.join('plots','explore latent yamnet','64 dimensions')
    layer_indices = range(14)
    dimensionalities = {}

    # Plot PCA
    plt.figure(figsize=(10,5)); plt.title(f"Principal Component Analysis")
    for layer_index in layer_indices:
        dimensionalities[layer_index], explained_variances = run(layer_index=layer_index)
        R = 0
        plt.gca().set_prop_cycle(None)
        for i, r in enumerate(explained_variances[layer_index]):
            plt.bar([str(layer_index)],[r], bottom=R, color='white', edgecolor='black')
            R += r

        plt.ylim(0,1)
    plt.ylabel('Explained Variance'); plt.xlabel('Layer')
    plt.savefig(os.path.join(figure_output_folder, f"Principal Component Analysis"))
    plt.show()

    # Plot dimensionalities
    plt.figure(figsize=(10,5)); plt.title("Original Dimensionalities")
    plt.bar(dimensionalities.keys(), dimensionalities.values(), color='white', edgecolor='black')
    plt.ylabel("Dimensionality"); plt.xlabel('Layer')
    plt.savefig(os.path.join(figure_output_folder, "Original Dimensionalities"))
    plt.show()

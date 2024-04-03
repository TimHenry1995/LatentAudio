"""This script creates a standard scaler for scaling before PCA, a complete pca model and a standard scaler for after PCA
on the latent yamnet representations. This script takes a long time and is not intended for all layers.

Requirements:
- requires the audio_to_latent_yamnet script to be executed apriori

Steps:
- for a given layer it loads a sample of latent yamnet representations
- fits a standard scaler to the sample
- performs a full principal component analysis (for invertibility)
- saves the scaler and PCA models as .pkl files

Side effects
- before saving the models into the specified folder, that folder (if exists) is deleted and any other files inside of it will thus be lost.
"""

import os, shutil
from latent_audio import utilities as utl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
import random

def run(layer_index: int,
        X_folder: str = os.path.join('data','latent yamnet','original'), 
        PCA_folder: str = os.path.join('models','Scaler and PCA')):
     
    print("Running create scalers and PCA model for latent yamnet")

    random.seed(42)
    X_layer_folder = os.path.join(X_folder, f'Layer {layer_index}')
    PCA_layer_folder = os.path.join(PCA_folder, f"Layer {layer_index}")
    if os.path.exists(PCA_layer_folder): shutil.rmtree(PCA_layer_folder)
    os.makedirs(PCA_layer_folder)

    # Load one instance to get shape
    X_tmp, _ = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=1) # Shape == [sample size = 1, dimensionality]
    dimensionality = X_tmp.shape[1] 
    sample_size = (int)(1.1 * dimensionality) # PCA needs dimensionality many UNIQUE data points. Here we take a few more data points to hopefully have this many unique ones
    del X_tmp

    # Load sample
    X_sample, Y_sample = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=sample_size)
    print(f"\tX_sample loaded. Shape == {X_sample.shape}")

    # Fit scaler
    pre_scaler = StandardScaler()
    X_sample = pre_scaler.fit_transform(X_sample)
    print("\tPre PCA Standard Scaler fit to X_sample")

    # Fit PCA
    pca = PCA(n_components=dimensionality)
    pca.fit(X_sample)
    print("\tPCA fit to X_sample")

    # Fit scaler
    post_scaler = StandardScaler()
    post_scaler.fit(pca.transform(X_sample))
    print("\tPost PCA Standard Scaler fit to X_sample")

    # Save
    with open(os.path.join(PCA_layer_folder, "Pre PCA Standard Scaler.pkl"),"wb") as file_handle:
        pkl.dump(pre_scaler, file_handle)
        
    with open(os.path.join(PCA_layer_folder, f"Complete PCA.pkl"),"wb") as file_handle:
        pkl.dump(pca, file_handle)

    with open(os.path.join(PCA_layer_folder, "Post PCA Standard Scaler.pkl"), "wb") as file_handle:
        pkl.dump(post_scaler, file_handle)
    print("\tRun Completed")

if __name__ == "__main__":
    for layer_index in range(14):
        run(layer_index=layer_index)
"""This script creates standard scaler and complete pca model for latent yamnet representations.

Requirements:
- requires the audio_to_latent_yamnet script to be executed apriori

Steps:
- for a given layer it loads a sample of latent yamnet representations
- fits a standard scaler to the sample
- performs a full principal component analysis (for invertibility)
- saves the scaler and PCA models as .pkl files
"""

import os
from latent_audio import utilities as utl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
import random

# Configuration
X_folder = os.path.join('data','latent yamnet') # The latent yamnet data
PCA_folder = os.path.join('models','Scaler and PCA')

for l in range(13):
    print(f"Layer {l}")

    # Configuration
    random.seed(42)
    X_layer_folder = os.path.join(X_folder, f'Layer {l}')
    PCA_layer_folder = os.path.join(PCA_folder, f"Layer {l}")
    if not os.path.exists(PCA_layer_folder): os.makedirs(PCA_layer_folder)
    
    # Load one instance to get shape
    X_tmp, _ = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=1) # Shape == [sample size = 1, dimensionality]
    dimensionality = X_tmp.shape[1] 
    sample_size = (int)(1.5 * dimensionality) # PCA needs dimensionality many UNIQUE data points. Here we take a few more data points to hopefully have this many unique ones
    del X_tmp

    # Load sample
    X_sample, Y_sample = utl.load_latent_sample(data_folder=X_layer_folder, sample_size=sample_size)
    print(f"X_sample loaded. Shape == {X_sample.shape}")

    # Fit scaler
    pre_scaler = StandardScaler()
    X_sample = pre_scaler.fit_transform(X_sample)
    print("Pre PCA Standard Scaler fit to X_sample")

    # Fit PCA
    pca = PCA(n_components=dimensionality)
    pca.fit(X_sample)
    print("PCA fit to X_sample")

    # Fit scaler
    post_scaler = StandardScaler()
    post_scaler.fit(pca.transform(X_sample))
    print("Post PCA Standard Scaler fit to X_sample")

    # Save
    with open(os.path.join(PCA_layer_folder, "Pre PCA Standard Scaler.pkl"),"wb") as file_handle:
        pkl.dump(pre_scaler, file_handle)
        
    with open(os.path.join(PCA_layer_folder, f"Complete PCA.pkl"),"wb") as file_handle:
        pkl.dump(pca, file_handle)

    with open(os.path.join(PCA_layer_folder, "Post PCA Standard Scaler.pkl"), "wb") as file_handle:
        pkl.dump(post_scaler, file_handle)


print("Script completed")
"""This takes the individual files from the audio_to_latent_yamnet script, projects them to a manageable dimensionality and
combines them into a single file for calibration of the flow network. It takes long as is not meant to be executed for all layers.

Requirements:
- the audio_to_latent_yamnet script needs to be executed apriori
- the create_scaler_and_PCA_model_for_latent_yamnet script needs to be executed a priori

Steps:
- for a given layer it loads each data point
- passes it through the pre PCA scaler, the complete PCA and the post PCA scaler
- takes the top k dimensions, where k is a number mangeable by the downstream models
- collects all data points in one file
- saves them all in one file called X.npy along with the labels in Y.npy

Side effects:
- The output files (if exist) will be deleted before the new files are created.
"""

from latent_audio import utilities as utl
import os, pickle as pkl, numpy as np, shutil

# Configuration
dimensionality = 64
input_data_path = os.path.join('data','latent yamnet','original')
pca_path = os.path.join('models','Scaler and PCA')
output_data_path = os.path.join('data','latent yamnet', f"{dimensionality} dimensions")
layer_index = 9

# Processing
input_layer_path = os.path.join(input_data_path, f'Layer {layer_index}')
output_layer_path = os.path.join(output_data_path, f'Layer {layer_index}')
pca_layer_path = os.path.join(pca_path, f"Layer {layer_index}")
if os.path.exists(output_layer_path): shutil.rmtree(pca_layer_path)
os.makedirs(output_layer_path)

# Loads models
with open(os.path.join(pca_layer_path, "Pre PCA Standard Scaler.pkl"),'rb') as fh:
    pre_scaler = pkl.load(fh)

with open(os.path.join(pca_layer_path, "Complete PCA.pkl"),'rb') as fh:
    pca = pkl.load(fh)

with open(os.path.join(pca_layer_path, "Post PCA Standard Scaler.pkl"), 'rb') as fh:
    post_scaler = pkl.load(fh)

print(f"The top {dimensionality} dimensions explain {np.round(100 * np.sum(pca.explained_variance_ratio_[:dimensionality]),2)} % of variance.")

# Transform each X
x_file_names = utl.find_matching_strings(strings=os.listdir(input_layer_path), token='_X_')
Xs = [None] * len(x_file_names); Ys = [None] * len(x_file_names)
for i, x_file_name in enumerate(x_file_names):
    X = np.load(os.path.join(input_layer_path, x_file_name))[np.newaxis,:]
    Xs[i] = post_scaler.transform(pca.transform(pre_scaler.transform(X)))[:,:dimensionality] 
    Ys[i] = np.load(os.path.join(input_layer_path, x_file_name.replace('_X_','_Y_')))[np.newaxis,:]
    if (i+1) % (int)(len(x_file_names)/10) == 0:
        print(f"{np.round(100*(i+1)/len(x_file_names),2)} % Finished")
# Save
np.save(os.path.join(output_layer_path, "X"), np.concatenate(Xs, axis=0))
np.save(os.path.join(output_layer_path, "Y"), np.concatenate(Ys, axis=0))
    
print("Script completed")
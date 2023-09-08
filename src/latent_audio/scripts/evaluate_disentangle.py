"""This script visualizes the model created by the disentangle script. It passes sample data through the model
and shows in scatter plots how well the matierial and action factors are disentangled."""

from latent_audio.scripts import disentangle as lsd
from latent_audio.yamnet import layer_wise as lyl
from typing import List, Any, OrderedDict, Callable, Generator, Tuple
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gyoza.modelling import flow_layers as mfl
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import pickle as pkl

# Define some functions
def scatter_plot(flow_network, Z, Y, material_labels, action_labels, plot_save_path) -> None:

    # Predict
    Z_tilde = flow_network(Z)
    Z_tilde_m = Z_tilde[:,-2] # Material dimension
    Z_tilde_a = Z_tilde[:,-1] # Action dimension

    # Plot
    plt.subplots(2, 4, figsize=(12,6), gridspec_kw={'width_ratios': [1,5,1,5], 'height_ratios': [5,1]})

    plt.suptitle(r"$\tilde{Z}$")

    # 1. Materials
    ms = list(set(Y[:,-2]))
    
    # 1.1 Vertical Boxplot
    plt.subplot(2,4,1)
    plt.boxplot([Z_tilde_a[Y[:,-2]==m] for m in ms])
    plt.xticks(list(range(1,len(ms)+1)), [material_labels[m] for m in ms])
    plt.ylabel("Action Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.yticks([])

    # 1.2 Horizontal Boxplot
    plt.subplot(2,4,6)
    plt.boxplot([Z_tilde_m[Y[:,-2]==m] for m in ms], vert=False)
    plt.yticks(list(range(1,len(ms)+1)), [material_labels[m] for m in ms])
    plt.xlabel("Material Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.xticks([])

    # 1.3 Scatter
    plt.subplot(2,4,2); plt.title("Materials")
    
    for m in ms:
        plt.scatter(Z_tilde_m[Y[:,-2]==m], Z_tilde_a[Y[:,-2]==m], marker='.')
    plt.legend([material_labels[m] for m in ms])
    
    # 2. Action
    As = list(set(Y[:,-1]))
    
    # 2.1 Vertical Boxplot
    plt.subplot(2,4,3)
    plt.boxplot([Z_tilde_a[Y[:,-1]==a] for a in As])
    plt.xticks(list(range(1,len(As)+1)), [action_labels[a] for a in As])
    plt.ylabel("Action Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.yticks([])

    # 2.2 Horizontal Boxplot
    plt.subplot(2,4,8)
    plt.boxplot([Z_tilde_m[Y[:,-1]==a] for a in As], vert=False)
    plt.yticks(list(range(1,len(As)+1)), [action_labels[a] for a in As])
    plt.xlabel("Material Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.xticks([])

    # 2.3 Scatter
    plt.subplot(2,4,4); plt.title("Actions")
    
    for a in As:
        plt.scatter(Z_tilde_m[Y[:,-1]==a], Z_tilde_a[Y[:,-1]==a], marker='.')
    plt.legend([action_labels[a] for a in As])
    
    # Remove other axes
    plt.subplot(2,4,5); plt.axis('off'); plt.subplot(2,4,7); plt.axis('off'); 
    plt.savefig(os.path.join(plot_save_path, f"materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Scatterplots.png"))
    plt.show()

def evaluate_factor_sensitivity(flow_network: mfl.SupervisedFactorNetwork, Z_tilde_to_semantic: Callable, iterator: Callable, sample_size: int, dimensions_per_factor: List[int], plot_save_path: str) -> None:

    # Compute distance between pairs
    delta_ma = compute_paired_differences(flow_network=flow_network, iterator=iterator, equal_material=True, equal_action=True, sample_size=sample_size) # ma means equal material, equal action
    delta_m = compute_paired_differences(flow_network=flow_network, iterator=iterator, equal_material=True, equal_action=False, sample_size=sample_size) # m means equal material, different action
    delta_a = compute_paired_differences(flow_network=flow_network, iterator=iterator, equal_material=False, equal_action=True, sample_size=sample_size) # a means different material, equal action

    count = 0
    distance = lambda x: tf.sqrt(tf.reduce_sum(x**2, axis=1)) # Shape of output == [instance count]
    factor_names = ['Residual','Material','Action']
    for f, dimension_count in enumerate(dimensions_per_factor):
        # Plot
        plt.figure(); plt.title(f'Sensitivity Of {factor_names[f]} Dimensions '+ r'Of $\tilde{Z}$' + '\nTo Changes In Z')
        plt.violinplot([distance(delta_ma[:,count:count+dimension_count]), distance(delta_m[:,count:count+dimension_count]), distance(delta_a[:,count:count+dimension_count])], showmeans=True)
        plt.xticks(ticks=[1,2,3], labels=['Same Material\nSame Action', 'Same Material\nOther Action', 'Other Material\nSame Action'])
        plt.ylabel('Distance'); plt.xlabel("Input Pair")
        count += dimension_count
        plt.savefig(os.path.join(plot_save_path, f' Sensitivity {factor_names[f]}.png'))
    plt.show()

def compute_paired_differences(flow_network: mfl.SupervisedFactorNetwork, Z_tilde_to_semantic: Callable, iterator: Callable, equal_material: bool, equal_action: bool, sample_size: int) -> None:

    # Initialization
    Z_ab = []
    current_sample_size = 0

    # Accumulate instances to meet the sample size
    while current_sample_size < sample_size:
        Z_ab_batch, Y_ab_batch = next(iterator)
        Z_ab.append(Z_ab_batch.numpy()[np.logical_and(Y_ab_batch[:,1] == equal_material, Y_ab_batch[:,2] == equal_action),:])
        
        current_sample_size += Z_ab[-1].shape[0]

    # Obtain proper shape
    Z_ab = np.concatenate(Z_ab, axis=0) # Instance axis
    Z_ab = Z_ab[:sample_size,:] # Shape == [sample_size, 2, ...] where 2 is for pair and ... is for a single instance

    # Distance vectors in disentangled space
    delta = tf.reshape(flow_network(Z_ab[:,0,:]) - flow_network(Z_ab[:,1,:]), [sample_size, -1]) # Shape == [instance count, dimension count]

    # Output
    return delta

# Configuration
inspection_layer_index = 8
data_path = os.path.join('data','pre-processed','16 PCA dimensions all in 1 file','Layer 8')
batch_size = 512
np.random.seed(850)
stage_count = 10
epoch_count = 100
dimensions_per_factor = [14,1,1]
materials_to_keep = [1,4]; actions_to_keep = [0,1]
materials_to_drop = list(range(6))
for m in reversed(materials_to_keep): materials_to_drop.remove(m)
actions_to_drop = list(range(4))
for a in reversed(actions_to_keep): actions_to_drop.remove(a)
m_string = ",".join(str(m) for m in materials_to_keep)
a_string = ",".join(str(a) for a in actions_to_keep)
model_save_path = os.path.join('models', f'Layer {inspection_layer_index}')
plot_save_path = os.path.join('plots','evaluate disentangle', f'Layer {inspection_layer_index}')
if not os.path.exists(plot_save_path): os.makedirs(plot_save_path)
material_labels = ['W','M','G','S','C','P']
action_labels = ['T','R','D','W']

# Load data iterators
train_iterator, test_iterator, batch_count, Z_train, Z_test, Y_train, Y_test = lsd.load_iterators(data_path=data_path, materials_to_drop=materials_to_drop, actions_to_drop=actions_to_drop, batch_size=batch_size)
Z_ab_sample, Y_ab_sample = next(train_iterator) # Sample

print("The data is fed to the model in batches of shape:\n","Z: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_sample: (instance count, factor count): \t', Y_ab_sample.shape)

# Create network
flow_network = lsd.create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
flow_network.load_weights(os.path.join(model_save_path, f'materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count}.h5'))

# Evaluate
scatter_plot(flow_network=flow_network, Z=Z_test, Y=Y_test, material_labels=material_labels, action_labels=action_labels, plot_save_path=plot_save_path)

with open(os.path.join(model_save_path, 'Standard Scaler.pkl'), 'rb') as file_handle:
    scaler = pkl.load(file_handle)
with open(os.path.join(model_save_path, f'PCA {np.sum(dimensions_per_factor)}.pkl'), 'rb') as file_handle:
    pca = pkl.load(file_handle)
'''
yamnet_second_half = lyl.LayerWiseYamnet()
with open(os.path.join(model_save_path, 'Standard Scaler.pkl'), 'rb') as file_handle:
    scaler = pkl.load(file_handle)
with open(os.path.join(model_save_path, f'{np.sum(dimensions_per_factor)} PCA.pkl'), 'rb') as file_handle:
    pca = pkl.load(file_handle)
    
create a new iterator
load scaler, full pca
load instance, scale it, transform full pca
pass the top 16 dims through flow net
change factor
invert flow net
replace top 16 dims
invert full pca, invert scaler
continue processing through yamnet
get a class distribution out and match that against some reference

# Need the standard scaler from all dims
# Need the pca from all dims to current dims
# Need to invert both transforms
Z_tilde_to_semantic = lambda x: yamnet_second_half.call_from_layer(latent: scaler.inverse_transform(pca.inverse_transform(x)), layer_index=inspection_layer_index)
evaluate_factor_sensitivity(flow_network=flow_network, latent_to_semantic=latent_to_semantic, iterator=test_iterator, sample_size=500, dimensions_per_factor=dimensions_per_factor, plot_save_path=plot_save_path)
'''
k=3
"""This script visualizes the model created by the disentangle script. It passes sample data through the model
and shows in scatter plots how well the matierial and action factors are disentangled."""

from latent_audio.scripts import disentangle as lsd
import latent_audio.utilities as utl
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
    plt.boxplot([Z_tilde_m[Y[:,-2]==m] for m in reversed(ms)], vert=False)
    plt.yticks(list(range(1,len(ms)+1)), [material_labels[m] for m in reversed(ms)])
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
    plt.boxplot([Z_tilde_m[Y[:,-1]==a] for a in reversed(As)], vert=False)
    plt.yticks(list(range(1,len(As)+1)), [action_labels[a] for a in reversed(As)])
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

def latent_transfer(X_sample: np.ndarray, dimensions_per_factor: List[int], switch_factors:int, standard_scaler: Callable, full_pca: Callable, flow_network: Callable, yamnet_from_layer: Callable) -> None:

    # Apply standard scaler(int)(np.sum(dimensions_per_factor[:factor])) and full pca
    X_sample = full_pca.transform(scaler.transform(X_sample))

    # Pass the top few dimensions through flow net
    dimension_count = np.sum(dimensions_per_factor)
    Z_tilde = flow_network(X_sample[:,:dimension_count]).numpy()

    # Swap the factors
    instance_count = X_sample.shape[0]
    Z_tilde_swapped = np.copy(Z_tilde)
    switch_factors = sorted(switch_factors)
    for factor in switch_factors:
        start = (int)(np.sum(dimensions_per_factor[:factor]))
        end = (int)(np.sum(dimensions_per_factor[:factor+1]))
        Z_tilde_swapped[:instance_count,start:end] = Z_tilde[instance_count:,start:end]
        Z_tilde_swapped[instance_count:,start:end] = Z_tilde[:instance_count,start:end]
    del Z_tilde

    # Invert flow net
    Z = flow_network.invert(Z_tilde_swapped)

    # Replace top few dimensions
    X_sample_swapped = np.copy(X_sample)
    X_sample_swapped[:,:dimension_count] = Z

    # Invert full pca, invert scaler
    X_sample_swapped = scaler.inverse_transform(pca.inverse_transform(X_sample_swapped))

    # Continue processing through yamnet
    P = yamnet_from_layer(X_sample_swapped)
    
    # Outputs
    return P

    # m1 a1
    # m2 a2
    # x_a, x_b  ss  sd  ds  dd
    # m1a1 m1a1 x
    # m1a1 m1a2     x
    # m1a1 m2a1         x
    # m1a1 m2a2             x
    # m1a2 m1a1     x
    # m1a2 m1a2 x
    # m1a2 m2a1             x
    # m1a2 m2a2         x
    # m2a1 m1a1         x
    # m2a1 m1a2             x
    # m2a1 m2a1 x
    # m2a1 m2a2     x
    # m2a2 m1a1             x
    # m2a2 m1a2         x
    # m2a2 m2a1     x
    # m2a2 m2a2 x

def plot_contribution_per_layer(network: mfl.SequentialFlowNetwork, s_range: Tuple[float, float], manifold_function: Callable, manifold_name:str, layer_steps: List[int], step_titles: List[str]):
    """Plots for each layer (or rather step of consecutive layers) the contribution to the data transformation. The plot is strucutred into three rows.
    The first row shows a stacked bar chart whose bottom segment is the contribution due to affine transformation and the top segment is the contribution
    due to higher order transformation. To better understand the mechanisms behind these contributions there is a pictogram in the bottom row for the
    actual affine transformation and in the middle row for the remaining higher order part. This separation is done to understand the complexity of the
    transformation, whereby affine is considered simple and higher order is considered complex. The decomposition into affine and higher order is obtained
    by means of a first order `Maclaurin series <https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables>`_.

    :param network: The network whose transfromation shall be visualized. It is expecetd to map 1 dimensional manifolds from the real 2-dimensional
      plane to the real 2-dimensional plane.
    :type network: :class:`gyoza.modelling.flow_layers.SequentialFlowNetwork`
    :param s_range: The lower and upper bounds for the position along the manifold, respectively.
    :type s_range: Tuple[float, float]
    :param manifold_function: A function that maps from position along manifold to coordinates on the manifold in the real two dimensional plane.
    :type manifold_function: :class:`Callable`
    :param manifold_name: The name of the manifold used for the figure title.
    :type manifold_name: str
    :param layer_steps: A list of steps across layers of the ``network``. If, for instance, the network has 7 layers and visualization shall be done for
      after the 1., 3. and 7, then ``layer_steps`` shall be set to [1,3,7]. The minimum entry shall be 1, then maximum entry shall be the number of layers
      in ``network`` and all entries shall be strictly increasing.
    :type layer_steps: List[int]
    :param step_titles: The titles associated with each step in ``layer_steps``. Used as titles in the figure.
    :type step_titles: List[str]
    """

    # Prepare plot
    #plt.figure(figsize=(12,3.5));
    layer_steps = [0] + layer_steps
    K = len(step_titles)
    fig, axs = plt.subplots(3, 1+K, figsize=(0.8+K,5), gridspec_kw={'height_ratios': [2,1,1], 'width_ratios':[0.3]+[1]*K})
    plt.suptitle(rf'Contribution per Layer on ${manifold_name}$')

    # Sample from s range
    S = np.linspace(s_range[0], s_range[1], len(gum.color_palette), dtype=tf.keras.backend.floatx())
    z_1, z_2 = manifold_function(S); Z = np.concatenate([z_1[:, np.newaxis], z_2[:, np.newaxis]], axis=1)
    max_bar_height = 0

    # Plot annotations on left
    gray = [0.8,0.8,0.8]
    #plt.subplot(3,1+K,1); plt.axis('off')
    plt.subplot(3,1+K,1+K+1); plt.bar([''],[1], color=gray, edgecolor='black', hatch='oo'); plt.ylim(0,1); plt.xticks([]); plt.yticks([]); plt.ylabel('Higher Order')
    plt.subplot(3,1+K,2*(1+K)+1); plt.bar([''],[1], color=gray, edgecolor='black', hatch='///'); plt.ylim(0,1); plt.xticks([]); plt.yticks([]); plt.ylabel('Affine')

    # Iterate layers
    for k in range(1, len(layer_steps)):

        # Set up 1st order Maclaurin decomposition https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables
        # Z_tilde ~= layer(0) + J(0) * Z, where J(0) is the jacobian w.r.t input evaluated at the origin
        origin = tf.Variable(tf.zeros([1] + list(Z[0].shape), dtype=tf.keras.backend.floatx())) # The extra 1 is the batch dimension
        Z_tilde = Z
        c = origin # Shape == [1, N]. The layer's shifting of the origin
        with tf.GradientTape() as tape:
          for layer in network.sequence[layer_steps[k-1]:layer_steps[k]]:
            c = layer(c)
            Z_tilde = layer(Z_tilde) # Shape == [instance count, N]

        J = tf.squeeze(tape.jacobian(c, origin)) # Shape == [N z_tilde dimensions, N z dimensions]. The layer's linear combination of input dimensions

        # Compute approximation error (contribution of higher order terms in the Maclaurin series)
        prediction = c + tf.linalg.matmul(Z, tf.transpose(J))
        P = prediction - Z # Shape == [instance count, N]. Arrows from Z to prediction
        E = Z_tilde - prediction # Shape == [instance count, N]. Arrows from prediction to Z_tilde

        # 2. Plot
        # 2.1 Bars
        plt.subplot(3,1+K,k+1); plt.title(step_titles[k-1], fontsize=10)
        E_norm = np.mean(np.sqrt(np.sum(E**2, axis=1)))
        P_norm = np.mean(np.sqrt(np.sum(P**2, axis=1)))
        plt.bar([''],[E_norm+P_norm], color = gray, edgecolor='black', hatch='oo')
        plt.bar([''],[P_norm], color = gray, edgecolor='black', hatch='///')
        max_bar_height = max(max_bar_height, E_norm+P_norm); plt.axis('off')

        # 2.1 Tails
        # 2.1.1 Error
        plt.subplot(3,1+K,1+K+k+1)
        plt.scatter(prediction[:,0], prediction[:,1], color=gray, marker='.',s=0.1)
        plt.quiver(prediction[:,0], prediction[:,1], E[:,0], E[:,1], angles='xy', scale_units='xy', scale=1., color=gray, zorder=3)
        plt.scatter(Z_tilde[:,0], Z_tilde[:,1], c=gum.color_palette/255.0, marker='.',s=1.5)
        plt.axis('equal'); plt.xticks([]); plt.yticks([]); plt.xlim(1.3*np.array(plt.xlim())); plt.ylim(1.3*np.array(plt.ylim()))

        # 2.1.2 Prediction
        plt.subplot(3,1+K,2*(1+K)+k+1)
        plt.scatter(Z[:,0], Z[:,1], color=gray, marker='.',s=0.1)
        plt.quiver(Z[:,0], Z[:,1], P[:,0], P[:,1], angles='xy', scale_units='xy', scale=1., color=gray, zorder=3)
        plt.scatter(prediction[:,0], prediction[:,1], c=gum.color_palette/255.0, marker='.',s=1.5)
        plt.axis('equal'); plt.xticks([]); plt.yticks([]); plt.xlim(1.3*np.array(plt.xlim())); plt.ylim(1.3*np.array(plt.ylim()))

        # Prepare next iteration
        Z=Z_tilde

    # Adjust bar heights
    for k in range(1, len(layer_steps)):
      plt.subplot(3,1+K,k+1); plt.ylim(0, max_bar_height)
    plt.subplot(3,1+K,1); plt.ylabel('Mean Change'); plt.ylim(0, max_bar_height); ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); plt.xticks([])
    ax.yaxis.tick_right(); ax.tick_params(axis="y",direction="in", pad=-12)

    plt.tight_layout()
    plt.show()

# Configuration
inspection_layer_index = 8
data_path = os.path.join('data','pre-processed','16 PCA dimensions all in 1 file','Layer 8')
batch_size = 512
np.random.seed(850)
stage_count = 3
epoch_count = 5
dimensions_per_factor = [14,1,1]
materials_to_keep = [0,1,3]; actions_to_keep = [0,1,3]
materials_to_drop = list(range(6))
for m in reversed(materials_to_keep): materials_to_drop.remove(m)
actions_to_drop = list(range(4))
for a in reversed(actions_to_keep): actions_to_drop.remove(a)
m_string = ",".join(str(m) for m in materials_to_keep)
a_string = ",".join(str(a) for a in actions_to_keep)
x_data_path = os.path.join('/Volumes/Untitled 2/pre-processed', f'Layer {inspection_layer_index}')
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
'''
# Load a sample of even size from yamnets latent space 
X_sample, Y_sample = utl.load_latent_sample(data_folder=data_folder_path, sample_size=sample_size)
    
with open(os.path.join(model_save_path, 'Full Standard Scaler.pkl'), 'rb') as file_handle:
    scaler = pkl.load(file_handle)
with open(os.path.join(model_save_path, f'Full PCA.pkl'), 'rb') as file_handle:
    pca = pkl.load(file_handle)

yamnet_second_half = lyl.LayerWiseYamnet()
with open(os.path.join(model_save_path, 'Standard Scaler.pkl'), 'rb') as file_handle:
    scaler = pkl.load(file_handle)
with open(os.path.join(model_save_path, f'{np.sum(dimensions_per_factor)} PCA.pkl'), 'rb') as file_handle:
    pca = pkl.load(file_handle)
    
# Need the standard scaler from all dims
# Need the pca from all dims to current dims
# Need to invert both transforms
Z_tilde_to_semantic = lambda x: yamnet_second_half.call_from_layer(latent: scaler.inverse_transform(pca.inverse_transform(x)), layer_index=inspection_layer_index)
evaluate_factor_sensitivity(flow_network=flow_network, latent_to_semantic=latent_to_semantic, iterator=test_iterator, sample_size=500, dimensions_per_factor=dimensions_per_factor, plot_save_path=plot_save_path)
'''
k=3
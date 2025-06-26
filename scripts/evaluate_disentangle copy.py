import sys
sys.path.append(".")

from LatentAudio.scripts import disentangle as lsd
import LatentAudio.utilities as utl
from LatentAudio.configurations import loader as configuration_loader
import tensorflow as tf
from LatentAudio.adapters import layer_wise as ylw
from typing import List, Any, OrderedDict, Callable, Generator, Tuple
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from gyoza.modelling import flow_layers as mfl
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import random
import pickle as pkl

# Define some functions
def scatter_plot_disentangled(flow_network, Z, Y, material_labels, action_labels, plot_save_path) -> None:

    # Predict
    Z_tilde = flow_network(Z)
    Z_tilde_m = Z_tilde[:,-2] # Material dimension
    Z_tilde_a = Z_tilde[:,-1] # Action dimension

    # Plot
    plt.subplots(2, 4, figsize=(12,6), gridspec_kw={'width_ratios': [1,5,1,5], 'height_ratios': [5,1]})

    plt.suptitle(r"Disentangled Materials and Actions")

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
        plt.scatter(Z_tilde_m[Y[:,-2]==m], Z_tilde_a[Y[:,-2]==m],s=1)
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
        plt.scatter(Z_tilde_m[Y[:,-1]==a], Z_tilde_a[Y[:,-1]==a], s=1)
    plt.legend([action_labels[a] for a in As])

    # Remove other axes
    plt.subplot(2,4,5); plt.axis('off'); plt.subplot(2,4,7); plt.axis('off')
    plt.savefig(plot_save_path)
    plt.show()

def plot_permutation_test(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, plot_save_path: str) -> None:

    # Swop each factor
    swops = {'Material':['material'], 'Action':['action'], 'Material and Action':['material','action']}
    dissimilarities = {}
    entropy = lambda P, Q: P*np.log(Q)
    dissimilarity_function = lambda P, Q: - np.sum(entropy(tf.nn.softmax(P, axis=1),tf.nn.softmax(Q, axis=1)),axis=1)
    plt.figure(figsize=(10,10)); plt.suptitle('Latent Transfer')
    b = 1
    x_min = np.finfo(np.float32).max
    x_max = np.finfo(np.float32).min
    
    for factor_name, switch_factors in swops.items():
        # Baseline
        P_d, P_r = latent_transfer(Z_prime=Z_prime, Y=Y, dimensions_per_factor=dimensions_per_factor, switch_factors=switch_factors, baseline=True, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index)
        baseline = dissimilarity_function(P_d,P_r) # P and Q are each of shape [instance count, class count]. cross entropy is of shape [instance count]
        # Experimental
        P_d, P_r = latent_transfer(Z_prime=Z_prime, Y=Y, dimensions_per_factor=dimensions_per_factor, switch_factors=switch_factors, baseline=False, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index)
        experimental = dissimilarity_function(P_d,P_r) # P and Q are each of shape [instance count, class count]. cross entropy is of shape [instance count]
        
        
        # Plot
        plt.subplot(len(swops),1,b); plt.title(factor_name)
        plt.boxplot([baseline, experimental], showmeans=True, vert=False, showfliers=False)
        plt.yticks([1,2], ['Within Class','Between Class'], rotation=90, va='center')
        x_min = min(x_min, plt.xlim()[0])
        x_max = max(x_max, plt.xlim()[1])
        
        b+=1
    
    # Set labels and range
    plt.xlabel(r"Crossentropy of $P_d$ and $P_r$")
    for i in range (1,b): 
        plt.subplot(b-1,1,i); plt.xlim([x_min, x_max])
        plt.grid(alpha=0.25)
        if i < b-1: plt.gca().tick_params(labelbottom=False) 
        
    
    plt.savefig(plot_save_path)
    plt.show()
    
def latent_transfer(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], switch_factors:[str], baseline:bool, pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int) -> None:

    instance_count = Z_prime.shape[0]
    #assert instance_count % 2 == 0, f"The number of instance was assumed to be even such that the first half of instances can be swopped with the second half. There were {instance_count} many instances provided."
    
    # Compute P
    layer_index_to_shape = [ [instance_count, 48, 32, 32],  [instance_count, 48, 32, 64],  [instance_count, 24, 16, 128],  [instance_count, 24, 16, 128],  [instance_count, 12, 8, 256],  [instance_count, 12, 8, 256], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 3, 2, 1024], [instance_count, 3, 2, 1024]]
    P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime, layer_index_to_shape[layer_index]), layer_index=layer_index+1).numpy()
    
    # Apply standard scalers and pca
    Z_prime = post_scaler.transform(pca.transform(pre_scaler.transform(Z_prime)))

    # Pass the top few dimensions through flow net
    dimension_count = np.sum(dimensions_per_factor)
    Z_tilde = flow_network(Z_prime[:,:dimension_count]).numpy()

    # Partition the data according to classes
    partition = {}

    ms = set(Y[:,-2]); As = set(Y[:,-1])
    for m in ms:
        partition[f"m{m}"] = np.copy(Z_tilde[Y[:,-2]==m])
    
    for a in As:
        partition[f'a{a}'] = np.copy(Z_tilde[Y[:,-1]==a])

    # Perform swops
    Z_tilde_swapped = np.copy(Z_tilde)
    for i in range(len(Z_tilde)):
        # Determine the material and action class
        m_i = Y[i,-2]; a_i = Y[i, -1]

        # Baseline
        if baseline:
            # In baseline mode swaps will only be made for the indicated factors with instances with the same class
            if 'material' in switch_factors:
                # Sample from the set of points with same material
                Z_tilde_swapped[i, -2] = random.choice(partition[f"m{m_i}"])[-2]
            if 'action' in switch_factors:
                # Sample from the set of points with same action
                Z_tilde_swapped[i, -1] = random.choice(partition[f"a{a_i}"])[-1]
        else:
            # In experimental mode swaps will only be made for indicated factors with instances that have a different class
            if 'material' in switch_factors:
                # Sample from the set of points with other material
                m_j = random.choice(list(ms.difference({m_i})))
                Z_tilde_swapped[i, -2] = random.choice(partition[f"m{m_j}"])[-2]
            if 'action' in switch_factors:
                # Sample from the set of points with other action
                a_j = random.choice(list(As.difference({a_i})))
                Z_tilde_swapped[i, -1] = random.choice(partition[f"a{a_j}"])[-1]
      
    # Invert flow net
    Z_swapped = flow_network.invert(Z_tilde_swapped)

    # Replace top few dimensions
    Z_prime_swapped = np.copy(Z_prime)
    Z_prime_swapped[:,:dimension_count] = Z_swapped

    # Invert full pca, invert scaler
    Z_prime_swapped = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_swapped)))

    # Continue processing through yamnet
    Q = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_swapped, layer_index_to_shape[layer_index]), layer_index=layer_index+1).numpy()
    
    # Outputs
    return P, Q

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


if __name__ == "__main__":
    
    # Configuration
    
    projected_data_path = os.path.join(configuration['latent_yamnet_data_folder'],'projected',f'{np.sum(configuration['dimensions_per_factor'])} dimensions',f'Layer {configuration['layer_index_full_PCA']}')
    original_data_path = os.path.join(configuration['latent_yamnet_data_folder'],'original',f'Layer {configuration['layer_index_full_PCA']}')
    flow_model_save_path = os.path.join(configuration['model_folder'], 'flow models',f'Layer {configuration['layer_index_full_PCA']}')
    pca_model_path = os.path.join(configuration['model_folder'], 'PCA and Standard Scalers',f"Layer {configuration['layer_index_full_PCA']}")
    plot_save_path = os.path.join(configuration['plots_folder'],'evaluate flow models', f'Layer {configuration['layer_index_full_PCA']}')
    
    if not os.path.exists(plot_save_path): os.makedirs(plot_save_path)
    flow_model_save_path = os.path.join(flow_model_save_path, f'Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count}.h5')
    material_labels=configuration['material_to_index'].keys(); action_labels = configuration['action_to_index'].keys()
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="create_scalers_and_PCA_model_for_latent_yamnet",
        description='''This script visualizes the model created by the disentangle script. It passes sample data through the model
                    and shows in scatter plots how well the matierial and action factors are disentangled.

                    There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                    The second way is to manually pass all these other arguments while calling the script.
                    For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                    When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')
    
    parser.add_argument("--random_seed", help="An int (in form of a json string) that is used to set the random module or Python, numpy and keras to make the model parameter initialization and instance sampling reproducible", type=str)
    parser.add_argument("--stage_count", help="An int (in form of a json string) that is used to set the number of stages in the flow model. The more stages are used, the more complex the model will be.", type=str)
    parser.add_argument("--batch_size", help="An int (in form of a json string) that is used to control the number of instances per batch that are fed to the flow model during training.", type=str)
    parser.add_argument("--epoch_count", help="An int (in form of a json string) indicating how many iterations through the dataset are made during training. An iteration is defined by finding one partner instance for each instance in the given training set.", type=str)
    parser.add_argument("--learning_rate", help="A float (in form of a json string) indicating the learning rate passed to the optimizer", type=str)
    parser.add_argument("--validation_proportion", help="A float (in form of a json string) indicating the proportions of the calibration data that should be used for validating the model.", type=str)
    parser.add_argument("--dimensions_per_factor", help="A list of ints (in form of a json string) indicating how many dimensions each factor should be allocated in the output of the flow model. The zeroth entry is for the residual factor, the first entry is for the first factor, the second entry for the second factor, etc. The sum of dimensions has to be equal to the dimensionality of the input to the flow model.", type=str)
    parser.add_argument("--factor_index_to_included_class_indices", help="A dictionary (in form of a json string) that maps the index of a factor to the indices of its classes that should be included in the training. This is a way to exclude unwanted classes from the calibration data if needed. The indexing of factors has to be in line with that in the Y files of the calibration data, i.e. the residual factor is excluded.", type=str)
    parser.add_argument("--calibration_data_set_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the projections are stored.", type=str)
    parser.add_argument("--flow_model_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder where the flow model shall be stored.", type=str)
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plot of the calibration trajectory should be saved.")

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.random_seed != None and args.stage_count != None and args.batch_size != None and args.epoch_count != None and args.learning_rate != None and args.validation_proportion != None and args.dimensions_per_factor != None and args.factor_index_to_included_class_indices != None and args.calibration_data_set_folder != None and args.flow_model_folder != None and args.figure_folder != None, "If no configuration file is provided, then all other arguments must be provided."
    
        random_seed = json.loads(args.random_seed)
        stage_count = json.loads(args.stage_count)
        batch_size = json.loads(args.batch_size)
        epoch_count = json.loads(args.epoch_count)
        learning_rate = json.loads(learning_rate)
        validation_proportion = json.loads(args.validation_proportion)
        dimensions_per_factor = json.loads(args.dimensions_per_factor)
        factor_index_to_included_class_indices = json.loads(args.factor_index_to_included_class_indices)
        calibration_data_set_folder = json.loads(args.calibration_data_set_folder)
        calibration_data_set_folder_path = os.path.join(*calibration_data_set_folder)
        flow_model_folder = json.loads(args.flow_model_folder)
        flow_model_folder_path = os.path.join(*flow_model_folder)
        figure_folder = json.loads(args.figure_folder)
        figure_folder_path = os.path.join(*figure_folder)
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.random_seed == None and args.stage_count == None and args.batch_size == None and args.epoch_count == None and args.learning_rate == None and args.validation_proportion == None and args.dimensions_per_factor == None and args.factor_index_to_included_class_indices == None and args.calibration_data_set_folder == None and args.flow_model_folder == None and args.figure_folder == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'disentangle' or configuration['script'] == 'disentangle.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'disentangle'."
        
        random_seed = configuration['arguments']['random_seed']
        stage_count = configuration['arguments']['stage_count']
        batch_size = configuration['arguments']['batch_size']
        epoch_count = configuration['arguments']['epoch_count']
        learning_rate = configuration['arguments']['learning_rate']
        validation_proportion = configuration['arguments']['validation_proportion']
        dimensions_per_factor = configuration['arguments']['dimensions_per_factor']
        factor_index_to_included_class_indices = configuration['arguments']['factor_index_to_included_class_indices']
        calibration_data_set_folder_path = os.path.join(*configuration['arguments']['calibration_data_set_folder'])
        flow_model_folder_path = os.path.join(*configuration['arguments']['flow_model_folder'])
        figure_folder_path = os.path.join(*configuration['arguments']['figure_folder'])
        
    print("\n\n\tStarting script disentangle")
    print("\t\tThe script parsed the following arguments:")
    print("\t\trandom_seed: ", random_seed)
    print("\t\tstage_count: ", stage_count)
    print("\t\tbatch_size: ", batch_size)
    print("\t\tepoch_count: ", epoch_count)
    print("\t\tlearning_rate: ",learning_rate)
    print("\t\tvalidation_proportion: ", validation_proportion)
    print("\t\tdimensions_per_factor: ", dimensions_per_factor)
    print("\t\tfactor_index_to_included_class_indices: ", factor_index_to_included_class_indices)
    print("\t\tcalibration_data_set_folder path: ", calibration_data_set_folder_path)
    print("\t\tflow_model_folder path: ", flow_model_folder_path)
    print("\t\tfigure_folder path: ", figure_folder_path)
    print("\n\tStarting script now:\n")
    
    ### Start actual data processing

    # Load Configuration
    configuration = configuration_loader.load()
    
    # Configuration
    batch_size = configuration['flow_model_batch_size']
    latent_transfer_sample_size = 2**12 # Needs to be large enough for samples of all conditions to appear
    np.random.seed(configuration['random_seed'])
    tf.keras.utils.set_random_seed(configuration['random_seed'])
    random.seed(configuration['random_seed'])
    stage_count = configuration['flow_model_stage_count']
    epoch_count = configuration['flow_model_epoch_count']
    dimensions_per_factor = configuration['flow_model_dimensions_per_factor']
    materials_to_keep = configuration['flow_model_materials_to_keep']; actions_to_keep = configuration['flow_model_actions_to_keep']
    materials_to_drop = list(range(6))
    for m in reversed(materials_to_keep): materials_to_drop.remove(m)
    actions_to_drop = list(range(4))
    for a in reversed(actions_to_keep): actions_to_drop.remove(a)
    m_string = ",".join(str(m) for m in materials_to_keep)
    a_string = ",".join(str(a) for a in actions_to_keep)
    
    projected_data_path = os.path.join(configuration['latent_yamnet_data_folder'],'projected',f'{np.sum(configuration['dimensions_per_factor'])} dimensions',f'Layer {configuration['layer_index_full_PCA']}')
    original_data_path = os.path.join(configuration['latent_yamnet_data_folder'],'original',f'Layer {configuration['layer_index_full_PCA']}')
    flow_model_save_path = os.path.join(configuration['model_folder'], 'flow models',f'Layer {configuration['layer_index_full_PCA']}')
    pca_model_path = os.path.join(configuration['model_folder'], 'PCA and Standard Scalers',f"Layer {configuration['layer_index_full_PCA']}")
    plot_save_path = os.path.join(configuration['plots_folder'],'evaluate flow models', f'Layer {configuration['layer_index_full_PCA']}')
    
    if not os.path.exists(plot_save_path): os.makedirs(plot_save_path)
    flow_model_save_path = os.path.join(flow_model_save_path, f'Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count}.h5')
    material_labels=configuration['material_to_index'].keys(); action_labels = configuration['action_to_index'].keys()
    
    tf.keras.backend.clear_session() # Need to clear session because otherwise yamnet cannot be loaded
    layer_wise_yamnet = ylw.LayerWiseYamnet()
    layer_wise_yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

    # Load data iterators
    train_iterator, test_iterator, batch_count, Z_train, Z_test, Y_train, Y_test = lsd.load_iterators(data_path=projected_data_path, materials_to_drop=materials_to_drop, actions_to_drop=actions_to_drop, batch_size=batch_size)
    Z_ab_sample, Y_ab_sample = next(train_iterator) # Sample

    print("The data is fed to the model in batches of shape:\n","Z: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_sample: (instance count, factor count): \t', Y_ab_sample.shape)

    # Create network
    flow_network = lsd.create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
    flow_network.load_weights(flow_model_save_path)

    # Evaluate
    scatter_plot_disentangled(flow_network=flow_network, Z=Z_test, Y=Y_test, material_labels=material_labels, action_labels=action_labels, plot_save_path=os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Calibrated Network Scatterplots.png"))

    # Load a sample of even size from yamnets latent space
    Z_prime_sample, Y_sample = utl.load_latent_sample(data_folder=original_data_path, sample_size=latent_transfer_sample_size)
    for material in materials_to_drop:
        Z_prime_sample = Z_prime_sample[Y_sample[:,-2] != material]
        Y_sample = Y_sample[Y_sample[:,-2] != material]

    for action in actions_to_drop:
        Z_prime_sample = Z_prime_sample[Y_sample[:,-1] != action]
        Y_sample = Y_sample[Y_sample[:,-1] != action]

    with open(os.path.join(pca_model_path, 'Pre PCA Standard Scaler.pkl'), 'rb') as file_handle:
        pre_scaler = pkl.load(file_handle)
    with open(os.path.join(pca_model_path, f'Complete PCA.pkl'), 'rb') as file_handle:
        pca = pkl.load(file_handle)
    with open(os.path.join(pca_model_path, 'Post PCA Standard Scaler.pkl'), 'rb') as file_handle:
        post_scaler = pkl.load(file_handle)

    plot_permutation_test(Z_prime=Z_prime_sample, Y=Y_sample, dimensions_per_factor=dimensions_per_factor, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=configuration['layer_index_full_PCA'], plot_save_path=os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Calibrated Network Latent Transfer.png"))
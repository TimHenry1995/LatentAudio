import sys
sys.path.append(".")
import json, argparse
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
from gyoza.utilities import math as gum
import time
from typing import Dict
from scipy import stats

# Define some functions
def scatter_plot_disentangled(flow_network, Z, Y, 
                              factor_index_to_included_class_indices_to_names,
                              factor_index_to_name, 
                              factor_index_to_z_tilde_dimension,
                              figure_file_path) -> None:
    # Convenience variables
    first_factor_index = list(factor_index_to_included_class_indices_to_names.keys())[0]
    first_factor_name = factor_index_to_name[first_factor_index]
    first_factor_dimension = factor_index_to_z_tilde_dimension[first_factor_index]
    first_factor_class_indices = list(factor_index_to_included_class_indices[first_factor_index])
    first_factor_class_labels = list(factor_index_to_included_class_indices_to_names[first_factor_index].values())
    first_factor_index = (int)(first_factor_index)
    
    second_factor_index = list(factor_index_to_included_class_indices_to_names.keys())[1]
    second_factor_name = factor_index_to_name[second_factor_index]
    second_factor_dimension = factor_index_to_z_tilde_dimension[second_factor_index]
    second_factor_class_indices = list(factor_index_to_included_class_indices[second_factor_index])
    second_factor_class_labels = list(factor_index_to_included_class_indices_to_names[second_factor_index].values())
    second_factor_index = (int)(second_factor_index)

    # Predict
    Z_tilde = flow_network(Z)
    first_factor_Z_tilde = Z_tilde[:,first_factor_dimension] # First factor's dimension
    second_factor_Z_tilde = Z_tilde[:,second_factor_dimension] # Second factor's dimension
    first_factor_class_means = [np.mean(first_factor_Z_tilde[Y[:,first_factor_index]==c]) for c in first_factor_class_indices]
    permutation = np.argsort(first_factor_class_means)
    first_factor_class_indices = [first_factor_class_indices[c] for c in permutation]
    first_factor_class_labels = [first_factor_class_labels[c] for c in permutation]
    second_factor_class_means = [np.mean(second_factor_Z_tilde[Y[:,second_factor_index]==c]) for c in second_factor_class_indices]
    permutation = np.argsort(second_factor_class_means)
    second_factor_class_indices = [second_factor_class_indices[c] for c in permutation]
    second_factor_class_labels = [second_factor_class_labels[c] for c in permutation]
    
    # Plot
    plt.subplots(2, 4, figsize=(12,6), gridspec_kw={'width_ratios': [1,5,1,5], 'height_ratios': [5,1]})

    plt.suptitle(f"Disentangled {first_factor_name}s and {second_factor_name}s")

    # 1. First factor
    # 1.1 Vertical Boxplot
    plt.subplot(2,4,1)
    plt.boxplot([second_factor_Z_tilde[Y[:,first_factor_index]==c] for c in first_factor_class_indices])
    plt.xticks(list(range(1,len(first_factor_class_indices)+1)), first_factor_class_labels)
    plt.ylabel(f"{second_factor_name} Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.yticks([])

    # 1.2 Horizontal Boxplot
    plt.subplot(2,4,6)
    plt.boxplot([first_factor_Z_tilde[Y[:,first_factor_index]==c] for c in reversed(first_factor_class_indices)], vert=False)
    plt.yticks(list(range(1,len(first_factor_class_indices)+1)), reversed(first_factor_class_labels))
    plt.xlabel(f"{first_factor_name} Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.xticks([])

    # 1.3 Scatter
    plt.subplot(2,4,2); plt.title(first_factor_name)

    for c in first_factor_class_indices:
        plt.scatter(first_factor_Z_tilde[Y[:,first_factor_index]==c], second_factor_Z_tilde[Y[:,first_factor_index]==c],s=1)
    plt.legend(first_factor_class_labels)
    
    # 2. Second factor
    # 2.1 Vertical Boxplot
    plt.subplot(2,4,3)
    plt.boxplot([second_factor_Z_tilde[Y[:,second_factor_index]==c] for c in second_factor_class_indices])
    plt.xticks(list(range(1,len(second_factor_class_indices)+1)), second_factor_class_labels)
    plt.ylabel(f"{second_factor_name} Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.yticks([])

    # 2.2 Horizontal Boxplot
    plt.subplot(2,4,8)
    plt.boxplot([first_factor_Z_tilde[Y[:,second_factor_index]==c] for c in reversed(second_factor_class_indices)], vert=False)
    plt.yticks(list(range(1,len(second_factor_class_indices)+1)), reversed(second_factor_class_labels))
    plt.xlabel(f"{first_factor_name} Dimension")
    ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    plt.xticks([])

    # 2.3 Scatter
    plt.subplot(2,4,4); plt.title(second_factor_name)

    for c in second_factor_class_indices:
        plt.scatter(first_factor_Z_tilde[Y[:,second_factor_index]==c], second_factor_Z_tilde[Y[:,second_factor_index]==c], s=1)
    plt.legend(second_factor_class_labels)

    # Remove other axes
    plt.subplot(2,4,5); plt.axis('off'); plt.subplot(2,4,7); plt.axis('off')
    if os.path.exists(figure_file_path): 
        print(f"\t\tFound existing figure at {figure_file_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(figure_file_path, (figure_file_path[:-4] + ' (old) ' + (str)(time.time()))[:256] + '.png')
    plt.tight_layout()
    
    plt.savefig(figure_file_path)

def plot_latent_transfer(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, figure_file_path: str, factor_index_to_name, factor_index_to_z_tilde_dimension: Dict[int,int],factor_index_to_y_dimension: Dict[int,int], swop_to_sample_size: Dict[str,int]) -> None:

    # Swop each factor
    swops = {list(factor_index_to_name.values())[0]:['first factor'], list(factor_index_to_name.values())[1]:['second factor'], f'{list(factor_index_to_name.values())[0]} and {list(factor_index_to_name.values())[1]}':['first factor','second factor']}
    
    dissimilarity_function = lambda P, Q: np.sqrt(np.sum(np.power(P-Q, 2), axis=1))# - np.sum(entropy(tf.nn.softmax(P, axis=1),tf.nn.softmax(Q, axis=1)),axis=1)
    #plt.figure(figsize=(10,10)); plt.suptitle('Latent Transfer')
    b = 1
    statistical_results = {"test":f"Wilcoxon Signed Rank test between pairwise Yamnet output logit distances before and after latent transfer."}

    for factor_name, switch_factors in swops.items():
        
        # Sample
        sample_size = swop_to_sample_size[factor_name]
        indices = random.sample(range(Z_prime.shape[0]), sample_size)
                
        # Latent Transfer
        P, Q_before, Q_after = latent_transfer(Z_prime=Z_prime, Y=Y, dimensions_per_factor=dimensions_per_factor, switch_factors=switch_factors, baseline=True, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index, factor_index_to_z_tilde_dimension=factor_index_to_z_tilde_dimension, factor_index_to_y_dimension=factor_index_to_y_dimension)
        
        before = dissimilarity_function(P[indices], Q_before[indices]) # P and Q are each of shape [instance count, class count]. cross entropy is of shape [instance count]
        after = dissimilarity_function(P[indices], Q_after[indices]) # P and Q are each of shape [instance count, class count]. cross entropy is of shape [instance count]
        
        # Compute t-test
        tmp = stats.wilcoxon(before, after, alternative='greater') # We expect the distance to be smaller after latent transfer
        statistical_results[f"{factor_name}"] = {"Mean of Difference": (str)(stats.trim_mean(before-after, 0.1)), "Standard Deviation of Difference": (str)(stats.mstats.trimmed_std(before-after, 0.1)), "Sample size": len(before), "W": tmp.statistic, "p (uncorrected)": (str)(tmp.pvalue)}
        
        b+=1
    
    plt.xlabel('Euclidean distance for Pair of instances before - after transfer')
    if os.path.exists(figure_file_path): 
        print(f"\t\tFound existing figure at {figure_file_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(figure_file_path, figure_file_path[:-4] + ' (old) ' + (str)(time.time())+'.png')
    plt.tight_layout()
    plt.savefig(figure_file_path)
    
    # Save results
    print_path = os.path.join(figure_folder_path, f"Latent {factor_name} Transfer Statistics.txt")
    if os.path.exists(print_path): 
        print(f"\t\tFound existing file at {print_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(print_path, print_path[:-4] + ' (old) ' + (str)(time.time()) + '.txt')

    with open(print_path, 'w') as file:
        json.dump(statistical_results, file)
    
def latent_transfer(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], switch_factors:List[str], baseline:bool, pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, factor_index_to_z_tilde_dimension: Dict[int,int], factor_index_to_y_dimension: Dict[int,int]) -> None:

    instance_count = Z_prime.shape[0]
    #assert instance_count % 2 == 0, f"The number of instance was assumed to be even such that the first half of instances can be swopped with the second half. There were {instance_count} many instances provided."
    
    ### Find partners for each instance in Z_tilde
    partner_indices = [None] * instance_count
        
    first_factor_index = list(factor_index_to_z_tilde_dimension.keys())[0]
    first_factor_z_tilde_dimension = factor_index_to_z_tilde_dimension[first_factor_index]
    first_factor_y_dimension = factor_index_to_y_dimension[first_factor_index]
    first_factor_classes = set(Y[:,first_factor_y_dimension])
    second_factor_index = list(factor_index_to_z_tilde_dimension.keys())[1]
    second_factor_z_tilde_dimension = factor_index_to_z_tilde_dimension[second_factor_index]
    second_factor_y_dimension = factor_index_to_y_dimension[second_factor_index]
    second_factor_classes = set(Y[:,second_factor_y_dimension])

    for i in range(instance_count):
        current_f1_class = Y[i,first_factor_y_dimension]
        current_f2_class = Y[i,second_factor_y_dimension]
    
        # If only one factor needs to be switched, we build pairs of instances that are different for the switch factor but constant for the other factor
        if switch_factors == ['first factor']:
            # Choose a partner with different first factor class while holding second factor constant
            partner_f1_class = random.sample(list(set(first_factor_classes) - set([current_f1_class])), k=1)[0]
            partner_indices[i] = random.sample(list(np.where(np.logical_and(Y[:, first_factor_y_dimension]==partner_f1_class, Y[:, second_factor_y_dimension]==current_f2_class))[0]),k=1)[0]

        elif switch_factors == ['second factor']:
            # Choose a partner with different first factor class while holding second factor constant
            partner_f2_class = random.sample(list(set(second_factor_classes) - set([current_f2_class])), k=1)[0]
            partner_indices[i] = random.sample(list(np.where(np.logical_and(Y[:, first_factor_y_dimension]==current_f1_class, Y[:, second_factor_y_dimension]==partner_f2_class))[0]),k=1)[0]

        # Otherwise, both factors need to be switched
        elif switch_factors == ['first factor','second factor']:
            partner_f1_class = random.sample(list(set(first_factor_classes) - set([current_f1_class])), k=1)[0]
            partner_f2_class = random.sample(list(set(second_factor_classes) - set([current_f2_class])), k=1)[0]
            partner_indices[i] = random.sample(list(np.where(np.logical_and(Y[:, first_factor_y_dimension]==partner_f1_class, Y[:, second_factor_y_dimension]==partner_f2_class))[0]),k=1)[0]

    # Compute P, which is the logits of Yamnet for the first instance of each of the pairs
    layer_index_to_shape = [ [instance_count, 48, 32, 32],  [instance_count, 48, 32, 64],  [instance_count, 24, 16, 128],  [instance_count, 24, 16, 128],  [instance_count, 12, 8, 256],  [instance_count, 12, 8, 256], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 3, 2, 1024], [instance_count, 3, 2, 1024]]
    P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # Compute Q_before which is the logits of the second instance of each pair before latent transfer
    Q_before = np.copy(P[partner_indices])
    
    ### Compute Q_after which is the logits of the second instance of each pair after latent transfer
    
    # Apply standard scalers and pca
    Z_prime = post_scaler.transform(pca.transform(pre_scaler.transform(Z_prime)))

    # Pass the top few dimensions through flow net
    dimension_count = np.sum(dimensions_per_factor)
    Z_tilde = flow_network(Z_prime[:,:dimension_count]).numpy()
    
    # Perform swops
    Z_tilde_after = np.copy(Z_tilde[partner_indices,:])
    for i in range(len(Z_tilde)):
        if 'first factor' in switch_factors:
            Z_tilde_after[i, first_factor_z_tilde_dimension] = Z_tilde[i, first_factor_z_tilde_dimension]
        if 'second factor' in switch_factors:
            Z_tilde_after[i, second_factor_z_tilde_dimension] = Z_tilde[i, second_factor_z_tilde_dimension]
      
    # Replace top few dimensions
    Z_prime_after = np.copy(Z_prime[partner_indices,:])
    Z_prime_after[:,:dimension_count] = flow_network.invert(Z_tilde_after)

    # Invert full pca, invert scaler
    Z_prime_after = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_after)))

    # Continue processing through yamnet
    Q_after = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_after, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # Outputs
    return P, Q_before, Q_after
    

if __name__ == "__main__":
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="evaluate_disentangle",
        description='''This script visualizes the model created by the disentangle script. This happens in two ways:
                    (1) Factor disentanglement: It passes the previously PCA projected data through the flow-model and then creates two scatterplots with the same dots, yet one with 
                    coloring for the first factor and one with coloring for the second factor. It is assumed that each of these two factors only has a single dimension in the flow-mdoel's output.
                    (2) Latent transfer: It passes the original Yamnet latent representations of the indicated layer through the provided full PCA model and standard scalers, then takes the top k dimensions
                    (where k is the number of dimenions that the flow-model expects) and passes them through the flow-model. In the disentangled space, for each instance (called a receptor instance), 
                    the script takes the value along one of the factors of interest from another instance (called the donor instance) and uses it to set the receptor instance's value along that factor.
                    It then passes both instances back via the inverse flow model, the inverse standard scalers and inverse PCA and then continues downard Yamnet processing.
                    It then checks to what extent the output logits of Yamnet for the receptor instance assimilate to those of the donor instance.
                    The data being passed through the flow model is the validation data from the disentangle script. Using test data would make the coordination of scripts too complicated and since the
                    flow-model is not going to be deployed for outside use, test data is not considered worth the additional reduction in the amount of training data.

                    There are two ways to use this script. The first way is to pass a configuration_step and a configuration_file_path which will then be used to read the values for all other arguments.
                    The second way is to manually pass all these other arguments while calling the script.
                    For the latter option, all arguments are expected to be json strings such that they can be parsed into proper Python types. 
                    When writing a string inside a json string, use the excape character and double quotes instead of single quotes to prevent common parsing errors.''')
    
    parser.add_argument("--stage_count", help="An int (in form of a json string) that is used to set the number of stages in the flow model. The more stages are used, the more complex the model will be.", type=str)
    parser.add_argument("--epoch_count", help="An int (in form of a json string) indicating how many iterations through the dataset were made during training.", type=str)
    parser.add_argument("--dimensions_per_factor", help="A list of ints (in form of a json string) indicating how many dimensions each factor should be allocated in the output of the flow model. The zeroth entry is for the residual factor, the first entry is for the first factor, the second entry for the second factor, etc. The sum of dimensions has to be equal to the dimensionality of the input to the flow model.", type=str)
    parser.add_argument("--random_seed", help="An int (in form of a json string) that is used to set the random module or Python in order to make the instance sampling reproducible. Note, this random seed as well as the validation_proportion should be the same as for the script that calibrated the flow model in order to make sure there is no overlap between the test set and the training/ validation sets.", type=str)
    parser.add_argument("--validation_proportion", help="A float (in form of a json string) indicating the proportion of the entire data that should be used for validating the model.", type=str)
    parser.add_argument("--factor_index_to_z_tilde_dimension", help="A dictionary (in form of a json string) that has two entires, namely one for the first factor of interest and one for the second factor of interest. For a given entry, the key is the index of the factor and the value is the index in the corresponding dimension in the flow model's output along the last axis. This is used to extract the coordinates from the output for plotting in a scatter plot.", type=str)
    parser.add_argument("--factor_index_to_y_dimension", help="A dictionary (in form of a json string) that has two entires, namely one for the first factor of interest and one for the second factor of interest. For a given entry, the key is the index of the factor and the value is the index in the corresponding dimension in the Y variable. Dimensions exclude the residual factor, thus dimension 0 is the first actual factor's dimension.", type=str)
    parser.add_argument("--factor_index_to_included_class_indices_to_names", help="A dictionary (in form of a json string) that maps the index of a factor to another dictionary. That other dictionary maps the indices of its classes to their corresponding display names. The indexing of factors has to be in line with that in the Y files of the test data, i.e. the residual factor is excluded.", type=str)
    parser.add_argument("--factor_index_to_name", help="A dictionary (in form of a json string) that maps the index of a factor to the name of that factor.", type=str)
    parser.add_argument("--latent_representations_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the original Yamnet latent representations, i.e. before PCA projection, are stored. This folder should directly include the data, not indirectly in e.g. a layer-specific subfolder.", type=str)
    parser.add_argument("--layer_index", help="The index (in form of a json string) pointing to the Yament layer for which the latent transfer shall be made.", type=str)
    parser.add_argument("--pca_projected_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the projections are stored. This folder should directly include the data, not indirectly in e.g. a layer-specific subfolder.", type=str)
    parser.add_argument("--PCA_and_standard_scaler_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the models are stored.", type=str)
    parser.add_argument("--flow_model_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the model weights are saved.", type=str)
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plot should be saved.")
    parser.add_argument("--file_name_prefix_to_factor_wise_label", help="A dictionary (as json string) that maps from a prefix of the file name of a latent representation to its factorwise numeric labels. For example, if a latent representation of a single sound is called CD_X_1.npy, then 'CD' would be be mapped to [1,3], if C is the abbreviation for the first factor's class whose index is 1 and D the second factor's class whose index is 3. Note that the factor-wise numeric labels do not include the residual factor but only the actual factors that would be of interest to a flow model.", type=str)
    parser.add_argument("--swop_to_sample_size", help="A dictionary (as json string) that maps from the swop name (here 'Material', 'Action' or 'Material and Action') to the sample size used for a paired samples t-test in the latent transfer to compare Yamnet's output distances before and after transfer.", type=str)

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.stage_count != None and args.epoch_count != None and args.dimensions_per_factor != None and args.random_seed != None and args.validation_proportion != None and args.factor_index_to_z_tilde_dimension != None and args.factor_index_to_y_dimension != None and args.factor_index_to_included_class_indices_to_names != None and args.factor_index_to_name != None and args.latent_representations_folder != None and args.layer_index != None and args.pca_projected_folder != None and args.PCA_and_standard_scaler_folder != None and args.flow_model_folder != None and args.figure_folder != None and args.file_name_prefix_to_factor_wise_label != None and args.swop_to_sample_size != None, "If no configuration file is provided, then all other arguments must be provided."
    
        stage_count = json.loads(args.stage_count)
        epoch_count = json.loads(args.epoch_count)
        dimensions_per_factor = json.loads(args.dimensions_per_factor)
        random_seed = json.loads(args.random_seed)
        validation_proportion = json.loads(args.validation_proportion)
        factor_index_to_z_tilde_dimension = json.loads(args.factor_index_to_z_tilde_dimension)
        factor_index_to_y_dimension = json.loads(args.factor_index_to_y_dimension)
        factor_index_to_included_class_indices_to_names = json.loads(args.factor_index_to_included_class_indices_to_names)
        factor_index_to_name = json.loads(args.factor_index_to_name)
        latent_representations_folder = json.loads(args.latent_representations_folder)
        latent_representations_folder_path = os.path.join(*latent_representations_folder)
        layer_index = json.loads(args.layer_index)
        pca_projected_folder = json.loads(args.pca_projected_folder)
        pca_projected_folder_path = os.path.join(*pca_projected_folder)
        PCA_and_standard_scaler_folder = json.loads(args.PCA_and_standard_scaler_folder)
        PCA_and_standard_scaler_folder_path = os.path.join(*PCA_and_standard_scaler_folder)
        flow_model_folder = json.loads(args.flow_model_folder)
        flow_model_folder_path = os.path.join(*flow_model_folder)
        figure_folder = json.loads(args.figure_folder)
        figure_folder_path = os.path.join(*figure_folder)
        file_name_prefix_to_factor_wise_label = json.loads(args.file_name_prefix_to_factor_wise_label)
        swop_to_sample_size = json.loads(args.swop_to_sample_size)
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        print(args)
        assert args.stage_count == None and args.epoch_count == None and args.dimensions_per_factor == None and args.random_seed == None and args.validation_proportion == None and args.factor_index_to_z_tilde_dimension == None and args.factor_index_to_y_dimension == None and args.factor_index_to_included_class_indices_to_names == None and args.factor_index_to_name == None and args.latent_representations_folder == None and args.layer_index == None and args.pca_projected_folder == None and args.PCA_and_standard_scaler_folder == None and args.flow_model_folder == None and args.figure_folder == None and args.file_name_prefix_to_factor_wise_label == None and args.swop_to_sample_size == None, "If a configuration file is provided, then no other arguments shall be provided."
        assert args.configuration_step != None, "If a configuration file is given, then also the configuration_step needs to be provided."

        # Load configuration      
        configuration = configuration_loader.load_configuration_step(file_path=args.configuration_file_path, step=args.configuration_step)
        
        # Ensure step corresponds to this script
        assert configuration['script'] == 'evaluate_disentangle' or configuration['script'] == 'evaluate_disentangle.py', "The configuration_step points to an entry in the configuration_file that does not pertain to the current script. Ensure the 'script' attribute is equal to 'evaluate_disentangle'."
        
        stage_count = configuration['arguments']['stage_count']
        epoch_count = configuration['arguments']['epoch_count']
        dimensions_per_factor = configuration['arguments']['dimensions_per_factor']
        random_seed = configuration['arguments']['random_seed']
        validation_proportion = configuration['arguments']['validation_proportion']
        factor_index_to_z_tilde_dimension = configuration['arguments']['factor_index_to_z_tilde_dimension']
        factor_index_to_y_dimension = configuration['arguments']['factor_index_to_y_dimension']
        factor_index_to_included_class_indices_to_names = configuration['arguments']['factor_index_to_included_class_indices_to_names']
        factor_index_to_name = configuration['arguments']['factor_index_to_name']
        latent_representations_folder_path = os.path.join(*configuration['arguments']['latent_representations_folder'])
        layer_index = configuration['arguments']['layer_index']
        pca_projected_folder_path = os.path.join(*configuration['arguments']['pca_projected_folder'])
        PCA_and_standard_scaler_folder_path = os.path.join(*configuration['arguments']['PCA_and_standard_scaler_folder'])
        flow_model_folder_path = os.path.join(*configuration['arguments']['flow_model_folder'])
        figure_folder_path = os.path.join(*configuration['arguments']['figure_folder'])
        file_name_prefix_to_factor_wise_label = configuration['arguments']['file_name_prefix_to_factor_wise_label']
        swop_to_sample_size = configuration['arguments']['swop_to_sample_size']

    print("\n\n\tStarting script evaluate_disentangle")
    print("\t\tThe script parsed the following arguments:")
    print("\t\tstage_count: ", stage_count)
    print("\t\tepoch_count: ", epoch_count)
    print("\t\tdimensions_per_factor: ", dimensions_per_factor)
    print("\t\trandom_seed: ", random_seed)
    print("\t\tvalidation_proportion: ", validation_proportion)
    print("\t\tfactor_index_to_z_tilde_dimension: ", factor_index_to_z_tilde_dimension)
    print("\t\tfactor_index_to_y_dimension: ", factor_index_to_y_dimension)
    print("\t\tfactor_index_to_included_class_indices_to_names: ", factor_index_to_included_class_indices_to_names)
    print("\t\tfactor_index_to_name: ", factor_index_to_name)
    print('\t\tlatent_representations_folder path: ', latent_representations_folder_path)
    print('\t\tlayer_index:', layer_index)
    print("\t\tpca_projected_folder path: ", pca_projected_folder_path)
    print("\t\tPCA_and_standard_scaler_folder path:", PCA_and_standard_scaler_folder_path)
    print("\t\tflow_model_folder path: ", flow_model_folder_path)
    print("\t\tfigure_folder path: ", figure_folder_path)
    print("\t\tfile_name_prefix_to_factor_wise_label: ", file_name_prefix_to_factor_wise_label)
    print("\t\tswop_to_sample_size: ", swop_to_sample_size)
    print("\n\tStarting script now:\n")
    
       
    ### Start actual data processing
    
    # Ensure figure folder exists
    if not os.path.exists(figure_folder_path): os.makedirs(figure_folder_path)
    
    # Load yamnet
    tf.keras.backend.clear_session() # Need to clear session because otherwise yamnet cannot be loaded
    layer_wise_yamnet = ylw.LayerWiseYamnet()
    layer_wise_yamnet.load_weights(os.path.join('LatentAudio','plugins','yamnet','yamnet.h5'))

    # Load data iterators
    factor_index_to_included_class_indices = {factor_index: [(int)(index) for index in class_index_to_names.keys()] for factor_index, class_index_to_names in factor_index_to_included_class_indices_to_names.items()}
    _, validation_iterator, _, Z_validation, _, Y_validation = lsd.load_iterators(data_path=pca_projected_folder_path, factor_index_to_included_class_indices=factor_index_to_included_class_indices, validation_proportion=validation_proportion, batch_size=1, random_seed=random_seed) # The batch_size does not matter here
    Z_ab_sample, Y_ab_sample = next(validation_iterator) # Sample

    # Create network
    flow_model_file_path = os.path.join(flow_model_folder_path, f'Flow model {stage_count} stages, {epoch_count} epochs, {dimensions_per_factor} dimensions per factor and {factor_index_to_included_class_indices} included class indices'.replace(":","="))
    flow_model_file_path = flow_model_file_path[:257] + '.h5' # Trim path if too long
    
    flow_network = lsd.create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
    flow_network.load_weights(flow_model_file_path)

    # Evaluate
    scatter_plot_file_path = os.path.join(figure_folder_path, f'Flow model scatter {stage_count} stages, {epoch_count} epochs, {dimensions_per_factor} dimensions per factor and {factor_index_to_included_class_indices} included class indices'.replace(":","="))
    scatter_plot_file_path = scatter_plot_file_path[:256] + '.png' # Trim path if too long

    # We are passing the validation data here instead of the test data because the model is not going to be deployed anyways and coordinating test data with the other scripts and other models would be very cumbersome
    # All of the validation data will be used for the scatter plot
    scatter_plot_disentangled(flow_network=flow_network, Z=Z_validation, Y=Y_validation, factor_index_to_included_class_indices_to_names=factor_index_to_included_class_indices_to_names,
                              factor_index_to_z_tilde_dimension=factor_index_to_z_tilde_dimension, factor_index_to_name=factor_index_to_name, figure_file_path=scatter_plot_file_path)
                            
    # Load a sample of even size from yamnets latent space
    x_file_names = utl.find_matching_strings(strings=os.listdir(latent_representations_folder_path), token='_X_')
    x_file_names.sort() # Needed to ensure that the file names are consistently ordered across runs
    random.seed(random_seed)
    taboo_list = random.sample(x_file_names, k=(int)((1-validation_proportion)*len(x_file_names)))
    x_file_names = list(set(x_file_names) - set(taboo_list))
    
    # Exclude unwanted classes
    for x_file_name in reversed(x_file_names):
        valid = False
        for prefix in file_name_prefix_to_factor_wise_label.keys():
            if prefix in x_file_name: valid = True
        if not valid: x_file_names.remove(x_file_name)
    x_file_names = x_file_names[:np.max(list(swop_to_sample_size.values()))]

    x_shape = np.load(os.path.join(latent_representations_folder_path, x_file_names[0])).shape
    Z_prime_sample = np.zeros([len(x_file_names)] + list(x_shape)); 
    Y_sample = np.zeros([len(x_file_names), len(list(file_name_prefix_to_factor_wise_label.values())[0])])
    
    for i, x_file_name in enumerate(x_file_names): 
        
        # Load
        Z_prime_sample[i,:] = np.load(os.path.join(latent_representations_folder_path, x_file_name))[np.newaxis,:]
        
        # Determine factorwise labels
        file_name_prefix = x_file_name.split(sep='_X_')[0]
        factorwise_label = file_name_prefix_to_factor_wise_label[file_name_prefix]
        Y_sample[i,:] = np.array(factorwise_label)

    with open(os.path.join(PCA_and_standard_scaler_folder_path, 'Pre PCA Standard Scaler.pkl'), 'rb') as file_handle:
        pre_scaler = pkl.load(file_handle)
    with open(os.path.join(PCA_and_standard_scaler_folder_path, f'PCA.pkl'), 'rb') as file_handle:
        pca = pkl.load(file_handle)
    with open(os.path.join(PCA_and_standard_scaler_folder_path, 'Post PCA Standard Scaler.pkl'), 'rb') as file_handle:
        post_scaler = pkl.load(file_handle)

    bar_plot_file_path = os.path.join(figure_folder_path, f'Flow model bar {stage_count} stages, {epoch_count} epochs, {dimensions_per_factor} dimensions per factor and {factor_index_to_included_class_indices} included class indices'.replace(":","="))
    bar_plot_file_path = bar_plot_file_path[:256] + '.png' # Trim path if too long

    plot_latent_transfer(Z_prime=Z_prime_sample, Y=Y_sample, dimensions_per_factor=dimensions_per_factor, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index, figure_file_path=bar_plot_file_path, factor_index_to_name=factor_index_to_name, factor_index_to_z_tilde_dimension=factor_index_to_z_tilde_dimension, factor_index_to_y_dimension=factor_index_to_y_dimension, swop_to_sample_size=swop_to_sample_size)
    
    # Log
    print("\n\n\Completed script evaluate disentangle")
    
    

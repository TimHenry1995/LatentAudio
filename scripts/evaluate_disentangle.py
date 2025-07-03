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
                              figure_file_path) -> None:
    # Convenience variables
    first_factor_index = list(factor_index_to_included_class_indices_to_names.keys())[0]
    first_factor_name = factor_index_to_name[first_factor_index]
    first_factor_dimension = factor_index_to_dimension_index[first_factor_index]
    first_factor_class_indices = list(factor_index_to_included_class_indices[first_factor_index])
    first_factor_class_labels = list(factor_index_to_included_class_indices_to_names[first_factor_index].values())
    first_factor_index = (int)(first_factor_index)

    second_factor_index = list(factor_index_to_included_class_indices_to_names.keys())[1]
    second_factor_name = factor_index_to_name[second_factor_index]
    second_factor_dimension = factor_index_to_dimension_index[second_factor_index]
    second_factor_class_indices = list(factor_index_to_included_class_indices[second_factor_index])
    second_factor_class_labels = list(factor_index_to_included_class_indices_to_names[second_factor_index].values())
    second_factor_index = (int)(second_factor_index)

    # Predict
    Z_tilde = flow_network(Z)
    first_factor_Z_tilde = Z_tilde[:,first_factor_dimension] # First factor's dimension
    second_factor_Z_tilde = Z_tilde[:,second_factor_dimension] # Second factor's dimension

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

def plot_permutation_test(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, figure_file_path: str, factor_index_to_name, factor_index_to_z_tilde_dimension: Dict[int,int],y_factor_index_to_dimension_index: Dict[int,int]) -> None:

    # Swop each factor
    swops = {list(factor_index_to_name.values())[0]:['first factor'], list(factor_index_to_name.values())[1]:['second factor'], f'{list(factor_index_to_name.values())[0]} and {list(factor_index_to_name.values())[1]}':['first factor','second factor']}
    entropy = lambda P, Q: P*np.log(Q)
    dissimilarity_function = lambda P, Q: np.sqrt(np.sum(np.power(P-Q, 2), axis=1))# - np.sum(entropy(tf.nn.softmax(P, axis=1),tf.nn.softmax(Q, axis=1)),axis=1)
    plt.figure(figsize=(10,10)); plt.suptitle('Latent Transfer')
    b = 1
    x_min = np.finfo(np.float32).max
    x_max = np.finfo(np.float32).min
    
    for factor_name, switch_factors in swops.items():
        # Baseline
        P, Q_before, Q_after = latent_transfer(Z_prime=Z_prime, Y=Y, dimensions_per_factor=dimensions_per_factor, switch_factors=switch_factors, baseline=True, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index, factor_index_to_z_tilde_dimension=factor_index_to_z_tilde_dimension, y_factor_index_to_dimension_index=y_factor_index_to_dimension_index)
        baseline = dissimilarity_function(P, Q_before) # P and Q are each of shape [instance count, class count]. cross entropy is of shape [instance count]
        experimental = dissimilarity_function(P, Q_after) # P and Q are each of shape [instance count, class count]. cross entropy is of shape [instance count]
        
        # Plot
        plt.subplot(len(swops),1,b); plt.title(factor_name)
        plt.violinplot([np.log(baseline),np.log(experimental)], vert=False, showmedians=True)#plt.boxplot([baseline, experimental], showmeans=True, vert=False, showfliers=False)
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
        
    if os.path.exists(figure_file_path): 
        print(f"\t\tFound existing figure at {figure_file_path}. Renaming that one with appendix ' (old) ' and time-stamp.")
        os.rename(figure_file_path, figure_file_path + ' (old) ' + (str)(time.time()))
    plt.tight_layout()
    plt.savefig(figure_file_path)
    
def latent_transfer(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], switch_factors:List[str], baseline:bool, pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, factor_index_to_z_tilde_dimension: Dict[int,int], y_factor_index_to_dimension_index: Dict[int,int]) -> None:

    instance_count = Z_prime.shape[0]
    #assert instance_count % 2 == 0, f"The number of instance was assumed to be even such that the first half of instances can be swopped with the second half. There were {instance_count} many instances provided."
    
    ### Find partners for each instance in Z_tilde
    partner_indices = [None] * instance_count
        
    first_factor_index = list(factor_index_to_z_tilde_dimension.keys())[0]
    first_factor_z_tilde_dimension = factor_index_to_z_tilde_dimension[first_factor_index]
    first_factor_y_dimension = y_factor_index_to_dimension_index[first_factor_index]
    first_factor_classes = set(Y[:,first_factor_y_dimension])
    second_factor_index = list(factor_index_to_z_tilde_dimension.keys())[1]
    second_factor_z_tilde_dimension = factor_index_to_z_tilde_dimension[second_factor_index]
    second_factor_y_dimension = y_factor_index_to_dimension_index[second_factor_index]
    second_factor_classes = set(Y[:,second_factor_y_dimension])

    if len(switch_factors) == 1:
        for i in range(instance_count):
            current_f1_class = Y[i,first_factor_y_dimension]
            current_f2_class = Y[i,second_factor_y_dimension]
        
            # If only one factor needs to be switched, we build pairs of instances that are different for the switch factor but constant for the other factor
            if switch_factors == ['first factor']:
                # Choose a partner with different first factor class while holding second factor constant
                partner_f1_class = random.sample(set(first_factor_classes) - set([current_f1_class]))
                partner_indices[i] = random.sample(np.where(np.logical_and(Y[:, first_factor_y_dimension]==partner_f1_class, Y[:, second_factor_y_dimension]==current_f2_class)))

            elif switch_factors == ['second factor']:
                # Choose a partner with different first factor class while holding second factor constant
                partner_f2_class = random.sample(set(second_factor_classes) - set([current_f2_class]))
                partner_indices[i] = random.sample(np.where(np.logical_and(Y[:, first_factor_y_dimension]==current_f1_class, Y[:, second_factor_y_dimension]==partner_f2_class)))

            # Otherwise, both factors need to be switched
            else:
                partner_f1_class = random.sample(set(first_factor_classes) - set([current_f1_class]))
                partner_f2_class = random.sample(set(second_factor_classes) - set([current_f2_class]))
                partner_indices[i] = random.sample(np.where(np.logical_and(Y[:, first_factor_y_dimension]==partner_f1_class, Y[:, second_factor_y_dimension]==partner_f2_class)))

    # Compute P, which is the logits of Yamnet for the first instance of each of the pairs
    layer_index_to_shape = [ [instance_count, 48, 32, 32],  [instance_count, 48, 32, 64],  [instance_count, 24, 16, 128],  [instance_count, 24, 16, 128],  [instance_count, 12, 8, 256],  [instance_count, 12, 8, 256], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 3, 2, 1024], [instance_count, 3, 2, 1024]]
    P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # Compute Q_before which is the logits of the second instance of each pair before latent transfer
    Q_before = np.copy([partner_indices])
    
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
      
    # Invert flow net
    Z_after = flow_network.invert(Z_tilde_after)

    # Replace top few dimensions
    Z_prime_after = np.copy(Z_prime)
    Z_prime_after[:,:dimension_count] = Z_after

    # Invert full pca, invert scaler
    Z_prime_after = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_after)))

    # Continue processing through yamnet
    Q_after = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_after, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # Outputs
    return P, Q_before, Q_after
    """
    # Partition the data according to classes
    partition = {}
    first_factor_index = (int)(list(factor_index_to_name.keys())[0])
    second_factor_index = (int)(list(factor_index_to_name.keys())[1])
    f1s = set(Y[:, first_factor_index]); f2s = set(Y[:,second_factor_index])
    for f1 in f1s:
        partition[f"f1{f1}"] = np.copy(Z_tilde[Y[:,first_factor_index]==f1])
    
    for f2 in f2s:
        partition[f'f2{f2}'] = np.copy(Z_tilde[Y[:,second_factor_index]==f2])

    # Perform swops
    Z_tilde_swapped = np.copy(Z_tilde)
    for i in range(len(Z_tilde)):
        # Determine the material and action class
        f1_i = Y[i,first_factor_index]; f2_i = Y[i, second_factor_index]

        # Baseline
        if baseline:
            # In baseline mode swaps will only be made for the indicated factors with instances with the same class
            if 'first factor' in switch_factors:
                # Sample from the set of points with same material
                Z_tilde_swapped[i, first_factor_index] = random.choice(partition[f"f1{f1_i}"])[first_factor_index]
            if 'second factor' in switch_factors:
                # Sample from the set of points with same action
                Z_tilde_swapped[i, second_factor_index] = random.choice(partition[f"f2{f2_i}"])[second_factor_index]
        else:
            # In experimental mode swaps will only be made for indicated factors with instances that have a different class
            if 'first factor' in switch_factors:
                # Sample from the set of points with other material
                m_j = random.choice(list(f1s.difference({f1_i})))
                Z_tilde_swapped[i, first_factor_index] = random.choice(partition[f"f1{m_j}"])[first_factor_index]
            if 'second factor' in switch_factors:
                # Sample from the set of points with other action
                a_j = random.choice(list(f2s.difference({f2_i})))
                Z_tilde_swapped[i, second_factor_index] = random.choice(partition[f"f2{a_j}"])[second_factor_index]
      
    # Invert flow net
    Z_swapped = flow_network.invert(Z_tilde_swapped)

    # Replace top few dimensions
    Z_prime_swapped = np.copy(Z_prime)
    Z_prime_swapped[:,:dimension_count] = Z_swapped

    # Invert full pca, invert scaler
    Z_prime_swapped = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_swapped)))

    # Continue processing through yamnet
    Q = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_swapped, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # Outputs
    return P, Q
    """
'''
def plot_latent_transfer(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, plot_save_path: str, factor_index_to_name: Dict[int,str], factor_index_to_z_tilde_dimension: Dict[int,int], y_factor_index_to_dimension_index: Dict[int,int]) -> None:

    dissimilarity_function = lambda P, Q: np.sqrt(np.sum((P-Q)**2, axis=1))#- np.sum(entropy(tf.nn.softmax(P, axis=1),tf.nn.softmax(Q, axis=1)),axis=1)
    plt.figure(figsize=(10,3)); plt.suptitle('        Latent Transfer')
    b = 1
    x_min = np.finfo(np.float32).max
    x_max = np.finfo(np.float32).min

    first_factor_name = list(factor_index_to_name.values())[0]
    first_factor_index = list(factor_index_to_name.keys())[0]
    first_factor_z_tilde_dimension = factor_index_to_z_tilde_dimension[first_factor_index]
    first_factor_y_dimension = y_factor_index_to_dimension_index[first_factor_index]
    second_factor_name = list(factor_index_to_name.values())[1]
    second_factor_index = list(factor_index_to_name.keys())[1]
    second_factor_z_tilde_dimension = factor_index_to_z_tilde_dimension[second_factor_index]
    second_factor_y_dimension = y_factor_index_to_dimension_index[second_factor_index]

    for current_factor_name, transfer_dimensions in {first_factor_name:[first_factor_z_tilde_dimension, first_factor_y_dimension, second_factor_y_dimension], second_factor_name:[second_factor_z_tilde_dimension, second_factor_y_dimension, first_factor_y_dimension]}.items():
        current_factor_z_tilde_dimension = transfer_dimensions[0]
        current_factor_y_dimension = transfer_dimensions[1]
        other_factor_y_dimension = transfer_dimensions[2]
        
        # initialize
        H_PM = np.array([])
        H_P_M_to_P = np.array([])
        H_P_M_to_Q = np.array([])

        for s in range(10):
            # Shuffle
            indices = list(range(Z_prime.shape[0])); np.random.shuffle(indices)
            Z_prime = Z_prime[indices]; Y = Y[indices]
            for c in set(Y[:,other_factor_y_dimension]):

                # Compute probability distributions
                P, M, M_to_P, M_to_Q = latent_transfer(Z_prime=Z_prime[Y[:,other_factor_y_dimension]==c], Y=Y[Y[:,other_factor_y_dimension]==c], dimensions_per_factor=dimensions_per_factor, transfer_dimension_y=current_factor_y_dimension, transfer_dimension_z_tilde=current_factor_z_tilde_dimension, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index)
                
                # Compute dissimilarities
                H_PM = np.concatenate([H_PM, dissimilarity_function(P,M)])
                H_P_M_to_P = np.concatenate([H_P_M_to_P, dissimilarity_function(P,M_to_P)]) 
                H_P_M_to_Q = np.concatenate([H_P_M_to_Q, dissimilarity_function(P,M_to_Q)]) 
        
        # Adjust sample size for t test.  This sample size was calculated using a small initial sample and this website https://sample-size.net/sample-size-study-paired-t-test/
        if b == 1: 
           sample_size_mp = 199 # alpha one tailed 0.025, beta=0.2, effect size = 0.02, std of change = 0.1
           sample_size_mq = 199 # alpha one tailed 0.025, beta=0.2, effect size = 0.02, std of change = 0.1
        if b == 2: 
            sample_size_mp = 787 # alpha one tailed 0.025, beta=0.2, effect size = 0.005, std of change = 0.05
            sample_size_mq = 309 #  alpha one tailed 0.025, beta=0.2, effect size = 0.008, std of change = 0.05
        assert len(H_PM) > max(sample_size_mp, sample_size_mq), "Sample sizes were too small to do a significance test." # The others Hs have the same number of instances
        indices = random.sample(range(len(H_PM)), max(sample_size_mp, sample_size_mq))
        
        
        Latent Transfer statististics Material
        H_PM and H_P_M_to_P have test results:
        TtestResult(statistic=2.785779995991498, pvalue=0.005858864225107021, df=198)
        H_PM and H_P_M_to_Q have test results:
        TtestResult(statistic=-2.9205989565475496, pvalue=0.0038989501277942227, df=198)


        Latent Transfer statististics Action
        H_PM and H_P_M_to_P have test results:
        TtestResult(statistic=2.5633447859516485, pvalue=0.010551890126375264, df=786)
        H_PM and H_P_M_to_Q have test results:
        TtestResult(statistic=-3.5004239493223115, pvalue=0.0005331221549660843, df=308)

        # Plot
        plt.subplot(1,2,b); plt.title(current_factor_name)
        means = [np.mean(H_PM),np.mean(H_P_M_to_P),np.mean(H_P_M_to_Q)]
        errors = [np.std(H_PM)/ np.sqrt(len(H_PM)), np.std(H_P_M_to_P)/ np.sqrt(len(H_P_M_to_P)), np.std(H_P_M_to_Q)/ np.sqrt(len(H_P_M_to_Q))]
        plt.bar([1,2,3], means, color=[0.1,0.1,0.1,0.1], edgecolor='black')
        plt.xticks([1,2,3],['P,M','P,M->P','P,M->Q'])
        plt.grid(alpha=0.25)
        if b==1:plt.ylabel(r'Euclidean Distance $\Delta$' + ' of \nYamnet Output Logits')
        plt.ylim(means[1]-1.8*errors[1], means[2]+1.5*errors[2])

        # Significance tests
        print('\n')
        print("Latent Transfer statististics", current_factor_name)
        print("H_PM and H_P_M_to_P have test results:")
        print(stats.ttest_rel(H_PM[indices[:sample_size_mp]], H_P_M_to_P[indices[:sample_size_mp]]))
        print("H_PM and H_P_M_to_Q have test results:")
        print(stats.ttest_rel(H_PM[indices[:sample_size_mq]], H_P_M_to_Q[indices[:sample_size_mq]]))

        if stats.ttest_rel(H_PM[indices[:sample_size_mp]], H_P_M_to_P[indices[:sample_size_mp]]).pvalue <= 0.025: # Bonferroni corrected
            plt.annotate('*', (1.99, means[1]+0.5*errors[1]))
        else: plt.annotate('o', (1.99, means[1]+0.5*errors[1]))
        if stats.ttest_rel(H_PM[indices[:sample_size_mq]], H_P_M_to_Q[indices[:sample_size_mq]]).pvalue <= 0.025: # Bonferroni corrected
            plt.annotate('*', (2.99, means[2]+0.5*errors[2]))
        else: plt.annotate('o', (2.99, means[2]+0.5*errors[2]))
        b+=1
        
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()

def latent_transfer(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], transfer_dimension_y: int, transfer_dimension_z_tilde: int, pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int) -> None:

    # 1. Disentangle

    # 1.1 Apply standard scalers and pca
    Z_prime_pca = post_scaler.transform(pca.transform(pre_scaler.transform(Z_prime)))

    # 1.2 Pass the top few dimensions through flow net
    dimension_count = np.sum(dimensions_per_factor)
    Z_tilde = flow_network(Z_prime_pca[:,:dimension_count]).numpy()

    # Choose two random classes of the current factor
    cs=list(set(Y[:,transfer_dimension_y])); cs=sorted(cs); random.shuffle(cs); p = cs[0]; q = cs[1] # Choose a random class for p and q
       
    # Choose the atypical P-instances. They are in the middle of the P and Q distributions
    mean_p = np.mean(Z_tilde[Y[:,transfer_dimension_y]==p])
    mean_q = np.mean(Z_tilde[Y[:,transfer_dimension_y]==q])
    mean_p_q = np.mean(Z_tilde[np.logical_or(Y[:,transfer_dimension_y]==p, Y[:,transfer_dimension_y]==q), transfer_dimension_z_tilde])
    mean_distance = np.abs(mean_p-mean_q)
    relative_error_tolerance = 0.1 # We will guarantee that for each triplet that we build, the distance from our middle instance to the chosen typical p instance is equal to the distance from the middle instance to a typical q instance, up to the relative error (relative w.r.t the distance between the means of the two distributions p and q) 
    absolute_error_tolerance = relative_error_tolerance * mean_distance

    
    tmp = max(0, (absolute_error_tolerance - np.abs(mean_p_q-mean_p) + np.abs(mean_p_q - mean_q))/4.0)
    assert tmp >= 0
    assert tmp <= np.abs(mean_p_q - mean_q)*2.0
    assert tmp <= np.abs(mean_p_q - mean_q)
    d = min(d, np.abs(mean_p_q-mean_p)) # Bound from above
    
        
    radius = min(absolute_error_tolerance - np.abs(mean_p_q - mean_p), absolute_error_tolerance - np.abs(mean_p_q - mean_p))/2.0 
    d needs to be positive
    and smaller than pq - P

    lower_valley_bound = mean_p_q - valley_size_percentage*mean_distance # All instance between these bounds fall within the valley and will be subject to latent transfer
    upper_valley_bound = mean_p_q + valley_size_percentage*mean_distance
    P_valley_indices = np.where(np.logical_and(Y[:,transfer_dimension_y]==p,
                                                lower_valley_bound <= Z_tilde[:, transfer_dimension_z_tilde],
                                                Z_tilde[:, transfer_dimension_z_tilde] <= upper_valley_bound))[0]
    
    # Choose typical P and Q instances (they are close to the respective means)

    t12, t23 = mean-0.2, mean+0.2
    #Q_indices = np.where(np.logical_and(Y[:,transfer_dimension_y] == q, # Instance needs to belong to Qs class 
    #                                    np.abs(Z_tilde[:,transfer_dimension_z_tilde] - np.mean(Z_tilde[Y[:,transfer_dimension_y] == q,transfer_dimension_z_tilde])) < 0.3))[0] # Instance needs to be close to centre of q
    M_indices = np.where(np.logical_and(t12 < Z_tilde[:,transfer_dimension_z_tilde], Z_tilde[:,transfer_dimension_z_tilde] <= t23, Y[:,transfer_dimension_y]==p))[0] # Points that are between p and q but belong to p
    #P_indices = np.where(np.logical_and(Y[:,transfer_dimension_y] == p, # Instance needs to belong to ps class 
    #                                    np.abs(Z_tilde[:,transfer_dimension_z_tilde] - np.mean(Z_tilde[Y[:,transfer_dimension_y] == p,transfer_dimension_z_tilde])) < 0.3))[0] # Instance needs to be close to centre of p
    
    # 2. Compute M,P via yamnet
    layer_index_to_shape = [ [instance_count, 48, 32, 32],  [instance_count, 48, 32, 64],  [instance_count, 24, 16, 128],  [instance_count, 24, 16, 128],  [instance_count, 12, 8, 256],  [instance_count, 12, 8, 256], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 3, 2, 1024], [instance_count, 3, 2, 1024]]
    M = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime[M_indices], layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime[P_indices], layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # 3. Perform latent transfer from M to Q and to P
    Z_tilde_M_to_Q = np.copy(Z_tilde[M_indices])
    Z_tilde_M_to_Q[:, transfer_dimension_z_tilde] = Z_tilde[Q_indices, transfer_dimension_z_tilde]
    Z_tilde_M_to_P = np.copy(Z_tilde[M_indices])
    Z_tilde_M_to_P[:, transfer_dimension_z_tilde] = Z_tilde[P_indices, transfer_dimension_z_tilde]

    # 4. Replace top few dimensions with inverse
    Z_prime_M_to_Q = np.copy(Z_prime_pca[M_indices])
    Z_prime_M_to_Q[:,:dimension_count] = flow_network.invert(Z_tilde_M_to_Q)
    Z_prime_M_to_P = np.copy(Z_prime_pca[M_indices])
    Z_prime_M_to_P[:,:dimension_count] = flow_network.invert(Z_tilde_M_to_P)

    # 5. Invert full pca, invert scaler
    Z_prime_M_to_Q = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_M_to_Q)))
    Z_prime_M_to_P = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_M_to_P)))
    
    # 6. Continue processing through yamnet
    M_to_Q = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_M_to_Q, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    M_to_P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_M_to_P, layer_index_to_shape[layer_index]), layer_index=layer_index+1, only_logits=True).numpy()
    
    # Outputs
    return P, M, M_to_P, M_to_Q
'''

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


if __name__ == "__main__":
    
    ### Parse input arguments
    parser = argparse.ArgumentParser(
        prog="create_scalers_and_PCA_model_for_latent_yamnet",
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
    parser.add_argument("--factor_index_to_y_dimension_index", help="A dictionary (in form of a json string) that has two entires, namely one for the first factor of interest and one for the second factor of interest. For a given entry, the key is the index of the factor and the value is the index in the corresponding dimension in the Y variable. Dimensions exclude the residual factor, thus dimension 0 is the first actual factor's dimension.", type=str)
    parser.add_argument("--factor_index_to_included_class_indices_to_names", help="A dictionary (in form of a json string) that maps the index of a factor to another dictionary. That other dictionary maps the indices of its classes to their corresponding display names. The indexing of factors has to be in line with that in the Y files of the test data, i.e. the residual factor is excluded.", type=str)
    parser.add_argument("--factor_index_to_name", help="A dictionary (in form of a json string) that maps the index of a factor to the name of that factor.", type=str)
    parser.add_argument("--latent_representations_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the original Yamnet latent representations, i.e. before PCA projection, are stored. This folder should directly include the data, not indirectly in e.g. a layer-specific subfolder.", type=str)
    parser.add_argument("--layer_index", help="The index (in form of a json string) pointing to the Yament layer for which the latent transfer shall be made.", type=str)
    parser.add_argument("--pca_projected_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the projections are stored. This folder should directly include the data, not indirectly in e.g. a layer-specific subfolder.", type=str)
    parser.add_argument("--PCA_and_standard_scaler_folder", help="A list of strings that, when concatenated using the os-specific separator, result in a path to a folder in which the models are stored.", type=str)
    parser.add_argument("--flow_model_folder", help="A list of strings (in form of a json string) that, when concatenated using the os-specific separator, result in a path to a folder in which the model weights are saved.", type=str)
    parser.add_argument("--figure_folder", help="A list of strings (in form of one json string) that, when concatenated using the os-specific separator, result in a path to a folder where the plot should be saved.")
    parser.add_argument("--file_name_prefix_to_factor_wise_label", help="A dictionary (as json string) that maps from a prefix of the file name of a latent representation to its factorwise numeric labels. For example, if a latent representation of a single sound is called CD_X_1.npy, then 'CD' would be be mapped to [1,3], if C is the abbreviation for the first factor's class whose index is 1 and D the second factor's class whose index is 3. Note that the factor-wise numeric labels do not include the residual factor but only the actual factors that would be of interest to a flow model.", type=str)

    parser.add_argument("--configuration_file_path", help=f'A path to a json configuration file.{configuration_loader.CONFIGURATION_FILE_SPECIFICATION}', type=str)
    parser.add_argument("--configuration_step", help="An int pointing to the step in the configuration_file that should be read.", type=int)

    # Parse args
    args = parser.parse_args()
    
    # User provided no configuration file
    if args.configuration_file_path == None:
        # Assert all other arguments (except configuration step) are provided
        assert args.stage_count != None and args.epoch_count != None and args.dimensions_per_factor != None and args.random_seed != None and args.validation_proportion != None and args.factor_index_to_z_tilde_dimension != None and args.factor_index_to_y_dimension != None and args.factor_index_to_included_class_indices_to_names != None and args.factor_index_to_name != None and args.latent_representations_folder != None and args.layer_index != None and args.pca_projected_folder != None and args.PCA_and_standard_scaler_folder != None and args.flow_model_folder != None and args.figure_folder != None and args.file_name_prefix_to_factor_wise_label != None, "If no configuration file is provided, then all other arguments must be provided."
    
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
        
    # User provided configuration file.
    else:
        # Make sure step is provided but no other arguments are.
        assert args.stage_count == None and args.epoch_count == None and args.dimensions_per_factor == None and args.random_seed == None and args.validation_proportion == None and args.factor_index_to_z_tilde_dimension == None and args.factor_index_to_y_dimension == None and args.factor_index_to_included_class_indices_to_names == None and args.factor_index_to_name == None and args.latent_representations_folder == None and args.layer_index == None and args.pca_projected_folder == None and args.PCA_and_standard_scaler_folder == None and args.flow_model_folder == None and args.figure_folder == None and args.file_name_prefix_to_factor_wise_label == None, "If a configuration file is provided, then no other arguments shall be provided."
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
    print("\n\tStarting script now:\n")
    """
    material_to_index = {"W":0,"G":1,"P":2,"S":3,"C":4}#,"M":1,
    action_to_index = {"T":0,"R":1,"W":2}#,"D":4}

    stage_count = 4
    epoch_count = 50
    dimensions_per_factor = [62,1,1] # In order of residual factor, material factor, action factor
    random_seed = 42
    validation_proportion = 0.3
    factor_index_to_z_tilde_dimension = {0:62, 1:63}
    factor_index_to_y_dimension = {0:0, 1:1}
    factor_index_to_included_class_indices_to_names = {0: {value: key for key, value in material_to_index.items()}, 1: {value: key for key, value in action_to_index.items()}}
    factor_index_to_name = {0:'Material',1:'Action'}
    latent_representations_folder_path = "E:\\LatentAudio\complete configuration\data\latent\original\Layer 9"
    layer_index = 9
    pca_projected_folder_path = "E:\\LatentAudio\complete configuration\data\latent\pca projected\Layer 9"
    PCA_and_standard_scaler_folder_path = "E:\\LatentAudio\complete configuration\models\pca and standard scalers\Layer 9"
    flow_model_folder_path = "E:\\LatentAudio\complete configuration\models\\flow"
    figure_folder_path =  "E:\\LatentAudio\complete configuration\\figures"
    file_name_prefix_to_factor_wise_label =  {f"{m}{a}" : [material_to_index[m], action_to_index[a]] for m in material_to_index.keys() for a in action_to_index.keys()} # File names are of the form MA, where M is the material abbreviation and A the action abbreviation
    """
    
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
                              factor_index_to_name=factor_index_to_name, figure_file_path=scatter_plot_file_path)
    """                        
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
    x_file_names = x_file_names[:100]
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

    plot_permutation_test(Z_prime=Z_prime_sample, Y=Y_sample, dimensions_per_factor=dimensions_per_factor, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index, figure_file_path=bar_plot_file_path, factor_index_to_name=factor_index_to_name, factor_index_to_z_tilde_dimension=factor_index_to_z_tilde_dimension, factor_index_to_y_dimension=factor_index_to_y_dimension)
    #plot_latent_transfer(Z_prime=Z_prime_sample, Y=Y_sample, dimensions_per_factor=dimensions_per_factor, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index, plot_save_path=bar_plot_file_path, factor_index_to_name=factor_index_to_name, factor_index_to_z_tilde_dimension=factor_index_to_z_tilde_dimension, y_factor_index_to_dimension_index=y_factor_index_to_dimension_index)

    # Log
    print("\n\n\Completed script disentangle")
    """
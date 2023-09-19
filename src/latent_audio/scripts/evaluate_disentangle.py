"""This script visualizes the model created by the disentangle script. It passes sample data through the model
and shows in scatter plots how well the matierial and action factors are disentangled."""

from latent_audio.scripts import disentangle as lsd
import latent_audio.utilities as utl
import tensorflow as tf
from latent_audio.yamnet import layer_wise as ylw
from typing import List, Any, OrderedDict, Callable, Generator, Tuple
import os, numpy as np
import tensorflow as tf, matplotlib.pyplot as plt
from gyoza.modelling import flow_layers as mfl
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import random
from scipy import stats
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

def plot_permutation_test_2(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int, plot_save_path: str) -> None:

    inverse_sigmoid = lambda x : np.log(1 / (1 + np.exp(-x)))
    entropy = lambda P, Q: P*np.log(Q)
    dissimilarity_function = lambda P, Q: np.sqrt(np.sum((inverse_sigmoid(P) - inverse_sigmoid(Q))**2, axis=1))#- np.sum(entropy(tf.nn.softmax(P, axis=1),tf.nn.softmax(Q, axis=1)),axis=1)
    plt.figure(figsize=(10,6)); plt.suptitle('\tLatent Transfer')
    b = 1
    x_min = np.finfo(np.float32).max
    x_max = np.finfo(np.float32).min
    
    for factor_name, transfer_dimensions in {'Material':[-2,-1],'Action':[-1,-2]}.items():
        current_dimension = transfer_dimensions[0]
        other_dimension = transfer_dimensions[1]

        # initialize
        H_PM = np.array([])
        H_P_M_to_P = np.array([])
        H_P_M_to_Q = np.array([])

        for s in range(10):
            # Shuffle
            indices = list(range(Z_prime.shape[0])); np.random.shuffle(indices)
            Z_prime = Z_prime[indices]; Y = Y[indices]
            for c in set(Y[:,other_dimension]):

                # Compute probability distributions
                P, M, M_to_P, M_to_Q = latent_transfer_2(Z_prime=Z_prime[Y[:,other_dimension]==c], Y=Y[Y[:,other_dimension]==c], dimensions_per_factor=dimensions_per_factor, transfer_dimension=current_dimension, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=layer_index)
                
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
        '''
        H_PM and H_P_M_to_P have test results:
        TtestResult(statistic=4.562010678256846, pvalue=8.873686878414838e-06, df=198)
        H_PM and H_P_M_to_Q have test results:
        TtestResult(statistic=-2.8269841308183516, pvalue=0.005181231511157646, df=198)
        H_PM and H_P_M_to_P have test results:
        TtestResult(statistic=1.6386363857835877, pvalue=0.1016892024009523, df=786)
        H_PM and H_P_M_to_Q have test results:
        TtestResult(statistic=-2.0608555263997435, pvalue=0.04015613967209464, df=308)'''
        # Plot
        plt.subplot(2,1,b); plt.title(factor_name)
        means = [np.mean(H_PM),np.mean(H_P_M_to_P),np.mean(H_P_M_to_Q)]
        errors = [np.std(H_PM)/ np.sqrt(len(H_PM)), np.std(H_P_M_to_P)/ np.sqrt(len(H_P_M_to_P)), np.std(H_P_M_to_Q)/ np.sqrt(len(H_P_M_to_Q))]
        plt.bar([1,2,3], means, color=[0.1,0.1,0.1,0.1], edgecolor='black')
        plt.xticks([1,2,3],['P,M','P,M->P','P,M->Q'])
        plt.grid(alpha=0.25)
        plt.ylabel('Mean Squred Error of \nYamnet Output Logit')
        plt.ylim(means[1]-1.8*errors[1], means[2]+1.5*errors[2])

        # Significance tests
        print("H_PM and H_P_M_to_P have test results:")
        print(stats.ttest_rel(H_PM[indices[:sample_size_mp]], H_P_M_to_P[indices[:sample_size_mp]]))
        print("H_PM and H_P_M_to_Q have test results:")
        print(stats.ttest_rel(H_PM[indices[:sample_size_mq]], H_P_M_to_Q[indices[:sample_size_mq]]))

        if stats.ttest_rel(H_PM[indices[:sample_size_mp]], H_P_M_to_P[indices[:sample_size_mp]]).pvalue <= 0.025: # Bonferroni corrected
            plt.annotate('*', (1.99, means[1]+1.1*errors[1]))
        else: plt.annotate('o', (1.99, means[1]+1.1*errors[1]))
        if stats.ttest_rel(H_PM[indices[:sample_size_mq]], H_P_M_to_Q[indices[:sample_size_mq]]).pvalue <= 0.025: # Bonferroni corrected
            plt.annotate('*', (2.99, means[2]+1.1*errors[2]))
        else: plt.annotate('o', (2.99, means[2]+1.1*errors[2]))
        b+=1
        
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()

def latent_transfer_2(Z_prime: np.ndarray, Y: np.ndarray, dimensions_per_factor: List[int], transfer_dimension: int, pre_scaler: Callable, pca: Callable, post_scaler: Callable, flow_network: Callable, layer_wise_yamnet: Callable, layer_index: int) -> None:

    # 1. Disentangle

    # 1.1 Apply standard scalers and pca
    Z_prime_pca = post_scaler.transform(pca.transform(pre_scaler.transform(Z_prime)))

    # 1.2 Pass the top few dimensions through flow net
    dimension_count = np.sum(dimensions_per_factor)
    Z_tilde = flow_network(Z_prime_pca[:,:dimension_count]).numpy()

    # Split data into 3 tertiles
    cs=list(set(Y[:,transfer_dimension])); cs=sorted(cs); random.shuffle(cs); p = cs[0]; q = cs[1] # Choose a random class for p and q
    mean = np.mean(Z_tilde[:, transfer_dimension])
    t12, t23 = mean-0.2, mean+0.2
    Q_indices = np.where(np.logical_and(Y[:,transfer_dimension] == q, # Instance needs to belong to Qs class 
                                        np.abs(Z_tilde[:,transfer_dimension] - np.mean(Z_tilde[Y[:,transfer_dimension] == q,transfer_dimension])) < 0.3))[0] # Instance needs to be close to centre of q
    M_indices = np.where(np.logical_and(t12 < Z_tilde[:,transfer_dimension], Z_tilde[:,transfer_dimension] <= t23, Y[:,transfer_dimension]==p))[0] # Points that are between p and q but belong to p
    P_indices = np.where(np.logical_and(Y[:,transfer_dimension] == p, # Instance needs to belong to ps class 
                                        np.abs(Z_tilde[:,transfer_dimension] - np.mean(Z_tilde[Y[:,transfer_dimension] == p,transfer_dimension])) < 0.3))[0] # Instance needs to be close to centre of p
    
    instance_count = np.min([len(Q_indices), len(M_indices), len(P_indices)]) # Crop all to equal length
    Q_indices = Q_indices[:instance_count]
    M_indices = M_indices[:instance_count]
    P_indices = P_indices[:instance_count]

    # 2. Compute M,P via yamnet
    layer_index_to_shape = [ [instance_count, 48, 32, 32],  [instance_count, 48, 32, 64],  [instance_count, 24, 16, 128],  [instance_count, 24, 16, 128],  [instance_count, 12, 8, 256],  [instance_count, 12, 8, 256], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 6, 4, 512], [instance_count, 3, 2, 1024], [instance_count, 3, 2, 1024]]
    M = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime[M_indices], layer_index_to_shape[layer_index]), layer_index=layer_index+1).numpy()
    P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime[P_indices], layer_index_to_shape[layer_index]), layer_index=layer_index+1).numpy()
    
    # 3. Perform latent transfer from M to Q and to P
    Z_tilde_M_to_Q = np.copy(Z_tilde[M_indices])
    Z_tilde_M_to_Q[:, transfer_dimension] = Z_tilde[Q_indices, transfer_dimension]
    Z_tilde_M_to_P = np.copy(Z_tilde[M_indices])
    Z_tilde_M_to_P[:, transfer_dimension] = Z_tilde[P_indices, transfer_dimension]

    # 4. Replace top few dimensions with inverse
    Z_prime_M_to_Q = np.copy(Z_prime_pca[M_indices])
    Z_prime_M_to_Q[:,:dimension_count] = flow_network.invert(Z_tilde_M_to_Q)
    Z_prime_M_to_P = np.copy(Z_prime_pca[M_indices])
    Z_prime_M_to_P[:,:dimension_count] = flow_network.invert(Z_tilde_M_to_P)

    # 5. Invert full pca, invert scaler
    Z_prime_M_to_Q = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_M_to_Q)))
    Z_prime_M_to_P = pre_scaler.inverse_transform(pca.inverse_transform(post_scaler.inverse_transform(Z_prime_M_to_P)))
    
    # 6. Continue processing through yamnet
    M_to_Q = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_M_to_Q, layer_index_to_shape[layer_index]), layer_index=layer_index+1).numpy()
    M_to_P = layer_wise_yamnet.call_from_layer(np.reshape(Z_prime_M_to_P, layer_index_to_shape[layer_index]), layer_index=layer_index+1).numpy()
    
    # Outputs
    return P, M, M_to_P, M_to_Q

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

def stage_wise_maclaurin(network: mfl.SequentialFlowNetwork, Z: np.ndarray, y: np.ndarray, layer_steps: List[int], step_titles: List[str], plot_save_path: str):
    
    # Prepare plot
    layer_steps = [0] + layer_steps
    K = len(step_titles)
    fig, axs = plt.subplots(3, 1+K, figsize=(0.8+K,5), gridspec_kw={'height_ratios': [2,1,1], 'width_ratios':[0.3]+[1]*K}, dpi=4*96)
    plt.suptitle(rf'Sequential Maclaurin Decomposition Of Flow Model')
    max_bar_height = 0

    # Plot annotations on left
    gray = [0.5,0.5,0.5]
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
        E_norm = np.mean(np.sqrt(np.sum(E[:,-2:]**2, axis=1)))
        P_norm = np.mean(np.sqrt(np.sum(P[:,-2:]**2, axis=1)))
        plt.bar([''],[E_norm+P_norm], color = gray, edgecolor='black', hatch='oo')
        plt.bar([''],[P_norm], color = gray, edgecolor='black', hatch='///')
        max_bar_height = max(max_bar_height, E_norm+P_norm); plt.axis('off')

        # 2.1 Tails
        # 2.1.1 Error
        cs = list(set(y)); cs = sorted(cs)
        plt.subplot(3,1+K,1+K+k+1); plt.gca().set_prop_cycle(None)
        plt.scatter(prediction[:,-2], prediction[:,-1], color=gray, marker='.',s=0.1)
        plt.quiver(prediction[:,-2], prediction[:,-1], E[:,-2], E[:,-1], angles='xy', scale_units='xy', scale=1., color=gray, zorder=1)
        for c in cs: plt.scatter(Z_tilde.numpy()[y==c,0], Z_tilde.numpy()[y==c,1], marker='.',s=1)
        plt.axis('equal'); plt.xticks([]); plt.yticks([]); plt.xlim(1.3*np.array(plt.xlim())); plt.ylim(1.3*np.array(plt.ylim()))

        # 2.1.2 Prediction
        plt.subplot(3,1+K,2*(1+K)+k+1); plt.gca().set_prop_cycle(None)
        plt.scatter(Z[:,-2], Z[:,-1], color=gray, marker='.',s=0.1)
        plt.quiver(Z[:,-2], Z[:,-1], P[:,-2], P[:,-1], angles='xy', scale_units='xy', scale=1., color=gray, zorder=1)
        for c in cs: plt.scatter(prediction.numpy()[y==c,-2], prediction.numpy()[y==c,-1], marker='.',s=1)
        plt.axis('equal'); plt.xticks([]); plt.yticks([]); plt.xlim(1.3*np.array(plt.xlim())); plt.ylim(1.3*np.array(plt.ylim()))

        # Prepare next iteration
        Z=Z_tilde

    # Adjust bar heights
    for k in range(1, len(layer_steps)):
      plt.subplot(3,1+K,k+1); plt.ylim(0, max_bar_height)
    plt.subplot(3,1+K,1); plt.ylabel('Mean Change'); plt.ylim(0, max_bar_height); ax = plt.gca();ax.spines['top'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); plt.xticks([])
    ax.yaxis.tick_right(); ax.tick_params(axis="y",direction="in", pad=-12)

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=4*96)
    plt.show()


# Configuration
inspection_layer_index = 9
batch_size = 512
latent_transfer_sample_size = 2**13 # Needs to be large enough for samples of all conditions to appear
np.random.seed(865)
tf.keras.utils.set_random_seed(895)
random.seed(248)
stage_count = 5
epoch_count = 10
dimensions_per_factor = [62,1,1]
materials_to_keep = [0,1,2,4]; actions_to_keep = [0,3]
materials_to_drop = list(range(6))
for m in reversed(materials_to_keep): materials_to_drop.remove(m)
actions_to_drop = list(range(4))
for a in reversed(actions_to_keep): actions_to_drop.remove(a)
m_string = ",".join(str(m) for m in materials_to_keep)
a_string = ",".join(str(a) for a in actions_to_keep)
projected_data_path = os.path.join('data','latent yamnet',f'{np.sum(dimensions_per_factor)} dimensions',f'Layer {inspection_layer_index}')
original_data_path = os.path.join('data','latent yamnet','original',f'Layer {inspection_layer_index}')
flow_model_save_path = os.path.join('models', 'flow models', f'Layer {inspection_layer_index}')
pca_model_path = os.path.join("models","Scaler and PCA",f"Layer {inspection_layer_index}")
plot_save_path = os.path.join('plots','evaluate flow models', f'Layer {inspection_layer_index}')
if not os.path.exists(plot_save_path): os.makedirs(plot_save_path)
flow_model_save_path = os.path.join(flow_model_save_path, f'Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count}.h5')
material_labels=['W','M','G','S','C','P']; action_labels = ['T','R','D','W']
layer_wise_yamnet = ylw.LayerWiseYamnet()
layer_wise_yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

# Load data iterators
train_iterator, test_iterator, batch_count, Z_train, Z_test, Y_train, Y_test = lsd.load_iterators(data_path=projected_data_path, materials_to_drop=materials_to_drop, actions_to_drop=actions_to_drop, batch_size=batch_size)
Z_ab_sample, Y_ab_sample = next(train_iterator) # Sample

print("The data is fed to the model in batches of shape:\n","Z: (instance count, pair, dimensionality): \t", Z_ab_sample.shape,'\nY_sample: (instance count, factor count): \t', Y_ab_sample.shape)

# Create network
flow_network = lsd.create_network(Z_sample=Z_ab_sample[:,0,:], stage_count=stage_count, dimensions_per_factor=dimensions_per_factor)
flow_network.load_weights(flow_model_save_path)

# Maclaurin series for materials and actions
indices = random.sample(range(len(Z_test)), 700)
stage_wise_maclaurin(network=flow_network, Z= Z_test[indices], y= Y_test[indices,-2], layer_steps=[7*(s+1) for s in range(stage_count)], step_titles= [f'Stage {s+1}' for s in range(stage_count)], plot_save_path = os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Calibrated Network Maclaurin materials.png"))
stage_wise_maclaurin(network=flow_network, Z= Z_test[indices], y= Y_test[indices,-1], layer_steps=[7*(s+1) for s in range(stage_count)], step_titles= [f'Stage {s+1}' for s in range(stage_count)], plot_save_path = os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Calibrated Network Maclaurin actions.png"))

# Scatterplots
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

plot_permutation_test_2(Z_prime=Z_prime_sample, Y=Y_sample, dimensions_per_factor=dimensions_per_factor, pre_scaler=pre_scaler, pca=pca, post_scaler=post_scaler, flow_network=flow_network, layer_wise_yamnet=layer_wise_yamnet, layer_index=inspection_layer_index, plot_save_path=os.path.join(plot_save_path, f"Materials {m_string} actions {a_string} stages {stage_count} epochs {epoch_count} Calibrated Network Latent Transfer.png"))

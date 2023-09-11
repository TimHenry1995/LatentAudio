"""This script requires the 
for the given layer it passes each data point through pca
- it then collects for all these Xs the top k dimensions 
- it saves them all in one file called X.npy along with the labels in Y.npy
It thus allows to set up a training data set"""



    with open('/Users/timdick/Documents/LatentAudio/models/Layer 8/Full Standard Scaler.pkl','rb') as fh:
        scaler = pkl.load(fh)
    
    with open('/Users/timdick/Documents/LatentAudio/models/Layer 8/Full PCA.pkl','rb') as fh:
        pca = pkl.load(fh)
    
    # Transform each X
    x_file_names = utl.find_matching_strings(strings=os.listdir(X_layer_folder), token='_X_')
    Xs = [None] * len(x_file_names); Ys = [None] * len(x_file_names)
    for i, x_file_name in enumerate(x_file_names):
        X = np.load(os.path.join(X_layer_folder, x_file_name))[np.newaxis,:]
        Xs[i] = pca.transform(scaler.transform(X))[:,:pca_dimensionality] # We transform the entire X, but for training we only keep the top few dimensions
        Ys[i] = np.load(os.path.join(X_layer_folder, x_file_name.replace('_X_','_Y_')))[np.newaxis,:]
        if i % (int)(len(x_file_names)/10) == 0:
            print(f"Layer {l}: {100*i/len(x_file_names)} % Finished")
    # Save
    np.save(os.path.join(layer_folder_ouput, "X"), np.concatenate(Xs, axis=0))
    np.save(os.path.join(layer_folder_ouput, "Y"), np.concatenate(Ys, axis=0))
    
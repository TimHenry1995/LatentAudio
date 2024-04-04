
import tensorflow as tf
from latent_audio.yamnet import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from scipy.signal import decimate

def run(
    layer_index: int,
    raw_folder_path: str = os.path.join('data','raw audio'),
    augmented_folder_path: str = os.path.join('data','augmented audio'),
    latent_data_path: str = os.path.join('data','latent yamnet','original')
    ):
    """This function can be used to convert the raw audio data from their waveform representation into their latent yamnet representations.

    Requirements:
    - The script augment_audio needs to be executed apriori
    - The raw and augmented audio data needs to be sampled at 48 Khz with int16 bit rate

    Steps:
    - iterates layer indices
    - loads raw and augmented waveforms (will be concatenated and normalized to float32 in range [-1,1])
    - passes through yamnet up until current layer
    - flattens each time frame into a vector
    - saves each time frame

    Side effects:
    - The script initially deletes the output folder (if exists) and the saves only the new data in it. Hence, any old data in that fodler will be lost.
    - Keras session is cleared at beginning of call.
        
    :param layer_index: The index of the Yamnet layer for which the latent Yamnet data shall be computed.
    :type layer_index: int
    :param raw_folder_path: The path to the folder containing the raw, i.e. waveform data before augmentation.
    :type raw_folder_path: str
    :param augmented_folder_path: The path to the folder containing the augmented, i.e. waveform data that augments the raw data.
    :type augmented_folder_path: str
    :param latent_data_path: The path to the folder in which the latent representations shall be stored.
    :type latent_data_path: str
    """

    print("Running script to convert audio to latent yamnet")
    
    # Initialization
    raw_file_names = os.listdir(raw_folder_path) # Assumed to have material as first letter and action as second letter
    for file_name in reversed(raw_file_names):
        if '.wav' not in file_name: raw_file_names.remove(file_name)

    if not os.path.exists(latent_data_path): os.makedirs(latent_data_path)
    
    tf.keras.backend.clear_session() # Need to clear session because otherwise yamnet cannot be loaded
    yamnet = ylw.LayerWiseYamnet()
    yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

    material_to_label = {'W':0,'M':1,'G':2,'S':3,'C':4,'P':5}
    action_to_label = {'T':0,'R':1,'D':2,'W':3}


    layer_path = os.path.join(latent_data_path,f'Layer {layer_index}')

    # This line deletes any content in the output folder 
    if os.path.exists(layer_path): shutil.rmtree(layer_path)
    os.makedirs(layer_path)

    # Preprocess all files
    for r, raw_file_name in enumerate(raw_file_names):
        # Load .wav file
        waveform_r, sampling_rate_r = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
        assert sampling_rate_r == 48000, f"The sampling rate of the raw audio was assumed to be 48000 in order to apply the decimation algorithm and achieve yamnets 16000. The provided audio has sampling rate {sampling_rate_r}. You need to use a different downsampling method, e.g. from sklearn to meet yamnet's requirement."
        waveform_a, sampling_rate_a = sf.read(os.path.join(augmented_folder_path, raw_file_name), dtype=np.int16)
        assert sampling_rate_a == 48000, f"The sampling rate of the augmented audio was assumed to be 48000 in order to apply the decimation algorithm and achieve yamnets 16000. The provided audio has sampling rate {sampling_rate_a}. You need to use a different downsampling method, e.g. from sklearn to meet yamnet's requirement."
        
        # Normalize
        waveform_r =  waveform_r.astype('float32')
        waveform_r = waveform_r / np.max(waveform_r)
        waveform_a = waveform_a.astype('float32')
        waveform_a = waveform_a / np.max(waveform_a)

        # Concatenate raw and augmented
        waveform = np.concatenate([waveform_r, waveform_a])
        
        # Down-sample to yamnet's 16000
        waveform = decimate(waveform, 3)
        sampling_rate = 16000
        
        # Pass through yamnet up until target layer
        latent = yamnet.call_until_layer(waveform=waveform, layer_index=layer_index).numpy()

        # Create y
        material = raw_file_name[0]; action = raw_file_name[1]
        y = np.array([material_to_label[material], action_to_label[action]])

        # Save
        for s, slice in enumerate(latent):
            np.save(os.path.join(layer_path, f"{raw_file_name[:-4]}_X_{s}.npy"), np.reshape(slice,[-1])) 
            np.save(os.path.join(layer_path, f"{raw_file_name[:-4]}_Y_{s}.npy"), y) 
        print(f"\r\t{np.round(100*(r+1)/len(raw_file_names))}% Completed", end='')

        # Delete singleton
        del latent

    print("\n\tRun Completed")

if __name__ == "__main__":
    for layer_index in range(14):
        run(layer_index=layer_index)       

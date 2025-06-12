
import tensorflow as tf
from LatentAudio.yamnet import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from scipy.signal import decimate
from typing import Dict, List

def run(
    layer_index: int,
    waveform_file_name_to_Y_vector: Dict[str, List[int]],
    raw_folder_path: str = os.path.join('LatentAudio','data','raw audio'),
    augmented_folder_path: str = os.path.join('LatentAudio','data','augmented audio'),
    latent_data_path: str = os.path.join('LatentAudio','data','latent yamnet','original')
    ) -> None:
    """This function loads each waveform file, converts it into a sequence of ca. 1 second long spectrogram slices and passes them through Yamnet up until `layer_index`.
    Each slice will then be flattened into a long vector and saved to disk with the name \<waveform file name>\_X\_\<slice index>.npy next to a small file called \<waveform file name>\_Y\_\<slice index>.npy storing the vector with the factor-wise class indices of that slice.
    For example, if the waveform file was called 'Metal Tapping.wav', the X files would be called 'Metal Tapping_X_1.npy', 'Metal Tapping_X_2.npy', etc. and the Y files 'Metal Tapping_Y_1.npy', 'Metal Tapping_Y_2.npy' etc.
    Note that this function initially deletes the output folder (if exists) and then saves only the new data in it. Hence, any old data in that folder will get lost.
    Apart from that, this function cleares the current Keras session and hence any variables stored in it will get lost.

    :param layer_index: The index of the Yamnet layer for which the latent Yamnet data shall be computed.
    :type layer_index: int
    :param waveform_file_name_to_Y_vector: A dictionary mapping the wavefomr file names (without their file extension) to Y vectors that will then be stored for each sound slice.
    :type waveform_file_name_to_Y_vector: Dict[str, int]
    :param raw_folder_path: The path to the folder containing the raw, i.e. waveform data before augmentation. This audio data needs to be sampled at 48 Khz with int16 bit rate.
    :type raw_folder_path: str
    :param augmented_folder_path: The path to the folder containing the augmented, i.e. waveform data that augments the raw data. Same sampling rate and bitrate is assumed. You can set this to None if you do not intend to use augmented data.
    :type augmented_folder_path: str
    :param latent_data_path: The path to the folder in which the latent representations shall be stored. The result will be a file called X.npy which stores a matrix of shape [instance count, dimensionality of latent yamnet at `layer_index`] and a file Y.npy of shape [instance count, 2] storing the material and action class, respectively, for every sound.
    :type latent_data_path: str
    """

    print(f"Running script to convert audio to latent yamnet layer {layer_index}.")
    
    # Initialization
    raw_file_names = os.listdir(raw_folder_path) # Assumed to have material as first letter and action as second letter
    for file_name in reversed(raw_file_names):
        if '.wav' not in file_name: raw_file_names.remove(file_name)

    if not os.path.exists(latent_data_path): os.makedirs(latent_data_path)
    
    tf.keras.backend.clear_session() # Need to clear session because otherwise yamnet cannot be loaded
    yamnet = ylw.LayerWiseYamnet()
    yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

    layer_path = os.path.join(latent_data_path,f'Layer {layer_index}')

    # This line deletes any content in the output folder 
    if os.path.exists(layer_path): shutil.rmtree(layer_path)
    os.makedirs(layer_path)

    # Preprocess all files
    for r, raw_file_name in enumerate(raw_file_names):
        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
        assert sampling_rate == 48000, f"The sampling rate of the raw audio was assumed to be 48000 in order to apply the decimation algorithm and achieve yamnets 16000. The provided audio has sampling rate {sampling_rate_r}. You need to use a different downsampling method, e.g. from sklearn to meet yamnet's requirement."
        if augmented_folder_path is not None:
            waveform_a, sampling_rate_a = sf.read(os.path.join(augmented_folder_path, raw_file_name), dtype=np.int16)
            assert sampling_rate_a == 48000, f"The sampling rate of the augmented audio was assumed to be 48000 in order to apply the decimation algorithm and achieve yamnets 16000. The provided audio has sampling rate {sampling_rate_a}. You need to use a different downsampling method, e.g. from sklearn to meet yamnet's requirement."
        
        # Normalize
        waveform =  waveform.astype('float32')
        waveform = waveform / np.max(waveform)
        if augmented_folder_path is not None:
            waveform_a = waveform_a.astype('float32')
            waveform_a = waveform_a / np.max(waveform_a)

            # Concatenate raw and augmented
            waveform = np.concatenate([waveform, waveform_a])
        
        # Down-sample to yamnet's 16000
        waveform = decimate(waveform, 3)
        sampling_rate = 16000
        
        # Pass through yamnet up until target layer
        latent = yamnet.call_until_layer(waveform=waveform, layer_index=layer_index).numpy()

        # Create y
        name = '.'.join(raw_file_name.split('.')[:-1]) # removes the file extension
        y = np.array(waveform_file_name_to_Y_vector[name])

        # Save
        for s, slice in enumerate(latent):
            np.save(os.path.join(layer_path, f"{name}_X_{s}.npy"), np.reshape(slice,[-1])) 
            np.save(os.path.join(layer_path, f"{name}_Y_{s}.npy"), y) 
        print(f"\r\t{np.round(100*(r+1)/len(raw_file_names))}% Completed", end='')

        # Delete singleton
        del latent

    print("\n\tRun Completed")

"""This module can be used to pre-process the raw audio data into instances that can be used for calibration and evaluation of the
disentangling flow network. The raw data is a collection of .wav files, whose first two letters of the filename indicate the condition
(see code for parsing details). Each .wav file is a continous recording of sound of the same condition sampled at 48 kHz and int16 
bit-rate. During pre-processing it will be normalized to the range [-1,1], decimated to 16 kHz and converted to 0.96 second long half-
overlapping spectrogram slices and fed through yamnet up until a specified layer (see code for layer choice) and flattened. After 
pre-processing, each .wav file has a set of corresponding .npy files with float32 precision storing the latent representations of 
these slices (<condition>_X_<index>.npy) and the material and action labels representing numerically with indices 
(<condition>_Y_<index>.npy) in the specified pre-processed data folder."""

from latent_audio.yamnet import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from scipy.signal import decimate

# Configuration
raw_folder_path = os.path.join('data','raw audio')
raw_file_names = os.listdir(raw_folder_path) # Assumed to have material as first letter and action as second letter
for file_name in reversed(raw_file_names):
    if '.wav' not in file_name: raw_file_names.remove(file_name)
pre_processed_path = os.path.join('data','pre-processed','All PCA dimensions')
yamnet = ylw.LayerWiseYamnet()
yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

material_to_label = {'W':0,'M':1,'G':2,'S':3,'C':4,'P':5}
action_to_label = {'T':0,'R':1,'D':2,'W':3}

for inspection_layer_index in range(14):
    layer_path = os.path.join(pre_processed_path,f'Layer {inspection_layer_index}')

    # This line deletes current pre-processed files
    if os.path.exists(layer_path): shutil.rmtree(layer_path)
    os.makedirs(layer_path)

    # Preprocess all files
    for raw_file_name in raw_file_names:

        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
        max = np.max(waveform)
        
        # Adjust format
        waveform = waveform / max#np.max(waveform)#np.iinfo(np.int16).max
        waveform = waveform.astype('float32')
        
        # Pass through yamnet up until target layer
        latent = yamnet.call_until_layer(waveform=waveform, layer_index=inspection_layer_index).numpy()
        
        # Create y
        material = raw_file_name[0]; action = raw_file_name[1]
        y = np.array([material_to_label[material], action_to_label[action]])

        # Save
        for s, slice in enumerate(latent):
            np.save(os.path.join(layer_path, f"{raw_file_name[:-4]}_X_{s}.npy"), np.reshape(slice,[-1])) 
            np.save(os.path.join(layer_path, f"{raw_file_name[:-4]}_Y_{s}.npy"), y) 
            

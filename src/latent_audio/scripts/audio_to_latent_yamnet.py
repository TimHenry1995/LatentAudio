"""This script can be used to convert the raw audio data into latent yamnet representations.

Requirements:
- The audio data needs to be sampled at 48 Khz with int16 bit rate

Steps:
- iterates layer indices
- loads waveforms (will be normalized to float32 in range [-1,1])
- passes through yamnet up until current layer
- flattens each time frame into a vector
- saves each time frame

Side effects:
- The script initially deletes the output folder (if exists) and the saves only the new data in it. Hence, any old data in that fodler will be lost.
"""

from latent_audio.yamnet import layer_wise as ylw
import os, soundfile as sf, numpy as np, shutil
from scipy.signal import decimate

# Configuration
raw_folder_path = os.path.join('data','raw audio')
raw_file_names = os.listdir(raw_folder_path) # Assumed to have material as first letter and action as second letter
for file_name in reversed(raw_file_names):
    if '.wav' not in file_name: raw_file_names.remove(file_name)
latent_data_path = os.path.join('data','latent yamnet')
yamnet = ylw.LayerWiseYamnet()
yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

material_to_label = {'W':0,'M':1,'G':2,'S':3,'C':4,'P':5}
action_to_label = {'T':0,'R':1,'D':2,'W':3}

for inspection_layer_index in range(14):
    print(f"Layer {inspection_layer_index}")

    layer_path = os.path.join(latent_data_path,f'Layer {inspection_layer_index}')

    # This line deletes any content in the output folder 
    if os.path.exists(layer_path): shutil.rmtree(layer_path)
    os.makedirs(layer_path)

    # Preprocess all files
    for raw_file_name in raw_file_names:
        # Load .wav file
        waveform, sampling_rate = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
        max = np.max(waveform)
        assert sampling_rate == 48000, f"The sampling rate was assumed to be 48000 in order to apply the decimation algortihm and achieve yamnets 16000. The provided audio has sampling rate {sampling_rate}. You need to use a different downsampling method, e.g. from sklearn to meet yamnet's requirement."
        print(f"Loaded {raw_file_name}")

        # Down-sample to yamnet's 16000
        waveform = decimate(waveform, 3)
        sampling_rate = 16000

        # Adjust format
        waveform = waveform / max#np.max(waveform)#np.iinfo(np.int16).max
        waveform = waveform.astype('float32')
        
        # Pass through yamnet up until target layer
        latent = yamnet.call_until_layer(waveform=waveform, layer_index=inspection_layer_index).numpy()
        print("Passed waveform through yamnet")

        # Create y
        material = raw_file_name[0]; action = raw_file_name[1]
        y = np.array([material_to_label[material], action_to_label[action]])

        # Save
        for s, slice in enumerate(latent):
            np.save(os.path.join(layer_path, f"{raw_file_name[:-4]}_X_{s}.npy"), np.reshape(slice,[-1])) 
            np.save(os.path.join(layer_path, f"{raw_file_name[:-4]}_Y_{s}.npy"), y) 
        print("Saved latent slices")

print("Script completed")            

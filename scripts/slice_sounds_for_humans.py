import os, soundfile as sf, numpy as np, shutil
from LatentAudio.plugins.yamnet import params as yamnet_params, yamnet as yamnet_model

# Configuration
raw_folder_path = os.path.join('LatentAudio','data','raw audio')
raw_file_names = os.listdir(raw_folder_path) # Assumed to have material as first letter and action as second letter
for file_name in reversed(raw_file_names):
    if '.wav' not in file_name: raw_file_names.remove(file_name)

save_path = os.path.join('LatentAudio','data','sound slices for humans')

if os.path.exists(save_path): shutil.rmtree(save_path) # This line deletes current sound slices 
os.makedirs(save_path)

block_count = 4
duration_of_slice = 5*0.96 # seconds

yamnet = yamnet_model.yamnet_frames_model(yamnet_params.Params())
yamnet.load_weights(os.path.join('LatentAudio','plugins','yamnet','yamnet.h5'))
yamnet_classes = yamnet_model.class_names(os.path.join('LatentAudio','plugins','yamnet','yamnet_class_map.csv'))
np.random.seed(312)
# Slicing
for b in range(block_count):
    block_path = os.path.join(save_path,f"Block {b}")
    os.makedirs(block_path)

    # Shuffle order of saved files
    order = list(range(len(raw_file_names)))
    np.random.shuffle(order)

    for r, raw_file_name in enumerate(raw_file_names):
        
        # Load
        waveform, sampling_rate = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
        waveform = waveform / np.max(waveform)#np.iinfo(np.int16).max
        
        silent = True
        while silent:
            # Sample a slice
            frames_per_slice = (int)(duration_of_slice*sampling_rate)
            start = np.random.randint(low=0, high=len(waveform)-frames_per_slice)
            slice = waveform[start:start+frames_per_slice]
            
            scores, embeddings, spectrogram = yamnet(slice)
            prediction = np.mean(scores, axis=0)
            silent = yamnet_classes[np.argmax(prediction)] == 'Silent'

        # Save
        condition = raw_file_name[:2]
        sf.write( os.path.join(block_path, f"{order[r]} {condition}.wav"), slice, sampling_rate,) 

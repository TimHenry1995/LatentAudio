"""This module can be used to pre-process the raw audio data into instances that can be used for calibration and evaluation of the
disentangling flow network. The raw data is a collection of .wav files, whose first two letters of the filename indicate the condition.
Each .wav file is a continous recording of sound sampled at 48 kHz and int16 bit-rate. During pre-processing it will be normalized to
the range [-1,1], separated into overlapping slices and the slices that yamnet judges non-silent are kept. After preprocessing, each
.wav file has a corresponding .npy file with float32 precision storing a tensor of these slices."""

from latent_audio.plugins.yamnet import yamnet as pyy, params as pyp
import os, soundfile as sf, numpy as np
from scipy.signal import decimate

# Configuration
raw_folder_path = os.path.join('data','raw audio')
raw_file_names = os.listdir(raw_folder_path)
for file_name in reversed(raw_file_names):
    if '.wav' not in file_name: raw_file_names.remove(file_name)
pre_processed_folder_path = os.path.join('data','pre processed audio')

seconds_per_slice = 0.96 # Required by Yamnet
desired_sample_rate = 16000 # Requried by Yamnet
frames_per_slice = (int)(desired_sample_rate*seconds_per_slice)
frames_per_hop = (int)(frames_per_slice/4)

yamnet = pyy.yamnet_frames_model(params=pyp.Params())
yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))
class_names = pyy.class_names(os.path.join('src','latent_audio','plugins','yamnet','yamnet_class_map.csv'))

# Preprocess all files
a = 0; r = 0 # Counters for accepted and rejected chunks
for raw_file_name in raw_file_names:

    # Load .wav file
    waveform, sampling_rate = sf.read(os.path.join(raw_folder_path, raw_file_name), dtype=np.int16)
    max = np.max(waveform)
    waveform = waveform[:(int)(0.05*len(waveform))]

    # Adjust format
    waveform = decimate(waveform, 3) # Relies on assumption that sampling rate of .wav file is 3x the sampling rate of yamnet
    waveform = waveform / max#np.max(waveform)#np.iinfo(np.int16).max
    waveform = waveform.astype('float32')
    
    # Split into slices and keep non-silence ones
    slices = []
    slice_count = (int)(len(waveform) / frames_per_slice)
    s=0
    while s < len(waveform) - frames_per_slice:
        # Split
        slice = waveform[s:s+frames_per_slice]
        class_probabilities, _, _ = yamnet(slice)
        inferred_class = class_names[class_probabilities.numpy().mean(axis=0).argmax()]
        
        # Select
        if inferred_class != 'Silence':
            slices.append(slice); a += 1
        else:
            r += 1
            
        # Prepare next iteration
        s += frames_per_hop

    print(f"{a} slices accepted and {r} rejected for {raw_file_name}")
    a=0;r=0

import unittest
import latent_audio.yamnet.layer_wise as ylw
import latent_audio.plugins.yamnet.yamnet as yamnet_lib
import latent_audio.plugins.yamnet.params as params_lib
from scipy.signal import decimate
import soundfile as sf
import tensorflow as tf, numpy as np, os

class TestLayerWiseYamnet(unittest.TestCase):
    """Collects unit tests for the class LayerWiseYamnet"""

    def test_equivalence_to_regular_yamnet(self):
        """Tests whether the layer-wise yamnet produces the same output as the regular yamnet."""

        # Initialize
        layer_wise_yamnet = ylw.LayerWiseYamnet()
        layer_wise_yamnet.load_weights(file_path=os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

        yamnet = yamnet_lib.yamnet_frames_model(params_lib.Params())
        yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))
    
        # Load file
        waveform, sampling_rate = sf.read(os.path.join("data","raw audio","WT.wav"), dtype=np.int16)
        assert sampling_rate == 48000, f"The sampling rate of the raw audio was assumed to be 48000 in order to apply the decimation algorithm and achieve yamnets 16000. The provided audio has sampling rate {sampling_rate}. You need to use a different downsampling method, e.g. from sklearn to meet yamnet's requirement."
        
        # Normalize
        waveform =  waveform.astype('float32')[:(int)(len(waveform)/10)] # The recording is 10 minutes in duration, one is sufficient
        waveform = waveform / np.max(waveform)
        
        # Down-sample to yamnet's 16000
        waveform = decimate(waveform, 3)
        sampling_rate = 16000

        for layer_index in range(14):
            probabilities_1 = layer_wise_yamnet.call_from_layer(layer_wise_yamnet.call_until_layer(waveform, layer_index=layer_index),layer_index=layer_index+1) 
            probabilities_2, _, _ = yamnet(waveform)

            # Evaluate
            self.assertTupleEqual(tuple1=tuple(probabilities_1.shape), tuple2=tuple(probabilities_2.shape))
            self.assertEqual(first=tf.reduce_sum((probabilities_1-probabilities_2)**2).numpy(), second=0)

if __name__ == '__main__':
    TestLayerWiseYamnet().test_equivalence_to_regular_yamnet()
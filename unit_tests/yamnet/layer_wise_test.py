import unittest
import latent_audio.yamnet.layer_wise as ylw
import latent_audio.plugins.yamnet.yamnet as yamnet_lib
import latent_audio.plugins.yamnet.params as params_lib
import soundfile as sf, tensorflow as tf, numpy as np
from scipy.signal import decimate

class TestLayerWiseYamnet(unittest.TestCase):
    def test_load(self):
        """Tests whether the call method of AdditiveCoupling can do 2-axes coupling."""

        # Initialize
        layer_wise_yamnet = ylw.LayerWiseYamnet()
        layer_wise_yamnet.load_weights(file_path='src/latent_audio/plugins/yamnet/yamnet.h5')

        wav_data, sr = sf.read("src/latent_audio/plugins/yamnet/WH0000_1270.wav", dtype=np.int16)
        waveform = wav_data / 32768.0
        waveform = waveform.astype('float32')
        waveform = decimate(waveform, 3)
        waveform = waveform[:(int)(0.05*len(waveform))]
        yamnet = yamnet_lib.yamnet_frames_model(params_lib.Params())
        yamnet.load_weights('src/latent_audio/plugins/yamnet/yamnet.h5')
        
        for layer_index in range(14):
            probabilities_1 = layer_wise_yamnet.call_from_latent(layer_wise_yamnet.call_until_latent(waveform, layer_index=layer_index),layer_index=layer_index) 
            
            probabilities_2, embeddings, spectrogram = yamnet(waveform)

            # Evaluate
            self.assertTupleEqual(tuple1=tuple(probabilities_1.shape), tuple2=tuple(probabilities_2.shape))
            self.assertEqual(first=tf.reduce_sum((probabilities_1-probabilities_2)**2).numpy(), second=0)

if __name__ == '__main__':
    TestLayerWiseYamnet().test_load()
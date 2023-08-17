import unittest
import latent_audio.yamnet.layer_wise as ylw
import latent_audio.plugins.yamnet.yamnet as yamnet_lib
import latent_audio.plugins.yamnet.params as params_lib
import tensorflow as tf, numpy as np, os

class TestLayerWiseYamnet(unittest.TestCase):
    """Collects unit tests for the class LayerWiseYamnet"""

    def test_equivalence_to_regular_yamnet(self):
        """Tests whether the layer-wise yamnet produces the same output as the regular yamnet."""

        # Initialize
        layer_wise_yamnet = ylw.LayerWiseYamnet()
        layer_wise_yamnet.load_weights(file_path=os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))

        waveform = np.sin(1000*np.pi*np.linspace(0,10,100000), dtype=np.float32)
        yamnet = yamnet_lib.yamnet_frames_model(params_lib.Params())
        yamnet.load_weights(os.path.join('src','latent_audio','plugins','yamnet','yamnet.h5'))
        
        for layer_index in range(14):
            probabilities_1 = layer_wise_yamnet.call_from_layer(layer_wise_yamnet.call_until_layer(waveform, layer_index=layer_index),layer_index=layer_index) 
            probabilities_2, _, _ = yamnet(waveform)

            # Evaluate
            self.assertTupleEqual(tuple1=tuple(probabilities_1.shape), tuple2=tuple(probabilities_2.shape))
            self.assertEqual(first=tf.reduce_sum((probabilities_1-probabilities_2)**2).numpy(), second=0)

if __name__ == '__main__':
    TestLayerWiseYamnet().test_equivalence_to_regular_yamnet()
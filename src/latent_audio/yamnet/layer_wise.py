from latent_audio.plugins.yamnet import params, features as features_lib, yamnet as yamnet_lib
import tensorflow as tf
from typing import Tuple
import numpy as np

class LayerWiseYamnet(tf.keras.Model):

    class BatchNormalization(tf.keras.Model):

        def __init__(self, name, configuration):
            
            # Super
            super(LayerWiseYamnet.BatchNormalization, self).__init__()

            # Attributes
            self.__batch_normalization__ = tf.keras.layers.BatchNormalization(name=name, center=configuration.batchnorm_center, scale=configuration.batchnorm_scale, epsilon=configuration.batchnorm_epsilon)

        def call(self, inputs):
            return self.__batch_normalization__(inputs)

    class Convolution(tf.keras.Model):

        def __init__(self, name, kernel_size: int, strides: int, filter_count:int, configuration):
            
            # Super
            super(LayerWiseYamnet.Convolution, self).__init__()

            # Attributes
            self.__convolution__ = tf.keras.layers.Conv2D(name='{}/conv'.format(name), filters=filter_count, kernel_size=kernel_size, strides=strides, padding=configuration.conv_padding, use_bias=False, activation=None)
            self.__batch_normalization__ = LayerWiseYamnet.BatchNormalization(name='{}/conv/bn'.format(name), configuration=configuration)
            self.__activation__ = tf.keras.layers.ReLU(name='{}/relu'.format(name))

        def call(self, inputs):
            return self.__activation__(self.__batch_normalization__(self.__convolution__(inputs)))

    class SeparableConvolution(tf.keras.Model):

        def __init__(self, name, kernel_size: int, strides: int, filter_count:int, configuration):
            
            # Super
            super(LayerWiseYamnet.SeparableConvolution, self).__init__()

            # Attributes
            self.__depth_wise_convolution__ = tf.keras.layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name), kernel_size=kernel_size, strides=strides, depth_multiplier=1, padding=configuration.conv_padding, use_bias=False, activation=None)
            self.__depth_wise_batch_normalization__ = LayerWiseYamnet.BatchNormalization(name='{}/depthwise_conv/bn'.format(name), configuration=configuration)
            self.__depth_wise_activation__ = tf.keras.layers.ReLU(name='{}/depthwise_conv/relu'.format(name))
            self.__point_wise_convolution__ = tf.keras.layers.Conv2D(name='{}/pointwise_conv'.format(name), filters=filter_count, kernel_size=(1,1), strides=1, padding=configuration.conv_padding, use_bias=False, activation=None)
            self.__point_wise_batch_normalization__ = LayerWiseYamnet.BatchNormalization(name='{}/pointwise_conv/bn'.format(name), configuration=configuration)
            self.__point_wise_activation__ = tf.keras.layers.ReLU(name='{}/pointwise_conv/relu'.format(name))

        def call(self, inputs):
            inputs = self.__depth_wise_activation__(self.__depth_wise_batch_normalization__(self.__depth_wise_convolution__(inputs)))
            return self.__point_wise_activation__(self.__point_wise_batch_normalization__(self.__point_wise_convolution__(inputs)))

    def __init__(self):
        
        # Super
        super(LayerWiseYamnet, self).__init__()

        # Convenience variables
        self.configuration = params.Params()

        # Attributes
        self.__reshape__ = tf.keras.layers.Reshape((self.configuration.patch_frames, self.configuration.patch_bands, 1), input_shape=(self.configuration.patch_frames, self.configuration.patch_bands))
        
        self.__convolutional_layers__ = [
            LayerWiseYamnet.Convolution(name='layer1', kernel_size=[3,3], strides=2, filter_count=32, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer2', kernel_size=[3,3], strides=1, filter_count=64, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer3', kernel_size=[3,3], strides=2, filter_count=128, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer4', kernel_size=[3,3], strides=1, filter_count=128, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer5', kernel_size=[3,3], strides=2, filter_count=256, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer6', kernel_size=[3,3], strides=1, filter_count=256, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer7', kernel_size=[3,3], strides=2, filter_count=512, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer8', kernel_size=[3,3], strides=1, filter_count=512, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer9', kernel_size=[3,3], strides=1, filter_count=512, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer10', kernel_size=[3,3], strides=1, filter_count=512, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer11', kernel_size=[3,3], strides=1, filter_count=512, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer12', kernel_size=[3,3], strides=1, filter_count=512, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer13', kernel_size=[3,3], strides=2, filter_count=1024, configuration=self.configuration),
            LayerWiseYamnet.SeparableConvolution(name='layer14', kernel_size=[3,3], strides=1, filter_count=1024, configuration=self.configuration),
        ]

        self.__pooling__ = tf.keras.layers.GlobalAveragePooling2D()
        self.__dense__ = tf.keras.layers.Dense(name='dense', units=self.configuration.num_classes, use_bias=True)
        self.__activation__ = tf.keras.layers.Activation(activation=self.configuration.classifier_activation)

        # Initialize weights in layerwise yamnet with example input
        waveform = tf.keras.layers.Input(batch_shape=(None,), dtype=tf.float32)
        self(waveform)
        self.built = True

    def load_weights(self, file_path: str) -> None:
        
        # Load weight into original yamnet
        yamnet = yamnet_lib.yamnet_frames_model(params.Params())
        yamnet.load_weights(file_path)

        
        # Transfer weights from original to layerwise yamnet
        for layer in self.layers:
            for variable_1 in layer.variables:
                did_load = False
                for layer in yamnet.layers:
                    for variable_2 in layer.variables:
                        if variable_2.name in variable_1.name: 
                            variable_1.assign(variable_2)
                            did_load=True
                if not did_load:
                    print("Unable to load variables for ", variable_1.name)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        # Create spectrogram
        waveform_padded = features_lib.pad_waveform(inputs, self.configuration)
        _, X = features_lib.waveform_to_log_mel_spectrogram_patches(waveform_padded, self.configuration)

        # Pass through layers
        X = self.__reshape__(X)
        for layer in self.__convolutional_layers__:
            X = layer(X)
        X = self.__pooling__(X) # Embeddings
        X = self.__dense__(X) # Logits
        class_probabilities = self.__activation__(X)

        return class_probabilities

    def call_until_latent(self, waveform: np.array, layer_index: int) -> tf.Tensor:
        
        # Create spectrogram
        waveform_padded = features_lib.pad_waveform(waveform, self.configuration)
        _, X = features_lib.waveform_to_log_mel_spectrogram_patches(waveform_padded, self.configuration)

        # Pass through layers
        X = self.__reshape__(X)
        for layer in self.__convolutional_layers__[:layer_index]:
            X = layer(X)
        
        return X

    def call_from_latent(self, latent: tf.Tensor, layer_index) -> tf.Tensor:
        X = latent
        for layer in self.__convolutional_layers__[layer_index:]:
            X = layer(X)
        X = self.__pooling__(X) # Embeddings
        X = self.__dense__(X) # Logits
        class_probabilities = self.__activation__(X)

        return class_probabilities

if __name__ == "__main__":
    import soundfile as sf
    from scipy.signal import decimate
    

    model = LayerWiseYamnet()
    model.load_weights(file_path='src/latent_audio/plugins/yamnet/yamnet.h5')

    wav_data, sr = sf.read("src/latent_audio/plugins/yamnet/WH0000_1270.wav", dtype=np.int16)
    waveform = wav_data / 32768.0
    waveform = waveform.astype('float32')
    waveform = decimate(waveform, 3)
    waveform = waveform[:(int)(0.05*len(waveform))]
    yamnet = yamnet_lib.yamnet_frames_model(params.Params())
    yamnet.load_weights('src/latent_audio/plugins/yamnet/yamnet.h5')
    
    for layer_index in range(14):
        scores1 = model.call_from_latent(model.call_until_latent(waveform, layer_index=layer_index),layer_index=layer_index) 
        
        scores2, embeddings, spectrogram = yamnet(waveform)
        print(np.sum((scores1-scores2)**2))
    k=3
    

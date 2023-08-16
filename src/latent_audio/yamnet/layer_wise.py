from latent_audio.plugins.yamnet import params, features as features_lib, yamnet as yamnet_lib
import tensorflow as tf
from typing import Tuple
import numpy as np

class LayerWiseYamnet(tf.keras.Model):
    """Provides an implementation of `yamnet <https://www.tensorflow.org/hub/tutorials/yamnet>`_ which allows to inspect the latent 
    representation of a particular layer.
    """

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
        """Loads weights from a .h5 file. The weights are supposed to come from a regular yamnet, e.g. from 
        `here <https://storage.googleapis.com/audioset/yamnet.h5>`_.

        :param file_path: The path pointing towards the file .h5 containing the model weights.
        :type file_path: str
        """

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
                    raise Exception("Unable to load variables for ", variable_1.name)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward propagation through this model. It provides the inference functionality of the regular 
        `yamnet <https://www.tensorflow.org/hub/tutorials/yamnet>`_. 

        :param inputs: The waveform to be sliced and passed through the model for sound event recognition. It is expected to be of 
            shape [time frame count] and of data type float32 in the range [-1,1].
        :type inputs: :class:`tensorflow.Tensor`
        :return: class_probabilities (:class:`tensorflow.Tensor`) - The probability for each slice to belong to each of the AudioSet 
            classes. Shape == [slice count, class count]. For slicing and class count refere to the regular yamnet.
        """

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

        # Output
        return class_probabilities

    def call_until_layer(self, waveform: tf.Tensor, layer_index: int) -> tf.Tensor:
        """Propagates the ``waveform`` through self up until the layer at index ``layer_index``.

        :param waveform: The waveform to be sliced and passed through the model for sound event recognition. It is expected to be of 
            shape [time frame count] and of data type float32 in the range [-1,1].
        :type inputs: :class:`tensorflow.Tensor`
        :param layer_index: The index of the layer up until which the ``waveform`` shall be propagated through self. This index shall 
            be in the range [0,14). The layer at index ``layer_index`` is not evaluated.
        :type layer_index: int
        :return: X (:class:`tensorflow.Tensor`) - The latent representation before the layer at index ``layer_index``. Its shape depends
            of the layer and can be looked up in the original yamnet 
            `implementation <https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet.py>`_.
        """

        # Create spectrogram
        waveform_padded = features_lib.pad_waveform(waveform, self.configuration)
        _, X = features_lib.waveform_to_log_mel_spectrogram_patches(waveform_padded, self.configuration)

        # Pass through layers
        X = self.__reshape__(X)
        for layer in self.__convolutional_layers__[:layer_index]:
            X = layer(X)
        
        # Output
        return X

    def call_from_layer(self, latent: tf.Tensor, layer_index: int) -> tf.Tensor:
        """Provides a continuation of the forward propagation executed by :py:meth:`call_until_layer`. It starts at the layer at index
        ``layer_index`` and stops at the last layer of yamnet. 

        :param latent: The representation of an input to the model right before the layer at index ``layer_index``. Its shape depends
            of the layer and can be looked up in the original yamnet 
            `implementation <https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet.py>`_.
        :type latent: :class:`tensorflow.Tensor`
        :param layer_index: The index of the layer at which processing shall continue.
        :type layer_index: int
        :return: class_probabilities (:class:`tensorflow.Tensor`) - The probability for each slice to belong to each of the AudioSet 
            classes. Shape == [slice count, class count]. For slicing and class count refere to the regular yamnet.
        """
        
        # Propagate
        X = latent
        for layer in self.__convolutional_layers__[layer_index:]:
            X = layer(X)
        X = self.__pooling__(X) # Embeddings
        X = self.__dense__(X) # Logits
        class_probabilities = self.__activation__(X)

        # Output
        return class_probabilities


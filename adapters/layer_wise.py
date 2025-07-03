from LatentAudio.plugins.yamnet import params, features as features_lib, yamnet as yamnet_lib
import tensorflow as tf
from typing import Tuple, Dict, Any
import numpy as np

class LayerWiseYamnet(tf.keras.Model):
    """Provides an implementation of `yamnet <https://www.tensorflow.org/hub/tutorials/yamnet>`_ which allows to inspect the latent 
    representation of a particular layer.
    """
    layer_names = ['Conv\n1'] + [f"Conv\n{i} pw" for i in range(1,14)] + ['Fully\nConnected']

    class Convolution(tf.keras.Model):
        """Provides an implementation of a convolution later with 2D convolution, batch normalization and ReLU activation.

        :param convolution_kwargs: The keyword arguments for the convolutional layer, excluding name, use_bias and activation. These 
            are managed internally. ``convolution_kwargs`` may for instance specify filters, kernel_size, strides and padding.
        :type convolution_kwargs: Dict[str, Any]
        :param batch_normalization_kwargs: The keyword arguments for the two batch normalization layers, excluding name.
        :type batch_normalization_kwargs:  Dict[str, Any]
        """

        def __init__(self, name:str, convolution_kwargs: Dict[str, Any], batch_normalization_kwargs: Dict[str, Any]):
            
            # Super
            super(LayerWiseYamnet.Convolution, self).__init__()

            # Attributes
            self.__convolution__ = tf.keras.layers.Conv2D(name='{}/conv'.format(name), use_bias=False, activation=None, **convolution_kwargs)
            self.__batch_normalization__ = tf.keras.layers.BatchNormalization(name='{}/conv/bn'.format(name), **batch_normalization_kwargs)
            self.__activation__ = tf.keras.layers.ReLU(name='{}/relu'.format(name))

        def call(self, inputs):
            return self.__activation__(self.__batch_normalization__(self.__convolution__(inputs)))

    class SeparableConvolution(tf.keras.Model):
        """Provides an implementation for a convolutional layer that applies depth-wise 2D convolution, batch normalization and ReLu 
        activation, followed by point-wise 2D convolution, batch normalization and ReLU activation.
    
        :param name: The name of the layer.
        :type name: str
        :param depth_wise_kernel_size: The size of convolutional filter for the depth-wise convolution. For point-wise convolution it 
            is set to 1.
        :type depth_wise_kernel_size: int
        :param depth_wise_strides: The stride for the depth-wise convolution. For point-wise convolution it is set to 1.
        :type depth_wise_strides: int
        :param point_wise_filters: The number of filter for the point-wise convolution. For depth-wise convolution it is managed by
            the depth_multiplier argument which is here fixed at 1.
        :type point_wise_filters: int
        :param padding: The padding applied to both convolution layers.
        :type padding: str
        :param batch_normalization_kwargs: The keyword arguments for the two batch normalization layers, excluding name.
        :type batch_normalization_kwargs:  Dict[str, Any]"""

        def __init__(self, name: str, depth_wise_kernel_size: int, depth_wise_strides: int, point_wise_filters: int, padding: str, batch_normalization_kwargs: Dict[str, Any]):
            
            # Super
            super(LayerWiseYamnet.SeparableConvolution, self).__init__()

            # Attributes
            self.__depth_wise_convolution__ = tf.keras.layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name), kernel_size=depth_wise_kernel_size, strides=depth_wise_strides, depth_multiplier=1, use_bias=False, activation=None, padding=padding)
            self.__depth_wise_batch_normalization__ = tf.keras.layers.BatchNormalization(name='{}/depthwise_conv/bn'.format(name), **batch_normalization_kwargs)
            self.__depth_wise_activation__ = tf.keras.layers.ReLU(name='{}/depthwise_conv/relu'.format(name))
            self.__point_wise_convolution__ = tf.keras.layers.Conv2D(name='{}/pointwise_conv'.format(name), kernel_size=(1,1), filters=point_wise_filters, strides=1, use_bias=False, activation=None, padding=padding)
            self.__point_wise_batch_normalization__ = tf.keras.layers.BatchNormalization(name='{}/pointwise_conv/bn'.format(name), **batch_normalization_kwargs)
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
        
        bn_kwargs = {'center': self.configuration.batchnorm_center, 'scale':self.configuration.batchnorm_scale, 'epsilon':self.configuration.batchnorm_epsilon}
        padding = self.configuration.conv_padding
        self.__convolutional_layers__ = [
            LayerWiseYamnet.Convolution(name='layer1', convolution_kwargs={'kernel_size':[3,3], 'strides':2, 'filters':32, 'padding':padding}, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer2', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=64, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer3', depth_wise_kernel_size=[3,3], depth_wise_strides=2, padding=padding, point_wise_filters=128, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer4', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=128, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer5', depth_wise_kernel_size=[3,3], depth_wise_strides=2, padding=padding, point_wise_filters=256, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer6', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=256, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer7', depth_wise_kernel_size=[3,3], depth_wise_strides=2, padding=padding, point_wise_filters=512, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer8', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=512, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer9', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=512, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer10', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=512, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer11', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=512, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer12', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=512, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer13', depth_wise_kernel_size=[3,3], depth_wise_strides=2, padding=padding, point_wise_filters=1024, batch_normalization_kwargs=bn_kwargs),
            LayerWiseYamnet.SeparableConvolution(name='layer14', depth_wise_kernel_size=[3,3], depth_wise_strides=1, padding=padding, point_wise_filters=1024, batch_normalization_kwargs=bn_kwargs),
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
        :type waveform: :class:`tensorflow.Tensor`
        :param layer_index: The index of the layer up until which the ``waveform`` shall be propagated through self. This index shall 
            be in the range [0,14). The layer at index ``layer_index`` is evaluated.
        :type layer_index: int
        :return: X (:class:`tensorflow.Tensor`) - The latent representation right after the layer at index ``layer_index``. Its shape 
            depends on the layer and can be looked up in the original yamnet 
            `implementation <https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet.py>`_.
        """

        # Create spectrogram
        waveform_padded = features_lib.pad_waveform(waveform, self.configuration)
        _, X = features_lib.waveform_to_log_mel_spectrogram_patches(waveform_padded, self.configuration)

        # Pass through layers
        X = self.__reshape__(X)
        for layer in self.__convolutional_layers__[:layer_index+1]:
            X = layer(X)
        
        # Output
        return X

    def call_from_layer(self, latent: tf.Tensor, layer_index: int, only_logits: bool=False) -> tf.Tensor:
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
        if only_logits: return X
        else:

            class_probabilities = self.__activation__(X)

            # Output
            return class_probabilities


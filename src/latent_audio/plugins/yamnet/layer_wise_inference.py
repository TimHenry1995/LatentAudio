"""This python file was added by the author of LatenAudio in order to inspect the latent representations of yamnet.
It is based on the code of the original yamnet"""

from __future__ import division, print_function

import sys

import numpy as np
#import resampy
import soundfile as sf
import tensorflow as tf
from scipy.signal import decimate
import params as yamnet_params
import yamnet as yamnet_model

import features as features_lib

def create_first_half_of_yamnet(up_until_layer: int):
    pass

def create_second_half_of_yamnet(from_layer: int):
    pass

def yamnet_forward(features, params):
  """Define the core YAMNet mode in Keras."""
  net = tf.keras.layers.Reshape(
      (params.patch_frames, params.patch_bands, 1),
      input_shape=(params.patch_frames, params.patch_bands))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(yamnet_model._YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
  embeddings = tf.keras.layers.GlobalAveragePooling2D()(net)
  logits = tf.keras.layers.Dense(units=params.num_classes, use_bias=True)(embeddings)
  predictions = tf.keras.layers.Activation(activation=params.classifier_activation)(logits)
  return predictions, embeddings


def yamnet_frames_model(params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of Params containing hyperparameters.

  Returns:
    A model accepting (num_samples,) waveform input and emitting:
    - predictions: (num_patches, num_classes) matrix of class scores per time frame
    - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
    - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
  """
  waveform = tf.keras.layers.Input(batch_shape=(None,), dtype=tf.float32)
  waveform_padded = features_lib.pad_waveform(waveform, params)
  log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
      waveform_padded, params)
  predictions, embeddings = yamnet_forward(features, params)
  frames_model = tf.keras.Model(
      name='yamnet_frames', inputs=waveform,
      outputs=[predictions, embeddings, log_mel_spectrogram])
  return frames_model


def main(argv):
  assert argv, 'Usage: inference.py <wav file> <wav file> ...'

  params = yamnet_params.Params()
  yamnet = yamnet_frames_model(params, up_until_layer=4)
  yamnet.load_weights('src/plugins/yamnet/yamnet.h5')
  yamnet_classes = yamnet_model.class_names('src/plugins/yamnet/yamnet_class_map.csv')

  for file_name in argv:
    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / np.max(wav_data)  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
      waveform = decimate(waveform, 3) #resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    waveform_padded = features_lib.pad_waveform(waveform, params)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(waveform_padded, params)
    scores, embeddings, spectrogram = yamnet(waveform)
    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]
    print(file_name, ':\n' +
          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))


if __name__ == '__main__':
  main(["src/plugins/yamnet/WH0000_1270.wav"])

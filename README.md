# LatentAudio
This repository collects code and data to disentangle the latent space of sound-event recognition model Yamnet into materials and actions. The data consists of 60K 1-second long sound snippets for which it is known which material and action was involved in making the sound. Yamnet is a 14 layer convolutional neural network that maps a sound's spectrogram to 521 common-sense auditory event classes. 

## Pre-processing
The pre-processing code first passes the sounds through yamnet to obtain the latent representation at each layer. It then uses principal component analysis to reduce the dimensionality of these representations. Researchers who are interested in verifying or adjusting this part of code can use the Preprocess.ipynb file as a starting point. This file walks the user through downloading the github repository, installing dependencies and running the pre-processing scripts and saving the results. It can for instance be opened in Google Colab.

## Analysis
Researchers who are interested in verifying or adjusting the actual analysis can skip the pre-processing and use the pre-processed data that is stored already in this github repository. The main analysis is demonstrated in Main.ipynb which also walks the user through download and installation of all necessary files. The analyses involves classification of the sounds into materials and actions at each latent layer as well as the disentanglement of the latent space using an invertible flow model. The latent representations of sounds are then perturbed and systematic changes in Yamnets output are demonstrated as a result. The notebook provides numerous statistical tests and figures for these analyses. It can also be run in Google Colab.


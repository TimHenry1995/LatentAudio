# LatentAudio
The goal of this work was to disentangle the latent space of sound-event recognition model [Yamnet](https://www.tensorflow.org/hub/tutorials/yamnet) into materials and actions. The data consists of 60K 1-second long sound snippets for which it is known which material and action was involved in making the sound. Yamnet is a 14 layer convolutional neural network that maps a sound's spectrogram to 521 common-sense auditory event classes. For detailed explanations of our experiments, refer to our award-winning [research paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4959687).

## Analysis
In order to replicate the work of our accompanying paper, open and execute the "Analysis.ipynb" notebook. 

### Installation
If you have a Windows or Intel-based Mac computer, then you can run the installation on locally. Alternatively, you can use the cloud computer of [Google Colab](https://colab.google). The "Analysis.ipynb" notebook runs you through the installation.

### Pre-processing
If you want to keep the same pre-processing as we did in our experiments, then you can skip the pre-process section and instead use the existing pre-processed data that is stored in this github repository for the main analysis (see below). Yet, if you are interested in adjusting the pre-processing, you should work with the pre-processing section of the "Analysis.ipynb" notebook. There, you can adjust how sounds are passed through Yamnet to obtain the latent representation at each layer and how principal component analysis is used to reduce the dimensionality of these representations per layer. 

### Actual Analysis 
Researchers who are interested in verifying or adjusting the actual analysis should work with the analysis section of "Analysis.ipynb". It takes the pre-processed layer-wise sound representations and passes them through t-SNE for visualization. It also uses KNN to classify them into materials and actions. Using our custom flow model library [Gyoza](https://pypi.org/project/gyoza/), it disentangles the dimensions of the latent space. The disentangled representations are then perturbed and systematic changes in Yamnets output are demonstrated as a result. Numerous statistical tests and figures for these analyses are also provided. 


# LatentAudio
The goal of this work was to disentangle the latent space of sound-event recognition model Yamnet into materials and actions. The data consists of 60K 1-second long sound snippets for which it is known which material and action was involved in making the sound. Yamnet is a 14 layer convolutional neural network that maps a sound's spectrogram to 521 common-sense auditory event classes. For detailed explanations of our experiments, refer to our award-winning [research paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4959687).

## Installation
The code for this project has been run successfully on Windows, Intel-based Mac and Linux machines that are capable of running Python 3.9.x or 3.10.x with tensorflow 2.9.x. Tests on Apple Silicon were unsuccesful and users are thus encouraged to switch platform, e.g. to [Google Colab](https://colab.google).

### Personal Machine (Windows or Intel-based Mac)
Make sure you have a code editor such as [Visual Studio Code](https://code.visualstudio.com) with Python support as well as the package managers [Conda](https://www.anaconda.com) and [Git](https://desktop.github.com/download/) installed. Then, open your code editor and set its project folder to one that is located on a disk with at least 15GB of free storage, e.g. an external hard-drive. Next, open the code editor's terminal, typically using a top left menu option. To create the conda environment and install all required software packages for LatentAudio, execute the below code in the terminal:
```
conda create -n "latent_audio_env" python=3.9 ipython
conda activate latent_audio_env
git clone https://github.com/TimHenry1995/LatentAudio.git
cd LatentAudio
pip install .
```

### Google Colab
Create a custom runtime by clicking the top left menu option "Runtime" > "Change Runtime Type" > "Custom". Then, select a Python 3.9 version and click "Save". You can then select this runtime from the "Runtime" menu. Next, it is recommended to link your [Google Drive](https://drive.google.com/drive/home) to Colab using the login menu on the left side. This will allow you to save results of the LatentAudio analysis to your personal drive in case Colab closes your runtime before you can download them. Next, in the currently open notebook, create a code cell and execute the follwoing code to download the LatentAudio Github repository:
```
!git clone https://github.com/TimHenry1995/LatentAudio.git
```
In a second code cell, execute the below code to install the required software packages. This code cell might prompt a pop-up window asking you to restart the session. Accept and after restart run the cell again. It should then provide as output that LatentAudio was installed successfully.
```
%cd LatentAudio
%pip install .
```

## Analysis
In order to replicate the work of our accompanying paper, open and execute the "Analysis.ipynb" notebook. It is divided into pre-processing and actual analysis.

### Pre-processing
The pre-processing code first passes the sounds through Yamnet to obtain the latent representation at each layer. It then uses principal component analysis to reduce the dimensionality of these representations. Researchers who are interested in verifying or adjusting this part of code can use the Preprocess.ipynb file as a starting point. This file walks the user through downloading the github repository, installing dependencies and running the pre-processing scripts and saving the results. It can for instance be opened in Google Colab.

### Actual Analysis 
Researchers who are interested in verifying or adjusting the actual analysis can skip the pre-processing and use the existing pre-processed data that is stored in this github repository. The main analysis is demonstrated in Process.ipynb which involves classification of the sounds into materials and actions at each of Yamnet's latent layers as well as the disentanglement of the latent space using an invertible flow model. The latent representations of sounds are then perturbed and systematic changes in Yamnets output are demonstrated as a result. The notebook provides numerous statistical tests and figures for these analyses. 


# Spec2VEC

## What is Spec2VEC?
Spec2VEC is an intelligent framework designed to extract objective, interpretable feature embeddings from any 1D Time Series Data or its Time-Frequency Representations (TFRs). The tool computes a suite of information-theoretic, statistical, and spatial metrics to generate a unique "spectral fingerprint" for a input signals.

Developed initially for geophysics and planetary exploration, Spec2VEC addresses the "input" problem by systematically extracting information content from signal representations and representing it through a suite of numerical identities. It provides a transparent, physics-aware alternative to "black-box" deep learning models, ensuring that signal characterization remains consistent, interpretable, and reproducible.

## Applications
Spec2VEC is highly versatile and can be used on various forms of time-series data. Currently, this repository includes applications for:
- **Distributed Acoustic Sensing (DAS):** Processing and characterizing fiber-optic DAS arrays. Available in [das_application](das_application/) folder.
- **Exotic Seismic Event Catalog Clustering (ESEC):** Analyzing and clustering tectonic and non-tectonic earthquakes based on signal textural, spatial and statistical properties. Available in [esec_catalog_clustering](esec_catalog_clustering/) folder.
- **Synthetic Geophysical Signal Clustering:** Generating catalogs of randomized geophysical waveforms and clustering them based on their time-frequency fingerprints. Available in [synth_geophysical_signal_clustering](synth_geophysical_signal_clustering/) folder.
- **Acoustic Signal Modeling:** Building synthetic time series of waveforms propagating through complex heteregenous physical mediums. Then showing how Spec2VEC process can distinguish such waveforms. Available in [acoustic_signal_modeling](acoustic_signal_modeling/) folder.
- **Simple Synthetics Signal Analysis:** Spec2VEC can also be used for simple synthetic signals clsutering and prototyping. The examples and few other helpful notebooks can be referred at [other_useful_notebooks](other_useful_notebooks/) folder.

## Required Libraries
To run Spec2VEC, you need the following core Python libraries:
- `numpy`, `scipy`, `pandas` (Core scientific computing and data manipulation)
- `matplotlib`, `seaborn` (Plotting and visualization)
- `scikit-learn` (Machine learning algorithms)
- `h5py` (HDF5 data format support for reading/writing data banks)
- `pywt` (`PyWavelets` for Continuous Wavelet Transforms)
- `numba` (JIT compilation for performance)
- `tsfel`, `antropy`, `ordpy` (Time-series feature extraction and entropy/complexity features)
- `hilbert` (Used for Hilbert curve encoding/decoding during spatial flattening of images)
- `shap` (For feature importance analysis and model explanations)
- `jwave`, `jax`, `jaxlib` (Used for physics-based acoustic simulation features, optional if not running acoustic modeling)

*Detailed hard and soft requirement files can be found in the `software_requirements` folder.*
(Note: It is highly recommended that users first create and activate a fresh virtual environment before running the installation command so it doesn't conflict with their system's Python packages.)

## Installation Guide (To Local Machine)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/URseismology/SPEC2VEC.git
   cd Spec2Vec
   ```

- **EITHER: Set up a Python virtual environment and install libararies sequentially:**

   ```bash
   conda create -n spec2vec_env python=3.10 numpy=2.2.5 matplotlib pandas scikit-learn scipy seaborn notebook jupyterlab

   # Install other dependencies:
   pip install h5py PyWavelets numba tsfel antropy ordpy pyradiomics numpy-hilbert-curve gstools structify-net networkx
   
   # Optional: For acoustic/complex waveform modeling
   pip install jwave jax jaxlib
   ```
   
- **OR: If you do not want to install the libraries one by one then you can run the following (Recommended):**
   Note: Please follow these steps as provided to obtain an error free installation.
   
   ```
   conda create -n spec2vec_env python=3.10.19
   conda activate spec2vec_env
   pip install numpy==2.2.5
   pip install pyfastnoisesimd==0.4.2 --no-build-isolation
   pip install -r software_requirements/spec2vec_requirements_hard.txt
   pip install pyradiomics==3.0.1 --no-build-isolation
   ```

## Alternative Installation Guide (To Local Machine): automated setup script
   We provide a bash script that automatically handles the environment creation and dependency installation order for you. Update the Environment Name and Software Requirements File Path.
   
   ```bash
   bash setup.sh
   ```
   
   Once it completes, activate the environment:
   ```bash
   conda activate spec2vec_env
   ```
   
   Note: After completing the above setup you may need to run the following command for the new environment to show up in your kernel list.
   ```
   ~/.conda/envs/spec2vec_env/bin/python -m ipykernel install --user --name=spec2vec_env --display-name "spec2vec_env"
   ```

## Installation Guide (Google Colab):
Follow the below steps to use the Spec2Vec library and associated codes in Google Colab. The installation would take about 1-2mins.

1. Clone the repo in your google drive. (You might need to use a notebook):
   ```
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/
   !git clone https://github.com/URseismology/SPEC2VEC.git
   ```

2. Open the notebook you want to work with and copy the following code block at the beginning of the file and run it. Please ignore any pip dependency resolver related warnings and restart session prompts. 
   ```
   %cd /content/drive/MyDrive/
   !pip install -r SPEC2VEC/software_requirements/spec2vec_requirements_google_colab.txt
   !pip install git+https://github.com/Radiomics/pyradiomics
   ```

***Note: While running the notebooks/scripts in google colab make sure to change the path vairable as per google drive paths (e.g. /content/drive/MyDrive/SPEC2VEC/...) for seamelss execution.***

## Key Components and Functions
The backend operations of Spec2VEC live under the `utils/` directory. Key components include:
- **Spectrogram Generation (`spectograms_lib.py`):** Functions like `stft_basic_spectogram()` and `cwt_simple()` to convert 1D time-series into 2D time-frequency representations.
- **Noise Generation & Synthesis (`noise_lib.py` & `data_preparations.py`):** Tools for colored noise generation (white, red, pink, etc.) and simple synthetic signal formulation (additive, multiplicative and autoregressive processes).
- **Pointwise Metrics (`gisqa_pointwise_metrics_updated.py`):** Extracts scalar features from a single spectrogram image (e.g., entropy, complexity, statistical moments) by treating the image as a flattened Hilbert curve.
- **Pairwise Metrics (`gisqa_pairwise_metrics.py`):** Advanced comparative operations that evaluate a target spectrogram against a reference noise bank (e.g., histogram alignment, GMM labeling, size-zone spatial metrics) to generate spatial features.
- **The Main Pipeline (`gisqa_compute_updated.py`):** The primary `GISQAPipeline` class that wraps both pointwise and pairwise operations into a simple, high-level API.

## Typical Workflows

### Example 1: 1D Time Series to Spec2Vec Features
Here is a minimal example demonstrating how to extract Spec2VEC Features from a randomly generated 1D time series signal.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SPEC2VEC.utils.gisqa_compute_updated import *

## Generate Signal
sr=100
t=5
x = np.linspace(0, t, sr*t)
y1 = np.sin(2 * np.pi * 35 * x) * np.exp(-((x - 4) ** 2) / (2 * 0.3 ** 2))
y2 = np.sin(2 * np.pi * 20 * x) * np.exp(-((x - 1) ** 2) / (2 * 0.3 ** 2))

giq = GISQAPipeline()
spec_pointwise_metrics_df = giq.compute_pointwise_metrics_from_spec(['example_1','example_2'], [y1, y2],
                                                                    is_hilbertize = 0,
                                                                    is_normalize_stat = 0,
                                                                    is_normalize_entropy = 0,
                                                                    is_best_features = 0,
                                                                    ordpydx = 4, antropydx = 4,
                                                                    metrics_list=None, hilbert_locs=None)

spec_pointwise_metrics_df
```

### Example 2: 2D inputs (TFRs/Images) to Spec2Vec Features
Another example demonstrating how to extract Spec2VEC Features from a 2D Spectrogram image of a 1D time series signal.

```python
## Compute Spectrograms
f1, t1, stft1 = stft_basic_spectogram(y1,sr,64,0.5,"hann",0,50,
                                     max_normalize=False, powerlog=False,vmin_percentile=2, vmax_percentile=98)
f1, t1, stft2 = stft_basic_spectogram(y2,sr,64,0.5,"hann",0,50,
                                     max_normalize=False, powerlog=False,vmin_percentile=2, vmax_percentile=98)
#plt.pcolormesh(t1,f1,stft1); plt.pcolormesh(t1,f1,stft2)

metrics_to_compute = ['permutation_entropy_antropy', 'spectral_entropy_antropy', 'svd_entropy_antropy', 'detrended_fluctuation_antropy']

giq = GISQAPipeline()
spec_pointwise_metrics_df = giq.compute_pointwise_metrics_from_spec(['example_1','example_2'], [stft1, stft2],
                                                                    is_hilbertize = 1,
                                                                    is_normalize_stat = 0,
                                                                    is_normalize_entropy = 0,
                                                                    is_best_features = 0,
                                                                    ordpydx = 5, antropydx = 5,
                                                                    metrics_list=metrics_to_compute, 
                                                                    hilbert_locs=None)

spec_pointwise_metrics_df
```

## Folder Structure
This repository is organized into backend utilities and specific domain applications. Each domain application folder might contain a related data folder to be used along with its respective notebooks and an app_readme to provide specific details relating those applications. 

*The following project tree privdes a high level outline of the project architecture and information regarding how you should navigate the folder structure:*

```
📦 SPEC2VEC
├─ utils (core functions)
├─ domain application (e.g. DAS, ESEC Catalog, Synthetics)
│  ├─ data
│  ├─ notebooks
│  └─ app_readme.md
├─ software requirements (libraries needed)
├─ available_features_list.txt
├─ readme.md
└─ license
```

*The following contains the details of each folder available in this repository:*

- **[utils](utils/)**: The core "engine" of the Spec2VEC algorithm. Contains all Python source code for metric computation, plotting, pipeline logic, noise generation, and spectrogram transformations. **Most users will import from here.**
- **[das_application](das_application/)**: Contains notebooks and data-processing scripts specific to Distributed Acoustic Sensing (DAS) experiments. 
- **[esec_catalog_clustering](esec_catalog_clustering/)**: Contains notebooks focused on evaluating and clustering earthquake events using spectral fingerprints.
- **[synth_geophysical_signal_clustering](synth_geophysical_signal_clustering/)**: Notebooks demonstrating the Spec2VEC workflow on algorithmically generated mock geophysical waveforms.
- **[acoustic_signal_modeling](acoustic_signal_modeling/)**: Complex waveform modeling and synthetic physics-based acoustic simulations.
- **[time_complexity_analysis](time_complexity_analysis/)**: Scripts used to benchmark the runtime performance of pointwise and pairwise metric extraction against different image dimensions and sample sizes.
- **[other_useful_notebooks](other_useful_notebooks/)**: Miscellaneous scripts and notebooks for data exploration, noise bank preparation, and side experiments.
- **[software_requirements](software_requirements/)**: Detailed text files containing hard and soft python dependencies to set up your environment.


## Contact: 
Sayan Swar (sswar@ur.rochester.edu; https://github.com/PSU-Geofluids-Lab) \
Tushar Mittal (tmittal@psu.edu; https://github.com/PSU-Geofluids-Lab) \
Tolulope Olugboji (tolulope.olugboji@rochester.edu)

## Citation
If you use this code, data, or software in your research, please cite both the accompanying paper (currently in review) and the repository:

#### 1. Main Paper (In Review)
```bibtex
@article{Swar2026spec2vec,
  author  = {Swar, Sayan and Mittal, Tushar and Olugboji, Tolulope},
  title   = {{Spec2VEC}: Interpretable Spectral Representations of Planetary Signals in Vector Spaces},
  journal = {Geophysical Journal International},
  year    = {2026},
  note    = {In review}
}
```

#### 2. Software Repository
```bibtex
@software{Swar2026spec2vec,
  author  = {Swar, Sayan and Mittal, Tushar and Olugboji, Tolulope},  
  title   = {Spec2Vec: Interpretable Spectral Representations of Planetary Signals in Vector Spaces},
  year    = {2026},
  url     = {https://github.com/URseismology/Spec2Vec},
  note    = {GitHub repository}
}
```
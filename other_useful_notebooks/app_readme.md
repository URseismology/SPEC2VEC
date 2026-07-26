# Other Useful Notebooks

This directory contains a collection of supplementary Jupyter notebooks designed to demonstrate the core functionality of Spec2VEC, generate reference datasets, and reproduce key figures from the manuscript.

## Overview of Notebooks

* **[`example_spec2vec_usage.ipynb`](example_spec2vec_usage.ipynb)**
  A straightforward, high-level tutorial demonstrating the basic usage of the Spec2VEC pipeline. This is a great starting point for new users wanting to see the algorithm in action on basic inputs.

* **[`reference_noisebankprep.ipynb`](reference_noisebankprep.ipynb)**
  This script demonstrates the methodology for generating and preparing different categories of noise referrences to build a "noise bank." The noise bank is used as a baseline reference by Spec2VEC to extract relative spatial features.

* **[`spec2vec_on_simple_synth_signal.ipynb`](spec2vec_on_simple_synth_signal.ipynb)**
  Focuses on applying the Spec2VEC algorithm to basic synthetic signals (like combined sines, chirps, and transients). It walks through the process of generating these signals and how Spec2VEC can be applied to extract discriminative features. The script can also be used to reproduce one of the figures presented in the manuscript.

* **[`spec2vec_on_synth_texture_categories.ipynb`](spec2vec_on_synth_texture_categories.ipynb)**
  Explores Spec2VEC's ability to differentiate complex image textures. This notebook runs the algorithm across distinct synthetic texture categories (e.g., Perlin noise, uniform noise) to demonstrate the discrimination capabilities of the framework. The script can also be used to reproduce one of the figures presented in the manuscript.

## Data Requirements
To successfully execute these notebooks and reproduce the manuscript results, certain data files are required. 
All necessary data should be located within the **[`other_data/`](other_data/)** folder. If the data is too large to host on GitHub, this folder will contain its own `README.md` with explicit instructions or links on how to fetch the required files before running the notebooks.

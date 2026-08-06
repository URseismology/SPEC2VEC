# Synthetic Geophysical Signal Clustering

## Overview
This directory focuses on the application of the Spec2VEC framework to synthetic geophysical records. The primary objective is to demonstrate the robust clustering capabilities of Spec2VEC on well-defined synthetic datasets representing various geological and environmental phenomena, such as:
- T-Phase earthquakes
- Strombolian explosions
- Surface wave dispersions
- ... and other distinct categorical waveforms.

By evaluating the algorithm on synthetics, we can validate how effectively Spec2VEC isolates the underlying physical source signatures from noise before applying it to real-world noisy catalogs.

## 📂 Provided Data
To allow for quick reproduction of results without needing to re-compute everything from scratch, the required data files have already been generated and are located in the [`data/`](data/) directory:
- **`synth_geophysical_waveform_catalog.npy`**: A NumPy array containing the raw catalog of synthesized time-series waveforms.
- **`spec2vec_synth_geophysical_signals.csv`**: A pre-computed CSV dataset containing the extracted Spec2VEC features and metrics for the waveforms in the catalog.

---

## 📊 Notebooks & Analysis Workflow

### 1. Deep Analysis & Feature Preparation
**Notebook:** [`synth_geophysical_signals_application.ipynb`](synth_geophysical_signals_application.ipynb)  
This notebook is the core engine for this module. Use this notebook if you want to understand the backend process. It walks you through:
- How to prepare and generate the mock synthetic geophysical waveforms with varying physical properties.
- How to transform those time-series records into Time-Frequency Representations (TFRs).
- How to compute the full suite of Spec2VEC features on those generated TFRs.

### 2. Result Reproduction & Visualization
**Notebook:** [`synth_geophysical_signals_application_figures.ipynb`](synth_geophysical_signals_application_figures.ipynb)  
This notebook focuses purely on the analysis of the extracted features. You can run this directly using the pre-computed data files to reproduce the specific clustering figures, statistical comparisons, and dimensionality reduction scatter plots shown in the Spec2VEC publication.

---

## 🚀 Quick Start / How to Run

1. **Reproduce Final Results Only:**  
   If you just want to reproduce the final clustering visualizations, open [`synth_geophysical_signals_application_figures.ipynb`](synth_geophysical_signals_application_figures.ipynb) and run it from top to bottom. It will automatically load the pre-computed metrics from the `data/` folder and generate the final plots.

2. **Experiment & Generate New Data:**  
   If you want to experiment with modifying the types of synthetic signals, tuning the signal-to-noise ratios, or tweaking the Spec2VEC metric extraction parameters, open [`synth_geophysical_signals_application.ipynb`](synth_geophysical_signals_application.ipynb). This will allow you to generate a brand new dataset and calculate a fresh set of metrics.

<br>



*Note: Make sure to change the file paths to match you local directory settings* 
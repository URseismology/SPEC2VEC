# ESEC & Tectonic Earthquake Catalog Clustering

## Overview
This directory contains the necessary scripts, notebooks, and reference catalogs to process, analyze, and cluster both Exotic Seismic Event Catalog (ESEC) events and standard Tectonic Earthquakes using the Spec2VEC framework. 

The **Exotic Seismic Event Catalog (ESEC)** is a curated database of unconventional seismic events (such as landslides, debris flows, and other non-tectonic surface events). You can find more detailed information about the ESEC catalog by visiting the [IRIS SPUD page](https://ds.iris.edu/spud/esec).

## Data Catalogs & Sourcing
The master catalogs for both types of events are included in this directory:
- **ESEC Data:** Reference catalog is available in [`Master_ESEC_Catalog_vel_all.csv`](Master_ESEC_Catalog_vel_all.csv).
- **Tectonic EQ Data:** Reference catalog is available in [`Master_tectonic_eq_catalog.csv`](Master_tectonic_eq_catalog.csv). Note that the Tectonic EQ events were manually curated and downloaded using the [IRIS Wilber 3](https://ds.iris.edu/wilber3/find_event) tool.

---

## 🚀 Step-by-Step Workflow

> [!WARNING]
> **Configuration Required:** Before running any of the scripts below, please open the Python files and update the global variables (e.g., file/directory paths) to match your local setup.

### Step 1: ESEC Data Download & Processing
Run the first step to automatically download and process the raw waveform data for the ESEC catalog.
- **Script:** [`step1_esec_catalog_dataprepandprocess.py`](step1_esec_catalog_dataprepandprocess.py)
- **Output:** The raw ESEC data will be downloaded and saved into the [`esec_raw_data`](esec_raw_data/) folder.

### Step 2: Tectonic EQ Processing
Since the tectonic earthquake data is manually downloaded, run the second step to apply standard signal processing routines to the waveforms.
- **Script:** [`step2_tectonic_eq_dataprepandprocess.py`](step2_tectonic_eq_dataprepandprocess.py) *(Run this after Step 1)*
- **Output:** The processed tectonic earthquake velocity traces will be saved into the [`eq_processed_vel`](eq_processed_vel/) folder.

*Note: The Raw and Processed Data files for the ESEC and Tectonic EQs are also available to directly download via the link provided at the end of this document.*

### Step 3: Spec2VEC Metrics Preparation
With both catalogs downloaded and processed, run the final preparation step to compute the Spec2VEC features (spectral fingerprints) for all events.
- **Script:** [`step3_spec2vec_metrics_prep.py`](step3_spec2vec_metrics_prep.py)
- **Output:** The computed feature datasets will be stored in the [`computed_dataset_files`](computed_dataset_files/) folder.

---

## 📊 Analysis & Reproducing Paper Results

Once the metrics have been computed (Step 3), you can explore the data and reproduce the clustering results discussed in the Spec2VEC publication.
- Head over to the **[`spec2vec_analysis_notebooks`](spec2vec_analysis_notebooks/)** directory. 
- We have also provided the computed metrics files in the **[`computed_dataset_files`](computed_dataset_files/)** directory. So one can directly run the [spec2vec_analysis_main.ipynb](spec2vec_analysis_notebooks/spec2vec_analysis_main.ipynb) notebook.
- These Jupyter notebooks will walk you through the statistical comparisons, dimensionality reduction, and clustering of the exotic vs. tectonic events.

*Note 1: Please update the path variables in the notebooks to match your setup* <br>
*Note 2: For some of the figures you will need the sac files which you can download as per instructions in the data_readme files*

---

## 🎨 Optional: Generating Figures & Plots

We also provide optional Python scripts if you wish to generate deeper visual analyses or standalone figures for reports.

- **Reference Input Figures:**  
  Run [`optional_input_figures_ref_prep.py`](optional_input_figures_ref_prep.py) to generate reference plots of the input signals. 
  - Outputs are saved in [`esec_spec2vec_ip_figures`](esec_spec2vec_ip_figures/) and [`eq_spec2vec_ip_figures`](eq_spec2vec_ip_figures/) folders.
  - We have provided the figures in these folders for immidiate analysis.

- **Comprehensive Metric Plots:**  
  Run [`optional_comprehensive_metrics_plots.py`](optional_comprehensive_metrics_plots.py) to generate detailed visualizations of the calculated features.
  - Outputs are saved in the `comprehensive_metrics_plots/` directory for review.

- **Spec2VEC on Rolling Basis:**  
  To run Spec2VEC on 30 sec (or N sec) chunks of time series and generate features accordingly we have provided a rolling version of the Spec2VEC metrics calculation in the [optional_esec_spec2vec_rolling_metrics_prep.py](optional_esec_spec2vec_rolling_metrics_prep.py) script.
  - Users can run this script for the ESEC catalog and analyze/cluster the results.
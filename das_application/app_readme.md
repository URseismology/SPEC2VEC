# DAS Application

## Overview
This directory demonstrates the application of the Spec2VEC algorithm to Distributed Acoustic Sensing (DAS) data. It showcases how Spec2VEC can be used to characterize a DAS data before and after earthquake arrival.

## 📂 Directory Structure

- **[`das_application.ipynb`](das_application.ipynb)**: The main Jupyter notebook. It provides a complete walk-through of reading DAS data (HDF5), plotting space–time representations, generating array-wide spectra, and extracting Spec2VEC features.
- **[`data`](data/)**: Contains a `data_readme.md` with instructions and links on where to download the required raw DAS data.

## 🚀 Quick Start
1. Navigate to the [`data`](data/) folder and follow the instructions in the `data_readme.md` to download the raw DAS dataset.
2. Open [`das_application.ipynb`](das_application.ipynb) and execute it to process the DAS array data and extract the spectral embeddings.

*Note: Ensure your file paths inside the notebook are updated to match your local directory structure before running.*

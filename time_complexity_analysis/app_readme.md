# Time Complexity Analysis

## Overview
This directory contains scripts and benchmark data used to evaluate the computational runtime complexity of the Spec2VEC algorithm. The tests systematically measure the execution time required to process pointwise (statistical/textural) and pairwise (spatial) metrics across spectrograms of varying dimensions (e.g., 128x128, 256x256, 512x512).

## 📂 Directory Structure

#### 1. [`codes_and_files`](codes_and_files/)
This folder houses the core benchmarking logic and the generated output files:
- **[`analyze_time.py`](codes_and_files/analyze_time.py)**: The primary Python script that runs the time complexity benchmarks. It processes mock signals against the pre-computed noise reference banks and records the execution time in seconds.
- **[`time_complexity_results_diff_dims.csv`](codes_and_files/time_complexity_results_diff_dims.csv)**: A CSV file storing the tabulated runtime results across the different image dimensions.
- **[`total_time_complexity_plot_diff_dims.png`](codes_and_files/total_time_complexity_plot_diff_dims.png)**: A graphical visualization showing how the Spec2VEC execution time scales with the resolution of the spectrogram image.

#### 2. [`data_noise_ref`](data_noise_ref/)
This folder contains pre-computed HDF5 noise reference banks used exclusively by the benchmarking script. Providing these upfront ensures that noise-generation overhead does not skew the performance benchmarks of the Spec2VEC metric calculation itself:
- `tc_noise_reference_128by128.h5`
- `tc_noise_reference_256by256.h5`
- `tc_noise_reference_512by512.h5`

---

## 🚀 How to Run

To run the analysis locally and re-generate the benchmarks or test on your own hardware, execute the Python script from the root repository folder:

```bash
python SPEC2VEC/time_complexity_analysis/codes_and_files/analyze_time.py
```

*Note: Make sure your Python environment is fully set up with the required dependencies and that you are executing the script from the root of the project so that relative imports resolve correctly.*

from numba import jit, types
import pandas as pd
import numpy as np

import sys
import os
import requests
from bs4 import BeautifulSoup
import io
import time
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

import scipy.signal as signal
from scipy.signal import stft, windows

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.append('/data/sswar_files/')
pd.set_option('display.max_columns', None)

from collections import Counter, defaultdict
from scipy.stats import norm
import shap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from scipy.signal import hilbert

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, root_mean_squared_error, r2_score, mean_squared_error, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

sys.path.append('/data/sswar_files/')
sys.path.append('/data/sswar_files/0_PUBLISHED_CODES/')

# from PRJ_SPEC_MASTER.src.utils.data_preparations import *
# from PRJ_SPEC_MASTER.src.utils.noise_lib import *
# from PRJ_SPEC_MASTER.src.utils.spectograms_lib import *
# from PRJ_GIS_QA.src.utils.gisqa_compute_updated import *
# from PRJ_GIS_QA.src.utils.helper import *

from SPEC2VEC.src.utils.data_preparations import *
from SPEC2VEC.src.utils.noise_lib import *
from SPEC2VEC.src.utils.spectograms_lib import *
from SPEC2VEC.src.utils.gisqa_compute_updated import *
from SPEC2VEC.src.utils.gisqa_helper import *

#---------------------------- GLOBAL VARIABLES------------------------------
# Set Global Variables
#---------------------------------------------------------------------------
CATALOG_PATH_ESEC = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/Master_ESEC_Catalog_vel_all.csv"
CATALOG_PATH_EQ = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/Master_tectonic_eq_catalog.csv"
ESEC_EVENTS_REF = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/metadata/IRIS_DMC_esecEventsDb.txt"

ESEC_CWT_DICT = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/computed_dataset_files/esec_spec2vec_input_cwt_dict.npy"
EQS_CWT_DICT = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/computed_dataset_files/tectonic_eq_spec2vec_input_cwt_dict.npy"

SEPC2VEC_METRICS_ESEC = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/computed_dataset_files/esec_spec2vec_metrics_spatial.csv"
SEPC2VEC_METRICS_EQS = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/computed_dataset_files/eq_spec2vec_metrics_spatial.csv"

DATA_DIR_LIST = ["/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/esec_processed_vel",
                "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/eq_processed_vel"]
FIG_SAVE_PATH = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/figures/forpaper_V1/comprehensive_metrics_plots"
COMPUTE_METRICS = ['permutation_entropy','svd_entropy','spectral_entropy','petrosian_fd','hjorth_complexity','detrended_fluctuation',
                    'fisher_shannon','higuchi_fd','renyi_complexity_entropy']

TOTAL_SAMPLES = 20
DX_SPEC = 5
DX_TS = 4
DX_FFT = 4
SPEC_NOISE_THRESH = 50

##---------------------------- Helper Functions ----------------------------
def _signalamp_norm(signal_amp):
    signal_amp = 2 * (signal_amp - signal_amp.min()) / (signal_amp.max() - signal_amp.min()) - 1
    return signal_amp

def _ghilbertize_asym_data(img):
        #locs = HelperFunc.gilbertize_image(width=self.img_height, height=self.img_width)
        locs = HelperFunc.gilbertize_image_optimized(width=img.shape[0], height=img.shape[1])
        return img[locs[:,0], locs[:,1]].reshape(-1)

def _compute_metric(data, metric_name, dx=3, sf=20):
    """
    Calculates a specific entropy, complexity, or fractal dimension metric 
    based on the provided parameter name.
    
    Parameters:
        data (array): The 1D signal/array to analyze.
        metric_name (str): The name of the metric to compute.
        dx (int): Embedding dimension/order for ordpy and antropy methods.
        sf (float): Sampling frequency, required for spectral entropy.
        
    Returns:
        float: The calculated metric.
    """
    # Normalize the parameter name to lowercase for safe matching
    metric_name = metric_name.lower().strip()
    
    if metric_name == 'permutation_entropy':
        return ordpy.complexity_entropy(data, dx=dx)[0]
        
    elif metric_name == 'svd_entropy':
        return ant.svd_entropy(data, order=dx, normalize=True)
        
    elif metric_name == 'spectral_entropy':
        return ant.spectral_entropy(data, sf=sf, method='welch', normalize=True)
        
    elif metric_name == 'petrosian_fd':
        return ant.petrosian_fd(data)
        
    elif metric_name == 'ant_perm_entropy':
        return ant.perm_entropy(data, order=dx, normalize=True)
        
    elif metric_name == 'hjorth_complexity':
        return ant.hjorth_params(data)[1]
        
    elif metric_name == 'detrended_fluctuation':
        return ant.detrended_fluctuation(data)
        
    elif metric_name == 'global_node_entropy':
        return ordpy.global_node_entropy(data, dx=dx)
        
    elif metric_name == 'fisher_shannon':
        return ordpy.fisher_shannon(data, dx=dx)[1]
        
    elif metric_name == 'missing_patterns':
        return ordpy.missing_patterns(data, dx=dx, return_fraction=True, return_missing=False)
        
    elif metric_name == 'renyi_complexity_entropy':
        return ordpy.renyi_complexity_entropy(data, dx=dx, alpha=3)[0]
        
    elif metric_name == 'tsallis_complexity_entropy':
        return ordpy.tsallis_complexity_entropy(data, dx=dx, q=0.5)[0]
        
    elif metric_name == 'higuchi_fd':
        return ant.higuchi_fd(data)
        
    else:
        raise ValueError(f"Unknown metric name provided: '{metric_name}'")

def _plot_comprehensive_4row_dashboard(archetype_labels, archetype_subtype, spectro_dict, data_dir_lst, fs=20.0, dx_spec=5, dx_ts=5, dx_fft=5, spec_noise_thresh=75, 
                                        metric_name="Complexity Entp", save_path=None):
    N = len(archetype_labels)
    
    # Create figure and a GridSpec layout
    # Width scales dynamically with N. Height increased to accommodate 4 rows.
    fig = plt.figure(figsize=(4 * N, 14))
    

    # Height ratios: TS (1), TS/FFT Bars (1.2), Spectro (1), CWT Bars (1.2)
    # gs = gridspec.GridSpec(4, 2 * N, figure=fig, height_ratios=[1.0, 1.2, 1.0, 1.2])
    gs = gridspec.GridSpec(5, 2 * N, figure=fig, height_ratios=[1.0, 1.0, 1.2, 1.0, 1.2])

    # Storage arrays for the 4 bar charts
    ts_entropies = []
    fft_entropies = []
    flat_entropies = []
    hilb_entropies = []
    clean_names = []
    
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#8f0b8a'][:N] # Red, Blue, Green, Orange
    
    # ==========================================
    # DATA EXTRACTION & ROW 1/ROW 3 PLOTTING
    # ==========================================
    for i, (label,subtyp) in enumerate(zip(archetype_labels,archetype_subtype)):
        # 1. Clean up the label for titles
        parts = label.split('_')
        clean_name = " ".join([p for p in parts[2:-5] if p.islower()]).title() if subtyp != "Tectonic Eathquakes" else "Tectonic Eathquakes"
        clean_names.append(clean_name)
        
        # 2. Fetch Time Series
        filename = label[label.find('_')+1:] + '.sac' if subtyp != "Tectonic Eathquakes" else label + '.sac'
        data_dir = data_dir_lst[0] if subtyp != "Tectonic Eathquakes" else data_dir_lst[1]
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            st = read(filepath)
            tr_data = st[0].copy()
            if fs != tr_data.stats.sampling_rate:
                tr_data.interpolate(sampling_rate=fs, method='lanczos', a=20)
            signal = tr_data.data
            signal =  _signalamp_norm(signal) 
            time_vector = np.linspace(0, len(signal)/fs, len(signal))
            
            # 3. Fetch FFT
            fft_f, sigfft_tmp = signal_spectra(signal, fs)
            fft_mask = (fft_f >= 1) & (fft_f <= 5)
            fft_f = fft_f[fft_mask]
            sigfft_tmp = sigfft_tmp[fft_mask]
        else:
            signal = np.zeros(100)
            time_vector = np.linspace(0, 100/fs, 100)
            sigfft_tmp = np.zeros(100)
            
        # 4. Fetch Spectro 
        spectro = spectro_dict.get(label)
        spectro_mag = np.abs(spectro)
        spectro_mag[spectro_mag < np.percentile(spectro_mag, spec_noise_thresh)] = 0
        flat_spec_data = spectro_mag.flatten()
        hilb_spec_data = _ghilbertize_asym_data(spectro_mag)

        # ------------------------------------------
        # COMPUTE METRICS
        # ------------------------------------------
        ts_len = len(signal)
        fft_len = len(sigfft_tmp)
        spec_len = len(flat_spec_data)

        ent_ts = _compute_metric(signal, metric_name, dx=dx_ts)
        ent_fft = _compute_metric(sigfft_tmp, metric_name, dx=dx_fft)
        ent_flat = _compute_metric(flat_spec_data, metric_name, dx=dx_spec)
        ent_hilb = _compute_metric(hilb_spec_data, metric_name, dx=dx_spec)
        
        ts_entropies.append(ent_ts)
        fft_entropies.append(ent_fft)
        flat_entropies.append(ent_flat)
        hilb_entropies.append(ent_hilb)
        
        # ------------------------------------------
        # ROW 1: TIME SERIES PLOT
        # ------------------------------------------
        ax_ts = fig.add_subplot(gs[0, i*2 : (i+1)*2])
        ax_ts.plot(time_vector, signal, color='black', linewidth=0.5)
        ax_ts.set_title(f"TS, {clean_name} (N={ts_len})", fontsize=11, fontweight='bold', pad=10)
        ax_ts.set_ylabel("Velocity")
        ax_ts.set_xlim([0, time_vector[-1]])
        ax_ts.set_xticks([]) # Keep clean for the dashboard look

        # ------------------------------------------
        # ROW 2: FFT PLOT (1-5 Hz)
        # ------------------------------------------
        ax_fft = fig.add_subplot(gs[1, i*2 : (i+1)*2])
        ax_fft.plot(fft_f, sigfft_tmp, color='navy', linewidth=1.2)
        ax_fft.set_title(f"FFT, {clean_name} (N={fft_len})", fontsize=11, fontweight='bold', pad=10)
        ax_fft.set_ylabel("Amplitude")
        ax_fft.set_xlim([1, 5])
        ax_fft.set_xticks([1, 2, 3, 4, 5])
        ax_fft.set_xticklabels(['1Hz', '2Hz', '3Hz', '4Hz', '5Hz'])
        

        # ------------------------------------------
        # ROW 3: SPECTROGRAM PLOT
        # ------------------------------------------
        ax_spec = fig.add_subplot(gs[3, i*2 : (i+1)*2])
        
        ax_spec.imshow(spectro_mag, aspect='auto', origin='lower', cmap='viridis')
        ax_spec.set_title(f"CWT {clean_name}\nN={spec_len}, Thresh {spec_noise_thresh}%", fontsize=11, fontweight='bold')
        ax_spec.set_ylabel("Freq Bins")
        ax_spec.set_xlabel("Time Bins")
        
        
    # ==========================================
    # ROW 2: TIME SERIES & FFT BAR CHARTS
    # ==========================================
    x_pos = np.arange(N)
    
    # Left: Time Series Metric
    ax_bar_ts = fig.add_subplot(gs[2, 0:N])
    ax_bar_ts.bar(x_pos, ts_entropies, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    ax_bar_ts.set_xticks(x_pos)
    ax_bar_ts.set_xticklabels(clean_names, rotation=15, ha='right', fontsize=10)
    ax_bar_ts.set_title(f"Time Series, {metric_name}", fontsize=11, fontweight='bold', pad=10)
    ax_bar_ts.set_ylabel(f"Value (dx={dx_ts})")
    ax_bar_ts.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar_ts.set_yscale('log')

    # Right: FFT Metric
    ax_bar_fft = fig.add_subplot(gs[2, N:2*N])
    ax_bar_fft.bar(x_pos, fft_entropies, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    ax_bar_fft.set_xticks(x_pos)
    ax_bar_fft.set_xticklabels(clean_names, rotation=15, ha='right', fontsize=10)
    ax_bar_fft.set_title(f"FFT, {metric_name}", fontsize=11, fontweight='bold', pad=10)
    ax_bar_fft.set_ylabel(f"Value (dx={dx_fft})")
    ax_bar_fft.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar_fft.set_yscale('log')

    # ==========================================
    # ROW 4: CWT FLATTEN VS HILBERT BAR CHARTS
    # ==========================================
    # Left: Standard Flatten
    ax_bar_flat = fig.add_subplot(gs[4, 0:N])
    ax_bar_flat.bar(x_pos, flat_entropies, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    ax_bar_flat.set_xticks(x_pos)
    ax_bar_flat.set_xticklabels(clean_names, rotation=15, ha='right', fontsize=10)
    ax_bar_flat.set_title(f"CWT Standard Flatten (Row-by-Row)\n{metric_name}", fontweight='bold', pad=10)
    ax_bar_flat.set_ylabel(f"Value (dx={dx_spec})")
    ax_bar_flat.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar_flat.set_yscale('log')

    # Right: Hilbert Curve
    ax_bar_hilb = fig.add_subplot(gs[4, N:2*N])
    ax_bar_hilb.bar(x_pos, hilb_entropies, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    ax_bar_hilb.set_xticks(x_pos)
    ax_bar_hilb.set_xticklabels(clean_names, rotation=15, ha='right', fontsize=10)
    ax_bar_hilb.set_title(f"CWT Hilbert Space-Filling Curve\n{metric_name}", fontweight='bold', pad=10)
    ax_bar_hilb.set_ylabel(f"Value (dx={dx_spec})")
    ax_bar_hilb.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar_hilb.set_yscale('log')

    # Cleanup and display
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6) # Ensure titles and x-labels don't overlap
    if save_path:
        plt.savefig(save_path)
        plt.close()
    plt.show()

def main():

    ### Arrange the ESEC and Tectonic EQ Master Catalogs
    master_cat = pd.read_csv(CATALOG_PATH_ESEC)
    master_cat['Net_Stn'] = master_cat['StationNetwork']+'.'+master_cat['StationName']
    master_cat_all_types = master_cat['Type']
    master_cat_all_SNR = master_cat['SNR']
    master_cat_stndist = master_cat['StationDistance']
    master_cat_net_stn = master_cat['Net_Stn']

    esec_event_cat = pd.read_csv(ESEC_EVENTS_REF, delimiter='|')
    esec_event_cat['associationId'] = pd.to_numeric(esec_event_cat['associationId'], errors='coerce')

    sepc2vec_df = pd.read_csv(SEPC2VEC_METRICS_ESEC)
    sepc2vec_df['Eventid'] = [label.split('_')[1] for label in sepc2vec_df['label']]
    sepc2vec_df['Eventid'] = pd.to_numeric(sepc2vec_df['Eventid'], errors='coerce')
    sepc2vec_df = sepc2vec_df.merge(esec_event_cat[['associationId', 'subtype']], left_on='Eventid', right_on='associationId', how='left')
    sepc2vec_df = sepc2vec_df.drop(columns=['Eventid','associationId']).copy()
    #print("sepc2vec_df", sepc2vec_df.shape)

    spec2vec_witheqs = pd.read_csv(SEPC2VEC_METRICS_EQS)
    spec2vec_witheqs = spec2vec_witheqs.drop(columns=['TS_0_Mean absolute deviation','FFT_0_Mean absolute deviation','SPEC_0_Mean absolute deviation',
                                                    'TS_0_Root mean square','FFT_0_Root mean square','SPEC_0_Root mean square',
                                                    'TS_renyi_complexity_entropy_short_ordpy','FFT_renyi_complexity_entropy_short_ordpy','SPEC_renyi_complexity_entropy_short_ordpy',
                                                    'TS_renyi_stat_complexity_short_ordpy','FFT_renyi_stat_complexity_short_ordpy','SPEC_renyi_stat_complexity_short_ordpy'])
    spec2vec_witheqs['subtype'] = 'Tectonic Eathquakes'
    spec2vec_witheqs['type'] = 'Tectonic Eathquakes'
    spec2vec_witheqs = spec2vec_witheqs.drop(columns=['type'])
    #print("spec2vec_witheqs", spec2vec_witheqs.shape)

    sepc2vec_df_concat = pd.concat([sepc2vec_df, spec2vec_witheqs], axis=0, ignore_index=True)
    #print("sepc2vec_df_concat", sepc2vec_df_concat.shape)

    ### Load the Saved CWTS Dict For Faster Processing
    save_spectro_dict = np.load(ESEC_CWT_DICT, allow_pickle=True).item()
    save_spectro_dict_eqs = np.load(EQS_CWT_DICT, allow_pickle=True).item()
    MERGED_SPECTRO_DICT = save_spectro_dict | save_spectro_dict_eqs
    del save_spectro_dict, save_spectro_dict_eqs

    ### Plot Figures Loop
    cnt=1
    while cnt<=TOTAL_SAMPLES:
        plot_df_sample = sepc2vec_df_concat[sepc2vec_df_concat['subtype']!='mine collapse'].groupby('subtype').sample(n=1)
        ARCHETYPE_LABELS_TMP = plot_df_sample['label'].tolist()
        ARCHETYPE_SUBTYP_TMP = plot_df_sample['subtype'].tolist()

        for metric in COMPUTE_METRICS:
            fig_file_name = "_".join([str(item) for item in plot_df_sample.index.tolist()])
            _FIG_SAVE_PATH = os.path.join(FIG_SAVE_PATH,f"idxs_{fig_file_name}_{metric}.png")

            _plot_comprehensive_4row_dashboard(ARCHETYPE_LABELS_TMP,ARCHETYPE_SUBTYP_TMP,MERGED_SPECTRO_DICT, fs=20.0, 
                                                dx_spec=DX_SPEC, dx_ts=DX_TS, dx_fft=DX_FFT, spec_noise_thresh=SPEC_NOISE_THRESH,
                                                data_dir_lst=DATA_DIR_LIST, metric_name=metric, save_path=_FIG_SAVE_PATH)
            print(f"Plot for metric {metric} and sample set {cnt} completed")
        
        cnt+=1
        print(f"All plots for sample set {cnt} completed")

    print("Plotting Script Completed")

##---------------------------- Main ----------------------------
if __name__ == "__main__":
    os.makedirs(FIG_SAVE_PATH, exist_ok=True)
    main()
    print("Comprehensive Metrics Dashboard Plotting Completed")


## How to Run
## nohup /home/software/miniconda3/envs/spec_master_dev/bin/python -u /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/optional_comprehensive_metrics_plots.py > /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/terminal_outputs/optional_comprehensive_metrics_plots_prod_test.out 2>&1 &
## 
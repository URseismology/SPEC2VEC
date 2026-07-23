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
from pathlib import Path
from datetime import datetime as dt_log

import scipy.signal as signal
from scipy.signal import stft, windows
from collections import Counter, defaultdict
from scipy.stats import norm
from scipy.fft import fft, ifft

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.append('/data/sswar_files/')
pd.set_option('display.max_columns', None)

from PRJ_SPEC_MASTER.src.utils.data_preparations import *
from PRJ_SPEC_MASTER.src.utils.noise_lib import *
from PRJ_SPEC_MASTER.src.utils.spectograms_lib import *
from PRJ_GIS_QA.src.utils.gisqa_compute_updated import *
from PRJ_GIS_QA.src.utils.helper import *


CATALOG_PATH = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/Master_ESEC_Catalog_vel_all.csv"
DATA_DIR = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/esec_processed_vel"
SPEC2VEC_SAVEPATH = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/computed_dataset_files/esec_spec2vec_rolling_metrics.csv"

TARGET_FS = 20.0
NSTNS_PER_EVENT = 5

SPEC2VEC_FEATURES = [
                    'permutation_entropy_antropy', 
                    'spectral_entropy_antropy',
                    'svd_entropy_antropy', 
                    'petrosian_fd_antropy', 
                    'detrended_fluctuation_antropy', 
                    'hjorth_mobility_antropy',
                    'hjorth_complexity_antropy', 
                    'higuchi_fd_antropy',
                    'normalized_permutation_entropy_ordpy',
                    'statistical_complexity_entropy_ordpy', 
                    'fisher_shannon_ordpy',
                    'global_node_entropy_ordpy', 
                    'renyi_complexity_entropy_short_ordpy',
                    'renyi_stat_complexity_short_ordpy',
                    'renyi_complexity_entropy_long_ordpy',
                    'renyi_stat_complexity_long_ordpy',
                    'renyi_complexity_entropy_3_ordpy',
                    'renyi_stat_complexity_3_ordpy',
                    'tsallis_complexity_entropy_long_ordpy',
                    'tsallis_stat_complexity_long_ordpy',
                    'weighted_permutation_entropy_ordpy', 
                    'missing_links_ordpy',
                    'Absolute energy', 
                    'Average power',
                    'Mean absolute deviation',
                    'Root mean square',
                    'Kurtosis', 
                    'Lempel-Ziv complexity', 
                    'Mean', 
                    'Median',
                    'Skewness', 
                    'Standard deviation',
                    '75th_percentile',
                    '25th_percentile',
                    'label']


def _signalamp_norm(signal_amp):
    signal_amp = 2 * (signal_amp - signal_amp.min()) / (signal_amp.max() - signal_amp.min()) - 1
    return signal_amp

def _compute_cwt(signal, fs, n_scales=80, omega0=5.0, norm='l2',vmin_percentile=2,vmax_percentile=98,f_min=0,f_max=10):
    """CWT with configurable L1/L2 norm."""
    n = len(signal)
    n_pad = int(2 ** np.ceil(np.log2(2 * n)))
    data_pad = np.zeros(n_pad)
    data_pad[:n] = signal
    data_fft = fft(data_pad)

    nyquist = fs / 2.0
    if f_max is None:
        f_max = nyquist * 0.95
    elif f_max > nyquist:
        print(f"Warning: Requested f_max ({f_max}Hz) exceeds the Nyquist limit ({nyquist}Hz). Capping f_max.")
        f_max = nyquist * 0.95
    

    #freqs = np.linspace(0.5, fs/2.0 * 0.95, n_scales)
    freqs = np.linspace(f_min, f_max, n_scales)
    scales = (omega0 * fs) / (2.0 * np.pi * freqs)
    output = np.zeros((len(scales), n), dtype=complex)

    for i, scale in enumerate(scales):
        x = np.arange(n_pad) - (n_pad - 1.0) / 2.0

        # Apply strict scaling norms
        if norm == 'l1':
            scale_factor = scale
        elif norm == 'l2':
            scale_factor = np.sqrt(scale)
        else:
            raise ValueError("Norm must be 'l1' or 'l2'")

        psi = (np.pi ** -0.25) * np.exp(1j * omega0 * x / scale) * np.exp(-0.5 * (x / scale) ** 2) / scale_factor
        psi_fft = fft(np.roll(psi, -(n_pad // 2)))
        output[i, :] = ifft(data_fft * np.conj(psi_fft))[:n]

    spectro = np.abs(output)
    vmax = np.percentile(spectro, vmax_percentile)  
    vmin = np.percentile(spectro, vmin_percentile)  

    spectro[spectro > vmax] = vmax
    spectro[spectro < vmin] = vmin

    return spectro, freqs

def main(filepath: str, outpath: str, spec2vec_cols: list, nstns_per_event=5):
    
    mastercat_vel = pd.read_csv(filepath)
    mastercat_vel_filt = mastercat_vel[(mastercat_vel['DetectHF']==1.0) & (mastercat_vel['if_processed']==True)].copy()
    mastercat_vel_filt = mastercat_vel_filt[(mastercat_vel_filt['SignalSamplingRate']>=20) & (mastercat_vel_filt['SignalSamplingRate']<=200)]
    mastercat_vel_filt = mastercat_vel_filt.sort_values(by=["Eventid","SNR","StationDistance"], ascending=[True, False, True])
    mastercat_vel_filt = mastercat_vel_filt.groupby("Eventid").head(nstns_per_event) 

    if 'Spec2VecProcessed' not in mastercat_vel.columns:
        mastercat_vel['Spec2VecProcessed'] = False

    spec2vec_metrics_df = pd.DataFrame([])
    giq = GISQAPipeline()

    for idx, row in mastercat_vel_filt.iterrows():
        event_id = str(row['Eventid']).replace('.0', '')
        event_type = str(row['Type']).replace(' ', '_').replace('/', '_')
        net = str(row['StationNetwork'])
        sta = str(row['StationName'])
        chan = str(row['StationChannel'])
        dist = str(row['StationDistance'])
        location_code = str(row.get('StationLocationCode'))
        
        spec2vec_label = f"{idx}_{event_id}_{event_type}_{net}_{sta}_{chan}_{location_code}_{dist}"
        filename = f"{event_id}_{event_type}_{net}_{sta}_{chan}_{location_code}_{dist}.sac"
        filepath = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(filepath):
            try:
                st = read(filepath)
                tr = st[0].copy()
                fs = TARGET_FS
                current_fs = tr.stats.sampling_rate
                f_min = 0.5
                f_max = 8

                ### TS Interpolation
                if TARGET_FS!=current_fs:
                    tr.interpolate(sampling_rate=TARGET_FS, method='lanczos', a=20)
                
                tr_data_full = _signalamp_norm(tr.data) 
                samples_per_window = int(TARGET_FS * 30)
                hann_window = np.hanning(samples_per_window)
                
                if len(tr_data_full) < samples_per_window:
                    print(f"Skipping {filepath}: Signal length {len(tr_data_full)/TARGET_FS:.2f}s is less than 30s")
                    continue
                
                ### Global CWT
                spectro, f = _compute_cwt(tr_data_full, fs=fs, n_scales=32, norm='l2',
                                            f_min=f_min,f_max=f_max,vmin_percentile=2,vmax_percentile=98)
                
                spectro_noisefree = spectro.copy()
                spectro_noisefree[spectro_noisefree<np.percentile(spectro_noisefree,50)]=0
                
                num_windows = int(len(tr_data_full) // samples_per_window)
                window_metrics_list = []
                
                for w in range(num_windows):
                    start_idx = w * samples_per_window
                    end_idx = (w + 1) * samples_per_window
                    tr_data = tr_data_full[start_idx:end_idx]
                    
                    ### FFT
                    tr_data_windowed = tr_data * hann_window
                    fft_f, sigfft_tmp = signal_spectra(tr_data_windowed, fs)
                    fft_mask = (fft_f>=1) & (fft_f<=5)
                    sigfft_tmp = sigfft_tmp[fft_mask]
                    
                    spectro_window = spectro_noisefree[:, start_idx:end_idx]

                    ### Spec2Vec Pipeline Pointwise Metrics
                    # TS dx=4, FFT dx=3, SPEC dx=5
                    ts_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[tr_data],
                                                                                    is_hilbertize=0,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                    ordpydx=4, antropydx=4,metrics_list=spec2vec_cols,hilbert_locs=None)
                    fft_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[sigfft_tmp],
                                                                                    is_hilbertize=0,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                    ordpydx=3, antropydx=3,metrics_list=spec2vec_cols,hilbert_locs=None)
                    spec_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[spectro_window],
                                                                                    is_hilbertize=1,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                    ordpydx=5, antropydx=5,metrics_list=spec2vec_cols,hilbert_locs=None)
                    
                    ts_pointwise_metrics.columns = [f"TS_{col}" if col != 'label' else col for col in ts_pointwise_metrics.columns]
                    fft_pointwise_metrics.columns = [f"FFT_{col}" if col != 'label' else col for col in fft_pointwise_metrics.columns]
                    spec_pointwise_metrics.columns = [f"SPEC_{col}" if col != 'label' else col for col in spec_pointwise_metrics.columns]

                    window_df = pd.concat([ts_pointwise_metrics, fft_pointwise_metrics, spec_pointwise_metrics], axis=1)
                    window_df = window_df.loc[:, ~window_df.columns.duplicated()]
                    window_metrics_list.append(window_df.drop(columns=['label']))
                    print()
                
                # Aggregate metrics across all windows
                all_windows_df = pd.concat(window_metrics_list, ignore_index=True)
                
                mean_metrics = all_windows_df.mean().add_prefix('mean_')
                max_metrics = all_windows_df.max().add_prefix('max_')
                
                if num_windows < 2:
                    var_metrics = pd.Series(0.0, index=all_windows_df.columns).add_prefix('var_')
                else:
                    var_metrics = all_windows_df.var().add_prefix('var_')
                
                aggregated_metrics = pd.concat([mean_metrics, max_metrics, var_metrics]).to_frame().T
                aggregated_metrics['label'] = spec2vec_label
                
                cols = [c for c in aggregated_metrics.columns if c != 'label'] + ['label']
                aggregated_metrics = aggregated_metrics[cols]
                
                spec2vec_metrics_df = pd.concat([spec2vec_metrics_df, aggregated_metrics], ignore_index=True)
                
                mastercat_vel.at[idx, 'Spec2VecProcessed'] = True
                print(f"Spec2Vec processed for -> {filepath} with {num_windows} windows")
            
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"{filepath} Do Not Exist")


    spec2vec_metrics_df.to_csv(outpath, index=None)
    mastercat_vel.to_csv(CATALOG_PATH, index=None)

if __name__ == "__main__":
    print(f"[{dt_log.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ESEC Spec2Vec Computation!!!")
    main(filepath=CATALOG_PATH, outpath=SPEC2VEC_SAVEPATH, spec2vec_cols=SPEC2VEC_FEATURES, nstns_per_event=NSTNS_PER_EVENT)  
    print(f"[{dt_log.now().strftime('%Y-%m-%d %H:%M:%S')}] ESEC Spec2Vec Process Completed!!!")  


## How to Run
## nohup /home/software/miniconda3/envs/spec_master_dev/bin/python -u /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/optional_esec_spec2vec_rolling_metrics_prep.py > /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/terminal_outputs/optional_esec_spec2vec_rolling_metrics_prep_prod_test.out 2>&1 &

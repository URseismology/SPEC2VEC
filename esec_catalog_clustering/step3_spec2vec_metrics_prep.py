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

## For Local
sys.path.append('/path/to/my/all_projects') ## e.g. sys.path.append('/data/sswar_files/0_PUBLISHED_CODES')

from SPEC2VEC.utils.simple_synth_data_models import *
from SPEC2VEC.utils.noise_lib import *
from SPEC2VEC.utils.spectograms_lib import *
from SPEC2VEC.utils.gisqa_compute_updated import *
from SPEC2VEC.utils.gisqa_helper import *


#---------------------------- GLOBAL VARIABLES------------------------------
# Set Global Variables
#---------------------------------------------------------------------------
CATALOG_PATH_ESEC = "SPEC2VEC/esec_catalog_clustering/Master_ESEC_Catalog_vel_all.csv"
DATA_DIR_ESEC = "SPEC2VEC/esec_catalog_clustering/esec_processed_vel"
SPEC2VEC_SAVEPATH_ESEC = "SPEC2VEC/esec_catalog_clustering/computed_dataset_files/esec_spec2vec_metrics.csv"
SPATIAL_SAVEPATH_ESEC = "SPEC2VEC/esec_catalog_clustering/computed_dataset_files/esec_spec2vec_metrics_spatial.csv"

CATALOG_PATH_EQ = "SPEC2VEC/esec_catalog_clustering/Master_tectonic_eq_catalog.csv"
DATA_DIR_EQ = "SPEC2VEC/esec_catalog_clustering/eq_processed_vel"
SPEC2VEC_SAVEPATH_EQ = "SPEC2VEC/esec_catalog_clustering/computed_dataset_files/eq_spec2vec_metrics.csv"
SPATIAL_SAVEPATH_EQ = "SPEC2VEC/esec_catalog_clustering/computed_dataset_files/eq_spec2vec_metrics_spatial.csv"


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
                    'tsallis_complexity_entropy_long_ordpy',
                    'tsallis_stat_complexity_long_ordpy',
                    'weighted_permutation_entropy_ordpy', 
                    'missing_links_ordpy',
                    'Absolute energy', 
                    'Average power',
                    'Kurtosis', 
                    'Lempel-Ziv complexity', 
                    'Mean', 
                    'Median',
                    'Skewness', 
                    'Standard deviation',
                    '75th_percentile',
                    '25th_percentile',
                    'label']

##---------------------------- Additional Helper Functions----------------------------
def _signalamp_norm(signal_amp):
    signal_amp = 2 * (signal_amp - signal_amp.min()) / (signal_amp.max() - signal_amp.min()) - 1
    return signal_amp

def _clusterimages(data_orig,ncomp=5,image_thresh=75,isplot=0,ts=None,freqs=None):
    #data = list_images[50]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data_orig.reshape(-1, 1))
    data_nonzero = data[data > np.percentile(data,image_thresh)]

    gmm = GaussianMixture(n_components=ncomp, random_state=6, covariance_type='diag', n_init=5, init_params="k-means++")
    gmm.fit(data_nonzero.reshape(-1, 1))

    plt.figure(figsize=(6, 4))
    plt.hist(data_nonzero, bins=256, density=True, alpha=0.5, label='Histogram (non-zero data)')
    x = np.linspace(data_nonzero.min(), data_nonzero.max(), 1000).reshape(-1, 1)
    pdf = np.zeros_like(x)

    for i in range(ncomp):
        pdf += gmm.weights_[i] * norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0]))

    if isplot:
        plt.loglog(x, pdf, 'r-', linewidth=2, label='GMM Fit (5 components)')
        [plt.axvline(x=gmmean[0], color='k', linestyle='--') for gmmean in gmm.means_]
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Histogram with 5-Component GMM Fit on Filtered Trace')
        plt.legend()
        plt.show()
    else:
        plt.close()

    cut_locations = sorted(gmm.means_.flatten())

    data_0to1 = data_orig
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_0to1 = scaler.fit_transform(data_0to1.reshape(-1, 1)).reshape(data_0to1.shape)

    labeled_spec_image = np.zeros(data_0to1.shape)
    for i in range (0,len(cut_locations)):
        if i==0:
            labeled_spec_image[data_0to1==0] = i
            labeled_spec_image[(data_0to1>0) & (data_0to1<cut_locations[i].item())] = i
        elif i==len(cut_locations)-1:
            labeled_spec_image[(data_0to1>=cut_locations[i].item())] = i+1
        else:
            labeled_spec_image[(data_0to1>=cut_locations[i].item()) & (data_0to1<cut_locations[i+1].item())] = i+1
    labeled_spec_image = labeled_spec_image.astype(int)

    pafe = pairwise_feature_extractor()
    glszm_stg1, glszm_metrics_stg1 = pafe.compute_pixel_localization_metric(labeled_spec_image)

    if isplot:
        if ts is None and freqs is None:
            plot_freqs = np.linspace(0,50,data_orig.shape[0])
            plot_t = np.linspace(0,10,2500) #np.linspace(0,60,60*100)
        else:
            plot_freqs = freqs
            plot_t = ts

        gisqa_plots.plot_labeled_spec(plot_t,plot_freqs, labeled_spec_image, isplot=1, issave=0)
        gisqa_plots.plot_sizezone_variance_of_labeled_spec(glszm_stg1, isplot=1, issave=0)
    else:
        plt.close()

    return glszm_metrics_stg1

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

def _compute_esec_spec2vec(catalogpath: str, outpath: str, spatial_outpath: str, spec2vec_cols: list, nstns_per_event=5):
    
    mastercat_vel = pd.read_csv(catalogpath)
    mastercat_vel_filt = mastercat_vel[(mastercat_vel['DetectHF']==1.0) & (mastercat_vel['if_processed']==True)].copy()
    mastercat_vel_filt = mastercat_vel_filt[(mastercat_vel_filt['SignalSamplingRate']>=20) & (mastercat_vel_filt['SignalSamplingRate']<=200)]
    mastercat_vel_filt = mastercat_vel_filt.sort_values(by=["Eventid","SNR","StationDistance"], ascending=[True, False, True])
    mastercat_vel_filt = mastercat_vel_filt.groupby("Eventid").head(nstns_per_event) 

    if 'Spec2VecProcessed' not in mastercat_vel.columns:
        mastercat_vel['Spec2VecProcessed'] = False

    spec2vec_metrics_df = pd.DataFrame([])
    spat_features_list = []
    for idx, row in mastercat_vel_filt.iterrows():
        event_id = str(row['Eventid']).replace('.0', '')
        event_type = str(row['Type']).replace(' ', '_').replace('/', '_')
        net = str(row['StationNetwork'])
        sta = str(row['StationName'])
        chan = str(row['StationChannel'])
        dist = str(np.round(row['StationDistance'],3))
        location_code = str(row.get('StationLocationCode'))
        
        spec2vec_label = f"{idx}_{event_id}_{event_type}_{net}_{sta}_{chan}_{location_code}_{dist}"
        filename = f"{event_id}_{event_type}_{net}_{sta}_{chan}_{location_code}_{dist}.sac"
        filepath = os.path.join(DATA_DIR_ESEC, filename)
        
        if os.path.exists(filepath):
            try:
                st = read(filepath)
                tr = st[0].copy()
                fs = TARGET_FS
                current_fs = tr.stats.sampling_rate
                f_min = 0.5
                f_max = 8

                ### TS
                if TARGET_FS!=current_fs:
                    tr.interpolate(sampling_rate=TARGET_FS, method='lanczos', a=20)
                tr_data = _signalamp_norm(tr.data) 

                ### FFT
                fft_f, sigfft_tmp = signal_spectra(tr_data, fs)
                fft_mask = (fft_f>=1) & (fft_f<=5)
                sigfft_tmp = sigfft_tmp[fft_mask]
                
                ### CWT
                spectro, f = _compute_cwt(tr_data, fs=fs, n_scales=32, norm='l2',
                                            f_min=f_min,f_max=f_max,vmin_percentile=2,vmax_percentile=98)
                t = np.linspace(0,np.ceil(len(tr_data)/fs), spectro.shape[1])
                
                spectro_noisefree = spectro.copy()
                #spectro_noisefree[spectro_noisefree<np.percentile(spectro_noisefree,50)]=0
                
                ### Spec2Vec Pipeline
                giq = GISQAPipeline()
                ts_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[tr_data],
                                                                                is_hilbertize=0,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                ordpydx=4, antropydx=4,metrics_list=spec2vec_cols,hilbert_locs=None)
                fft_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[sigfft_tmp],
                                                                                is_hilbertize=0,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                ordpydx=4, antropydx=4,metrics_list=spec2vec_cols,hilbert_locs=None)
                spec_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[spectro_noisefree],
                                                                                is_hilbertize=1,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                ordpydx=5, antropydx=5,metrics_list=spec2vec_cols,hilbert_locs=None)
                ts_pointwise_metrics.columns = [f"TS_{col}" if col != 'label' else col for col in ts_pointwise_metrics.columns]
                fft_pointwise_metrics.columns = [f"FFT_{col}" if col != 'label' else col for col in fft_pointwise_metrics.columns]
                spec_pointwise_metrics.columns = [f"SPEC_{col}" if col != 'label' else col for col in spec_pointwise_metrics.columns]

                final_pointwise_metrics = pd.concat([ts_pointwise_metrics, fft_pointwise_metrics, spec_pointwise_metrics], axis=1)
                final_pointwise_metrics = final_pointwise_metrics.loc[:, ~final_pointwise_metrics.columns.duplicated()]
                cols = [c for c in final_pointwise_metrics.columns if c != 'label'] + ['label']
                final_pointwise_metrics = final_pointwise_metrics[cols]
                
                tmp_spat_features = _clusterimages(spectro.astype('float64'), ncomp=5, isplot=0, image_thresh=75)
                
                spec2vec_metrics_df = pd.concat([spec2vec_metrics_df,final_pointwise_metrics],ignore_index=True)
                spat_features_list.append(tmp_spat_features)
                mastercat_vel.at[idx, 'Spec2VecProcessed'] = True
                print(f"Spec2Vec processed for -> {filepath}")
            
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"{filepath} Do Not Exist")

    spec2vec_metrics_df.to_csv(outpath, index=None)
    mastercat_vel.to_csv(catalogpath, index=None)

    spat_features_df = pd.concat(spat_features_list, ignore_index=True)
    spec2vec_metrics_spat_df = pd.concat([spec2vec_metrics_df, spat_features_df], axis=1)
    spec2vec_metrics_spat_df.to_csv(spatial_outpath, index=None)

def _compute_tectonic_eq_spec2vec(catalogpath: str, outpath: str, spatial_outpath: str, spec2vec_cols: list):
    
    mastercat_vel = pd.read_csv(catalogpath)

    if 'Spec2VecProcessed' not in mastercat_vel.columns:
        mastercat_vel['Spec2VecProcessed'] = False

    spec2vec_metrics_df = pd.DataFrame([])
    spat_features_list = []
    
    for idx, row in mastercat_vel.iterrows():
        filepath = str(row["FilePath"])
        spec2vec_label = filepath.split("/")[-1][:-4]

        if os.path.exists(filepath):
            try:
                st = read(filepath)
                sttime = st[0].stats.starttime
                ettime = st[0].stats.endtime
                st.trim(sttime, ettime-15*60, pad='true', fill_value=0)
                tr = st[0].copy()

                fs = TARGET_FS
                current_fs = tr.stats.sampling_rate
                f_min = 0.5
                f_max = 8

                ### TS
                if TARGET_FS!=current_fs:
                    tr.interpolate(sampling_rate=TARGET_FS, method='lanczos', a=20)
                tr_data = _signalamp_norm(tr.data) 

                ### FFT
                fft_f, sigfft_tmp = signal_spectra(tr_data, fs)
                fft_mask = (fft_f>=1) & (fft_f<=5)
                sigfft_tmp = sigfft_tmp[fft_mask]
                
                ### CWT
                spectro, f = _compute_cwt(tr_data, fs=fs, n_scales=32, norm='l2',
                                            f_min=f_min,f_max=f_max,vmin_percentile=2,vmax_percentile=98)
                t = np.linspace(0,np.ceil(len(tr_data)/fs), spectro.shape[1])
                
                spectro_noisefree = spectro.copy()
                #spectro_noisefree[spectro_noisefree<np.percentile(spectro_noisefree,50)]=0
                
                ### Spec2Vec Pipeline
                giq = GISQAPipeline()
                ts_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[tr_data],
                                                                                is_hilbertize=0,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                ordpydx=4, antropydx=4,metrics_list=spec2vec_cols,hilbert_locs=None)
                fft_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[sigfft_tmp],
                                                                                is_hilbertize=0,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                ordpydx=4, antropydx=4,metrics_list=spec2vec_cols,hilbert_locs=None)
                spec_pointwise_metrics = giq.compute_pointwise_metrics_from_spec([spec2vec_label],[spectro_noisefree],
                                                                                is_hilbertize=1,is_normalize_stat=0,is_normalize_entropy=0,is_best_features=0,
                                                                                ordpydx=5, antropydx=5,metrics_list=spec2vec_cols,hilbert_locs=None)
                ts_pointwise_metrics.columns = [f"TS_{col}" if col != 'label' else col for col in ts_pointwise_metrics.columns]
                fft_pointwise_metrics.columns = [f"FFT_{col}" if col != 'label' else col for col in fft_pointwise_metrics.columns]
                spec_pointwise_metrics.columns = [f"SPEC_{col}" if col != 'label' else col for col in spec_pointwise_metrics.columns]

                final_pointwise_metrics = pd.concat([ts_pointwise_metrics, fft_pointwise_metrics, spec_pointwise_metrics], axis=1)
                final_pointwise_metrics = final_pointwise_metrics.loc[:, ~final_pointwise_metrics.columns.duplicated()]
                cols = [c for c in final_pointwise_metrics.columns if c != 'label'] + ['label']
                final_pointwise_metrics = final_pointwise_metrics[cols]
                
                tmp_spat_features = _clusterimages(spectro.astype('float64'), ncomp=5, isplot=0, image_thresh=75)
                
                spec2vec_metrics_df = pd.concat([spec2vec_metrics_df,final_pointwise_metrics],ignore_index=True)
                spat_features_list.append(tmp_spat_features)
                mastercat_vel.at[idx, 'Spec2VecProcessed'] = True
                print(f"Spec2Vec processed for -> {filepath}")
            
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"{filepath} Do Not Exist")

    spec2vec_metrics_df.to_csv(outpath, index=None)
    mastercat_vel.to_csv(catalogpath, index=None)

    spat_features_df = pd.concat(spat_features_list, ignore_index=True)
    spec2vec_metrics_spat_df = pd.concat([spec2vec_metrics_df, spat_features_df], axis=1)
    spec2vec_metrics_spat_df.to_csv(spatial_outpath, index=None)


##---------------------------- Main ----------------------------
if __name__ == "__main__":
    
    print(f"[{dt_log.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ESEC Spec2Vec Computation!!!")  
    _compute_esec_spec2vec(catalogpath=CATALOG_PATH_ESEC, outpath=SPEC2VEC_SAVEPATH_ESEC, spatial_outpath=SPATIAL_SAVEPATH_ESEC, spec2vec_cols=SPEC2VEC_FEATURES, nstns_per_event=NSTNS_PER_EVENT)  
    print(f"[{dt_log.now().strftime('%Y-%m-%d %H:%M:%S')}] ESEC Spec2Vec Process Completed!!!")  
    
    print(f"[{dt_log.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting EQ Spec2Vec Computation!!!")  
    _compute_tectonic_eq_spec2vec(catalogpath=CATALOG_PATH_EQ, outpath=SPEC2VEC_SAVEPATH_EQ, spatial_outpath=SPATIAL_SAVEPATH_EQ, spec2vec_cols=SPEC2VEC_FEATURES)  
    print(f"[{dt_log.now().strftime('%Y-%m-%d %H:%M:%S')}] Tectonic EQ Spec2Vec Process Completed!!!")  


##---------------------------- How to Run ----------------------------
## nohup /home/software/miniconda3/envs/spec_master_dev/bin/python -u /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/step3_spec2vec_metrics_prep.py > /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/terminal_outputs/step3_spec2vec_metrics_prep_prod_test.out 2>&1 &
## 1310695
import pandas as pd
import numpy as np

import sys
import os
import requests
import io
import time
import re
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import scipy.signal as signal
from scipy.signal import stft, windows
from scipy.stats import norm
from scipy.fft import fft, ifft
from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.append('/data/sswar_files/')
from PRJ_SPEC_MASTER.src.utils.data_preparations import *

#---------------------------- GLOBAL VARIABLES------------------------------
# Set Global Variables
#---------------------------------------------------------------------------
CATALOG_PATH_ESEC = "SPEC2VEC/esec_catalog_clustering/Master_ESEC_Catalog_vel_all.csv"
CATALOG_PATH_EQ = "SPEC2VEC/esec_catalog_clustering/Master_tectonic_eq_catalog.csv"

DATA_DIR_ESEC = "SPEC2VEC/esec_catalog_clustering/esec_processed_vel"
DATA_DIR_EQ = "SPEC2VEC/esec_catalog_clustering/eq_processed_vel"

SAVE_FIG_DIR_ESEC = "SPEC2VEC/esec_catalog_clustering/esec_spec2vec_ip_figures"
SAVE_FIG_DIR_EQ = "SPEC2VEC/esec_catalog_clustering/eq_spec2vec_ip_figures"

SAVE_CWT_ARRAYS = "SPEC2VEC/esec_catalog_clustering/computed_dataset_files"

TARGET_FS = 20.0
NSTNS_PER_EVENT = 5

##---------------------------- Helper Functions ----------------------------
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

def _esec_prepare_spec2vec_input_figures_ref():
    mastercat_vel = pd.read_csv(CATALOG_PATH_ESEC)
    all_events = mastercat_vel['Eventid'].unique().tolist()
    mastercat_vel_filt = mastercat_vel[(mastercat_vel['DetectHF']==1.0) & (mastercat_vel['if_processed']==True)].copy()
    mastercat_vel_filt = mastercat_vel_filt[(mastercat_vel_filt['SignalSamplingRate']>=20) & (mastercat_vel_filt['SignalSamplingRate']<=200)]
    mastercat_vel_filt = mastercat_vel_filt.sort_values(by=["Eventid","SNR","StationDistance"], ascending=[True, False, True])
    mastercat_vel_filt = mastercat_vel_filt.groupby("Eventid").head(NSTNS_PER_EVENT) 
    save_spectro_dict = {}

    for evid in all_events:
        
        subplt_cnt=0
        tmp_stft_df = mastercat_vel_filt[mastercat_vel_filt['Eventid']==evid]
        plt.figure(figsize=(14,7))

        for idx, row in tmp_stft_df.iterrows():
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
                    
                    ##TS
                    if TARGET_FS!=current_fs:
                        tr.interpolate(sampling_rate=TARGET_FS, method='lanczos', a=20)
                    tr_data = _signalamp_norm(tr.data)
                    
                    ##FFT
                    fft_f, sigfft_tmp = signal_spectra(tr_data, fs)
                    fft_mask = (fft_f>=1) & (fft_f<=5)
                    
                    ##CWT
                    spectro, cwt_f = _compute_cwt(tr_data, fs=fs, n_scales=32, norm='l2',
                                            f_min=f_min,f_max=f_max,vmin_percentile=2,vmax_percentile=98)
                    cwt_t = np.linspace(0,np.ceil(len(tr_data)/fs), spectro.shape[1])
                    save_spectro_dict[spec2vec_label] = spectro
                    
                    ## Plots
                    plt.subplot(3,5,subplt_cnt+1)
                    plt.plot(tr_data)
                    plt.title(f"{event_id}.{net}.{sta}.{chan}.{location_code}.{dist}", fontsize=8)

                    plt.subplot(3,5,subplt_cnt+6)
                    plt.plot(fft_f[fft_mask],sigfft_tmp[fft_mask])
                    plt.title(f"{event_id}.{net}.{sta}.{chan}.{location_code}.{dist}", fontsize=8)

                    plt.subplot(3,5,subplt_cnt+11)
                    plt.pcolormesh(cwt_t, cwt_f, spectro, cmap='plasma')
                    plt.title(f"{event_id}.{net}.{sta}.{chan}.{location_code}.{dist}", fontsize=8)
                    
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
            else:
                print(f"{filepath} Do Not Exist")
            
            subplt_cnt+=1
        
        savefigname =f"{event_id}_{event_type}.png"
        plt.tight_layout(); 
        plt.savefig(f"{SAVE_FIG_DIR_ESEC}/{savefigname}")
        plt.close()
        print(f"Completed ESEC Figure Prep for {event_id}")
    
    np.save(f"{SAVE_CWT_ARRAYS}/esec_spec2vec_input_cwt_dict.npy",save_spectro_dict)
    print(f"Figure saved at -> {SAVE_FIG_DIR_ESEC}\nand CWT dictionary saved at {SAVE_CWT_ARRAYS}")

def _tectonic_eq_prepare_spec2vec_input_figures_ref():
    mastercat_vel = pd.read_csv(CATALOG_PATH_EQ)
    save_spectro_dict = {}

    for idx, row in mastercat_vel.iterrows():
        plt.figure(figsize=(8,6))

        filepath = str(row["FilePath"])
        savefigname = filepath.split("/")[-1][:-4]
        
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
                sigfft_tmp = sigfft_tmp[fft_mask] ## Adding this filter from CWT V1 Run. Not there in STFT and CWT V0
                fft_f = fft_f[fft_mask]

                ### CWT
                spectro, cwt_f = _compute_cwt(tr_data, fs=fs, n_scales=32, norm='l2',
                                            f_min=f_min,f_max=f_max,vmin_percentile=2,vmax_percentile=98)
                cwt_t = np.linspace(0,np.ceil(len(tr_data)/fs), spectro.shape[1])
                save_spectro_dict[savefigname] = spectro

                ## Prepare Figure
                plt.subplot(3,1,1)
                plt.plot(tr_data)
                
                plt.subplot(3,1,2)
                plt.plot(fft_f,sigfft_tmp)

                plt.subplot(3,1,3)
                plt.pcolormesh(cwt_t, cwt_f, spectro, cmap='plasma')
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"{filepath} Do Not Exist")
        
        plt.suptitle(savefigname, fontsize=10)
        plt.tight_layout(); 
        plt.savefig(f'{SAVE_FIG_DIR_EQ}/{savefigname}',dpi=150); 
        plt.close()
        print(f"Figure saved at -> {SAVE_FIG_DIR_EQ}/{savefigname}")
    
    np.save(f"{SAVE_CWT_ARRAYS}/tectonic_eq_spec2vec_input_cwt_dict.npy",save_spectro_dict)
    print(f"Figure saved at -> {SAVE_FIG_DIR_EQ}\nand CWT dictionary saved at {SAVE_CWT_ARRAYS}")


##---------------------------- Main ----------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_CWT_ARRAYS, exist_ok=True)
    
    os.makedirs(SAVE_FIG_DIR_ESEC, exist_ok=True)
    _esec_prepare_spec2vec_input_figures_ref()

    os.makedirs(SAVE_FIG_DIR_EQ, exist_ok=True)
    _tectonic_eq_prepare_spec2vec_input_figures_ref()
    print("All Input Reference Figures Prepared and Saved")

##---------------------------- How to Run ----------------------------
## nohup /home/software/miniconda3/envs/spec_master_dev/bin/python -u /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/optional_input_figures_ref_prep.py > /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/terminal_outputs/optional_input_figures_ref_prep_prod_test.out 2>&1 &
## 1283811
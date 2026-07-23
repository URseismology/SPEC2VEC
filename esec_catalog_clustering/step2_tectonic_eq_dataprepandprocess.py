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
from scipy.fft import fft, ifft

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from collections import Counter, defaultdict
from scipy.stats import norm
import shap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from scipy.signal import hilbert


#---------------------------- GLOBAL VARIABLES------------------------------
# Set Global Variables
#---------------------------------------------------------------------------
EQ_RAW_DATA_DIR = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/eq_raw_data"
EQ_PROC_DATA_DIR = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/eq_processed_vel"
CATALOG_PATH = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/Master_tectonic_eq_catalog.csv"

##---------------------------- Helper Functions ----------------------------
def _tectonic_eq_dataprocessing():
    
    client = Client("IRIS")
    processed_data_list = []
    fmin = 1.0
    fmax = 5.0

    all_raw_files = glob.glob(os.path.join(EQ_RAW_DATA_DIR,"*.miniseed"))
    assert len(all_raw_files)>0, 'This code needs raw eq sac files to process. Please provide a path that contains raw eq sac files.'

    for raw_file in all_raw_files:
        st_tmp = read(raw_file)
        st_tmp.merge(fill_value='latest')
        st_tmp.sort()
        tr = st_tmp[0]
        
        net = tr.stats.network
        sta = tr.stats.station
        stloc = tr.stats.location
        st_sampling_rate = tr.stats.sampling_rate
        st_npts = tr.stats.npts
        st_chan = tr.stats.channel

        filename_tmp = raw_file.split("/")[-1]
        filename_tmp = filename_tmp[:filename_tmp.find(".")]
        evt_orgn_lst = filename_tmp.split("_")[3:]
        evt_orgn = UTCDateTime(*map(int, evt_orgn_lst))

        processed_filepath = os.path.join(EQ_PROC_DATA_DIR,f"{filename_tmp}.sac")

        try:
            inv = client.get_stations(network=net, station=sta, location='*', 
                                        channel="BHZ", starttime=UTCDateTime(tr.stats.starttime), 
                                        endtime=UTCDateTime(tr.stats.endtime), level="response")
            tr.detrend("demean")
            tr.detrend("linear")
            tr.remove_response(inventory=inv, output="VEL", 
                                pre_filt=[0.005, 0.01, st_sampling_rate/3, st_sampling_rate/2], 
                                zero_mean=True, taper=True, taper_fraction=0.05)
            tr.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
            tr.taper(max_percentage=.05)

            tr.write(processed_filepath, format="SAC")
            
            processed_data_list.append([evt_orgn,net,sta,st_chan,stloc,st_sampling_rate,st_npts,processed_filepath])
            print(f"Processed at -> {processed_filepath}")
        except Exception as e:
            print(f"Error {e} Processing {raw_file}")
            continue

    processed_data_df = pd.DataFrame(processed_data_list,columns=["Event_OriginTime","Network","Station","Channel","Location","Sampling_Rate","NPTS","FilePath"])
    processed_data_df.to_csv(CATALOG_PATH, index=None)


##---------------------------- Main ----------------------------
if __name__ == "__main__":
    os.makedirs(EQ_PROC_DATA_DIR, exist_ok=True)
    _tectonic_eq_dataprocessing()
    print("Tectonic EQ. Data Prep and Processing Completed")

##---------------------------- How to Run ----------------------------
## nohup /home/software/miniconda3/envs/spec_master_dev/bin/python -u /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/step2_tectonic_eq_dataprepandprocess.py > /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/terminal_outputs/step2_tectonic_eq_dataprepandprocess_prod_test.out 2>&1 &
## 1274877
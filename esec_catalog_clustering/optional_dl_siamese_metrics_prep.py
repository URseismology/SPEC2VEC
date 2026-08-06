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
import math
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import scipy.signal as signal
from scipy.signal import stft, windows
from collections import Counter, defaultdict
from scipy.stats import norm
from scipy.fft import fft, ifft

from obspy import read
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

from lightly.data import ImageCollateFunction, LightlyDataset, collate
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimCLRTransform

## Update For Local
sys.path.append('/path/to/my/all_projects') ## e.g. sys.path.append('/data/sswar_files/0_PUBLISHED_CODES')

from SPEC2VEC.utils.simple_synth_data_models import *
from SPEC2VEC.utils.noise_lib import *
from SPEC2VEC.utils.spectograms_lib import *
from SPEC2VEC.utils.gisqa_compute_updated import *
from SPEC2VEC.utils.gisqa_helper import *

CATALOG_PATH = "esec_catalog_clustering/Master_ESEC_Catalog_vel_all.csv"
DATA_DIR = "esec_catalog_clustering/esec_processed_vel"
FIG_SAVE_DIR = "esec_catalog_clustering/figures/siamese_inputs_cwt" ## create this folder if not present already
SIAMESE_SAVE_DIR_EMBED = "esec_catalog_clustering/computed_dataset_files/siam_embeddings_cwt.npy"
SIAMESE_SAVE_DIR_FLNAME = "esec_catalog_clustering/computed_dataset_files/siam_filenames_cwt.npy"
TARGET_FS = 20.0
NSTNS_PER_EVENT = 5
RUN_DATA_PREP = True

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

def _prepare_siamese_dataset(catalog_path: str, fig_save_dir: str, nstns_per_event=5):
    
    mastercat_vel = pd.read_csv(catalog_path)
    mastercat_vel_filt = mastercat_vel[(mastercat_vel['DetectHF']==1.0) & (mastercat_vel['if_processed']==True)].copy()
    mastercat_vel_filt = mastercat_vel_filt[(mastercat_vel_filt['SignalSamplingRate']>=20) & (mastercat_vel_filt['SignalSamplingRate']<=200)]
    mastercat_vel_filt = mastercat_vel_filt.sort_values(by=["Eventid","SNR","StationDistance"], ascending=[True, False, True])
    mastercat_vel_filt = mastercat_vel_filt.groupby("Eventid").head(nstns_per_event) 

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

                ### TS
                if TARGET_FS!=current_fs:
                    tr.interpolate(sampling_rate=TARGET_FS, method='lanczos', a=20)
                tr_data = _signalamp_norm(tr.data) 
                
                ### CWT
                spectro, f = _compute_cwt(tr_data, fs=fs, n_scales=32, norm='l2',
                                            f_min=f_min,f_max=f_max,vmin_percentile=2,vmax_percentile=98)
                t = np.linspace(0,np.ceil(len(tr_data)/fs), spectro.shape[1])

                fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=False, gridspec_kw={'height_ratios': [3, 1], 'hspace':1e-5},  layout='constrained', dpi=300)
                axs[0].axis('off')
                axs[1].axis('off')
                pcm = axs[0].pcolormesh(t, f, spectro, shading='auto', cmap='viridis')
                axs[1].plot(tr_data, color='tab:blue')
                
                plt.savefig(f"{fig_save_dir}/{spec2vec_label}.png")
                plt.close()
                print(f"Simese Data prepared for -> {filepath}")      
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"{filepath} Do Not Exist")
    print('All Inputs for Siamese Completed Successfully')

class SimSiam(nn.Module):
    def __init__(self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)

    def forward(self, x):
        # Get representations
        f = self.backbone(x).flatten(start_dim=1)
        # Get projections
        z = self.projection_head(f)
        # Get predictions
        p = self.prediction_head(z)
        # Stop gradient
        z = z.detach()
        return z, p

def main(embedding_save_dir:str=None, filename_save_dir:str=None):

    ##################
    # Data Load
    ##################
    print("Starting to Prep Data for Siamese Training==============================================")
    collate_fn = ImageCollateFunction(
                                        input_size=256,
                                        normalize=False,
                                        
                                        # flips and rotations
                                        hf_prob=0.5,
                                        vf_prob=0.5,
                                        rr_prob=0.5,
                                        
                                        # slight random cropping
                                        min_scale=0.5,
                                        
                                        # weak color jitter
                                        cj_prob=0.2,
                                        cj_bright=0.1,
                                        cj_contrast=0.1,
                                        cj_hue=0.1,
                                        cj_sat=0.1
                                    )

    transform1 = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    transform2 = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize["mean"],
            std=collate.imagenet_normalize["std"],
        )
    ])

    batch_size = 64
    dataset_train = LightlyDataset(input_dir=FIG_SAVE_DIR)
    traindataloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=24, shuffle=True,  drop_last=False, collate_fn=collate_fn)

    test_data = LightlyDataset(input_dir=FIG_SAVE_DIR, transform=transform1)
    test_data_load = DataLoader(test_data, batch_size=batch_size, num_workers=24, shuffle=False,  drop_last=False)


    ##################
    # Hyperparameters and configuration settings
    ##################
    num_workers = 8
    batch_size = 64
    epochs = 50
    input_size = 256

    # Dimension of the output embeddings
    num_ftrs = 512
    # Dimension of the output of the prediction and projection heads
    out_dim = proj_hidden_dim = 512
    # The prediction head uses a bottleneck architecture
    pred_hidden_dim = 128


    # ResNet18 backbone
    resnet = models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    # Instantiate the SimSiam model
    model = SimSiam(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)

    # Define the loss function
    criterion = NegativeCosineSimilarity()

    # Optimizer
    lr = 0.05 * batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Set the device to GPU if available, otherwise use CPU
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Running on GPU') if device == 'cuda' else print("Running on CPU")
    model.to(device)

    # Variables to track the average loss and output standard deviation
    avg_loss = 0.0
    avg_output_std = 0.0
    filename_list = []
    loss_track = []
    collapse_level_track = []

    ##################
    # Training Loop
    ##################
    print("Starting Training Loop==============================================")
    for e in range(epochs):
        for (x0, x1), _, filenames in traindataloader:

            # Move images to the gpu
            x0 = x0.to(device)
            x1 = x1.to(device)
            filename_list.append(filenames)

            # Get projections (z0 and z1)
            # Get predictions (p0 and p1)
            z0, p0 = model(x0)
            z1, p1 = model(x1)

            # Calculate the loss using negative cosine similarity
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize the output and calculate the standard deviation
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)
            output_std = torch.std(output, 0)
            output_std = output_std.mean()

            # Weighted moving average for loss and output standard deviation
            w = 0.9
            avg_loss = w * avg_loss + (1 - w) * loss.item()
            avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

        # Calculate the collapse level and print the results for each epoch
        collapse_level = max(0.0, 1 - math.sqrt(out_dim) * avg_output_std)
        loss_track.append(avg_loss)
        collapse_level_track.append(collapse_level)
        print(
            f"[Epoch {e:3d}] "
            f"Loss = {avg_loss:.2f} | "
            f"Collapse Level: {collapse_level:.2f} / 1.00"
        )
    
    ##################
    # Create Embedding
    ##################
    print("Training Completed !!! Saving Embedding==============================================")
    embeddings = []
    filenames = []

    # Set the model to evaluation mode (no gradient computation)
    model.eval()
    with torch.no_grad():
        for i, (x, _, fnames) in enumerate(test_data_load):
            # Move images to the gpu
            x = x.to(device)
            # Embed the images with the pre-trained backbone
            y = model.backbone(x).flatten(start_dim=1)
            # Store the embeddings and filenames
            embeddings.append(y)
            filenames = filenames + list(fnames)

    # Concatenate embeddings and convert them to a NumPy array
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    # Saving
    np.save(f'{embedding_save_dir}', embeddings)
    np.save(f'{filename_save_dir}', filenames)


if __name__ == "__main__":
    
    if RUN_DATA_PREP:
        _prepare_siamese_dataset(catalog_path=CATALOG_PATH, fig_save_dir=FIG_SAVE_DIR, nstns_per_event=NSTNS_PER_EVENT)  
   
    main(embedding_save_dir=SIAMESE_SAVE_DIR_EMBED, filename_save_dir=SIAMESE_SAVE_DIR_FLNAME)
    print("Siamese Process Completed!!!")  
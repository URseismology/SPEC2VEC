##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, chirp
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from scipy.spatial.distance import pdist,squareform,cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numba as nb
from scipy import signal
import os
import tempfile
import random
from collections import Counter
from scipy.stats import norm
from math import floor, log as mlog
from warnings import warn
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from scipy.signal import chirp, gausspulse, spectrogram
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from SPEC2VEC.utils.gisqa_compute_updated import *

@dataclass
class SignalConfig:
    """Configuration dataclass for basic signal generation parameters."""
    fs: float = 100.0
    duration: float = 60.0
    seed: int = 42
    noise_level: float = 0.05

class AdvancedSeismicSyntheticsParams:
    """
    Generator for advanced synthetic seismic waveforms simulating various physical phenomena.
    """
    def __init__(self, fs=100.0, duration=60.0, seed=101):
        """
        Initialize the synthetic seismogram generator.

        Args:
            fs (float): Sampling frequency in Hz.
            duration (float): Duration of the signal in seconds.
            seed (int): Random seed for reproducibility.
        """
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, duration, 1/fs)
        np.random.seed(seed)

    def _bandpass(self, data, lowcut, highcut, order=4):
        """
        Applies a Butterworth bandpass filter to the data.
        
        Args:
            data (np.ndarray): Input 1D signal array.
            lowcut (float): Lower frequency bound in Hz.
            highcut (float): Upper frequency bound in Hz.
            order (int): Filter order.
            
        Returns:
            np.ndarray: Bandpass-filtered signal.
        """
        nyq = 0.5 * self.fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)

    def impulsive_earthquake(self, p_onset=15.0, p_dur=3.0, p_freq=12.0, p_decay=2.0, p_amp=0.5,
                            s_onset=22.0, s_dur=8.0, s_freq=4.0, s_decay=1.0, s_amp=1.0,
                            noise_level=0.02):
        """
        Simulates an impulsive earthquake waveform with distinct P and S wave arrivals.
        
        Args:
            p_onset (float): Start time of the P-wave in seconds.
            p_dur (float): Duration of the P-wave in seconds.
            p_freq (float): Dominant frequency of the P-wave in Hz.
            p_decay (float): Exponential decay rate for the P-wave.
            p_amp (float): Amplitude of the P-wave.
            s_onset (float): Start time of the S-wave in seconds.
            s_dur (float): Duration of the S-wave in seconds.
            s_freq (float): Dominant frequency of the S-wave in Hz.
            s_decay (float): Exponential decay rate for the S-wave.
            s_amp (float): Amplitude of the S-wave.
            noise_level (float): Gaussian noise amplitude to add.
            
        Returns:
            tuple: (np.ndarray signal, str label)
        """
        sig = np.zeros_like(self.t)
        p_idx, p_len = int(p_onset * self.fs), int(p_dur * self.fs)
        p_wave = np.sin(2 * np.pi * p_freq * self.t[:p_len]) * np.exp(-self.t[:p_len] * p_decay)
        sig[p_idx:p_idx+p_len] += p_wave * p_amp

        s_idx, s_len = int(s_onset * self.fs), int(s_dur * self.fs)
        s_wave = np.sin(2 * np.pi * s_freq * self.t[:s_len]) * np.exp(-self.t[:s_len] * s_decay)
        sig[s_idx:s_idx+s_len] += s_wave * s_amp
        return sig + np.random.randn(len(self.t)) * noise_level, "Impulsive EQ"

    def volcanic_tornillo(self, onset=10.0, freq=2.5, decay=0.08, amp=1.0, noise_level=0.01):
        """
        Simulates a volcanic tornillo event characterized by a long, monochromatic, slowly decaying coda.
        
        Args:
            onset (float): Start time of the event in seconds.
            freq (float): Dominant frequency of the monochromatic signal in Hz.
            decay (float): Exponential decay rate of the coda.
            amp (float): Amplitude of the signal.
            noise_level (float): Gaussian noise amplitude to add.
            
        Returns:
            tuple: (np.ndarray signal, str label)
        """
        sig = np.zeros_like(self.t)
        onset_idx = int(onset * self.fs)
        coda_len = len(self.t) - onset_idx
        tone = np.sin(2 * np.pi * freq * self.t[:coda_len])
        envelope = np.exp(-self.t[:coda_len] * decay)
        sig[onset_idx:] = amp * tone * envelope
        return sig + np.random.randn(len(self.t)) * noise_level, "Volcanic Tornillo"

    def surface_wave_dispersion(self, start=10.0, end=50.0, f0=0.5, f1=8.0, amp=1.0, noise_level=0.03):
        """
        Simulates dispersive surface waves where frequency changes over time (chirp).
        
        Args:
            start (float): Start time of the dispersion in seconds.
            end (float): End time of the dispersion in seconds.
            f0 (float): Initial frequency at start time in Hz.
            f1 (float): Final frequency at end time in Hz.
            amp (float): Amplitude of the signal.
            noise_level (float): Gaussian noise amplitude to add.
            
        Returns:
            tuple: (np.ndarray signal, str label)
        """
        mask = (self.t >= start) & (self.t <= end)
        sig = np.zeros_like(self.t)
        t_sub = self.t[mask] - start
        dispersed = chirp(t_sub, f0=f0, f1=f1, t1=(end-start), method='linear')
        sig[mask] = amp * dispersed * np.sin(np.pi * t_sub / (end-start))
        #print('noise_level:',noise_level)
        return sig + np.random.randn(len(self.t)) * noise_level, "Surface Dispersion"

    def t_phase_earthquake(self, envelope_center=30.0, envelope_width=10.0, envelope_scale=2.0,
                          hf_low=4.0, hf_high=15.0, amp=2.0, noise_level=0.05):
        """
        Simulates an earthquake T-phase consisting of bandpassed high-frequency noise 
        shaped by a Gaussian envelope.
        
        Args:
            envelope_center (float): Peak time of the Gaussian envelope in seconds.
            envelope_width (float): Width of the Gaussian envelope in seconds.
            envelope_scale (float): Scaling factor for the envelope.
            hf_low (float): Lower frequency bound for the bandpass filter in Hz.
            hf_high (float): Upper frequency bound for the bandpass filter in Hz.
            amp (float): Amplitude of the signal.
            noise_level (float): Gaussian noise amplitude to add.
            
        Returns:
            tuple: (np.ndarray signal, str label)
        """
        noise = np.random.randn(len(self.t))
        hf_signal = self._bandpass(noise, hf_low, hf_high)
        envelope = np.exp(-((self.t - envelope_center) / envelope_width)**2) * (self.t / self.duration)
        sig = hf_signal * envelope * amp
        return sig + np.random.randn(len(self.t)) * noise_level, "T-Phase EQ"

    def strombolian_explosion(self, onset=20.0, decay=2.5, amp=1.5, noise_level=0.05):
        """
        Simulates a Strombolian explosion characterized by a broadband burst with rapid decay.
        
        Args:
            onset (float): Start time of the explosion in seconds.
            decay (float): Exponential decay rate.
            amp (float): Amplitude of the signal.
            noise_level (float): Gaussian noise amplitude to add.
            
        Returns:
            tuple: (np.ndarray signal, str label)
        """
        sig = np.zeros_like(self.t)
        mask = self.t >= onset
        if np.any(mask):
            t_sub = self.t[mask] - onset
            burst = np.random.randn(len(t_sub))
            sig[mask] = burst * np.exp(-t_sub * decay) * amp
        return sig + np.random.randn(len(self.t)) * noise_level, "Strombolian Explosion"

    def hybrid_eruption2(self, tremor_freq=3.0, tremor_amp=0.4, exp_onset=25.0, exp_decay=2.5, exp_amp=2.0, noise_level=0.05):
        """
        Simulates a hybrid eruption consisting of a continuous monochromatic tremor followed by a broadband blast.
        
        Args:
            tremor_freq (float): Frequency of the precursory tremor in Hz.
            tremor_amp (float): Amplitude of the precursory tremor.
            exp_onset (float): Start time of the explosion in seconds.
            exp_decay (float): Exponential decay rate of the explosion.
            exp_amp (float): Amplitude of the explosion.
            noise_level (float): Gaussian noise amplitude to add.
            
        Returns:
            tuple: (np.ndarray signal, str label)
        """
        sig = tremor_amp * np.sin(2 * np.pi * tremor_freq * self.t)
        exp_idx = int(exp_onset * self.fs)
        exp_len = len(self.t) - exp_idx
        if exp_len > 0:
            sig[exp_idx:] += np.random.randn(exp_len) * np.exp(-self.t[:exp_len] * exp_decay) * exp_amp
        return sig + np.random.randn(len(self.t)) * noise_level, "Tremor + Blast"

class GeophysicalSynthesizer:
    """Comprehensive hydroacoustic source library for SOFAR/CTBTO monitoring."""
    
    def __init__(self, fs=100, duration=30):
        """
        Initialize the hydroacoustic signal synthesizer.
        
        Args:
            fs (int): Sampling frequency in Hz.
            duration (float): Length of the simulation in seconds.
        """
        self.fs = fs
        self.dt = 1/fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(fs*duration), dtype=np.float32)
        self.n_samples = len(self.t)
        
    def tphase_eq(self):
        """
        Simulates an Earthquake T-phase signal characterized by hyperbolic dispersion 
        traveling through the SOFAR channel.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        signal = np.zeros_like(self.t)
        # Multiple modal arrivals (dispersive)
        for m in range(1, 4):
            a = 5 + m * 2  # Arrival time increases with mode
            b = 100 * m    # Dispersion strength
            
            f_inst = np.zeros_like(self.t)
            valid = (self.t > a) & (self.t < a + 15)
            # Hyperbolic dispersion: high freq arrives first
            f_inst[valid] = np.sqrt(b / np.maximum(self.t[valid] - a + 0.1, 0.01))
            f_inst = np.clip(f_inst, 2, 50)
            
            phase = 2 * np.pi * np.cumsum(f_inst) / self.fs
            env = np.exp(-((self.t - (a + 3)) / 4)**2)
            signal += env * np.sin(phase) * (0.6 ** m)
        return signal.astype(np.float32)
    
    def volcanic_sofar(self):
        """
        Simulates a volcanic eruption in the SOFAR channel consisting of a 
        long-duration harmonic tremor and a subsequent explosion.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        # Harmonic tremor (resonance in magma conduit)
        tremor = (np.sin(2 * np.pi * 4 * self.t) + 
                 0.5 * np.sin(2 * np.pi * 8 * self.t) +
                 0.25 * np.sin(2 * np.pi * 12 * self.t))
        
        # Amplitude modulation (volcanic "beats")
        modulation = 1 + 0.5 * np.sin(2 * np.pi * 0.3 * self.t)
        tremor = tremor * modulation
        
        # Explosion onset at t=10s
        explosion = np.exp(-((self.t - 10)**2) / 0.5) * np.random.randn(len(self.t))
        explosion = gaussian_filter1d(explosion, sigma=1)
        
        # Combine with different time windows
        env_tremor = np.exp(-((self.t - 15)**2) / 40)
        env_exp = np.exp(-((self.t - 10)**2) / 2)
        
        signal = tremor * env_tremor * 0.3 + explosion * env_exp * 0.8
        return signal.astype(np.float32)
    
    def whale_fin(self):
        """
        Simulates Fin whale vocalizations, typically characterized by 20-Hz pulses 
        with approximately 1 Hz repetition rate.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        pulse_duration = 1.0
        t_pulse = np.linspace(0, pulse_duration, int(self.fs * pulse_duration))
        
        # 20 Hz tone with slight downsweep
        pulse = chirp(t_pulse, f0=22, f1=18, t1=pulse_duration, method='linear')
        pulse *= np.exp(-((t_pulse - 0.5)**2) / 0.1)
        
        # Repeat every 1 second starting at t=5
        signal = np.zeros_like(self.t)
        for start in np.arange(5, 25, 1.0):
            idx = int(start * self.fs)
            if idx + len(pulse) < len(signal):
                signal[idx:idx+len(pulse)] += pulse * 0.7
        
        return signal.astype(np.float32)
    
    def whale_blue(self):
        """
        Simulates Blue whale B-call vocalizations, modeled as a quadratic downsweep 
        from 20 Hz to 12 Hz.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        call = np.zeros_like(self.t)
        # B-call at t=10s
        start = int(10 * self.fs)
        duration_samples = int(2 * self.fs)
        
        if start + duration_samples < len(call):
            t_local = np.linspace(0, 2, duration_samples)
            freq = chirp(t_local, f0=20, f1=12, t1=2, method='quadratic')
            env = np.exp(-((t_local - 1)**2) / 0.5)
            call[start:start+duration_samples] = freq * env * 0.8
        
        return call.astype(np.float32)
    
    def icequake(self):
        """
        Simulates an iceberg calving (icequake) event consisting of impulsive, 
        broadband, short-duration transient arrivals (Ricker wavelets).
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        # Multiple impulsive events
        signal = np.zeros_like(self.t)
        times = [5, 12, 20]
        
        for t0 in times:
            idx = int(t0 * self.fs)
            if idx < len(signal) - 100:
                # Ricker wavelet
                t_local = np.arange(-50, 50) / self.fs
                ricker = (1 - 2*(t_local/0.01)**2) * np.exp(-(t_local/0.01)**2)
                signal[idx-50:idx+50] += ricker * 0.9
        
        return signal.astype(np.float32)
    
    def airgun(self):
        """
        Simulates a seismic airgun array, producing impulsive, periodic, 
        high-amplitude pulses with bubble oscillations.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        signal = np.zeros_like(self.t)
        # Every 10 seconds
        for t0 in np.arange(2, self.duration, 10):
            idx = int(t0 * self.fs)
            if idx < len(signal):
                # Sharp impulse with bubble oscillation
                t_local = np.arange(0, int(0.5*self.fs)) / self.fs
                bubble = np.exp(-t_local/0.1) * np.sin(2 * np.pi * 10 * t_local)
                if idx + len(bubble) < len(signal):
                    signal[idx:idx+len(bubble)] += bubble * 0.8
        return signal.astype(np.float32)
    
    def shipping(self):
        """
        Simulates shipping noise in the ocean, dominated by low-frequency continuous 
        sound with propeller blade-rate modulation.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        # Broadband low freq
        noise = np.random.randn(len(self.t))
        noise = gaussian_filter1d(noise, sigma=5)  # <10 Hz
        
        # Propeller modulation (blade rate ~2 Hz)
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * self.t)
        
        # Slow variations
        slow_env = 1 + 0.2 * np.sin(2 * np.pi * 0.1 * self.t)
        
        return (noise * modulation * slow_env * 0.4).astype(np.float32)
    
    def calibration_tone(self):
        """
        Simulates an intermittent pure 37 Hz calibration tone, a frequency commonly 
        used in hydrophone calibration.
        
        Returns:
            np.ndarray: Simulated acoustic waveform.
        """
        tone = np.sin(2 * np.pi * 37 * self.t)
        # Intermittent
        env = np.zeros_like(self.t)
        for start in np.arange(0, self.duration, 5):
            mask = (self.t >= start) & (self.t < start + 2)
            env[mask] = 1
        return (tone * env * 0.5).astype(np.float32)
    
    def generate_all(self):
        """
        Generates all configured hydroacoustic sources and returns them in a labeled dictionary.
        
        Returns:
            dict: Mapping of event labels to their corresponding simulated waveform arrays.
        """
        return {
            'T-Phase (Eq)': self.tphase_eq(),
            'Volcanic Tremor': self.volcanic_sofar(),
            'Fin Whale': self.whale_fin(),
            'Blue Whale': self.whale_blue(),
            'Icequake': self.icequake(),
            'Airgun': self.airgun(),
            'Shipping': self.shipping(),
            'Calib 37Hz': self.calibration_tone()
        }


 

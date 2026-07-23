##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import chirp
from scipy.fft import fft
import datetime
import pandas as pd
import h5py


def signal_spectra(sig,fs,L=None):
    """ 
    Generate the FFT of a Signal:
    Cite: https://www.mathworks.com/help/matlab/ref/fft.html
    
    Args:
        sig (np.ndarray): signal vector
        fs (float): sampling frequency in Hz
        L (int): number of samples in the signal
        
    Returns:
        f (np.ndarray): frequency vector.
        sig_fft (np.ndarray): signal spectra vector.
    """
    L = L if L else len(sig)
    f = (fs/L)*np.array(range(0,int(L/2)))
    sig_fft = np.abs(fft(sig)/L)
    sig_fft = sig_fft[0:int(L/2)]
    sig_fft[1:-1] = 2*sig_fft[1:-1]
    return f, sig_fft

def gen_additive_synthetic_signal(fs,T,w,a):
    """ 
    Generate Synthetic Additive Signal: sin(A1) + sin(A2) + .... + sin(AN)
    
    Args:
        fs (float): Sampling frequency in Hz.
        T (float): Length of signal in seconds.
        w (float or np.ndarray): Angular Frequency (omega), scalar or 1D array.
        a (float or np.ndarray): Amplitude, scalar or 1D array matching w.
        
    Returns:
        t (np.ndarray): Time vector.
        additive_signal (np.ndarray): Resulting additive signal.
    """
    assert np.isscalar(w) or isinstance(w, np.ndarray), "w must be a scalar or a numpy array"
    assert np.isscalar(a) or isinstance(a, np.ndarray), "a must be a scalar or a numpy array"
    
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    L = int(fs*T)

    if np.isscalar(w) and np.isscalar(a):
        additive_signal =  a * np.sin(2 * np.pi * w * t)
    else:
        assert np.shape(a) == np.shape(w), "a and w must have the same dimensions"
        w = np.reshape(w, (-1, 1))
        a = np.reshape(a, (-1, 1))
        additive_signal = (a * np.sin(2 * np.pi * w * t)).sum(axis=0)
    
    return t, additive_signal

def gen_multiplicative_synthetic_signal(fs,T,w,a,type):
    """ 
    Generate Synthetic Multiplicative Signal:
    
    Args:
        fs (float): Sampling frequency in Hz.
        T (float): Length of signal in seconds.
        w (float or np.ndarray): Frequency (omega), scalar or 1D array.
        a (float or np.ndarray): Amplitude, scalar or 1D array matching w.
        type (int): can be either 1 or 2 and generates signal as per below equations
            1: x(t) = sinAsinB...sinN
            2: x(t) = sinA(1 + eps*sinB)
    Returns:
        t (np.ndarray): Time vector.
        mul_signal (np.ndarray): Resulting multiplicative signal.
    """

    assert isinstance(w, np.ndarray), "w must be a numpy array of shape (1, N)"
    assert isinstance(a, np.ndarray), "a must be a numpy array of shape (1, N)"
    assert np.shape(a) == np.shape(w), "a and w must have the same dimensions"

    t = np.linspace(0, T, int(T * fs), endpoint=False)
    L = int(fs*T)
    w = np.reshape(w, (-1, 1))
    a = np.reshape(a, (-1, 1))

    if type==1:
        mul_signal = a * np.sin(2 * np.pi * w * t)
        mul_signal = np.prod(mul_signal, axis=0)
    elif type==2:
        eps = 0.2
        assert len(w)==2, "Type 2 can only work for two different frequencies"
        assert w[0]>w[1], "w[0] must be greater than w[1]"
        mul_signal = np.sin(2 * np.pi * w[0] * t)*(1 + eps*np.sin(2 * np.pi * w[1] * t))

    return t, mul_signal

def gen_chirp_synthetic_signal(fs,T,w):
    """ 
    Generate Synthetic Chirp Signal:
    
    Args:
        fs (float): Sampling frequency in Hz.
        T (float): Length of signal in seconds.
        w (np.ndarray): Frequency (omega), 1D array of length 2.
        
    Returns:
        t (np.ndarray): Time vector.
        chirp_signal (np.ndarray): Resulting chirp signal.
    """

    assert isinstance(w, (list, np.ndarray)) and len(w) == 2, "w must be a list or array of length 2"
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    chirp_signal = chirp(t, f0=w[0], f1=w[1], t1=T, method='linear')

    return t, chirp_signal

def gen_narrowband_amfm_signal(fs,T,type):
    """
    Generate a Synthetic Narrowband AM-FM Signal.
    
    Based on the nonstationary signal models from:
    "Data-driven nonstationary signal decomposition approaches: a comparative analysis"
    Eriksen & Rehman, https://www.nature.com/articles/s41598-023-28390-w#Sec2

    Args:
        fs (float): Sampling frequency in Hz.
        T (float): Length of signal in seconds.
        type (int): Placeholder argument for signal type (currently unused in implementation).
        
    Returns:
        t (np.ndarray): Time vector.
        s (np.ndarray): Resulting composite AM-FM synthetic signal.
    """

    t = np.linspace(0, T, int(fs * T), endpoint=False)

    s11 = (1 + 0.2 * np.cos(t)) * np.cos(30 * np.pi * (2 * t + 0.3 * np.cos(t)))
    s12 = (1 + 0.3 * np.cos(2 * t)) * np.exp(-t / 15) * np.cos(30 * np.pi * (2.4 * t + 0.5 * t**1.2 + 0.3 * np.sin(t)))
    s13 = np.cos(30 * np.pi * (5.3 * t + 0.2 * t**1.3))
    s = s11 + s12 + s13

    return t, s

def float_to_grayscale(data, scale_255=True):
  """
  Converts a floating point 2D array to a grayscale image array.

  Parameters:
      data (np.ndarray): 2D array of floats
      scale_255 (bool): If True, output is uint8 in range [0, 255]
                        If False, output remains float in range [0, 1]

  Returns:
      np.ndarray: Normalized grayscale array
  """
  norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize to [0,1]
  if scale_255:
      return (norm_data * 255).astype(np.uint8)
  else:
      return norm_data




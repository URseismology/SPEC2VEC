##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
from scipy import signal
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
import numba as nb
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d,convolve1d


def stft_basic_spectogram(t_seg_data, sr, nperseg, overlap, window, f_min, f_max, max_normalize=True, 
                          powerlog=True, normalize_range=(1e-8,1), vmin_percentile=5, vmax_percentile=100):
    """
    Computes the Short-Time Fourier Transform (STFT) spectrogram of a given signal with scipy spectrogram function.

    Args:
        t_seg_data (np.ndarray): The 1D input time-series signal.
        sr (int or float): Sampling rate of the signal in Hz.
        nperseg (int): Number of samples per segment for the STFT.
        overlap (float): Fractional overlap between segments (e.g., 0.5 for 50% overlap).
        window (str or tuple): Desired window to use (e.g., 'hann').
        f_min (float): Minimum frequency to retain in the output spectrogram.
        f_max (float): Maximum frequency to retain in the output spectrogram.
        max_normalize (bool): If True, normalizes the spectrogram values using MinMaxScaler.
        powerlog (bool): If True, converts the magnitude spectrogram to a decibel (log) scale.
        normalize_range (tuple): The (min, max) range for normalization.
        vmin_percentile (int): Percentile to use as the lower bound (clips lower values).
        vmax_percentile (int): Percentile to use as the upper bound (clips higher values).

    Returns:
        f (np.ndarray): Frequency array.
        t (np.ndarray): Time array.
        spectro (np.ndarray): 2D spectrogram matrix.
    """
    f, t, spectro = signal.spectrogram(t_seg_data, fs=sr, nperseg=nperseg, noverlap=int(nperseg * overlap), 
                                       window=window, mode='magnitude') # scaling='spectrum'

    f_min_idx = (np.abs(f_min - f)).argmin()
    spectro = spectro[f_min_idx:]
    f = f[f_min_idx:]

    if f_max:
        f_max_idx = max((np.abs(f_max - f)).argmin(), f_min_idx)
        spectro = spectro[:f_max_idx]
        f = f[:f_max_idx]

    spectro_shape = spectro.shape
    if max_normalize:
      spectro = MinMaxScaler(feature_range=normalize_range).fit_transform(spectro.reshape(-1, 1)).reshape(spectro_shape)

    if powerlog:
      spectro = 10 * np.log10(spectro) 

    vmax = np.percentile(spectro, vmax_percentile)  
    vmin = np.percentile(spectro, vmin_percentile)  

    spectro[spectro > vmax] = vmax
    spectro[spectro < vmin] = vmin
    f = f[::-1]
    spectro = spectro[::-1]

    return f, t, spectro

def cwt_simple(signal:np.ndarray, sr:int=1000, dt:float=None, fscale:dict={'start':1, 'end':512, 'num':100}, wavelet:str="cmor1.5-1.0", 
                vmin_percentile:int=2, vmax_percentile:int=98, f_min:int=0, f_max:int=None, max_normalize=True, 
                fscaletype='linear', powerlog=False,
                normalize_range=(1e-8,1), decimate_factor=None):
    """
    Computes a Continuous Wavelet Transform (CWT) spectrogram using pywt library.

    Args:
        signal (np.ndarray): 1D input time-series signal.
        sr (int): Sampling rate in Hz. Default is 1000.
        dt (float): Sampling period. If None, calculated from `sr`.
        fscale (dict): Dictionary specifying frequency scale parameters ('start', 'end', 'num').
        wavelet (str): Name of the continuous wavelet to use (default: 'cmor1.5-1.0').
        vmin_percentile (int): Lower bound clipping percentile.
        vmax_percentile (int): Upper bound clipping percentile.
        f_min (int): Minimum frequency to retain.
        f_max (int): Maximum frequency to retain.
        max_normalize (bool): If True, scales values using MinMaxScaler.
        fscaletype (str): 'linear' or 'log' frequency scaling.
        powerlog (bool): If True, applies 10*log10 scaling.
        normalize_range (tuple): Min/max target range for normalization.
        decimate_factor (int, optional): Factor by which to decimate the output time axis.

    Returns:
        f (np.ndarray): Frequency array.
        t (np.ndarray): Time array.
        spectro (np.ndarray): 2D real-valued magnitude spectrogram matrix.
        cwtmatr_cmplx (np.ndarray): Raw complex CWT coefficient matrix.
    """
    assert signal.ndim==1, 'signal must be 1 dimesnional' 
    length = int(np.ceil(len(signal)/sr))

    if dt:
       sampling_period = dt
       t = np.arange(0,len(signal)*dt,dt)
    else:
      t = np.linspace(0, length, sr*length)
      sampling_period = dt if dt else np.diff(t).mean()
    
    if fscaletype=='linear':
      widths = np.arange(fscale['start'], fscale['end'], fscale['num'])
    elif fscaletype=='log':
      widths = np.geomspace(fscale['start'], fscale['end'], num=fscale['num'])
    else:
      raise ValueError('Freq. Scale type must be either linear or log')
    
    cwtmatr_cmplx, f = pywt.cwt(signal, widths, wavelet, sampling_period=sampling_period)
    spectro = np.abs(cwtmatr_cmplx[:-1, :-1]); 
    t = t[:-1]; f = f[:-1]
    #print(f.min(), f.max())

    if decimate_factor:
      spectro = spectro[:, ::decimate_factor]
      t = np.linspace(0, length, spectro.shape[1])
      #t = t[::decimate_factor]

    spectro = spectro[::-1]; 
    f = f[::-1]
    
    f_min_idx = (np.abs(f_min - f)).argmin()
    spectro = spectro[f_min_idx:]
    f = f[f_min_idx:]

    if f_max:
      f_max_idx = max((np.abs(f_max - f)).argmin(), f_min_idx)
      spectro = spectro[:f_max_idx]
      f = f[:f_max_idx]

    spectro_shape = spectro.shape
    if max_normalize:
      spectro = MinMaxScaler(feature_range=normalize_range).fit_transform(spectro.reshape(-1, 1)).reshape(spectro_shape)

    if powerlog:
      spectro = 10 * np.log10(spectro)

    vmax = np.percentile(spectro, vmax_percentile)  
    vmin = np.percentile(spectro, vmin_percentile)    

    spectro[spectro > vmax] = vmax
    spectro[spectro < vmin] = vmin

    f = f[::-1]
    spectro = spectro[::-1]

    return f, t, spectro, cwtmatr_cmplx

def cwt_analytic(signal, fs, n_scales=80, omega0=5.0, norm='l2',vmin_percentile=2,vmax_percentile=98,f_min=0,f_max=10):
    """
    Computes the Continuous Wavelet Transform (CWT) using an analytic Morlet wavelet via FFT, 
    with configurable L1/L2 norm scaling.
    
    Args:
        signal (np.ndarray): 1D input signal.
        fs (float): Sampling frequency.
        n_scales (int): Number of frequency scales to generate.
        omega0 (float): Non-dimensional frequency parameter of the Morlet wavelet.
        norm (str): Wavelet scaling norm, either 'l1' or 'l2'.
        vmin_percentile (int): Lower clipping percentile.
        vmax_percentile (int): Upper clipping percentile.
        f_min (float): Minimum frequency to compute.
        f_max (float): Maximum frequency to compute.
        
    Returns:
        spectro (np.ndarray): 2D magnitude spectrogram matrix.
        freqs (np.ndarray): 1D array of calculated frequencies.
    """
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

def _get_gray_order(level):
    """
    Generates a sequency-ordered index array (Gray code mapping) for wavelet packet nodes.
    Used to reorder the wavelet packet frequency bands monotonically. Used for Maximal Overlap Wavelet Packet Transform (MOWPT)
    computations.
    
    Args:
        level (int): Decomposition level.
        
    Returns:
        order (list): Reordered indices for the nodes at the given level.
    """
    order = [0]
    for i in range(1, level + 1):
        order = order + [2**i - 1 - x for x in order]
    return order

def _upsample_filter(f, level):
    """
    Upsamples a discrete filter by inserting zeros between elements (à trous algorithm).
    Used for Maximal Overlap Wavelet Packet Transform (MOWPT) computations.
    
    Args:
        f (np.ndarray): 1D filter array.
        level (int): Target wavelet decomposition level determining zero-padding length.
        
    Returns:
        up (np.ndarray): Upsampled filter array.
    """
    zeros = 2**(level - 1) - 1
    if zeros == 0: return np.array(f)
    up = np.zeros(len(f) + (len(f) - 1) * zeros)
    up[::zeros + 1] = f
    return up

@nb.njit(fastmath=True)
def _atrous_circular_convolve(x, h, step):
    """
    Numba JIT compiled periodic circular convolution using the 'à trous'
    stride algorithm. Bypasses explicit zero-padding for massive speedup.
    
    Args:
        x (np.ndarray): Input 1D array.
        h (np.ndarray): Filter array.
        step (int): Stride interval (2**(j-1) for level j).
        
    Returns:
        out (np.ndarray): Convolved output array of the same length as x.
    """
    N = len(x)
    K = len(h)
    out = np.zeros(N)
    for i in range(N):
        val = 0.0
        for j in range(K):
            # Circular boundary condition with dynamic stride
            idx = (i - j * step) % N
            val += x[idx] * h[j]
        out[i] = val
    return out

def compute_modwt_matrix(signal, level=5, wavelet='sym4'):
    """
    Computes the Maximal Overlap Discrete Wavelet Transform (MODWT).
    MODWT is inherently L2 normalized by the QMF filters.
    
    Args:
        signal (np.ndarray): 1D input signal.
        level (int): Number of decomposition levels.
        wavelet (str): PyWavelets wavelet name (e.g., 'sym4').
        
    Returns:
        np.ndarray: 2D matrix of MODWT detail coefficients.
    """
    n = len(signal)
    mod = n % (2 ** level)
    sig_pad = np.pad(signal, (0, (2**level) - mod), mode='symmetric') if mod > 0 else signal
    swt_coeffs = pywt.swt(sig_pad, wavelet, level=level, trim_approx=True)
    return np.vstack([np.abs(c)[:n] for c in swt_coeffs])

def compute_mowpt_matrix(signal, level=5, wavelet='sym4', norm='l2'):
    """
    Computes the Maximal Overlap Wavelet Packet Transform (MOWPT) utilizing explicitly 
    upsampled filters and periodic circular convolution.
    
    Args:
        signal (np.ndarray): 1D input signal.
        level (int): Number of decomposition levels.
        wavelet (str): PyWavelets wavelet name.
        norm (str): Wavelet scaling norm, either 'l1' or 'l2'.
        
    Returns:
        np.ndarray: 2D matrix of MOWPT coefficients, reordered by frequency.
    """
    wavelet_obj = pywt.Wavelet(wavelet)
    h = np.array(wavelet_obj.dec_lo) / np.sqrt(2)
    g = np.array(wavelet_obj.dec_hi) / np.sqrt(2)

    nodes = [signal]
    for j in range(1, level + 1):
        h_up = _upsample_filter(h, j)
        g_up = _upsample_filter(g, j)

        next_nodes = []
        for node in nodes:
            approx = convolve1d(node, h_up, mode='wrap', origin=0)
            detail = convolve1d(node, g_up, mode='wrap', origin=0)
            next_nodes.extend([approx, detail])
        nodes = next_nodes

    freq_order = _get_gray_order(level)
    matrix = np.vstack([np.abs(nodes[i]) for i in freq_order])

    if norm == 'l1':
        matrix *= (2 ** (level / 2.0))

    return matrix

def compute_mowpt_jit(signal, level=5, wavelet='sym4', norm='l2'):
    """
    Computes the Maximal Overlap Wavelet Packet Transform (MOWPT) utilizing a fast, 
    JIT-compiled à trous algorithm for circular convolution.
    
    Args:
        signal (np.ndarray): 1D input signal.
        level (int): Number of decomposition levels.
        wavelet (str): PyWavelets wavelet name.
        norm (str): Wavelet scaling norm, either 'l1' or 'l2'.
        
    Returns:
        np.ndarray: 2D matrix of raw linear MOWPT coefficients, reordered by frequency.
    """
    wavelet_obj = pywt.Wavelet(wavelet)
    h = np.array(wavelet_obj.dec_lo) / np.sqrt(2)
    g = np.array(wavelet_obj.dec_hi) / np.sqrt(2)

    nodes = [signal]
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        next_nodes = []
        for node in nodes:
            approx = _atrous_circular_convolve(node, h, step)
            detail = _atrous_circular_convolve(node, g, step)
            next_nodes.extend([approx, detail])
        nodes = next_nodes

    freq_order = _get_gray_order(level)
    matrix = np.vstack([np.abs(nodes[i]) for i in freq_order])

    if norm == 'l1':
        matrix *= (2 ** (level / 2.0))
    
    # Return raw linear coefficients
    return matrix


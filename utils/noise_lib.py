##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
import numpy as np
from scipy import signal
from scipy.signal import chirp
from scipy.interpolate import interp1d
from scipy.fft import fft


def white_noise_series(L, amp = 1, mu = 0, sd = 1):
    """ 
    Generate White Noise Time Series Data.

    Args:
        L (int): Length of the time series (number of samples).
        amp (float): Amplitude multiplier of the noise.
        mu (float): Mean of the normal distribution.
        sd (float): Standard deviation of the normal distribution.

    Returns:
        np.ndarray: Generated white noise signal array.
    """

    return amp*np.random.normal(mu,sd,L)

def ar_driven_noise_series(T = 2, sampling_rate = 1000, P = 1, 
                           a_k = [1, -2.2137, 2.9403, -2.1697, 0.9606], 
                           fmin = None, fmax = None):
    """ 
    Generate Noise Time Series Data from a linear system whose parameters are autoregressive 
    coefficients and its output is driven by white noise.

    Args:
        T (float): Length of the time series in seconds.
        sampling_rate (int): Sampling rate of the output time series in Hz.
        P (float): Variance of the driving white noise.
        a_k (list or np.ndarray): Autoregressive coefficients (starting with 1).
        fmin (float, optional): Minimum frequency to compute PSD for.
        fmax (float, optional): Maximum frequency to compute PSD for.

    Returns:
        time_series (np.ndarray): The synthesized time series signal.
        times (np.ndarray): The corresponding time vector.
        frequency_series (np.ndarray): Complex Fourier series of the signal.
        frequencies (np.ndarray): Frequency vector.
        psd (np.ndarray): Power Spectral Density (PSD) array.
    """

    assert ((not any(isinstance(i, list) for i in a_k)) and len(a_k)>1), 'a_k must be a vector of type numpy array or list starting with 1'
    if a_k[0] != 1:
        assert isinstance(a_k, np.ndarray), "a_k must be a numpy array type vector starting with 1"
        a_k = np.r_[1, -a_k]
    assert np.all(np.abs(np.roots(a_k)) < 1), 'The roots of the ar coeffcients must be less than 1 to generate stable system'

    dt = 1/sampling_rate
    df = 1 / T
    N = int(sampling_rate * T)
    times = np.linspace(0., T , N) 
    if fmin == None: fmin = 0
    if fmax == None: fmax = (N / 2) / T
    kmin = int(fmin/df)
    kmax = int(fmax/df) + 1

    # Formula spectrum S(f) from 
    # https://www.r-bloggers.com/2023/07/autoregressive-moving-average-models-and-power-spectral-densities/ 
    # https://www.mathworks.com/help/signal/ug/parametric-methods.html#f12-19188
    frequencies = df * np.linspace(kmin, kmax, int(N / 2 + 1))
    den = np.fft.fft(a_k, n=N)
    spectrum = dt * P / (np.abs(den) ** 2)
    f_spec, spec = np.fft.fftfreq(N,dt), spectrum

    psd = np.interp(frequencies, f_spec[:int(N/2+0.5)], spec.real[:int(N/2+0.5)], left = 0., right = 0.)
    sigma = np.sqrt(psd /  df * .5) # Still a real number
    
    # Performing element wise multiplication with sigma vector, making it into complex exponential, 
    # Adding random phase, building the whole spectra
    frequency_series = np.einsum('ij,j -> ij', 
                                 np.random.normal(0, 1, (1,len(sigma))) + 1j * np.random.normal(0, 1, (1,len(sigma))), 
                                 sigma)
    time_series = np.fft.irfft(frequency_series) * df * N
    time_series = np.squeeze(time_series)
    frequency_series = np.squeeze(frequency_series)
    
    return time_series, times, frequency_series, frequencies, psd

def synthesize_from_psd_rfft(a_k, P, sampling_rate, T, seed=None):
    """
    Synthesize a real time series of duration T (s) and sampling_rate whose PSD is 
    S(f) = dt*P / |A(e^{j*omega})|^2 where A is defined by a_k.
    
    This is a modified version of ar_driven_noise_series function above. 
    Kept here mainly for educational purposes.

    Args:
        a_k (np.ndarray or list): 1D array of AR denominator coefficients (e.g., [1, -a1, -a2, ...]).
        P (float): Driving white noise variance.
        sampling_rate (int): Sampling frequency in Hz.
        T (float): Duration of the signal in seconds.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        time_series (np.ndarray): The synthesized real time series signal.
        freqs (np.ndarray): Positive frequency axis (rfft bins).
        S_pos (np.ndarray): One-sided PSD estimate at the frequency bins.
    """

    rng = np.random.default_rng(seed)
    dt = 1.0 / sampling_rate
    N = int(round(sampling_rate * T))
    df = 1.0 / T
    n_pos = N // 2 + 1                # number of rfft bins (0 ... Nyquist)
    freqs = np.arange(n_pos) * df     # positive freq axis, length n_pos

    # Frequency response of A at rfft bins (use rfft for direct positive bins)
    A_pos = np.fft.rfft(a_k, n=N)     # length n_pos
    S_pos = (dt * P) / (np.abs(A_pos) ** 2)   # one-sided PSD estimate at these bins

    # Build rfft-spectrum `spec` such that irfft(spec) yields a real time series
    spec = np.zeros(n_pos, dtype=np.complex128)

    # DC term (k=0): real-valued Gaussian with variance = S_pos[0] * N * df
    # The scaling S_pos[0] * N * df here follows the Parseval/DFT-energy bookkeeping:
    # the variance of Fourier bin amplitudes times appropriate scale leads to the expected PSD. Same reasoning applies in other places.
    spec[0] = np.sqrt(S_pos[0] * N * df) * rng.normal()

    # Nyquist term (only if N is even): real-valued
    if N % 2 == 0:
        spec[-1] = np.sqrt(S_pos[-1] * N * df) * rng.normal()

    # Positive frequencies (1 .. n_pos-2): complex Gaussian with variance such that
    # E[|spec[k]|^2] = (S_pos[k] * N * df)  (full two-sided energy allocation)
    # For each positive k we create (a+jb) where a,b ~ N(0, 0.5 * S_pos[k] * N * df)
    if n_pos > 2:
        amp = np.sqrt(0.5 * S_pos[1:-1] * N * df)
        a = rng.normal(size=amp.shape)
        b = rng.normal(size=amp.shape)
        spec[1:-1] = amp * (a + 1j * b)

    # Inverse real FFT -> time series
    time_series = np.fft.irfft(spec, n=N)

    return time_series, freqs, S_pos

def psd_driven_noise_series(f:np.ndarray, PSD:np.ndarray, T = 2, sampling_rate = 1000, fmin = None, fmax = None):
    """ 
    Generate Noise Time Series Data directly from a custom designed PSD series.

    Args:
        f (np.ndarray): A vector of uniformly separated frequencies over which PSD is defined.
        PSD (np.ndarray): A vector of PSD values corresponding to `f`.
        T (float): Total length of the desired signal in seconds.
        sampling_rate (int): Sampling rate of the desired signal in Hz.
        fmin (float, optional): Minimum frequency to compute.
        fmax (float, optional): Maximum frequency to compute.

    Returns:
        times (np.ndarray): The corresponding time vector.
        time_series (np.ndarray): The synthesized real time series signal.
        frequencies (np.ndarray): Frequency vector matching the signal's spectral length.
        frequency_series (np.ndarray): Complex Fourier series of the signal.
        psd (np.ndarray): Power Spectral Density interpolated at the frequency bins.
    """
    
    psd_int = interp1d(f, PSD, bounds_error=False, fill_value='extrapolate')
    df = 1/T
    N = int(sampling_rate * T)
    times = np.linspace(0, T, N) 
    if fmin == None: fmin = 0
    if fmax == None: fmax = (N / 2) / T
    kmin = int(fmin/df)
    kmax = int(fmax/df) + 1
    
    frequencies = df * np.arange(kmin, kmax)
    frequency_series = np.zeros(len(frequencies), dtype = np.complex128)


    sigma = np.sqrt(psd_int(frequencies) /  df * .5) 
    frequency_series = sigma * (np.random.normal(0, 1, len(sigma)) + 1j * np.random.normal(0, 1, len(sigma)))

    time_series = np.fft.irfft(frequency_series, n=N) * df * N
    return times, time_series, frequencies, frequency_series, psd_int(frequencies)

class ColoredNoiseGenerator:
    """ 
    Generate Noise Time Series of Different Colors Such as Pink, Red, Blue, etc.
    Available colors: white, blue, violet, brownian, pink, and higher_order_noise (3, 4, 5).
    """

    def __init__(self, N):
        """
        Args:
            N (int): Number of samples of the desired series.
        """
        self.N = N
        self.freqs = np.fft.rfftfreq(N)

    def generate(self, psd_func):
        """
        Generates the colored noise given a PSD function.

        Args:
            psd_func (callable): Function that takes frequency array and returns PSD values.

        Returns:
            np.ndarray: Generated time series of colored noise.
        """
        X_white = np.fft.rfft(np.random.randn(self.N))
        S = psd_func(self.freqs)
        S = S / np.sqrt(np.mean(S**2))  # normalize PSD
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

    def white(self):
        """Generate white noise with flat PSD."""
        return self.generate(lambda f: 1)

    def blue(self):
        """Generate blue noise with PSD proportional to f."""
        return self.generate(lambda f: np.sqrt(f))

    def violet(self):
        """Generate violet noise with PSD proportional to f^2."""
        return self.generate(lambda f: f)

    def brownian(self):
        """Generate Brownian (red) noise with PSD proportional to 1/f^2."""
        return self.generate(lambda f: 1 / np.where(f == 0, float('inf'), f))

    def pink(self):
        """Generate pink noise with PSD proportional to 1/f."""
        return self.generate(lambda f: 1 / np.where(f == 0, float('inf'), np.sqrt(f)))
    
    def higher_order_noise_3(self):
        """Generate noise with PSD proportional to 1/f^3."""
        return self.generate(lambda f: 1 / np.where(f == 0, float('inf'), f**1.5))

    def higher_order_noise_4(self):
        """Generate noise with PSD proportional to 1/f^4."""
        return self.generate(lambda f: 1 / np.where(f == 0, float('inf'), f**2.0))

    def higher_order_noise_5(self):
        """Generate noise with PSD proportional to 1/f^5."""
        return self.generate(lambda f: 1 / np.where(f == 0, float('inf'), f**2.5))

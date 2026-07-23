##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
# Systems and Standard lib
import requests
import io
import os
import glob
import datetime
import random
import h5py
import inspect
from itertools import combinations
import warnings

# Data Manipulation and Visualization Std. lib
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import ticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Imaging and CV lib
from PIL import Image
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
import SimpleITK as sitk
from skimage.measure import label, regionprops
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Signal Processing lib
from scipy.spatial import distance_matrix
from scipy.spatial.distance import jensenshannon
from scipy.io import wavfile
from scipy import signal
from scipy.signal import chirp, find_peaks
from scipy.fft import fft
from scipy.interpolate import interp1d

# Metrics lib
from scipy.stats import wasserstein_distance_nd, wasserstein_distance
from radiomics.glszm import RadiomicsGLSZM
from scipy.optimize import minimize_scalar, minimize, Bounds, differential_evolution
from SPEC2VEC.utils.gisqa_helper import HelperFunc


class pairwise_feature_extractor:
    """
    Extracts spatial features by comparing input spectrogram distributions 
    against a bank of reference noise distributions, utilizing histogram alignment, and Gaussian Mixture Models (GMM).
    Filters the input spectrogram accordingly and the computes the spatial features with radiomics library.
    """

    def __init__(self):
        """
        Initialize the pairwise feature extractor with scaling attributes.
        """
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.noise_norm = None
        self.spec_norm = None
    
    def read_noisebank_image(self, noisebnk_path: str, noisebnk_key: str, binary_keys: list):
        """
        Reads a noise image from an HDF5 database or generates it dynamically from binary keys.

        Args:
            noisebnk_path (str): File path to the HDF5 noise bank.
            noisebnk_key (str): Key of the specific noise dataset to load.
            binary_keys (list): List of keys that trigger dynamic Gaussian noise generation.

        Returns:
            np.ndarray: The loaded or generated noise image array.
        """
        std_dev_list = np.linspace(0.05, 0.1, len(binary_keys))
        mean = 0
        dataset = np.array([])
        with h5py.File(noisebnk_path, 'r') as f:
            if noisebnk_key in f and 'image' in f[noisebnk_key]:
                dataset = f[noisebnk_key]['image'][:]
            if noisebnk_key in binary_keys:
                std_dev = std_dev_list[binary_keys.index(noisebnk_key)]
                one_locations = np.where(dataset >= 0)
                gaussian_values = np.random.normal(mean, std_dev, size=len(one_locations[0]))
                dataset[one_locations] = gaussian_values
        return dataset
    
    def find_nearest_imgbank_neighbour_bysignalfit(self, spec_img: np.ndarray, noisebnk_path: str, 
                                                   noisebnk_keys: list, binary_keys: list, 
                                                   corr_only: bool=True, noise_cutoff:int=90, hist_bins:int=512, 
                                                   optimization_params={'type':'exponential','start':0.1,'end':100,'slope':10},
                                                   scale_percentile_range=(1,100), 
                                                   **kwargs):
        """
        Scans a spectrogram distribution against an image bank, correlating and aligning 
        the distributions to find the best similarity match. If noise model is known and provided
        then fits the noise model to the input spectrogram and then aligns the distribution to filter out the noise.

        Args:
            spec_img (np.ndarray): 2D array where rows are batches and columns are flattened pixel intensities.
            noisebnk_path (str): Path to the HDF5 noise bank.
            noisebnk_keys (list): Keys in the noise bank to search through.
            binary_keys (list): Keys indicating dynamic noise generation.
            corr_only (bool, optional): If True, returns only correlation and difference values.
            noise_cutoff (int, optional): Percentile cutoff for noise thresholding.
            hist_bins (int, optional): Number of histogram bins.
            optimization_params (dict, optional): Parameters for weighting the histogram optimization.
            scale_percentile_range (tuple, optional): Range for percentile scaling.
            **kwargs: Can include 'best_neighbour_key', 'cross_correlation_values', 'area_diff_values'.

        Returns:
            Depending on `corr_only` and `best_neighbour_key`:
            - If searching (corr_only=True): (cross_correlation_values, area_diff_values, best_neighbour_key)
            - If aligning: (realigned_histograms, original_histrograms, global_bins, additional_params)
        """
        assert spec_img.ndim == 2, "spec img must be a 2D array with rows as batches and all columns as pixel intensity distribution"
        warnings.warn("make sure that each row corresponds to a batch of an image") if spec_img.shape[0] > 1 else None
        spec_img = np.array([row.reshape(-1, 1).flatten() for row in spec_img])
        
        best_neighbour_key = kwargs.get('best_neighbour_key', None)
        cross_correlation_values = kwargs.get('cross_correlation_values',None)
        area_diff_values = kwargs.get('area_diff_values', None)
        #if best_neighbour_key:
        #    assert cross_correlation_values, 'pass cross orrelation values dict for best key'

        if not best_neighbour_key:
            cross_correlation_values = {}
            area_diff_values = {}
            for key in noisebnk_keys:
                noisebnk_img = self.read_noisebank_image(noisebnk_path,key,binary_keys)
                noisebnk_img = np.array([row.reshape(-1, 1).flatten() for row in noisebnk_img])
                max_cross_corr, areadiff = self.correlate_and_align_histograms(noisebnk_img, spec_img, 
                                                                               corr_only=True, 
                                                                               optimization_params=optimization_params,
                                                                               scale_percentile_range=scale_percentile_range)
                cross_correlation_values[key] = max_cross_corr
                area_diff_values[key] = areadiff
            best_neighbour_key = min(area_diff_values, key=area_diff_values.get) #max(cross_correlation_values, key=cross_correlation_values.get)
            print(f'Best neightbour for this distribution is: {best_neighbour_key}')

            if corr_only:
                return cross_correlation_values, area_diff_values, best_neighbour_key

        noisebnk_img = self.read_noisebank_image(noisebnk_path,best_neighbour_key,binary_keys)
        noisebnk_img = np.array([row.reshape(-1, 1).flatten() for row in noisebnk_img])
        #print(noisebnk_img.shape)
        realigned_histograms, original_histrograms, global_bins, additional_params = self.correlate_and_align_histograms(noisebnk_img, spec_img, 
                                                                                                                            noise_cutoff=noise_cutoff, 
                                                                                                                            num_bins=hist_bins,
                                                                                                                            optimization_params=optimization_params,
                                                                                                                            scale_percentile_range=scale_percentile_range)
        additional_params['max_crosscorr_key'] = best_neighbour_key
        additional_params['all_crosscorr_details'] = cross_correlation_values
        additional_params['all_areadiff_details'] = area_diff_values

        return realigned_histograms, original_histrograms, global_bins, additional_params 
    
    def correlate_and_align_histograms(self, noise_img: np.ndarray, spec_img: np.ndarray, 
                                        num_bins=512, corr_only=False, noise_cutoff=90, 
                                        optimization_params={'type':'exponential','start':0.1,'end':100,'slope':10},
                                        verbose=0, scale_percentile_range=(1,100)):
        """
        Takes two distributions, correlates and aligns as per the peak of the distributions. 
        It expects the images to be flattened across the second dimension. 
        Both the reference noise and spectrogram images can be of two dimensions where the first 
        dimension indicates the batch and the second dimension indicates the flattened pixels.
        
        Args:
            noise_img (np.ndarray): A 2D array representing the noise image data.
            spec_img (np.ndarray): A 2D array representing the spectrogram image data.
            num_bins (int, optional): The number of bins to use for creating the histograms. Defaults to 512.
            corr_only (bool, optional): If True, only returns the max cross-correlation and area difference. Defaults to False.
            noise_cutoff (int, optional): The percentile of the reference noise image to use as a cutoff. Defaults to 90.
            optimization_params (dict, optional): Parameters defining the weighting function for optimization.
            verbose (int, optional): Verbosity level.
            scale_percentile_range (tuple, optional): Bounds for percentile scaling.

        Returns:
            (If corr_only=False):
            realigned_histograms (dict): Contains 'aligned_hist1', 'aligned_hist2', and 'noise_img_cutoff_realigned'.
            original_histrograms (dict): Contains 'hist1', 'hist2', and 'noise_img_cutoff'.
            global_bins (np.ndarray): The bin edges used for creating the histograms.
            additional_params (dict): Contains 'max_cross_corr_val' and 'area_diff'.
            
            (If corr_only=True):
            max_cross_corr (float): Normalized max cross correlation.
            bestfit_area_diff (float): The optimal area difference between distributions.
        """

        assert noise_img.ndim == 2, "noise img must be a 2D array with rows as batches and columns as pixel intensity distribution"
        assert spec_img.ndim == 2, "spec img must be a 2D array with rows as batches and all columns as pixel intensity distribution"

        def objective(scale, noise_data, ref_hist, global_bins, weights, verbose=0):
            param1, param2 = scale
            param2 = int(np.floor(param2))
            min_max_val = np.round(param1,3)

            set1 = np.array([MinMaxScaler(feature_range=(-min_max_val,min_max_val)).fit_transform(row.reshape(-1, 1)).flatten() for row in noise_data])
            hist1, _ = np.histogram(set1, bins=global_bins, density=True)
            
            cross_corr = signal.correlate(hist1, ref_hist, mode='full')
            #lag = np.argmax(cross_corr) + param2  - (len(ref_hist) - 1) #uncomment for lag optimization
            lag = np.argmax(cross_corr) - (len(ref_hist) - 1) #uncomment for lag optimization is not needed
            aligned_hist1 = np.roll(hist1, -lag)

            ### ORIGINAL LOGIC (USE WHEN THE DISTRIBUTION IS NOT RIGHT SKEWED WHEN MORE WEIGHT NEEDS TO BE GIVEN ON THE LEFT SIDE OF THE DIST.) ###
            # aligned_peak_location1 = np.argmax(aligned_hist1)
            # aligned_peak_location2 = np.argmax(ref_hist)
            # peak_bin_index = min(aligned_peak_location1,aligned_peak_location2)
            
            # areadiff = np.abs(aligned_hist1[:peak_bin_index] - ref_hist[:peak_bin_index])
            # areadiff = np.sum(areadiff * weights[:peak_bin_index])
            ### ORIGINAL LOGIC END ###

            ### NEW LOGIC (USE WHEN THE DISTRIBUTION IS RIGHT SKEWED) ###
            areadiff_array = np.abs(aligned_hist1 - ref_hist)
            areadiff = np.sum(areadiff_array)
            ### NEW LOGIC END ###

            if verbose:
                print(f'min_max_val={min_max_val}, areadiff={areadiff}')
            
            return areadiff

        ## Test Code to Check if vmin logic helps
        # flat = spec_img[0]
        # mask = np.ones_like(flat, dtype=bool)
        # min_indices = np.where(flat == flat.min())[0]
        # if len(min_indices) > 0:
        #     mask[min_indices[1:]] = False  # keep only the first min
        # flat_modified = flat[mask]
        # spec_img = flat_modified.reshape(1,-1)

        ## Prepare Histogram Bins and Good Data, lower_percentile is usually 1
        global_bins = np.linspace(-4, 4, num_bins + 1)
        #set2 = np.array([self.scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in spec_img]) 
        set2 = np.array([HelperFunc.scale_percentile(row.reshape(-1, 1),
                                                     lower_percentile=scale_percentile_range[0],
                                                     upper_percentile=scale_percentile_range[1],
                                                     scale_type='-1to1') for row in spec_img])
        hist2, bin_edges2 = np.histogram(set2, bins=global_bins, density=True)

        ## Compute the Weighting Factor for Histogram Optimization
        total_siglen = len(hist2)
        all_nonzero_idx = np.where(hist2>0)[0]
        hist2_nonzero = hist2[all_nonzero_idx[0]:all_nonzero_idx[-1]]
        weights = np.flipud(HelperFunc.weighting_function(N=len(hist2_nonzero), function_type=optimization_params['type'], start_value=optimization_params['start'], end_value=optimization_params['end'], k=optimization_params['slope']))
        noise_weights1 = np.ones(all_nonzero_idx[0]-0-1)*weights[0]
        noise_weights2 = np.ones(total_siglen-all_nonzero_idx[-1]+2)*weights[-1]
        final_weights = np.concatenate([noise_weights1,weights,noise_weights2], axis=0)

        ## Optimize for the Best Noise
        res = differential_evolution(objective, 
                                        args=(noise_img, hist2, global_bins, final_weights, verbose), 
                                        #bounds=Bounds([0.2, 0], [2, 20]), # Use this Bound Logged Specs, Gaussian type Distribution
                                        bounds=Bounds([0.2, 0], [1, 20]),  # Use this Bound for Right Skew Distribution, Non Logged Specs
                                        strategy = 'best1bin',
                                        maxiter=1000,
                                        integrality = [False, True],
                                        popsize=2, seed=42)

        if not res['success']:
            raise AssertionError('The noise fitting did not converge')
        noise_scale_factor = np.round(res.x[0],3)
        bestfit_area_diff = np.round(res['fun'],4)
        shift_factor = int(res.x[1])
        set1 = np.array([MinMaxScaler(feature_range=(-noise_scale_factor,noise_scale_factor)).fit_transform(row.reshape(-1, 1)).flatten() for row in noise_img])
        
        ## Preapre Noise Data Histogram as per Optimization
        noise_img_cutoff = np.percentile(set1, noise_cutoff)
        bin_index_90_percentile = np.searchsorted(global_bins, noise_img_cutoff) - 1
        bin_index_90_percentile = max(0, min(bin_index_90_percentile, len(global_bins) - 2))
        hist1, bin_edges1 = np.histogram(set1, bins=global_bins, density=True)
        
        ## Final Cross Correlate
        cross_corr = signal.correlate(hist1, hist2, mode='full')
        #lag = np.argmax(cross_corr) + shift_factor - (len(hist2) - 1) #uncomment when lag optimization is needed
        lag = np.argmax(cross_corr) - (len(hist2) - 1)  #uncomment when lag optimization is not needed
        cross_corr_norm = cross_corr/np.sqrt(np.sum(np.abs(hist2)**2)*np.sum(np.abs(hist1)**2))
        max_cross_corr = np.max(cross_corr_norm)
        if corr_only:
            return max_cross_corr, bestfit_area_diff

        aligned_hist1 = np.roll(hist1, -lag)
        aligned_hist2 = hist2
        bin_index_90_percentile = bin_index_90_percentile-lag
        noise_img_cutoff_aligned = global_bins[bin_index_90_percentile]
        
        self.noise_norm = set1
        self.spec_norm = set2
        realigned_histograms = {}
        realigned_histograms['aligned_hist1'] = aligned_hist1
        realigned_histograms['aligned_hist2'] = aligned_hist2
        realigned_histograms['noise_img_cutoff_realigned'] = noise_img_cutoff_aligned
        original_histrograms = {}
        original_histrograms['hist1'] = hist1
        original_histrograms['hist2'] = hist2
        original_histrograms['noise_img_cutoff'] = noise_img_cutoff
        additional_params={}
        additional_params['max_cross_corr_val'] = max_cross_corr
        additional_params['area_diff'] = bestfit_area_diff

        return realigned_histograms, original_histrograms, global_bins, additional_params

    def find_hist_area_difference_at_right_tail(self, realigned_histograms, global_bins):
        """
        Calculates the area difference between the aligned noise and spectrogram 
        histograms strictly at the right tail (values beyond the aligned peak).

        Args:
            realigned_histograms (dict): Dictionary containing 'aligned_hist1' and 'aligned_hist2'.
            global_bins (np.ndarray): The bin edges used for the histograms.

        Returns:
            float: Absolute area difference in the right tail.
        """
        aligned_hist1 = realigned_histograms['aligned_hist1']
        aligned_hist2 = realigned_histograms['aligned_hist2']
        aligned_peak_location1 = np.argmax(aligned_hist1)
        aligned_peak_location2 = np.argmax(aligned_hist2)

        peak_bin_index = min(aligned_peak_location1,aligned_peak_location2)

        intersections = np.where(np.diff(np.sign(aligned_hist1 - aligned_hist2)))[0]
        right_intersections = intersections[intersections > peak_bin_index]

        return np.abs(np.sum(aligned_hist1[right_intersections[0]:])*(global_bins[1] - global_bins[0])
                    -np.sum(aligned_hist2[right_intersections[0]:])*(global_bins[1] - global_bins[0]))

    def gmm_optimal_func(self, img_gray, ncomp=20, nbins=128, gmm_method='std'):
        """
        Fits a Gaussian Mixture Model (GMM) to the grayscale intensity values of an image.
        This GMM fit is specifically performed on the residual of the image after noise is filtered out.
        
        This function flattens the intensity values and fits a GMM with a specified 
        number of components, allowing for both standard and Bayesian approaches.

        Args:
            img_gray (np.ndarray): A 1D or 2D array representing the grayscale image.
            ncomp (int, optional): Number of GMM components. Defaults to 20.
            nbins (int, optional): Number of bins for PDF calculation. Defaults to 128.
            gmm_method (str, optional): 'std' for standard GMM, 'bayes' for Bayesian GMM.

        Returns:
            gmm_optimal: The fitted Gaussian Mixture Model object.
            pdf (np.ndarray): The probability density function of the fitted GMM.
            x (np.ndarray): X-axis values for the PDF.
            intensity_values (np.ndarray): Flattened array of original intensity values.
        """

        if img_gray.ndim >= 2:
            intensity_values = img_gray.flatten().reshape(-1, 1)
        else:
            intensity_values = img_gray.reshape(-1, 1)

        if gmm_method == 'std':
            gmm_optimal = GaussianMixture(n_components=ncomp, random_state=6, covariance_type='diag', n_init=5, init_params="k-means++")
        elif gmm_method == 'bayes':
            gmm_optimal = BayesianGaussianMixture(n_components=ncomp, random_state=6, n_init=5, covariance_type='diag', 
                                                  init_params="k-means++", weight_concentration_prior=2)
        else:
            raise ValueError('Invalid GMM Method. Choose between "std" or "bayes".')
        
        gmm_optimal.fit(intensity_values)
        x = np.linspace(intensity_values.min(), intensity_values.max(), nbins).reshape(-1, 1)
        logprob = gmm_optimal.score_samples(x)
        pdf = np.exp(logprob)
        
        return gmm_optimal, pdf, x.flatten(), intensity_values

    def label_image_with_gmm_fitting(self, realigned_histograms, global_bins, spec_img, gmm_ncomp=5,
                                    gmm_mean_threshold=0.05, gmm_nbins=256, gmm_method='std', 
                                    spec_normalize='full', scale_percentile_range=(1,100)):
        """
        Labels a spectrogram image based on the difference between its histogram
        and a noise histogram, using Gaussian Mixture Model (GMM) fitting on
        the distribution of pixel value differences.

        This function identifies regions in a spectrogram that deviate significantly 
        from a reference noise distribution by fitting a GMM to the difference distribution, 
        and using the GMM components to define labeling thresholds.

        Args:
            realigned_histograms (dict): Dict with 'aligned_hist1', 'aligned_hist2', and 'noise_img_cutoff_realigned'.
            global_bins (np.ndarray): The bins used for creating the histograms. Should be scaled between -1 and 1.
            spec_img (np.ndarray): The spectrogram image data.
            gmm_ncomp (int, optional): Number of components for the GMM fitting. Defaults to 5.
            gmm_mean_threshold (float, optional): Threshold for filtering GMM means. Defaults to 0.05.
            gmm_nbins (int, optional): Number of bins for GMM PDF.
            gmm_method (str, optional): 'std' or 'bayes'.
            spec_normalize (str, optional): Normalization method ('full' or 'columns').
            scale_percentile_range (tuple, optional): Range for percentile scaling.

        Returns:
            labeled_spec_image (np.ndarray): The spectrogram image labeled with integer values based on GMM cut locations.
            params_dict (dict): Dictionary containing calculated thresholds, GMM parameters, and intermediate distributions.
        """

        ## assert required conditions
        assert int(np.min(global_bins)) >= -4 and int(np.max(global_bins)) <= 4, "histograms must be scaled between -1 and 1"
        noise_hist = realigned_histograms['aligned_hist1']
        img_hist = realigned_histograms['aligned_hist2']
        noise_img_cutoff = realigned_histograms['noise_img_cutoff_realigned']

        ## compute histogram difference
        if noise_hist.ndim>1:
            noise_hist = noise_hist.flatten()
        if img_hist.ndim>1:
            img_hist = img_hist.flatten()
        hist_diff = np.abs((noise_hist-img_hist) * (global_bins[1] - global_bins[0]))
        
        ## compute the distribution of pixels for the histogram difference
        scaler_freq = MinMaxScaler(feature_range=(1, 100))
        frequencies = scaler_freq.fit_transform(hist_diff.reshape(-1, 1)).flatten()
        pixels = global_bins[:-1] 
        scaling_factor = 1.0
        if np.min(frequencies) > 0:
            base_freq = 10
            scaling_factor = base_freq / np.min(frequencies)
        scaled_frequencies = (frequencies * scaling_factor).astype(int)
        pixel_vector = np.repeat(pixels, scaled_frequencies)
        np.random.shuffle(pixel_vector)
        pixel_vector_filtered = pixel_vector[(pixel_vector>=noise_img_cutoff) & (pixel_vector<=1)]

        ## fit the differenced histogram with a GMM Model
        gmm_opt_val, fit_pdf, fit_pdf_axis, gmm_intensities = self.gmm_optimal_func(pixel_vector_filtered, ncomp=gmm_ncomp, nbins=gmm_nbins, gmm_method=gmm_method)
        means_gmm = gmm_opt_val.means_
        std_devs_gmm = np.sqrt(gmm_opt_val.covariances_)

        ## compute the GMM cluster means and filter out means closer than threshhold
        sorted_indices = np.argsort(means_gmm, axis=0)
        sorted_means_gmm = means_gmm[sorted_indices].flatten()
        sorted_std_devs_gmm = std_devs_gmm[sorted_indices].flatten()
        means_to_keep = np.diff(sorted_means_gmm,append=2) >= gmm_mean_threshold
        sorted_means_gmm_filt = sorted_means_gmm[means_to_keep]
        sorted_std_devs_gmm_filt = sorted_std_devs_gmm[means_to_keep]

        ## compute the locations of the high energy pixels. 
        ## if the 2*std is very small, that is a very high peak then provide a constant std which is small but resonable.
        ## if the 2*std is very big, that is a very spread out then provide a constant std which is big but limited.
        ## New Cut Location Code, cite: https://gist.github.com/ben741/d8c70b608d96d9f7ed231086b237ba6b
        
        cut_locations = []
        # for i in range(len(sorted_means_gmm_filt)-1):
        #     mean = sorted_means_gmm_filt[i]
        #     std_dev = sorted_std_devs_gmm_filt[i]
        #     std_dev = 0.05 if 2*std_dev > 0.2 else std_dev #std_dev = 0.0625 if 2*std_dev > 0.25 else std_dev #std_dev = 0.075 if 2*std_dev > 0.25 else std_dev
        #     std_dev = 0.025 if 2*std_dev < 0.05 else std_dev #std_dev = 0.025 if 2*std_dev < 0.05 else std_dev
        #     right_tail_2std = mean + 2 * std_dev
        #     cut_locations.append(right_tail_2std)
        # cut_locations.insert(0, noise_img_cutoff)

        ## As per Github
        # cut_dips = np.where((fit_pdf[1:-1] <= fit_pdf[0:-2]) * (fit_pdf[1:-1] <= fit_pdf[2:]))[0] + 1
        # cut_locations = list(fit_pdf_axis[cut_dips])

        ## As per Second Derivative
        y_second_derivative = np.gradient(np.gradient(fit_pdf, fit_pdf_axis), fit_pdf_axis)
        cut_dips, _ = find_peaks(y_second_derivative)
        cut_dips = cut_dips.tolist()
        cut_locations = list(fit_pdf_axis[cut_dips])[1:]

        ## if cut locations are very nearby to each other take the mean of the nearest points
        cut_diffs = np.abs(np.diff(cut_locations))
        cut_close_indices = np.where(cut_diffs < 0.01)[0]
        for i in reversed(cut_close_indices):
            mean_val = (cut_locations[i] + cut_locations[i+1]) / 2
            cut_locations[i:i+2] = [mean_val]

        ## label the spectogram as per the high energy pixels. make the pixels below noise_cutoff to be 0.
        if spec_normalize=='full':
            spec_image_shape = spec_img.shape
            #spec_img = self.scaler.fit_transform(spec_img.reshape(-1,1)).reshape(spec_image_shape)
            spec_img = HelperFunc.scale_percentile(spec_img.reshape(-1, 1),
                                                   lower_percentile=scale_percentile_range[0], 
                                                   upper_percentile=scale_percentile_range[1],
                                                   scale_type='-1to1').reshape(spec_image_shape)
        elif spec_normalize=='columns':
            spec_img = self.scaler.fit_transform(spec_img)
            
        labeled_spec_image = np.zeros(spec_img.shape)-1
        for i in range (0,len(cut_locations)):
            if i==0:
                labeled_spec_image[spec_img<cut_locations[i].item()] = i
                labeled_spec_image[(spec_img>=cut_locations[i].item()) & (spec_img<cut_locations[i+1].item())] = i+1
            elif i==len(cut_locations)-1:
                labeled_spec_image[(spec_img>=cut_locations[i].item())] = i+1
            else:
                labeled_spec_image[(spec_img>=cut_locations[i].item()) & (spec_img<cut_locations[i+1].item())] = i+1
        labeled_spec_image = labeled_spec_image.astype(int)

        ## prepare argument dictionary for plotting and analysis
        params_dict = {}
        params_dict['cut_locations'] = cut_locations
        params_dict['gmm_means'] = sorted_means_gmm
        params_dict['gmm_covars'] = gmm_opt_val.covariances_
        params_dict['gmm_stdevs'] = sorted_std_devs_gmm
        params_dict['filtered_gmm_means'] = sorted_means_gmm_filt
        params_dict['filtered_gmm_stdevs'] = sorted_std_devs_gmm_filt
        params_dict['gmm_model'] = gmm_opt_val
        params_dict['gmm_fit_pdf'] = fit_pdf
        params_dict['gmm_fit_pdf_xaxis'] = fit_pdf_axis
        params_dict['gmm_intensities'] = gmm_intensities
        params_dict['gmm_nbins'] = gmm_nbins
        params_dict['hist_diff_distribution'] = pixel_vector
        params_dict['hist_diff_distribution_filtered'] = pixel_vector_filtered
        params_dict['hist_diff'] = hist_diff

        return labeled_spec_image, params_dict

    def getWeightedZoneSizePercentageFeatureValue(self, glszm, min_zone_size_thrshld=5):
        """
        Calculates the Weighted Zone Size Percentage feature value from a GLSZM object. 
        This is an additional metric not available in radiomics library.

        This feature measures the proportion of the total number of pixels that are part
        of zones with a size greater than a specified threshold, weighted by their size.

        Args:
            glszm (RadiomicsGLSZM): The object containing the calculated Gray Level Size Zone Matrix.
            min_zone_size_thrshld (int, optional): Minimum zone size to be included. Defaults to 5.

        Returns:
            float: The calculated Weighted Zone Size Percentage feature value.
        """

        pix_size_dist_idx = np.where(glszm.coefficients['jvector']>min_zone_size_thrshld)
        pix_size_dist = glszm.coefficients['jvector'][pix_size_dist_idx]
        ps_filt_idx = len(glszm.coefficients['jvector'])-len(pix_size_dist)
        Np = np.sum(glszm.coefficients['ps'][0,ps_filt_idx:]*pix_size_dist)

        pix_size_tot=0
        for i, pix_size_idx in enumerate(pix_size_dist_idx):
            try:
                pix_size_tot += np.sum(glszm.P_glszm[0,:,pix_size_idx]*pix_size_dist[i])
            except:
                continue
        return pix_size_tot/Np

    def filter_labeled_spec_with_pixel_zone_size(self, labeled_spec_image, min_zone_size_thrshld=5):
        """
        Filters a labeled spectrogram image based on the size of connected zones.

        Identifies connected components for each unique label, calculates their size, 
        and removes zones smaller than the specified threshold.

        Args:
            labeled_spec_image (np.ndarray): A 2D array representing the labeled spectrogram image.
            min_zone_size_thrshld (int, optional): Minimum size a zone must have to be kept. Defaults to 5.

        Returns:
            np.ndarray: The filtered labeled spectrogram image.
        """
        zone_sizes_image = np.zeros_like(labeled_spec_image, dtype=int)
        unique_labels = np.unique(labeled_spec_image)
        unique_labels = unique_labels[unique_labels != 0]

        for current_label in unique_labels:
            binary_image = (labeled_spec_image == current_label).astype(int)
            labeled_components = label(binary_image)
            regions = regionprops(labeled_components)

            rows, cols = np.where(labeled_spec_image == current_label)
            for r, c in zip(rows, cols):
                component_label = labeled_components[r, c]
                if component_label > 0:
                    region = regions[component_label - 1]
                    zone_size = region.area
                    zone_sizes_image[r, c] = zone_size

        zone_size_mask = (zone_sizes_image>=min_zone_size_thrshld).astype(int)
        return labeled_spec_image*zone_size_mask

    def compute_pixel_localization_metric(self, labeled_spectrogram: np.ndarray, min_zone_size_thrshld=5):
        """
        Computes spatial localization metrics on a labeled spectrogram using Radiomics GLSZM.
        This function computes the spatial features on=f the input TFR.

        Args:
            labeled_spectrogram (np.ndarray): Filtered and labeled spectrogram image matrix.
            min_zone_size_thrshld (int, optional): Threshold for the custom weighted zone size percentage.

        Returns:
            glszm (RadiomicsGLSZM): The computed GLSZM object.
            df (pd.DataFrame): DataFrame containing extracted localization emphasis metrics.
        """
        matrix = labeled_spectrogram
        mask = np.ones(matrix.shape)
        mask[np.where(matrix==0)]=0
        mask = sitk.GetImageFromArray(mask)
        matrix_sitk = sitk.GetImageFromArray(matrix)
        # settings = {"voxelBased": False, "kernelRadius": 1, "binCount": len(np.unique(matrix))-1, 
        #             "normalize": True, "resampledPixelSpacing": None}
        settings = {"voxelBased": False, "kernelRadius": 1, "binWidth": 1,
                    "normalize": False, "resampledPixelSpacing": None}
        glszm = RadiomicsGLSZM(matrix_sitk,mask,**settings)
        glszm._initCalculation()
        sizezone_metrics={}

        sizezone_metrics['SmallAreaEmphasis'] = glszm.getSmallAreaEmphasisFeatureValue()
        sizezone_metrics['LargeAreaEmphasis'] = glszm.getLargeAreaEmphasisFeatureValue()
        sizezone_metrics['GrayLevelNonUniformityNormalized'] = glszm.getGrayLevelNonUniformityNormalizedFeatureValue()
        sizezone_metrics['SizeZoneNonUniformityNormalized'] = glszm.getSizeZoneNonUniformityNormalizedFeatureValue()
        sizezone_metrics['LargeAreaHighGrayLevelEmphasis'] = glszm.getLargeAreaHighGrayLevelEmphasisFeatureValue()
        sizezone_metrics['WeightedZoneSizePercentageFeatureValue'] = self.getWeightedZoneSizePercentageFeatureValue(glszm, min_zone_size_thrshld)
        return glszm, pd.DataFrame.from_dict(sizezone_metrics)



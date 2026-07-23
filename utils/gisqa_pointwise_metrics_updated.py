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
import math

# Data Manipulation and Visualization Std. lib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import ticker
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Imaging and CV lib
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d

# Signal Processing lib
from scipy.spatial import distance_matrix
from scipy.spatial.distance import jensenshannon
from scipy import signal
from scipy.signal import chirp
from scipy.fft import fft
from scipy.interpolate import interp1d

# Metrics lib
from scipy.stats import wasserstein_distance_nd, wasserstein_distance
import antropy as ant
import tsfel
import ordpy
from hilbert import decode, encode
from SPEC2VEC.utils.gisqa_helper import *


class antropy_pointwise_feature_extractor:
  """
  Extracts pointwise entropy and fractal dimension features from a 1D Array
  using the antropy library.
  """

  def __init__(self, dx: int = 5):
      """
      Initialize the feature extractor.

      Args:
          dx (int): Embedding dimension (order) for the entropy calculation.
      """
      self.dx = dx
      df = None 

  def generate_entropy_measures(self, x, metrics=None):
      """
      Generate entropy measures and fractal dimension from a given time series.

      Args:
          x (array_like): 1D Time series data.
          metrics (list, optional): List of metric names to compute. If None, computes all.

      Returns:
          df (pd.DataFrame): DataFrame containing the computed entropy and fractal dimension measures.
      """
      dict_entropy = {}
      
      # Helper to check if a metric should be computed
      def should_compute(name):
          return metrics is None or name in metrics

      # Permutation entropy
      if should_compute('permutation_entropy'):
          dict_entropy['permutation_entropy'] = ant.perm_entropy(x, order=self.dx, normalize=True) #order=3
      # Spectral entropy, teh SF parameter does not change the result, Checked for TS, SPEC all
      if should_compute('spectral_entropy'):
          dict_entropy['spectral_entropy'] = ant.spectral_entropy(x, sf=100, method='welch', normalize=True) ##250
      # Singular value decomposition entropy, This metrics will yield different result if the input is scaled
      if should_compute('svd_entropy'):
          dict_entropy['svd_entropy'] = ant.svd_entropy(x, order=self.dx, normalize=True) #order=3
      # Approximate entropy
      if should_compute('approximate_entropy'):
          dict_entropy['approximate_entropy'] = ant.app_entropy(x, order=self.dx) #order=2, #order=3 is standard as per paper in antropy webpage
      # Sample entropy
      if should_compute('sample_entropy'):
          dict_entropy['sample_entropy'] = ant.sample_entropy(x, order=self.dx) #order=2, #order=3 is standard
      # Hjorth mobility and complexity
      if should_compute('hjorth_mobility') or should_compute('hjorth_complexity'):
           hm, hc = ant.hjorth_params(x)
           if should_compute('hjorth_mobility'): dict_entropy['hjorth_mobility'] = hm
           if should_compute('hjorth_complexity'): dict_entropy['hjorth_complexity'] = hc
      # Number of zero-crossings - Does this take a lot of time?
      if should_compute('num_zerocross'):
          dict_entropy['num_zerocross'] = ant.num_zerocross(x)

      dict_fractal = {}
      # Petrosian fractal dimension
      if should_compute('petrosian_fd'):
          dict_fractal['petrosian_fd'] = ant.petrosian_fd(x)
      # Katz fractal dimension
      if should_compute('katz_fd'):
          dict_fractal['katz_fd'] = ant.katz_fd(x)
      # Higuchi fractal dimension
      if should_compute('higuchi_fd'):
          dict_fractal['higuchi_fd'] = ant.higuchi_fd(x)
      # Detrended fluctuation analysis
      if should_compute('detrended_fluctuation'):
          dict_fractal['detrended_fluctuation'] = ant.detrended_fluctuation(x)
      
      # Combine the two dictionaries into a single DataFrame
      df = pd.concat([pd.DataFrame(dict_entropy, index=[0]),
                      pd.DataFrame(dict_fractal, index=[0])], axis=1)
      
  
      cols_to_rename = df.columns
      df = df.rename(columns={col: col + '_antropy' for col in cols_to_rename})
      df.columns = df.columns.str.replace(' ', '_')
      
      return df

class ordpy_pointwise_feature_extractor:
  """
  Extracts pointwise complexity and entropy features based on ordinal patterns 
  (Bandt-Pompe permutation approach) using the ordpy library.
  """

  def __init__(self, dx: int = 7, taux: int = 1):
      """
      Initialize the feature extractor.

      Args:
          dx (int): Embedding dimension (order) of the ordinal patterns.
          taux (int): Embedding delay (time lag) of the ordinal patterns.
      """
      self.dx = dx
      self.taux = taux

  # ... (methods kept same, but generate_ordpy_features updated) ...

  def generate_ordpy_features(self, x: np.ndarray, metrics=None):
    """
    Computes various ordinal pattern-based entropy and complexity measures.

    Args:
        x (np.ndarray): 1D Time series data.
        metrics (list, optional): List of metric names to compute. If None, computes all.

    Returns:
        df (pd.DataFrame): DataFrame containing the computed ordpy features.
    """
    dict_entropy = {}
    
    # Helper to check if a metric should be computed
    def should_compute(name):
          return metrics is None or name in metrics

    # Complexity Entropy (returns normalized_permutation_entropy and statistical_complexity_entropy)
    if should_compute('normalized_permutation_entropy') or should_compute('statistical_complexity_entropy'):
        complexity_entropy_val = ordpy.complexity_entropy(x, dx = self.dx, taux=self.taux)
        if should_compute('normalized_permutation_entropy'):
            dict_entropy['normalized_permutation_entropy'] = complexity_entropy_val[0]
        if should_compute('statistical_complexity_entropy'):
            dict_entropy['statistical_complexity_entropy'] = complexity_entropy_val[1]

    if should_compute('fisher_shannon'):
        dict_entropy['fisher_shannon'] = ordpy.fisher_shannon(x, dx = self.dx, taux=self.taux)[1]
    
    if should_compute('global_node_entropy'):
        dict_entropy['global_node_entropy'] = ordpy.global_node_entropy(x, dx=self.dx, taux=self.taux)
    
    if should_compute('missing_links'):
        dict_entropy['missing_links'] = ordpy.missing_links(x, dx = self.dx, return_fraction=True, return_missing=False)
    
    if should_compute('missing_patterns'):
        dict_entropy['missing_patterns'] = ordpy.missing_patterns(x, dx = self.dx, return_fraction=True, return_missing=False)
    
    # Renyi Short
    if should_compute('renyi_complexity_entropy_short') or should_compute('renyi_stat_complexity_short'):
        renyi_complexity_entropy_short_val = ordpy.renyi_complexity_entropy(x, dx=self.dx, taux=self.taux, alpha=0.5)
        if should_compute('renyi_complexity_entropy_short'):
            dict_entropy['renyi_complexity_entropy_short'] = renyi_complexity_entropy_short_val[0]
        if should_compute('renyi_stat_complexity_short'):
            dict_entropy['renyi_stat_complexity_short'] = renyi_complexity_entropy_short_val[1]

    # Renyi Long
    if should_compute('renyi_complexity_entropy_long') or should_compute('renyi_stat_complexity_long'):
        renyi_complexity_entropy_long_val = ordpy.renyi_complexity_entropy(x, dx=self.dx, taux=self.taux, alpha=1.5)
        if should_compute('renyi_complexity_entropy_long'):
            dict_entropy['renyi_complexity_entropy_long'] = renyi_complexity_entropy_long_val[0]
        if should_compute('renyi_stat_complexity_long'):
            dict_entropy['renyi_stat_complexity_long'] = renyi_complexity_entropy_long_val[1]

    # Renyi 3
    if should_compute('renyi_complexity_entropy_3') or should_compute('renyi_stat_complexity_3'):
        renyi_complexity_entropy_3_val = ordpy.renyi_complexity_entropy(x, dx=self.dx, taux=self.taux, alpha=3)
        if should_compute('renyi_complexity_entropy_3'):
            dict_entropy['renyi_complexity_entropy_3'] = renyi_complexity_entropy_3_val[0]
        if should_compute('renyi_stat_complexity_3'):
            dict_entropy['renyi_stat_complexity_3'] = renyi_complexity_entropy_3_val[1]

    # Tsallis Short
    if should_compute('tsallis_complexity_entropy_short') or should_compute('tsallis_stat_complexity_short'):
        tsallis_complexity_entropy_short_val = ordpy.tsallis_complexity_entropy(x, dx=self.dx, taux=self.taux, q=0.5)
        if should_compute('tsallis_complexity_entropy_short'):
            dict_entropy['tsallis_complexity_entropy_short'] = tsallis_complexity_entropy_short_val[0]
        if should_compute('tsallis_stat_complexity_short'):
            dict_entropy['tsallis_stat_complexity_short'] = tsallis_complexity_entropy_short_val[1]

    # Tsallis Long
    if should_compute('tsallis_complexity_entropy_long') or should_compute('tsallis_stat_complexity_long'):
        tsallis_complexity_entropy_long_val = ordpy.tsallis_complexity_entropy(x, dx=self.dx, taux=self.taux, q=1.5)
        if should_compute('tsallis_complexity_entropy_long'):
            dict_entropy['tsallis_complexity_entropy_long'] = tsallis_complexity_entropy_long_val[0]
        if should_compute('tsallis_stat_complexity_long'):
            dict_entropy['tsallis_stat_complexity_long'] = tsallis_complexity_entropy_long_val[1]

    if should_compute('weighted_permutation_entropy'):
        dict_entropy['weighted_permutation_entropy'] = ordpy.weighted_permutation_entropy(x, dx=self.dx, taux=self.taux, normalized=False)

    df = pd.DataFrame(dict_entropy, index=[0])
    cols_to_rename = df.columns
    df = df.rename(columns={col: col + '_ordpy' for col in cols_to_rename})
    df.columns = df.columns.str.replace(' ', '_')
    return df

class tsfel_stat_pointwise_feature_extractor:
  """
    Class to compute a predefined set of temporal and statistical features
    from time series data using the TSFEL library.
  """

  def __init__(self):
    """
    Initializes the TSFEL extractor by loading a hardcoded custom configuration 
    dictionary defining specific temporal and statistical features.
    """
    self.cfg_file_metrics_timeseries = {'temporal': {'Lempel-Ziv complexity': {'complexity': 'linear',
    'description': "Computes the Lempel-Ziv's (LZ) complexity index, normalized by the signal's length.",
    'function': 'tsfel.lempel_ziv',
    'parameters': {'threshold': None},
    'n_features': 1,
    'use': 'yes'},
    'Signal distance': {'complexity': 'constant',
    'description': 'Computes signal traveled distance.',
    'function': 'tsfel.distance',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'}},
    'statistical': {'Absolute energy': {'complexity': 'log',
    'description': 'Computes the absolute energy of the signal.',
    'function': 'tsfel.abs_energy',
    'parameters': '',
    'n_features': 1,
    'use': 'yes',
    'tag': 'audio'},
    'Average power': {'complexity': 'constant',
    'description': 'Computes the average power of the signal.',
    'function': 'tsfel.average_power',
    'parameters': {'fs': 20}, #250 #change this accordingly
    'n_features': 1,
    'use': 'yes',
    'tag': 'audio'},
    'ECDF Percentile': {'complexity': 'log',
    'description': 'Determines the percentile value of the ECDF.',
    'function': 'tsfel.ecdf_percentile',
    'parameters': {'percentile': '[0.25, 0.5, 0.75]'},
    'n_features': 'percentile',
    'use': 'yes'},
    'Histogram mode': {'complexity': 'log',
    'description': "Computes the mode of the signal's histogram.",
    'function': 'tsfel.hist_mode',
    'parameters': {'nbins': 25},
    'n_features': 1,
    'use': 'yes'},
    'Interquartile range': {'complexity': 'constant',
    'description': 'Computes interquartile range of the signal.',
    'function': 'tsfel.interq_range',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Kurtosis': {'complexity': 'constant',
    'description': 'Computes kurtosis of the signal.',
    'function': 'tsfel.kurtosis',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Mean': {'complexity': 'constant',
    'description': 'Computes the mean value of the signal.',
    'function': 'tsfel.calc_mean',
    'parameters': '',
    'n_features': 1,
    'use': 'yes',
    'tag': 'inertial'},
    'Mean absolute deviation': {'complexity': 'log',
    'description': 'Computes mean absolute deviation of the signal.',
    'function': 'tsfel.mean_abs_deviation',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Median': {'complexity': 'constant',
    'description': 'Computes median of the signal.',
    'function': 'tsfel.calc_median',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Median absolute deviation': {'complexity': 'constant',
    'description': 'Computes median absolute deviation of the signal.',
    'function': 'tsfel.median_abs_deviation',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Peak to peak distance': {'complexity': 'constant',
    'description': 'Computes the peak to peak distance.',
    'function': 'tsfel.pk_pk_distance',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Root mean square': {'complexity': 'constant',
    'description': 'Computes root mean square of the signal.',
    'function': 'tsfel.rms',
    'parameters': '',
    'n_features': 1,
    'use': 'yes',
    'tag': ['emg', 'inertial']},
    'Skewness': {'complexity': 'constant',
    'description': 'Computes skewness of the signal.',
    'function': 'tsfel.skewness',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Standard deviation': {'complexity': 'constant',
    'description': 'Computes standard deviation of the signal.',
    'function': 'tsfel.calc_std',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'},
    'Variance': {'complexity': 'constant',
    'description': 'Computes variance of the signal.',
    'function': 'tsfel.calc_var',
    'parameters': '',
    'n_features': 1,
    'use': 'yes'}}}

  def generate_tsfel_stat_features(self, x, metrics=None):
      """
      Extracts and combines statistical features from a time series using TSFEL.

      Args:
          x (array_like): The 1D time series data from which features are to be extracted.
          metrics (list, optional): List of metric names to compute. If None, computes all features defined in config.

      Returns:
          pd.DataFrame: DataFrame containing the computed TSFEL statistical features.
      """
      
      # Filter configuration based on requested metrics
      if metrics is not None:
          # Note: TSFEL config structure is {'domain': {'feature': {...}}}
          # We need to filter this structure.
          
          # Mapping from flattened name to TSFEL name might be tricky if names are different.
          # Based on existing code, TSFEL returns names like 'Lempel-Ziv complexity'.
          # We'll filter the config by checking if the feature key is in our requested metrics list.
          
          filtered_cfg = {}
          for domain, features in self.cfg_file_metrics_timeseries.items():
              domain_features = {}
              for feat_name, feat_cfg in features.items():
                  if feat_name in metrics:
                      domain_features[feat_name] = feat_cfg
              if domain_features:
                  filtered_cfg[domain] = domain_features
          
          if not filtered_cfg:
              return pd.DataFrame() # No metrics matched
              
          cfg_to_use = filtered_cfg
      else:
          cfg_to_use = self.cfg_file_metrics_timeseries

      features_set1 = tsfel.time_series_features_extractor(cfg_to_use, x, fs=1, verbose=0)
      
      # Percentiles are custom added in the original code
      if metrics is None or '75th_percentile' in metrics:
          features_set1['75th_percentile'] = np.percentile(x, 75)
      if metrics is None or '25th_percentile' in metrics:
          features_set1['25th_percentile'] = np.percentile(x, 25)
          
      return features_set1

class pointwise_features_integration:
    """
    Master class that integrates antropy, ordpy, and TSFEL pointwise feature extractors.
    Handles data flattening (including Hilbert curve mapping for 2D images) and feature scaling.
    """

    def __init__(self, img, is_hilbertize=1, **kwargs):
        """
        Initializes the integration pipeline, flattens the input array, and applies optional normalization.

        Args:
            img (np.ndarray): Input 1D or 2D array representing the signal or image.
            is_hilbertize (int/bool): If True, uses Hilbert (or generalized Hilbert) space-filling curve 
                                      to map 2D images into 1D arrays while preserving spatial locality.
            **kwargs: Additional optional arguments for scaling ('scale_type', 'is_normalize_stat', 
                      'is_normalize_entropy') and embedding dimensions ('ordpydx', 'antropydx').
        """
        self.img = img
        self.is_normalize_entropy = kwargs.get('is_normalize_entropy',0)
        self.is_normalize_stat = kwargs.get('is_normalize_stat',1)
        self.normalize_feature_range =  kwargs.get('normalize_feature_range',(-1,1)) 
        hilbert_locs = kwargs.get('hilbert_locs',None) 
        scale_type = kwargs.get('scale_type', 'GenValScale') 
        ordpydx = kwargs.get('ordpydx',7)
        antropydx = kwargs.get('antropydx',5)
        self.verbose = kwargs.get('verbose', 0)

        if hilbert_locs is None:
            if is_hilbertize:
                self.img_height, self.img_width = img.shape
                if self.img_height == self.img_width:
                    self.img_flat = self._hilbertize_sym_data().reshape(1,-1)
                else:
                    self.img_flat = self._ghilbertize_asym_data().reshape(1,-1)
            else:
                self.img_flat = img.flatten().reshape(1,-1)
        else:
            print('Using precomputed Hilbert Trace')
            self.img_flat = self.img[hilbert_locs[:,0], hilbert_locs[:,1]].reshape(1,-1)

        if self.is_normalize_stat or self.is_normalize_entropy:
            if scale_type == 'GenValScale':
              scaler = MinMaxScaler(feature_range=self.normalize_feature_range)
              self.img_flat_norm = scaler.fit_transform(self.img_flat.reshape(-1,1)).T
            elif scale_type == 'PercentileValScale':
              #print('test: pcentile scaling')
              self.img_flat_norm =  HelperFunc.scale_percentile(self.img_flat,1,100,scale_type='-1to1')
              self.img_flat_norm = self.img_flat_norm.reshape(1,-1)
            elif scale_type == 'StandardizedPercentileValScale':
              #print('test: standard pcentile scaling')
              self.img_flat_norm =  HelperFunc.standardize_percentile(self.img_flat,1,100,scale_type='-1to1')
              self.img_flat_norm = self.img_flat_norm.reshape(1,-1)
            else:
              raise AssertionError('scale_type must be withinn [GenValScale, PercentileValScale]')

        self.first_ord_sta = tsfel_stat_pointwise_feature_extractor()
        self.antrop_textures = antropy_pointwise_feature_extractor(dx=antropydx)
        self.ordpy_textures = ordpy_pointwise_feature_extractor(dx=ordpydx) 
        
    def _hilbertize_sym_data(self):
        """
        Flattens a symmetric 2D image (power of 2 dimensions) into a 1D array 
        using a standard Hilbert space-filling curve.

        Returns:
            np.ndarray: 1D hilbertized array.
        """
        assert self.img_height == 2**(math.log2(self.img_height)), 'Image height/width must be a power of 2'
        num_dims = 2
        num_bits = int(math.log2(self.img_height))
        max_h = 2**(num_dims*num_bits)
        hilberts = np.arange(max_h)
        locs = decode(hilberts, num_dims, num_bits) 
        if self.verbose:
          print('Image has been hilbertized')
        return self.img[locs[:,0], locs[:,1]]
      
    def _ghilbertize_asym_data(self):
        """
        Flattens an asymmetric 2D image into a 1D array using a generalized 
        Hilbert (Ghilbert) space-filling curve.

        Returns:
            np.ndarray: 1D ghilbertized array.
        """
        #locs = HelperFunc.gilbertize_image(width=self.img_height, height=self.img_width)
        locs = HelperFunc.gilbertize_image_optimized(width=self.img_height, height=self.img_width)
        if self.verbose:
            print('Image has been Ghilbertized')
        return self.img[locs[:,0], locs[:,1]].reshape(1,-1)

    def compute_all_pointwise_features(self, is_best_features=1, metrics_list=None, **kwargs):
        """
        Executes the full pipeline to compute pointwise features across antropy, ordpy, and TSFEL,
        optionally filtered by a specific list of metrics.

        Args:
            is_best_features (int/bool): If True and `metrics_list` is None, returns only a hardcoded 
                                         subset of the best performing features.
            metrics_list (list, optional): List of specific feature strings to extract. For antropy and ordpy, 
                                           these should match the keys generated by the extractor 
                                           (with suffixes '_antropy' and '_ordpy'). For TSFEL, they match config keys.
            **kwargs: Additional keyword arguments (e.g., 'filtered_features' fallback).

        Returns:
            texture_features_master (pd.DataFrame): Combined DataFrame of extracted features.
            texture_features_cols (dict): Dictionary categorizing column names by their origin 
                                          ('antrop', 'ordpy', 'fostat').
        """
        
        antropy_metrics = None
        ordpy_metrics = None
        tsfel_metrics = None
        
        if metrics_list is not None:
            antropy_metrics = []
            ordpy_metrics = []
            tsfel_metrics = []
            
            for m in metrics_list:
                if m.endswith('_antropy'):
                    antropy_metrics.append(m.replace('_antropy', ''))
                elif m.endswith('_ordpy'):
                    ordpy_metrics.append(m.replace('_ordpy', ''))
                else: 
                     tsfel_metrics.append(m)

        
        if self.is_normalize_stat:
            features_tsfel_stat = self.first_ord_sta.generate_tsfel_stat_features(self.img_flat_norm[0], metrics=tsfel_metrics)
        else:
            features_tsfel_stat = self.first_ord_sta.generate_tsfel_stat_features(self.img_flat[0], metrics=tsfel_metrics)
        
        if self.is_normalize_entropy:
            features_antrop_textures = self.antrop_textures.generate_entropy_measures(self.img_flat_norm[0], metrics=antropy_metrics)
            features_ordpy_textures = self.ordpy_textures.generate_ordpy_features(self.img_flat_norm[0], metrics=ordpy_metrics)
        else:
            features_antrop_textures = self.antrop_textures.generate_entropy_measures(self.img_flat[0], metrics=antropy_metrics)
            features_ordpy_textures = self.ordpy_textures.generate_ordpy_features(self.img_flat[0], metrics=ordpy_metrics)

        texture_features_master = pd.concat([features_antrop_textures,features_ordpy_textures,features_tsfel_stat], axis=1)
        texture_features_cols={}
        texture_features_cols['antrop'] = features_antrop_textures.columns
        texture_features_cols['ordpy'] = features_ordpy_textures.columns
        texture_features_cols['fostat'] = features_tsfel_stat.columns
        
        if is_best_features and metrics_list is None:
            best_determined_features = ['detrended_fluctuation_antropy','fisher_shannon_ordpy','0_Lempel-Ziv complexity',
                                        'svd_entropy_antropy','approximate_entropy_antropy']
            texture_features_master = texture_features_master[best_determined_features]
        elif metrics_list is not None:
             # Since we supposedly only computed what was asked, we just return.
             # However, extra check: TSFEL returns '0_Lempel-Ziv complexity' sometimes?
             # TSFEL prefixes feature names with window index if multiple windows?
             # Here fs=1 and input is 1D, tsfel usually returns just the feature name.
             pass
        else:
            filtered_features = kwargs.get('filtered_features',[])
            texture_features_master = texture_features_master[filtered_features] if filtered_features else texture_features_master
            
        return texture_features_master, texture_features_cols

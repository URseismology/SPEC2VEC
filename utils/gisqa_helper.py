##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
# Systems and Standard lib
import argparse
import requests
import io
import os
import glob
import datetime
import random
import h5py
import inspect
import math
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from hilbert import decode, encode
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.spatial import distance_matrix
from scipy.spatial.distance import mahalanobis
from itertools import combinations, product
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler

from SPEC2VEC.utils.noise_lib import *
from SPEC2VEC.utils.spectograms_lib import *
from SPEC2VEC.utils.simple_synth_data_models import *


class HelperFunc:

    @staticmethod
    def read_noisebank(noisebnk_path:str, noisebnk_key:str, imgtype:str):
        """
        Reads a specific dataset from an HDF5 noise bank.

        Args:
            noisebnk_path (str): File path to the HDF5 noise bank.
            noisebnk_key (str): The specific key/group in the HDF5 file.
            imgtype (str): Type of the image/data to extract ('hilbert_image', 'image', 'time_series').

        Returns:
            np.ndarray: The extracted dataset.
        """
        assert imgtype in ['hilbert_image','image', 'time_series'], 'Invalid image type. Must be among "hilbert_image","image","time_series"'
        with h5py.File(noisebnk_path, 'r') as f:
            dataset = f[noisebnk_key][imgtype][:]
        return dataset

    @staticmethod
    def calculate_2dintercluster_distnces(data:pd.DataFrame, x:np.ndarray, features:list=None, group_by:list=['label']):
        """
        Calculates the Euclidean distance between a given vector `x` and the centroids 
        of different clusters in a DataFrame.

        Args:
            data (pd.DataFrame): The input data containing features and labels.
            x (np.ndarray): The target vector (1D) to compare against cluster centroids.
            features (list, optional): List of feature column names to use.
            group_by (list, optional): Columns to group by for centroid calculation. Defaults to ['label'].

        Returns:
            tuple: (Index of the closest cluster, pd.Series of all distances).
        """
        if features:
            data = data[features+group_by]
        cluster_centroids = data.groupby(group_by).mean()
        assert len(cluster_centroids.columns)==len(x.flatten()), 'number of features mismatch between data and x'
        distances = np.sqrt(np.sum((cluster_centroids - x)**2, axis=1).astype('float'))
        return distances.idxmin(), distances

    @staticmethod
    def calculate_best_spectrogram_by_rmse(noise_fos_df:pd.DataFrame, spec_fos_df:pd.DataFrame, target_cols:list=None, 
                                           label_col:str='label', 
                                            method:str='mean', spec_technique:list=None, 
                                            pair_normalize:bool=True,
                                            normalize_with_ref:bool=False, distance_type:str='l2', 
                                            normalize_method:str='MinMax', 
                                            removeoutliers = False, **kwargs):
        """
        Calculates distance metrics between reference noise features and target spectrogram features. 
        Supports various distance metrics and aggregation methods.

        Args:
            noise_fos_df (pd.DataFrame): DataFrame containing reference noise features.
            spec_fos_df (pd.DataFrame): DataFrame containing target spectrogram features.
            target_cols (list, optional): Specific feature columns to compute distance on.
            label_col (str, optional): The column denoting class/labels. Defaults to 'label'.
            method (str, optional): Aggregation method ('mean', 'median', 'min', 'max', 'quantile', 'mode'). Defaults to 'mean'.
            spec_technique (list, optional): Corresponding noise label for each spectrogram label to explicitly map them.
            pair_normalize (bool, optional): Whether to joint-normalize noise and spec features. Defaults to True.
            normalize_with_ref (bool, optional): Standardize based on reference features instead of pair normalizing. Defaults to False.
            distance_type (str, optional): Distance metric to use ('l2', 'cosine', 'cosine_distance', 'corrcoef', 'mahalanobis'). Defaults to 'l2'.
            normalize_method (str, optional): Normalization technique ('MinMax', 'ZScore'). Defaults to 'MinMax'.
            removeoutliers (bool, optional): If True, removes outliers beyond 3 standard deviations before aggregating. Defaults to False.
            **kwargs: Additional arguments such as 'returnall' (bool) and 'quantile_level' (float).

        Returns:
            dict or tuple: A dictionary of aggregated distances mapped by label. If `returnall` is True, 
                           also returns a dictionary of the full unaggregated distance arrays.
        """
        
        spec_labels = spec_fos_df[label_col].unique()
        assert len(spec_labels)==spec_fos_df.shape[0], 'the spectogram dataframe must have one unique row for each label'
        if not target_cols:
            target_cols = ['0_Kurtosis','0_Skewness', '0_Absolute energy','0_Average power','0_Mean absolute deviation','0_Standard deviation']

        returnall = kwargs.get('returnall', False)
        if not spec_technique:
            if pair_normalize:
                y, x = HelperFunc.pair_normalize(noise_fos_df,spec_fos_df,target_cols, method=normalize_method) #not needed to pair normalize if both Spec and Noise are generated by following the same process
            else:
                y, x = noise_fos_df[target_cols].values[:,:], spec_fos_df[target_cols].values[:,:]
            dist_tmp = {}
            for idx, lab in enumerate(spec_labels):
                dist_arr = pd.Series(np.sqrt(np.sum((y-x[idx,:])**2, axis=1)))
                if method=='quantile':
                    quantile_level = kwargs.get('quantile_level',0.1)
                    dist_tmp[lab] = getattr(dist_arr, method)(quantile_level)
                elif method=='median':
                    dist_tmp[lab] = np.median(dist_arr)
                elif method=='mode':
                    dist_tmp[lab] = stats.mode(dist_arr)
                elif method=='min':
                    dist_tmp[lab] = np.min(dist_arr)
                else:
                    dist_tmp[lab] = getattr(np, method)(dist_arr)
                
            return dist_tmp
        
        else:
            assert len(spec_labels)==len(spec_technique), 'Each technique must be associated with only one type of spectrogram'

            ## Uncomment the below if pair normalization need to be performed before
            # noise_fos_df = noise_fos_df.reset_index(drop=True);spec_fos_df = spec_fos_df.reset_index(drop=True)
            # noiselabels = noise_fos_df['label']; speclabels = spec_fos_df['label']
            # ynoisedf, xspecdf = HelperFunc.pair_normalize(noise_fos_df,spec_fos_df,target_cols)
            # noise_fos_df = pd.DataFrame(ynoisedf, columns=target_cols); noise_fos_df['label']=noiselabels
            # spec_fos_df = pd.DataFrame(xspecdf, columns=target_cols); spec_fos_df['label']=speclabels

            dist_tmp = {}
            dist_all = {}
            print(f'Computing for {spec_technique} spectrograms and {spec_labels} labels')
            for lab, key in zip(spec_labels, spec_technique):
                #print(lab,key)
                noise_df = noise_fos_df[noise_fos_df['label']==key][target_cols].reset_index(drop=True)
                specdf = spec_fos_df[spec_fos_df['label']==lab][target_cols].reset_index(drop=True)

                if pair_normalize:
                    noise_df = noise_df.values[:,:]
                    specdf = specdf.values[:,:]
                    noise_df, specdf = HelperFunc.pair_normalize(noise_df,specdf,method=normalize_method)
                elif normalize_with_ref:
                    noise_df, specdf = HelperFunc.standardize_with_reference(noise_df, specdf, target_cols)
                    noise_df = noise_df.values[:,:]
                    specdf = specdf.values[:,:]
                else:
                    noise_df = noise_df.values[:,:]
                    specdf = specdf.values[:,:]

                if distance_type == 'l2':    
                    dist_arr = pd.Series(distance_matrix(noise_df,specdf).flatten())
                    if removeoutliers:
                        dist_arr = HelperFunc.remove_outliers(dist_arr, m=3)

                elif distance_type == 'cosine':
                    dist_arr = pd.Series(cosine_similarity(noise_df,specdf).flatten())
                    if removeoutliers:
                        dist_arr = HelperFunc.remove_outliers(dist_arr, m=3)
                
                elif distance_type == 'cosine_distance':
                    dist_arr = pd.Series(cosine_distances(noise_df,specdf).flatten())
                    if removeoutliers:
                        dist_arr = HelperFunc.remove_outliers(dist_arr, m=3)
                
                elif distance_type == 'corrcoef':
                    dist_arr =  np.array([np.corrcoef(row, specdf[0])[0, 1] for row in noise_df])
                    if removeoutliers:
                        dist_arr = HelperFunc.remove_outliers(dist_arr, m=3)
                
                elif distance_type == 'mahalanobis':
                    # https://jamesmccaffreyblog.com/2022/11/29/mahalanobis-distance-example-using-python/
                    #VIBase = np.concatenate((noise_df,specdf), axis=0)
                    VIBase = noise_df
                    VI = np.linalg.inv(np.cov(VIBase, rowvar=False))
                    #mu = np.mean(noise_df, axis=0)
                    #dist_arr = mahalanobis(specdf[0], mu, VI=VI)
                    dist_arr = np.array([mahalanobis(row, specdf[0], VI=VI) for row in noise_df])
                    if removeoutliers:
                        dist_arr = HelperFunc.remove_outliers(dist_arr, m=3)
                
                if returnall:
                    dist_all[lab] = dist_arr

                if method=='quantile':
                    quantile_level = kwargs.get('quantile_level',0.1)
                    dist_tmp[lab] = getattr(dist_arr, method)(quantile_level)
                elif method=='median':
                    dist_tmp[lab] = np.median(dist_arr)
                elif method=='mode':
                    dist_tmp[lab] = stats.mode(dist_arr)
                elif method=='min':
                    dist_tmp[lab] = np.min(dist_arr)
                elif method=='max':
                    dist_tmp[lab] = np.max(dist_arr)
                elif method=='mean':
                    dist_tmp[lab] = np.mean(dist_arr)
                else:
                    dist_tmp[lab] = getattr(np, method)(dist_arr)
            
            if returnall:
                return dist_tmp, dist_all
            else:
                return dist_tmp
                
    @staticmethod
    def pair_normalize(df1, df2, target_cols:list=None, normalization_range:tuple=(0,1), method='MinMax'):
        """
        Normalizes two DataFrames or arrays jointly to ensure they are on the same scale.

        Args:
            df1 (pd.DataFrame or np.ndarray): First dataset.
            df2 (pd.DataFrame or np.ndarray): Second dataset.
            target_cols (list, optional): Columns to select if DataFrames are passed.
            normalization_range (tuple, optional): Range for MinMax scaling. Defaults to (0, 1).
            method (str, optional): Scaling method ('MinMax' or 'ZScore'). Defaults to 'MinMax'.

        Returns:
            tuple: (df1_scaled, df2_scaled) as numpy arrays.
        """

        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            assert target_cols, 'Please pass the list of columns to work on'
            df1 = df1[target_cols].values
            df2 = df2[target_cols].values

        n_df2 = df2.shape[0]
        stacked_arr = np.concatenate((df1,df2), axis=0)
        if method=='ZScore':
            stacked_arr = stats.zscore(stacked_arr, axis=0, ddof=0)
        elif method=='MinMax':
            stacked_arr = MinMaxScaler(feature_range=normalization_range).fit_transform(stacked_arr)
        df1_scaled = stacked_arr[:-n_df2,:]
        df2_scaled = stacked_arr[-n_df2:,:]
        
        #print('Pair normaliztion completed')
        return df1_scaled, df2_scaled

    @staticmethod
    def standardize_with_reference(ref_df, spec_df, columns:list=None):
        """
        Standardizes two DataFrames using the mean and standard deviation of a reference DataFrame.

        Args:
            ref_df (pd.DataFrame): Reference DataFrame used to compute mean and std.
            spec_df (pd.DataFrame): Target DataFrame to be standardized.
            columns (list): List of columns to standardize.

        Returns:
            tuple: (ref_df_std, spec_df_std) Standardized DataFrames.
        """
        ref_df_std = ref_df.copy()
        spec_df_std = spec_df.copy()
        for col in columns:
            mean = ref_df[col].mean()
            std = ref_df[col].std()
            if std == 0:
                raise ValueError(f"Standard deviation for column '{col}' in reference dataframe is zero. Cannot standardize.")
            else:
                ref_df_std[col] = (ref_df[col] - mean) / std
                spec_df_std[col] = (spec_df[col] - mean) / std
        
        return ref_df_std, spec_df_std

    @staticmethod
    def remove_outliers(arr, m=3):
        """
        Removes outliers from an array based on standard deviation.

        Args:
            arr (array_like): Input data array.
            m (float, optional): Number of standard deviations to use as the threshold. Defaults to 3.

        Returns:
            np.ndarray: Array with outliers removed.
        """
        # Q1 = np.percentile(arr, 25)
        # Q3 = np.percentile(arr, 75)
        # IQR = Q3 - Q1
        # lower = Q1 - 1.5*IQR
        # upper = Q3 + 1.5*IQR
        # filtered_arr = arr[(arr >= lower) & (arr <= upper)]

        arr = np.asarray(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        filtered = arr[np.abs((arr - mean) / std) < m]
        return filtered

    @staticmethod
    def scale_percentile(data, lower_percentile=1, upper_percentile=99, scale_type='-1to1'):
        """
        Scales the input data to the range [-1, 1] based on percentile values.

        Args:
            data (np.ndarray): The input data to be scaled. Can be any shape.
            lower_percentile (float): The percentile to use as the minimum value for scaling.
                                    Must be between 0 and 100. Defaults to 1.
            upper_percentile (float): The percentile to use as the maximum value for scaling.
                                    Must be between 0 and 100. Defaults to 98.

        Returns:
            np.ndarray: The scaled data with values between -1 and 1.
        """
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError("lower_percentile must be less than upper_percentile and both must be between 0 and 100.")

        flat_data = data.flatten()
        min_val = np.percentile(flat_data, lower_percentile)
        max_val = np.percentile(flat_data, upper_percentile)

        # Avoid division by zero if max_val is equal to min_val
        if max_val == min_val:
            # If all values are the same (or within the percentile range),
            # return a tensor of zeros scaled appropriately.
            return np.zeros_like(flat_data, dtype=data.dtype)

        if scale_type == '-1to1':
            scaled_data = 2 * (flat_data - min_val) / (max_val - min_val) - 1
            scaled_data = np.clip(scaled_data, -1, 1)
        
        elif scale_type == 'MinMax':
            ## Apply sclaing formula to scale x from [X_min, X_max] to [Y_min, Y_max]: Y = Y_min + ((x - X_min) * (Y_max - Y_min)) / (X_max - X_min)
            scaled_data = min_val + ((flat_data - flat_data.min())*((max_val - min_val))) / (flat_data.max() - flat_data.min())
        
        elif scale_type == '0to1':
            ## Apply 0-1 scaling with scale percentile
            scaled_data = (flat_data - min_val) / (max_val - min_val)
            scaled_data = np.clip(scaled_data, 0, 1)
        else:
            raise AssertionError('Invalid scale type, scale type must be between [-1to1, 0to1, MinMax]')

        return scaled_data

    @staticmethod
    def standardize_percentile(data, lower_percentile=1, upper_percentile=99, scale_type='-1to1'):
        """
        Scales the input data to the range [-1, 1] based on percentile values.

        Args:
            data (np.ndarray): The input data to be scaled. Can be any shape.
            lower_percentile (float): The percentile to use as the minimum value for scaling.
                                    Must be between 0 and 100. Defaults to 1.
            upper_percentile (float): The percentile to use as the maximum value for scaling.
                                    Must be between 0 and 100. Defaults to 98.

        Returns:
            np.ndarray: The scaled data with values between -1 and 1.
        """
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError("lower_percentile must be less than upper_percentile and both must be between 0 and 100.")

        flat_data = data.flatten()
        min_val = np.percentile(flat_data, lower_percentile)
        max_val = np.percentile(flat_data, upper_percentile)

        # Avoid division by zero if max_val is equal to min_val
        if max_val == min_val:
            # If all values are the same (or within the percentile range),
            # return a tensor of zeros scaled appropriately.
            return np.zeros_like(flat_data, dtype=data.dtype)

        if scale_type == '-1to1':
            scaled_data = 2 * (flat_data - min_val) / (max_val - min_val) - 1
            scaled_data = np.clip(scaled_data, -1, 1)
            
            mean = np.mean(scaled_data)
            std = np.std(scaled_data)
            standardized = (scaled_data - mean) / std
        
        elif scale_type == 'MinMax':
            ## Apply sclaing formula to scale x from [X_min, X_max] to [Y_min, Y_max]: Y = Y_min + ((x - X_min) * (Y_max - Y_min)) / (X_max - X_min)
            scaled_data = min_val + ((flat_data - flat_data.min())*((max_val - min_val))) / (flat_data.max() - flat_data.min())

            mean = np.mean(scaled_data)
            std = np.std(scaled_data)
            standardized = (scaled_data - mean) / std
        
        elif scale_type == '0to1':
            ## Apply 0-1 scaling with scale percentile
            scaled_data = (flat_data - min_val) / (max_val - min_val)
            scaled_data = np.clip(scaled_data, 0, 1)

            mean = np.mean(scaled_data)
            std = np.std(scaled_data)
            standardized = (scaled_data - mean) / std

        else:
            raise AssertionError('Invalid scale type, scale type must be between [-1to1, 0to1, MinMax]')

        return standardized

    @staticmethod
    def gilbertize_image(width:int=128, height:int=128):

        def gilbert_d2xy(idx, w, h):
            """
            Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
            2D rectangular grids. Takes a position along the gilbert curve and returns
            its 2D (x,y) coordinate.

            # SPDX-License-Identifier: BSD-2-Clause
            # Copyright (c) 2024 abetusk
            """

            if w >= h:
                return gilbert_d2xy_r(idx,0, 0,0, w,0, 0,h)
            return gilbert_d2xy_r(idx,0, 0,0, 0,h, w,0)

        def sgn(x):
            return -1 if x < 0 else (1 if x > 0 else 0)

        def gilbert_d2xy_r(dst_idx, cur_idx, x,y, ax,ay, bx,by):

            w = abs(ax + ay)
            h = abs(bx + by)

            (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
            (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

            dx = dax + dbx
            dy = day + dby
            di = dst_idx - cur_idx

            if h == 1: return (x + dax*di, y + day*di)
            if w == 1: return (x + dbx*di, y + dby*di)

            (ax2, ay2) = (ax//2, ay//2)
            (bx2, by2) = (bx//2, by//2)

            w2 = abs(ax2 + ay2)
            h2 = abs(bx2 + by2)

            if 2*w > 3*h:
                if (w2 % 2) and (w > 2):
                    # prefer even steps
                    (ax2, ay2) = (ax2 + dax, ay2 + day)


                # long case: split in two parts only
                nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by))
                if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
                    return gilbert_d2xy_r(dst_idx, cur_idx,  x, y, ax2, ay2, bx, by)
                cur_idx = nxt_idx

                return gilbert_d2xy_r(dst_idx, cur_idx, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

            if (h2 % 2) and (h > 2):
                # prefer even steps
                (bx2, by2) = (bx2 + dbx, by2 + dby)

            # standard case: one step up, one long horizontal, one step down
            nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2))
            if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
                return gilbert_d2xy_r(dst_idx, cur_idx, x,y, bx2,by2, ax2,ay2)
            cur_idx = nxt_idx

            nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)))
            if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
                return gilbert_d2xy_r(dst_idx, cur_idx, x+bx2, y+by2, ax,ay, bx-bx2,by-by2)
            cur_idx = nxt_idx

            return gilbert_d2xy_r(dst_idx, cur_idx,
                                x+(ax-dax)+(bx2-dbx),
                                y+(ay-day)+(by2-dby),
                                -bx2, -by2,
                                -(ax-ax2), -(ay-ay2))


        locs = []
        for idx in range(width*height):
            (x,y) = gilbert_d2xy(idx, width, height)
            locs.append([x,y])
        return np.array(locs)

    @staticmethod
    def read_ref_pointwise_metrics_df(filelist:list=None, target_cols:list=None, ref_noise_df:pd.DataFrame=None, iscolrename:bool=False):
        """
        Reads and optionally concatenates reference pointwise metrics from a list of CSV files.

        Args:
            filelist (list, optional): List of file paths to CSV files containing metrics.
            target_cols (list, optional): Columns to keep from the loaded DataFrames.
            ref_noise_df (pd.DataFrame, optional): Pre-loaded DataFrame to use if filelist is not provided.
            iscolrename (bool, optional): If True, appends specific suffixes to columns based on file order. Defaults to False.

        Returns:
            pd.DataFrame: The concatenated and filtered reference metrics DataFrame.
        """
        assert filelist or ref_noise_df, 'Either a filepath or a dataframe must be provided for reference noise data'
        
        if not ref_noise_df:
            nfiles = len(filelist)
            ref_noise_df = pd.DataFrame()
            for idx, file_path in enumerate(filelist):
                if os.path.exists(file_path):
                    temp_df = pd.read_csv(file_path)
                    if 'Unnamed: 0' in temp_df.columns:
                        temp_df = temp_df.drop('Unnamed: 0', axis=1)

                    if iscolrename:
                        colm_suffix = ['_antropy','_ordpy','']
                        cols_to_rename = temp_df.columns[:-1]
                        temp_df = temp_df.rename(columns={col: col + colm_suffix[idx] for col in cols_to_rename})
                        if colm_suffix[idx]:
                            temp_df.columns = temp_df.columns.str.replace(' ', '_')
                    
                    temp_df = temp_df.sort_values(by='label').reset_index(drop=True)
                    ref_noise_df = pd.concat([ref_noise_df, temp_df], axis=1)
                else:
                    print(f"The file does not exist.")
        
        if ref_noise_df['label'].ndim>1 and ref_noise_df['label'].columns.duplicated().any():
            print('Duplicate label columns found, checking integrity...')
            labels_are_identical = ref_noise_df['label'].apply(lambda row: row[0] == row[1] == row[2], axis=1).all()
            if labels_are_identical:
                ref_noise_df = ref_noise_df.loc[:,~ref_noise_df.columns.duplicated()].copy()
                cols = ref_noise_df.columns.tolist()
                cols.remove('label'); cols.append('label')
                ref_noise_df = ref_noise_df[cols]
            else:
                raise ValueError("Integrity error. Labels are not identical across datasets")

        if target_cols:
            ref_noise_df = ref_noise_df[target_cols] if 'label' in target_cols else ref_noise_df[target_cols+['label']]
        
        return ref_noise_df

    @staticmethod
    def train_randomforest_on_noisebnk(inputdf, labelcol='label', standardize=True, test_size=0.3, plot_cm=False, 
                                       labelmap:dict=None, savepath:str=None, rf_params:dict=None):
        """
        Trains a Random Forest classifier on a provided dataset of noise or spectrogram features.
        Not specific to any application. Can be used on any data to train a RF.

        Args:
            inputdf (pd.DataFrame): The input data containing features and labels.
            labelcol (str, optional): The column name for the target labels. Defaults to 'label'.
            standardize (bool, optional): Whether to scale features using StandardScaler. Defaults to True.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.3.
            plot_cm (bool, optional): If True, plots the confusion matrix on the test set. Defaults to False.
            labelmap (dict, optional): Dictionary to map label codes back to descriptive strings for plotting.
            savepath (str, optional): File path to save the confusion matrix plot.
            rf_params (dict, optional): Dictionary of parameters for the RandomForestClassifier.

        Returns:
            tuple: (classifier, all_features, label_to_code_dict, test_data_tuple, train_data_tuple)
        """
        
        if rf_params is None:
            print("Using Default Random Forest parameters")
            rf_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }

        df = inputdf.copy(); del inputdf
        df['y'] = df[labelcol].astype('category').cat.codes
        #all_features = df.columns.tolist()[:-2]
        all_features = [col for col in df.columns if col not in [labelcol, 'y']] #list(set(df.columns.tolist()) - set([labelcol, 'y']))
        X = df[all_features]
        y = df['y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
        label_to_code_dict = df[[labelcol, 'y']].drop_duplicates().set_index(labelcol)['y'].to_dict()

        ## Standardize
        if standardize:
            scaler = StandardScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            print('Standardization applied safely')


        #classifier = RandomForestClassifier(random_state=142)
        classifier = RandomForestClassifier(**rf_params)
        classifier.fit(X_train, y_train)
        print(f'Model Finished training on {len(all_features)} features and {X_train.shape[0]} data points')
        y_test_predict = classifier.predict(X_test)
        y_train_predict = classifier.predict(X_train)

        accuracy = accuracy_score(y_test, y_test_predict) * 100
        print("Accuracy from test set: " + str(accuracy) + "%")
        print(f"Accuracy from train set: {accuracy_score(y_train, y_train_predict) * 100}%")

        if plot_cm:
            unique_classes = sorted(list(set(y_test) | set(y_test_predict)))
            code_to_label_dict = {v: k for k, v in label_to_code_dict.items()}

            if labelmap:
                #filtered_keys = [labelmap[x] for x in list(label_to_code_dict.keys())]
                filtered_keys = [labelmap[code_to_label_dict[c]] for c in unique_classes]
            else:
                #filtered_keys = dict(sorted(label_to_code_dict.items(), key=lambda item: item[1]))
                #filtered_keys = list(filtered_keys.keys())
                filtered_keys = [code_to_label_dict[c] for c in unique_classes]

            #cm = confusion_matrix(y_test, y_test_predict)
            cm = confusion_matrix(y_test, y_test_predict, labels=unique_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',annot_kws={"size": 7},
                        xticklabels=filtered_keys,
                        yticklabels=filtered_keys)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(fontsize=7);plt.yticks(fontsize=7)
            plt.title('Confusion Matrix (Test Set)')
            plt.tight_layout()
            if savepath:
                plt.savefig(savepath, dpi=300)
            plt.show()

        return classifier, all_features, label_to_code_dict, (X_test.index, X_test, y_test, y_test_predict), (X_train.index, X_train, y_train, y_train_predict)

    @staticmethod
    def print_feature_importances_shap_values(shap_values, features, isplot=0):
        """
        Calculates and optionally plots feature importances based on mean absolute SHAP values.

        Args:
            shap_values (shap.Explanation or np.ndarray): Calculated SHAP values for the dataset.
            features (list): List of feature names corresponding to the SHAP values.
            isplot (int/bool, optional): If True, displays a bar plot of the feature importances. Defaults to false/0.

        Returns:
            tuple: (feature_importances_dict, feature_importances_df) sorted by descending importance.
        """
        
        # importances = []
        # for i in range(shap_values.values.shape[1]):
        #     importances.append(np.mean(np.abs(shap_values.values[:, i])))
        # feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
        # feature_importances_df = pd.DataFrame.from_dict(feature_importances, orient='index', columns=['vals']).reset_index()

        if len(shap_values.values.shape) == 3:
            print("Shap Importance Computation for Multiclass problem")
            #importances = np.mean(np.abs(shap_values.values), axis=(0, 2))
            mean_abs_shap_per_class = np.mean(np.abs(shap_values.values), axis=0)
            importances = mean_abs_shap_per_class.sum(axis=1)

        else:
            print("Shap Importance Computation for Binary/Regression problem")
            importances = np.mean(np.abs(shap_values.values), axis=0)
        
        feature_importances_df = pd.DataFrame({'index': features,'vals': importances})
        feature_importances_df = feature_importances_df.sort_values(by='vals', ascending=False).reset_index(drop=True)
        feature_importances = dict(zip(feature_importances_df['index'], feature_importances_df['vals']))

        if isplot:
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importances_df.sort_values(by='vals', ascending=False), x='vals', y='index', orient='h')
            plt.yticks(fontsize=8)
            plt.title('Feature Importance (based on Mean Absolute SHAP Across Samples and Classes', fontsize = 10)
            plt.tight_layout()
            plt.show()

        return feature_importances, feature_importances_df.sort_values(by='vals', ascending=False)

    @staticmethod
    def gen_wavelet_names(Fs, num_wavelets=5):
        """
        Generates a list of continuous Morlet wavelet names (cmor) with varying base and detail frequencies.

        Args:
            Fs (float): Sampling frequency of the signal.
            num_wavelets (int, optional): Number of wavelet names to generate. Defaults to 5.

        Returns:
            list: List of generated wavelet name strings formatted for PyWavelets.
        """
        wavelet_names = []
        base_ratios = [0.015, 0.025, 0.050, 0.075, 0.100]
        detail_ratios = [0.010, 0.015, 0.030, 0.040, 0.070]

        for i in range(num_wavelets):
            base_freq = Fs * base_ratios[i]
            detail_freq = Fs * detail_ratios[i]
            wavelet_name = f'cmor{base_freq:.1f}-{detail_freq:.1f}'
            wavelet_names.append(wavelet_name)
        
        return wavelet_names

    @staticmethod
    def compute_any_spec(signal:np.ndarray, sr:int, specname:str='stft', **kwargs):
        """
        Computes a spectrogram (STFT or CWT) for a given signal using dynamically provided parameters.

        Args:
            signal (np.ndarray): 1D time-domain signal.
            sr (int): Sampling rate of the signal.
            specname (str, optional): Type of spectrogram ('stft' or 'cwt'). Defaults to 'stft'.
            **kwargs: Configuration arguments such as 'overlap', 'winlen', 'window', 'fscale', 
                      'wavelet', 'vmin', 'vmax', 'f_min', 'f_max', 'max_normalize', 'powerlog'.

        Returns:
            tuple: (f, t, spectro) Frequency array, time array, and 2D spectrogram matrix.
        """
        
        if specname == 'stft':
            overlap = kwargs.get(overlap, 0.5)
            winlen = kwargs.get(winlen, 64)
            window = kwargs.get(window,  "hamming")
            vmin = kwargs.get(vmin, -500)
            vmax = kwargs.get(vmax, 500)
            f_min = kwargs.get(f_min, 0)
            f_max = kwargs.get(f_max, 128)
            max_normalize = kwargs.get(max_normalize, 'True')
            powerlog = kwargs.get(powerlog, 'True')

            f, t, spectro = stft_ohasisbio_basic_spectogram(signal,sr,winlen,overlap,window,vmin,vmax,f_min,f_max,max_normalize=max_normalize,powerlog=powerlog)
            return f, t, spectro

        elif specname =='cwt':
            fscale = kwargs.get(fscale, (1,128,1))
            wavelet = kwargs.get(wavelet, None)
            fscaletype = kwargs.get(fscaletype, 'Linear')
            vmin = kwargs.get(vmin, -500)
            vmax = kwargs.get(vmax, 500)
            f_min = kwargs.get(f_min, 0)
            f_max = kwargs.get(f_max, 128)
            max_normalize = kwargs.get(max_normalize, 'True')
            powerlog = kwargs.get(powerlog, 'True')

            f, t, spectro, _ = cwt_simple(signal=signal, sr=sr, fscale={'start':fscale[0], 'end':fscale[1], 'num':fscale[2]}, wavelet=wavelet, fscaletype=fscaletype,
                                        vmin=vmin, vmax=vmax, f_min=f_min, f_max=f_max, max_normalize=max_normalize, powerlog=powerlog)

            return f, t, spectro

        else:
            raise AssertionError('Not a valid spectrogram')

    @staticmethod
    def weighting_function(N: int, function_type: str = 'linear', start_value: float = 0.1, end_value: float = 1.0, k: float = 2.0) -> np.ndarray:
        """
        Generates a 1D weighting function of a given length N.

        Args:
            N (int): The length of the weighting function array.
            function_type (str): The type of function to generate. Options are 'linear',
                                'exponential', or 'gamma'. Defaults to 'linear'.
            start_value (float): The starting value of the weighting function. Defaults to 0.1.
            end_value (float): The ending value of the weighting function. Defaults to 1.0.
            k (float): A steepness parameter for the 'exponential' and 'gamma' types.
                    For 'exponential', k controls the growth rate.
                    For 'gamma', k is the power. Defaults to 2.0.

        Returns:
            np.ndarray: A 1D NumPy array representing the weighting function.
        
        Raises:
            ValueError: If an unknown function_type is provided.
        """
        if N <= 0:
            return np.array([])
        
        # Create a normalized linear space from 0 to 1
        t = np.linspace(0, 1, N)

        if function_type == 'linear':
            # Linearly interpolate between start_value and end_value
            return np.linspace(start_value, end_value, N)

        elif function_type == 'exponential':
            # Generate an exponential curve and scale it to the desired range
            # The base exponential curve starts at 1 and grows to exp(k)
            exponential_curve = np.exp(k * t)
            
            # Normalize and scale the curve to the desired [start_value, end_value] range
            normalized_curve = (exponential_curve - exponential_curve.min()) / (exponential_curve.max() - exponential_curve.min())
            return start_value + normalized_curve * (end_value - start_value)
        
        elif function_type == 'gamma':
            # Generate a power-law curve (t^k) and scale it to the desired range
            # k > 1 makes the curve concave up (slow start, fast end)
            # k < 1 makes the curve concave down (fast start, slow end)
            gamma_curve = t ** k
            
            # Scale the curve to the desired [start_value, end_value] range
            # Since t^k already goes from 0 to 1, this is a simple linear transformation
            return start_value + gamma_curve * (end_value - start_value)

        else:
            raise ValueError(f"Unknown function_type: '{function_type}'. Choose from 'linear', 'exponential', or 'gamma'.")

    @staticmethod
    def gini_index(spectrogram, method='trapezoidal', hilbertize=False, denoise_thresh=False):
        """
        Calculates the Gini index to quantify the energy concentration of a spectrogram.

        For small n, the trapezoidal method provides a more accurate estimation of the 
        theoretical Gini coefficient since it accounts for bin-centered areas. For large n, 
        both the Riemann sum and trapezoidal methods converge.

        Args:
            spectrogram (np.ndarray): 2D spectrogram array.
            method (str, optional): 'trapezoidal' or 'reimann'. Defaults to 'trapezoidal'.
            hilbertize (bool, optional): If True, flattens using a Hilbert curve. Defaults to False.
            denoise_thresh (bool, optional): If True, zeros out values below the 75th percentile. Defaults to False.

        Returns:
            float: The computed Gini index (between 0 and 1).
        """
        s = np.array([HelperFunc.scale_percentile(row.reshape(-1, 1),
                                                    lower_percentile=1,upper_percentile=100,
                                                    scale_type='0to1').flatten() 
                                                    for row in spectrogram.reshape(1,-1)]).reshape(spectrogram.shape)

        if denoise_thresh:
            s[s<np.percentile(s,75)]=0

        if hilbertize:
            locs = HelperFunc.gilbertize_image(width=spectrogram.shape[0], height=spectrogram.shape[1])
            s = s[locs[:,0], locs[:,1]].flatten()
        else:
            s = s.flatten()
            s = np.sort(s) 

        n = len(s)
        s_norm = s / np.sum(s)  # Normalize to probability distribution
        indices = np.arange(1, n + 1)

        if method == 'reimann':
            ## Edge Point Version, Reimann Sum based Lorenz Curve area approximation
            G = (1/n)*(n + 1 - 2*(np.sum((n - indices + 1)*s_norm)))
        
        elif method == 'trapezoidal':
            ## Mid Point Version, Trapezoidal Rule based Lorenz Curve area approximation
            G =  1 - 2 * np.sum(s_norm * (n - indices + 0.5) / n)

        return G
    
    @staticmethod
    def energy_decay_frac(spectrogram, energy_frac=0.90):
        """
        Calculates the fraction of sorted coefficients needed to capture a target 
        percentage of the total energy in a spectrogram.

        A lower fraction indicates that the energy is concentrated in fewer coefficients.

        Args:
            spectrogram (np.ndarray): 2D spectrogram array.
            energy_frac (float, optional): Target energy fraction (e.g., 0.90 for 90%). Defaults to 0.90.

        Returns:
            float: Fraction of total coefficients required to reach the target energy.
        """
        s = np.array([HelperFunc.scale_percentile(row.reshape(-1, 1),
                                                    lower_percentile=1, upper_percentile=100,
                                                    scale_type='0to1').flatten() 
                                                    for row in spectrogram.reshape(1,-1)]).reshape(spectrogram.shape)
         
        s = s.flatten()
        s_sorted = np.sort(s)[::-1]
        energy = s_sorted #s_sorted** 2 # Not using squared values since spectrogram values are already squared
        cum_energy = np.cumsum(energy)
        total_energy = cum_energy[-1]
        cum_energy_normalized = cum_energy / total_energy
        n_coeffs = np.searchsorted(cum_energy_normalized, energy_frac) + 1
        return n_coeffs / len(s)

    @staticmethod
    def compute_series_of_tfrs(final_signal, fs=250, T=10, vmin=1, vmax=99, f_min=0, f_max=125, interp_shape=None):
        """
        Computes a standardized series of STFT and CWT time-frequency representations (TFRs)
        for a given signal, primarily for SpecMaster validation experiments.

        Args:
            final_signal (np.ndarray): 1D time-domain signal.
            fs (int, optional): Sampling frequency. Defaults to 250.
            T (int, optional): Total duration of the signal in seconds. Defaults to 10.
            vmin (int, optional): Minimum percentile for scaling. Defaults to 1.
            vmax (int, optional): Maximum percentile for scaling. Defaults to 99.
            f_min (int, optional): Minimum frequency bound. Defaults to 0.
            f_max (int, optional): Maximum frequency bound. Defaults to 125.
            interp_shape (tuple, optional): Target (height, width) to interpolate all TFRs to.

        Returns:
            tuple: (imgs, names, ts, fs) Lists containing the 2D arrays, their string labels, 
                   time vectors, and frequency vectors respectively.
        """

        ## Spec Params
        window = "hamming"
                        
        ## Construct Signals
        f1, t1, spectro = stft_basic_spectogram(final_signal,fs,64,0.5,window,f_min,f_max,max_normalize=True, powerlog=True,vmin_percentile=vmin, vmax_percentile=vmax)
        image_spectro1 = spectro.copy()

        f2, t2, spectro = stft_basic_spectogram(final_signal,fs,128,0.5,window,f_min,f_max,max_normalize=True, powerlog=True,vmin_percentile=vmin, vmax_percentile=vmax)
        image_spectro2 = spectro.copy()

        f3, t3, spectro = stft_basic_spectogram(final_signal,fs,128,0.75,window,f_min,f_max,max_normalize=True, powerlog=True,vmin_percentile=vmin, vmax_percentile=vmax)
        image_spectro3 = spectro.copy()

        f4, t4, spectro = stft_basic_spectogram(final_signal,fs,250,0.5,window,f_min,f_max,max_normalize=True, powerlog=True,vmin_percentile=vmin, vmax_percentile=vmax)
        image_spectro4 = spectro.copy()
    
        f5, t5, spectro, _ = cwt_simple(signal=final_signal, sr=fs, fscale={'start':1, 'end':128, 'num':1}, wavelet="cmor2.0-2.0", fscaletype='linear',
                                        vmin_percentile=vmin, vmax_percentile=vmax, f_min=f_min, f_max=f_max, max_normalize=True, powerlog=True, decimate_factor=20)
        image_spectro5 = spectro.copy()

        f6, t6, spectro, _ = cwt_simple(signal=final_signal, sr=fs, fscale={'start':1, 'end':128, 'num':1}, wavelet="cmor3.0-2.5", fscaletype='linear',
                                        vmin_percentile=vmin, vmax_percentile=vmax, f_min=f_min, f_max=f_max, max_normalize=True, powerlog=True, decimate_factor=20)
        image_spectro6 = spectro.copy()

        f7, t7, spectro, _ = cwt_simple(signal=final_signal, sr=fs, fscale={'start':1, 'end':128, 'num':1}, wavelet="cmor5.0-3.0", fscaletype='linear',
                                        vmin_percentile=vmin, vmax_percentile=vmax, f_min=f_min, f_max=f_max, max_normalize=True, powerlog=True, decimate_factor=20)
        image_spectro7 = spectro.copy()

        f8, t8, spectro, tmp = cwt_simple(signal=final_signal, sr=fs, fscale={'start':1, 'end':128, 'num':1}, wavelet="cmor8.0-4.0", fscaletype='linear',
                                        vmin_percentile=vmin, vmax_percentile=vmax, f_min=f_min, f_max=f_max, max_normalize=True, powerlog=True, decimate_factor=20)
        image_spectro8 = spectro.copy()

        all_imgs = [image_spectro1, image_spectro2, image_spectro3, image_spectro4, 
                    image_spectro5, image_spectro6, image_spectro7, image_spectro8]
        all_names = ['TFR1','TFR2','TFR3','TFR4',
                     'TFR5','TFR6','TFR7','TFR8']
        all_t = [t1,t2,t3,t4,
                 t5,t6,t7,t8]
        all_f = [f1,f2,f3,f4,
                 f5,f6,f7,f8]
        
        interp_imgs = []
        interp_ts = []
        interp_fs = []

        if interp_shape:
            print(f'Interpolating all TFRs to shape: {interp_shape}')
            for img, t, f in zip(all_imgs, all_t, all_f):
                # Interpolate image
                t_new = np.linspace(t.min(), t.max(), interp_shape[1])
                f_new = np.linspace(f.min(), f.max(), interp_shape[0])
                interp_func = RegularGridInterpolator((f, t), img, method='slinear', bounds_error=False, fill_value=None) # cant do nearest interpolatation becuase it makes the smallest image the best becuase of repeating values #using nearest since it maintains the texture; using pchip since we have overshoot in the images
                # meshgrid for new coordinates
                T_new, F_new = np.meshgrid(t_new, f_new)
                points = np.array([F_new.flatten(), T_new.flatten()]).T
                img_interp = interp_func(points).reshape(interp_shape)
                # img_interp = interp_func(t_new, f_new)
                interp_imgs.append(img_interp)
                interp_ts.append(np.linspace(0, T, interp_shape[1]))
                interp_fs.append(np.linspace(f_min, f_max, interp_shape[0]))
            
            return interp_imgs, all_names, interp_ts, interp_fs

        return all_imgs, all_names, all_t, all_f

    @staticmethod
    def interpolate_tfrs(t, f, img, interp_shape=(128,148), T=10, f_min=0, f_max=125):
        """
        Interpolates a time-frequency representation (TFR) to a specific target shape.
        
        Uses 'slinear' (1st order spline) interpolation as it maintains texture while smoothing 
        the image, providing a balance between nearest-neighbor (repeating values) and 
        higher-order splines (overshooting).

        Args:
            t (np.ndarray): Original time vector.
            f (np.ndarray): Original frequency vector.
            img (np.ndarray): 2D TFR array.
            interp_shape (tuple, optional): Target (height, width) for the interpolated image. Defaults to (128,148).
            T (int, optional): Total time duration. Defaults to 10.
            f_min (int, optional): Minimum frequency bound. Defaults to 0.
            f_max (int, optional): Maximum frequency bound. Defaults to 125.

        Returns:
            tuple: (interp_t, interp_f, img_interp) The new time vector, new frequency vector, 
                   and the interpolated 2D image.
        """
        
        t_new = np.linspace(t.min(), t.max(), interp_shape[1])
        f_new = np.linspace(f.min(), f.max(), interp_shape[0])
        interp_func = RegularGridInterpolator((f, t), img, method='slinear', bounds_error=False, fill_value=None) 

        T_new, F_new = np.meshgrid(t_new, f_new)
        points = np.array([F_new.flatten(), T_new.flatten()]).T
        img_interp = interp_func(points).reshape(interp_shape) 

        interp_t = np.linspace(0, T, interp_shape[1])
        interp_f = np.linspace(f_min, f_max, interp_shape[0])  
        return interp_t, interp_f, img_interp

    @staticmethod
    def get_file_details(file_path:str=None):
        """
        Prints and extracts file metadata such as size, creation time, and modification time.

        Args:
            file_path (str, optional): The absolute or relative path to the file.
        
        Raises:
            ValueError: If the file path is None or does not exist.
        """
        if not file_path or not os.path.exists(file_path):
            raise ValueError("A valid file path must be provided.")
        
        file_stats = os.stat(file_path)
        file_details = {
            'File Path': file_path,
            'Size (bytes)': file_stats.st_size,
            'Last Modified': datetime.datetime.fromtimestamp(file_stats.st_mtime),
            'Created': datetime.datetime.fromtimestamp(file_stats.st_ctime)
        }

        for key, value in file_details.items():
            print(f"{key}: {value}")

    @staticmethod
    def gilbertize_image_optimized(width: int = 128, height: int = 128):
        """
        Optimized Generalized Hilbert ('gilbert') space-filling curve generator for arbitrary-sized
        2D rectangular grids.
        
        This implementation uses a recursive generator approach to produce coordinates in order,
        avoiding the O(N log N) complexity of the original indexed implementation.
        Complexity: O(N) where N = width * height.
        """
        
        def sgn(x):
            return -1 if x < 0 else (1 if x > 0 else 0)

        def generate(x, y, ax, ay, bx, by):
            w = abs(ax + ay)
            h = abs(bx + by)

            (dax, day) = (sgn(ax), sgn(ay))
            (dbx, dby) = (sgn(bx), sgn(by))

            if h == 1:
                # Base case: just a line along the major axis
                for i in range(w):
                    yield (x + dax * i, y + day * i)
                return

            if w == 1:
                # Base case: just a line along the orthogonal axis
                for i in range(h):
                    yield (x + dbx * i, y + dby * i)
                return

            (ax2, ay2) = (ax // 2, ay // 2)
            (bx2, by2) = (bx // 2, by // 2)

            w2 = abs(ax2 + ay2)
            h2 = abs(bx2 + by2)

            if 2 * w > 3 * h:
                if (w2 % 2) and (w > 2):
                    (ax2, ay2) = (ax2 + dax, ay2 + day)

                # Long case: split in two
                yield from generate(x, y, ax2, ay2, bx, by)
                yield from generate(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)

            else:
                if (h2 % 2) and (h > 2):
                    (bx2, by2) = (bx2 + dbx, by2 + dby)

                # Standard case: split in three
                # 1. First part (swapped axes)
                yield from generate(x, y, bx2, by2, ax2, ay2)
                
                # 2. Second part (standard orientation)
                yield from generate(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
                
                # 3. Third part (swapped axes, adjusted position)
                yield from generate(
                    x + (ax - dax) + (bx2 - dbx),
                    y + (ay - day) + (by2 - dby),
                    -bx2, -by2,
                    -(ax - ax2), -(ay - ay2)
                )

        if width >= height:
            gen = generate(0, 0, width, 0, 0, height)
        else:
            gen = generate(0, 0, 0, height, width, 0)
            
        return np.array(list(gen))

    @staticmethod
    def estimate_snr(trace_data):
        """
        Estimates the Signal-to-Noise Ratio (SNR) of a 1D trace.

        The noise floor is estimated as the median of the absolute amplitudes, 
        and the signal amplitude is estimated as the 95th percentile.

        Args:
            trace_data (np.ndarray): 1D array representing the signal trace.

        Returns:
            float: The estimated SNR ratio (not in decibels). Returns 0 if noise floor is 0.
        """
        max_val = np.max(np.abs(trace_data))
            
        trace_norm = trace_data / max_val
        abs_data = np.abs(trace_norm)    
        noise_floor = np.median(abs_data)
        
        if noise_floor == 0:
            return 0 
            
        signal_amp = np.percentile(abs_data, 95)
        snr_estimate = signal_amp / noise_floor
        return snr_estimate


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
from itertools import combinations
from hilbert import decode, encode
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from jwave.geometry import *
import porespy as ps
import inspect
inspect.signature(ps.generators.fractal_noise)

# UPDATED IMPORT
from SPEC2VEC.utils.gisqa_pairwise_metrics import *
from SPEC2VEC.utils.gisqa_plotting import *
from SPEC2VEC.utils.gisqa_pointwise_metrics_updated import *
from SPEC2VEC.utils.gisqa_helper import *
from SPEC2VEC.utils.complex_waveform_models import Simulator

class GISQAPipeline:
    def __init__(self):
        ## update the pointwise or pairwise class when invoked
        self.pafe = None 
        self.pofe = None
        self.best_pointwise_metrics = ['detrended_fluctuation_antropy','fisher_shannon_ordpy','0_Lempel-Ziv complexity',
                                        'svd_entropy_antropy','approximate_entropy_antropy']
    
    def execute_pairwise_pipleine(self, t_plot:np.ndarray, f_plot:np.ndarray, spec_img: np.ndarray, 
                                  noisebnk_path: str, noisebnk_keys: list, binary_keys: list, 
                                  hist_bins:int=512, noise_cutoff:int=90, corr_only: bool=False, 
                                  gmm_ncomp:int=5, gmm_mean_threshold:float=0.05, 
                                  gmm_nbins:int=256, spec_normalize='full', min_zone_size_thrshld:int=5, 
                                  filename:str=None, save_path:str=None, 
                                  save_all_plots:bool=True, best_neighbour_key:str=None,
                                  optimization_params={'type':'exponential','start':0.1,'end':100,'slope':10},
                                  scale_percentile_range=(1,100), **plotkwargs):
        """
        Compute GLZM Metrics based on best Histogra Fit
        Optinally Plots all the Steps    
        """
        ## initialize the pairwise operations class
        pafe = pairwise_feature_extractor()
        spec_img_reshaped = spec_img.reshape(1,-1)
        spec_image_updated = spec_img_reshaped.copy()
        spec_image_noise_filt = spec_img.copy()
        
        ## for a spectrogram, find the nearest img bank and correlate to allign distribution
        realigned_histograms, original_histrograms, global_bins, additional_params = pafe.find_nearest_imgbank_neighbour_bysignalfit(spec_img_reshaped,noisebnk_path,noisebnk_keys,
                                                                                                                                        binary_keys,corr_only,noise_cutoff,hist_bins,
                                                                                                                                        optimization_params=optimization_params,
                                                                                                                                        best_neighbour_key=best_neighbour_key, 
                                                                                                                                        scale_percentile_range=scale_percentile_range)

        ## perfrom a GMM fitting and obtain a spectrogram with energy labels 
        labeled_spec_image, arguments_dict = pafe.label_image_with_gmm_fitting(realigned_histograms, global_bins, spec_img, gmm_ncomp=gmm_ncomp,
                                                                                gmm_mean_threshold=gmm_mean_threshold, gmm_nbins=gmm_nbins, spec_normalize=spec_normalize)

        ## compute pixel localization through size zone determinations
        glszm_stg1, glszm_metrics_stg1 = pafe.compute_pixel_localization_metric(labeled_spec_image, min_zone_size_thrshld=min_zone_size_thrshld)
        glszm_metrics_stg1.columns = [f'{col}_stg1' for col in glszm_metrics_stg1.columns]

        ## refine labeld spectrogram to remove speckle noise and recompute metric for refines size zone values
        refined_labeled_spec_image = pafe.filter_labeled_spec_with_pixel_zone_size(labeled_spec_image, min_zone_size_thrshld=min_zone_size_thrshld)
        glszm_stg2, glszm_metrics_stg2 = pafe.compute_pixel_localization_metric(refined_labeled_spec_image, min_zone_size_thrshld=min_zone_size_thrshld)
        glszm_metrics_stg2.columns = [f'{col}_stg2' for col in glszm_metrics_stg2.columns]
        
        ## prepare a master size zone metric table
        glszm_metrics = pd.concat([glszm_metrics_stg1,glszm_metrics_stg2], axis=1)
        glszm_metrics['label'] = filename

        ## compute the noise cleaned version of a spectrogram based on histogram fit
        spec_image_updated = np.array([HelperFunc.scale_percentile(row.reshape(-1, 1),lower_percentile=1,upper_percentile=100,scale_type='-1to1').flatten() for row in spec_image_updated]).reshape(spec_img.shape)
        #spec_image_updated = np.array([MinMaxScaler(feature_range=(-1,1)).fit_transform(row.reshape(-1, 1)).flatten() for row in spec_image_updated]).reshape(spec_img.shape)
        #spec_image_updated[spec_image_updated<realigned_histograms['noise_img_cutoff_realigned']] = 0
        spec_image_noise_filt[spec_image_updated<realigned_histograms['noise_img_cutoff_realigned']] = spec_img.min() #1e-4 #0

        ## save all plots
        if save_all_plots:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            image_extension = plotkwargs.get('image_extension', '.png')

            filename_tmp = filename+'_S1_Spectrogram_and_GMM'+image_extension
            gisqa_plots.plot_spec_with_gmm_distribution(t_plot, f_plot, spec_img, isplot=0, issave=1, filename=filename_tmp, save_path=save_path, **plotkwargs)

            filename_tmp = filename+'_S2_Peak_Aligned_Spectrogram'+image_extension
            gisqa_plots.plot_spec_aligned_distributions(realigned_histograms, original_histrograms, global_bins, additional_params, isplot=0, issave=1, 
                                                            filename=filename_tmp, save_path=save_path)
            
            filename_tmp = filename+'_S3_Distribution_Difference'+image_extension
            gisqa_plots.plot_distribution_diff_analysis(global_bins, arguments_dict, realigned_histograms, isplot=0, issave=1, filename=filename_tmp, save_path=save_path)

            filename_tmp = filename+'_S4_GMM_Cuts'+image_extension
            gisqa_plots.plot_distribution_cut_analysis(arguments_dict, isplot=0, issave=1, filename=filename_tmp, save_path=save_path)

            filename_tmp = filename+'_S5A_Labeled_Spectrogram'+image_extension
            gisqa_plots.plot_labeled_spec(t_plot, f_plot, labeled_spec_image, isplot=0, issave=1, filename=filename_tmp, save_path=save_path)
            
            filename_tmp = filename+'_S6B_Size_Zone_Variation_Stg1'+image_extension
            gisqa_plots.plot_sizezone_variance_of_labeled_spec(glszm_stg1, isplot=0, issave=1, filename=filename_tmp, save_path=save_path)

            filename_tmp = filename+'_S6A_Filtered_Labeled_Spectrogram'+image_extension
            gisqa_plots.plot_labeled_spec(t_plot, f_plot, refined_labeled_spec_image, isplot=0, issave=1, filename=filename_tmp, save_path=save_path)
            
            filename_tmp = filename+'_S6B_Size_Zone_Variation_Stg2'+image_extension
            gisqa_plots.plot_sizezone_variance_of_labeled_spec(glszm_stg2, isplot=0, issave=1, filename=filename_tmp, save_path=save_path)

            filename_tmp = filename+'_S7_Noise_Filtered_Spectrogram'+image_extension
            gisqa_plots.plot_compare_spectrograms(t_plot, f_plot, [spec_img, spec_image_noise_filt], isplot=0, issave=1, filename=filename_tmp, save_path=save_path, 
                                                    title_list=['Original', 'After Noise Filter'])

        self.pafe = pafe
        updated_spec_details = {'t_plot':t_plot, 'f_plot':f_plot, 'labeled_spec_image':labeled_spec_image, 'arguments_dict':arguments_dict,
                                'refined_labeled_spec_image':refined_labeled_spec_image, 'spec_image_noise_cleaned': spec_image_noise_filt}
        return glszm_metrics, updated_spec_details

    def filter_Npercentile_noise_from_img(self, spec_img: np.ndarray, noisebnk_path: str, noisebnk_keys: list, binary_keys: list, 
                                            noise_cutoff:int=90, isplot:int=0, best_neighbour_key:str=None, 
                                            optimization_params:dict={'type':'exponential','start':0.1,'end':100,'slope':10}):
        pwfe = pairwise_feature_extractor()

        spec_image_reshaped = spec_img.reshape(1,-1)
        realigned_histograms, original_histrograms, global_bins, additional_params = pwfe.find_nearest_imgbank_neighbour_bysignalfit(spec_image_reshaped, noisebnk_path,noisebnk_keys,
                                                                                                                                        binary_keys, False, noise_cutoff, 
                                                                                                                                        optimization_params=optimization_params,
                                                                                                                                        best_neighbour_key=best_neighbour_key)

        if isplot:
            gisqa_plots.plot_spec_aligned_distributions(realigned_histograms, original_histrograms, global_bins, additional_params, isplot=isplot)

        spec_image_updated = spec_image_reshaped.copy()
        #spec_image_updated = np.array([MinMaxScaler(feature_range=(-1,1)).fit_transform(row.reshape(-1, 1)).flatten() for row in spec_image_updated]).reshape(spec_img.shape)
        spec_image_updated = np.array([HelperFunc.scale_percentile(row.reshape(-1, 1),lower_percentile=1,upper_percentile=100,scale_type='-1to1').flatten() for row in spec_image_updated]).reshape(spec_img.shape)
        spec_image_updated[spec_image_updated<realigned_histograms['noise_img_cutoff_realigned']]=0
        
        return spec_image_updated, (realigned_histograms, original_histrograms, global_bins, additional_params)

    def execute_pointwise_integration(self, spec_img:np.ndarray, is_hilbertize:int=1, is_normalize_stat:int=1, 
                                      is_normalize_entropy:int=0, is_best_features:int=1, 
                                        filename:str=None, filtered_features:list=None, metrics_list:list=None, **setupkwargs):

        assert filename, 'a name must be provided for each spectrogram in filename argument'

        pntf = pointwise_features_integration(spec_img, is_hilbertize=is_hilbertize, is_normalize_stat=is_normalize_stat, is_normalize_entropy=is_normalize_entropy, **setupkwargs)
        
        # Pass metrics_list to compute_all_pointwise_features
        pointwise_metrics_master_df, all_cols = pntf.compute_all_pointwise_features(is_best_features=is_best_features, metrics_list=metrics_list, filtered_features=filtered_features)
        
        pointwise_metrics_master_df['label'] = filename
        self.pofe = pntf

        return pointwise_metrics_master_df, all_cols

    def compute_pointwise_metrics_from_spec(self, spec_names:list,spec_images:list,
                                            is_hilbertize=1,is_normalize_stat=1,
                                            is_best_features=1,is_normalize_entropy=0, metrics_list=None, **kwargs):
        """
        metrics_list: Optional list of metrics to compute. If provided, overrides is_best_features logic effectively 
                      (or works in conjunction depending on implementation).
        """
        assert len(spec_names)==len(spec_images), 'number of elements in spec_names and spec_images must be the same'
        
        spec_pointwise_metrics_df = pd.DataFrame([])
        c=0
        for name, spec_img in zip(spec_names, spec_images):
            temp_df, _ = self.execute_pointwise_integration(spec_img, is_hilbertize=is_hilbertize, 
                                                            is_normalize_entropy=is_normalize_entropy,
                                                            is_normalize_stat=is_normalize_stat, 
                                                            is_best_features=is_best_features,
                                                            metrics_list=metrics_list, 
                                                            filename=name, **kwargs)    
            spec_pointwise_metrics_df = pd.concat([spec_pointwise_metrics_df, temp_df], axis=0) 
            if c%50==0:
                print(f'{c+1} number of row processed')
            c+=1
        spec_pointwise_metrics_df.reset_index(drop=True,inplace=True)
        return spec_pointwise_metrics_df

    def execute_pointwise_featurediff_pipeline(self, filelist:list=None, ref_noise_metrics_df:pd.DataFrame=None, spec_metrics_df:pd.DataFrame=None, 
                                                target_cols:list=None):
        
        ref_noise_metrics_df = HelperFunc.read_ref_pointwise_metrics_df(filelist=filelist, ref_noise_df=ref_noise_metrics_df, target_cols=target_cols) if ref_noise_metrics_df.empty else ref_noise_metrics_df 
        assert not spec_metrics_df.empty, 'Must pass the spectrogram metrics dataframe'
        assert not ref_noise_metrics_df.empty, 'Reference noise metrics Dataframe is empty'
        assert all(x == y for x, y in zip(spec_metrics_df.columns, ref_noise_metrics_df.columns)), 'feature set mismatch between the reference and target spectrogram dataframe'

        all_spec_labels = spec_metrics_df['label'].unique()
        all_ref_noise_classes = ref_noise_metrics_df['label'].unique()

        feature_diff_df = pd.DataFrame()
        for features in target_cols:
            refnoise = ref_noise_metrics_df.groupby('label')[features].mean().values.reshape(-1,1)
            specval = spec_metrics_df[features].values.reshape(1,-1)
            diff = abs(refnoise-specval)
            diff_df = pd.DataFrame(np.hstack([refnoise, diff]), index=all_ref_noise_classes, 
                                                columns=['Noise']+all_spec_labels.tolist())
            diff_df = diff_df.reset_index(names=['noise_class'])
            diff_df['feature'] = features

            feature_diff_df = pd.concat([feature_diff_df, diff_df], axis=0)

        return feature_diff_df, ref_noise_metrics_df
    
    def compute_imgbank_dict(self, spec_types:list=None, noiselist:list=None, Fs:int=100, time_in_sec:int=10, 
                             total_realizations:int=50, dt:float=None, save_path:str=None,
                             interp_size=None, normalize_range=(1e-8,1), **kwargs):

        vmin = kwargs.get('vmin', 1)
        vmax = kwargs.get('vmax', 100)
        f_min = kwargs.get('f_min', 0)
        f_max = kwargs.get('f_max', 128)
        max_normalize = kwargs.get('max_normalize', 'True')
        powerlog = kwargs.get('powerlog', 'True')
        L = int(Fs*time_in_sec)
        
        spec_dict = {}
        for specname in spec_types:
            if specname in ['stft','cwt']:
                for noise in noiselist:
                    if specname=='stft':
                        stft_win_list = kwargs.get('stft_win_list',None)
                        stft_olap_list = kwargs.get('stft_olap_list',None)
                        is_stft_param_permute = kwargs.get('is_stft_param_permute',False)
                        stft_win_name = kwargs.get('stft_win_name',"hamming")

                        if not stft_win_list:
                            stft_win_list = [32 * (2**i) for i in range(int(np.log2(Fs/32)) + 1)]
                            if Fs-stft_win_list[-1]<32:
                                stft_win_list[-1]=Fs
                            else:
                                stft_win_list+[Fs]

                        if not stft_olap_list:
                            stft_olap_list = [0.5, 0.75, 0.8, 0.9]

                        stft_combs = list(product(stft_win_list, stft_olap_list)) if is_stft_param_permute else list(zip(stft_win_list, stft_olap_list))
                        for stft_comb in stft_combs:
                            dftwin, olap = stft_comb
                            noise_ts_bank_tmp = []
                            noise_hilbertspec_bank_tmp = []
                            noise_img = []

                            for idx in range(0,total_realizations):
                                gen = ColoredNoiseGenerator(L)
                                if noise == 'white':
                                    noise_ts = gen.white()
                                elif noise == 'red':
                                    noise_ts = gen.brownian()
                                elif noise == 'pink':
                                    noise_ts = gen.pink()
                                elif noise == 'blue':
                                    noise_ts = gen.blue()
                                elif noise == 'violet':
                                    noise_ts = gen.violet()
                                elif noise == 'f3':
                                    noise_ts = gen.higher_order_noise_3()
                                elif noise == 'f4':
                                    noise_ts = gen.higher_order_noise_4()

                                noise_ts = MinMaxScaler(feature_range=(-1,1)).fit_transform(noise_ts.reshape(-1,1)).flatten()
                                f, t, spectro = stft_basic_spectogram(noise_ts,Fs,dftwin,olap,stft_win_name,f_min,f_max,
                                                                    max_normalize=max_normalize,powerlog=powerlog,
                                                                    normalize_range=normalize_range,
                                                                    vmin_percentile=vmin, vmax_percentile=vmax)
                                                
                                noise_spectro = spectro.copy()
                                if interp_size:
                                    t, f, noise_spectro = HelperFunc.interpolate_tfrs(t, f, noise_spectro, interp_shape=interp_size, 
                                                                                                T=time_in_sec, f_min=f_min, f_max=f_max)

                                locs = HelperFunc.gilbertize_image(width=noise_spectro.shape[0], height=noise_spectro.shape[1])
                                noise_hilbert =  noise_spectro[locs[:,0], locs[:,1]].flatten()
                                
                                noise_ts_bank_tmp.append(noise_ts)
                                noise_hilbertspec_bank_tmp.append(noise_hilbert)
                                noise_img.append(noise_spectro)

                            grp_name = f"{noise}_stft_w{str(dftwin)}_olap{str(olap)}"
                            spec_dict[grp_name] = {'time_series':np.array(noise_ts_bank_tmp), 'image':np.expand_dims(np.array(noise_img),axis=1),
                                                        'hilbert_image':np.array(noise_hilbertspec_bank_tmp), 'f':f, 't':t}
                            print(f"Generated Image for {grp_name}")

                    elif specname=='cwt':  
                        cwt_wavelet_list = kwargs.get('cwt_wavelet_list', None)
                        cwt_fscale = kwargs.get('cwt_fscale', (1,128,1))
                        cwt_fscaletype = kwargs.get('cwt_fscaletype', 'linear')
                        cwt_decimate_factor = kwargs.get('cwt_ts_decimate_factor', time_in_sec)

                        if not cwt_wavelet_list:
                            cwt_wavelet_list = HelperFunc.gen_wavelet_names(Fs)
            
                        for wavelet in cwt_wavelet_list:
                            noise_ts_bank_tmp = []
                            noise_hilbertspec_bank_tmp = []
                            noise_img = [] 

                            for idx in range(0,total_realizations):
                                gen = ColoredNoiseGenerator(L)
                                if noise == 'white':
                                    noise_ts = gen.white()
                                elif noise == 'red':
                                    noise_ts = gen.brownian()
                                elif noise == 'pink':
                                    noise_ts = gen.pink()
                                elif noise == 'blue':
                                    noise_ts = gen.blue()
                                elif noise == 'violet':
                                    noise_ts = gen.violet()
                                elif noise == 'f3':
                                    noise_ts = gen.higher_order_noise_3()
                                elif noise == 'f4':
                                    noise_ts = gen.higher_order_noise_4()

                                noise_ts = MinMaxScaler(feature_range=(-1,1)).fit_transform(noise_ts.reshape(-1,1)).flatten()
                                f, t, spectro, _ = cwt_simple(signal=noise_ts, sr=Fs, dt=dt, 
                                                            fscale={'start':cwt_fscale[0], 'end':cwt_fscale[1], 'num':cwt_fscale[2]}, 
                                                            wavelet=wavelet, fscaletype=cwt_fscaletype,
                                                            vmin_percentile=vmin, vmax_percentile=vmax, f_min=f_min, f_max=f_max, 
                                                            max_normalize=max_normalize, powerlog=powerlog, 
                                                            normalize_range=normalize_range, decimate_factor=cwt_decimate_factor)
                                noise_spectro = spectro.copy()
                                #noise_spectro = noise_spectro[:,::cwt_decimate_factor]
                                if interp_size:
                                    t, f, noise_spectro = HelperFunc.interpolate_tfrs(t, f, noise_spectro, interp_shape=interp_size, 
                                                                                    T=time_in_sec, f_min=f_min, f_max=f_max)
                                
                                locs = HelperFunc.gilbertize_image(width=noise_spectro.shape[0], height=noise_spectro.shape[1])
                                noise_hilbert =  noise_spectro[locs[:,0], locs[:,1]].flatten()
                                
                                noise_ts_bank_tmp.append(noise_ts)
                                noise_hilbertspec_bank_tmp.append(noise_hilbert)
                                noise_img.append(noise_spectro)
                            
                            grp_name = f"{noise}_cwt_w{str(wavelet)}"
                            spec_dict[grp_name] = {'time_series':np.array(noise_ts_bank_tmp), 'image':np.expand_dims(np.array(noise_img),axis=1),
                                                        'hilbert_image':np.array(noise_hilbertspec_bank_tmp), 'f':f, 't':t} #'t':t[::cwt_decimate_factor]
                            print(f"Generated Image for {grp_name}")

                    else: 
                        print('Spectrogram type is not valid')
                        break
            
            elif specname in ['perlin_noise','erdos_renyi_graph']:
                std_dev_list = np.linspace(0.05, 0.1, 2)
                mean = 0
                vals_list=np.linspace(0.4,0.8,3)
                domain = Domain((130, 130), (1, 1))
                field_type='Graph'
                loopcnt = 0
                for vals in vals_list:
                    noise_hilbertspec_bank_tmp = []
                    noise_img = []

                    for idx in range(0,total_realizations):
                        simulator = Simulator(domain,pml_size=1,num_air_grid=1,base_density=3000,defect_density=1000)
                        simulator.graph_type = specname
                        simulator.p_vals = vals
                        simulator.run_basic_setup(save_summary_plot=False,field_type=field_type)
                        spatial_struct = simulator.G_grid_array
                        
                        # std_dev = std_dev_list[loopcnt]
                        # one_locations = np.where(spatial_struct > 0)
                        # gaussian_values = np.random.normal(mean, std_dev, size=len(one_locations[0]))
                        # spatial_struct[one_locations] = gaussian_values

                        locs = HelperFunc.gilbertize_image(width=spatial_struct.shape[0], height=spatial_struct.shape[1])
                        noise_hilbert =  spatial_struct[locs[:,0], locs[:,1]].flatten()
                    
                        noise_hilbertspec_bank_tmp.append(noise_hilbert)
                        noise_img.append(spatial_struct)

                    grp_name = f"{specname}_p{str(round(vals,3))}"
                    spec_dict[grp_name] = {'image':np.expand_dims(np.array(noise_img),axis=1),
                                           'hilbert_image':np.array(noise_hilbertspec_bank_tmp)}
                    print(f"Generated Image for {grp_name}")
                loopcnt+=1

            elif specname in ['Gaussian','Exponential']:
                vals_list = np.linspace(5,20,3)
                domain = Domain((130, 130), (1, 1))
                field_type='Spatial'
                for vals in vals_list:
                    noise_hilbertspec_bank_tmp = []
                    noise_img = []

                    for idx in range(0,total_realizations):
                        simulator = Simulator(domain,pml_size=1,num_air_grid=1,base_density=3000,defect_density=1000)
                        simulator.cov_model_name = specname
                        simulator.cov_model_len_scale = vals
                        simulator.run_basic_setup(save_summary_plot=False,field_type=field_type)
                        spatial_struct = simulator.G_grid_array
                        locs = HelperFunc.gilbertize_image(width=spatial_struct.shape[0], height=spatial_struct.shape[1])
                        noise_hilbert =  spatial_struct[locs[:,0], locs[:,1]].flatten()
                    
                        noise_hilbertspec_bank_tmp.append(noise_hilbert)
                        noise_img.append(spatial_struct)

                    grp_name = f"{specname}_var{str(round(vals,3))}"
                    spec_dict[grp_name] = {'image':np.expand_dims(np.array(noise_img),axis=1),
                                           'hilbert_image':np.array(noise_hilbertspec_bank_tmp)}
                    print(f"Generated Image for {grp_name}")
            
            elif specname == 'pspy_fractal_noise':
                freq_list = np.linspace(0.01,0.2,3)
                shape = [interp_size[0], interp_size[1]]
                octaves = 2
                for vals in freq_list:
                    noise_hilbertspec_bank_tmp = []
                    noise_img = []

                    for idx in range(0, total_realizations):
                        spatial_struct = ps.generators.fractal_noise(shape=shape, frequency=vals, octaves=octaves, uniform=False)
                        locs = HelperFunc.gilbertize_image(width=spatial_struct.shape[0], height=spatial_struct.shape[1])
                        noise_hilbert =  spatial_struct[locs[:,0], locs[:,1]].flatten()
                    
                        noise_hilbertspec_bank_tmp.append(noise_hilbert)
                        noise_img.append(spatial_struct)
                    
                    grp_name = f"{specname}_freq{str(round(vals,3))}"
                    spec_dict[grp_name] = {'image':np.expand_dims(np.array(noise_img),axis=1),
                                           'hilbert_image':np.array(noise_hilbertspec_bank_tmp)}
                    print(f"Generated Image for {grp_name}")

            else:
                raise ValueError('Spectrogram type is not valid')

        if save_path:
            with h5py.File(save_path, "w") as h5f:
                for key in spec_dict.keys():
                    grp = h5f.create_group(key)
                    grp.create_dataset('image', data=spec_dict[key]['image'])
                    grp.create_dataset('hilbert_image', data=spec_dict[key]['hilbert_image'])
                    if 'time_series' in spec_dict[key]:
                        grp.create_dataset('time_series', data=spec_dict[key]['time_series'])
                        grp.attrs["f"] = spec_dict[key]['f']
                        grp.attrs["t"] = spec_dict[key]['t']

        return spec_dict

    def compute_noisebnk_metrics(self, noise_bnk_path:str='', img_bank_dict:dict={}, 
                                 max_realizations:int=100, save_path:str=None,
                                 is_hilbertize=1, is_normalize_entropy=0, 
                                 is_normalize_stat=1, is_best_features=0, verbose=0, metrics_list=None, **setupkwargs):
        
        if (not noise_bnk_path) and (not img_bank_dict):
            raise AssertionError('Both noise_bnk_path and img_bank_dict cant be empty')

        if noise_bnk_path:
            with h5py.File(noise_bnk_path, 'r') as f:
                spec_pointwise_metrics_df = pd.DataFrame([])
                for key in list(f.keys()):
                    spec_img_arr = f[key]['image'][:]
                    for idx in range(0, spec_img_arr.shape[0]):
                        if idx > max_realizations-1: 
                            break
                        spec_img = spec_img_arr[idx,0,:,:]
                        temp_df, _ = self.execute_pointwise_integration(spec_img, is_hilbertize=is_hilbertize, 
                                                                        is_normalize_entropy=is_normalize_entropy,
                                                                        is_normalize_stat=is_normalize_stat, 
                                                                        is_best_features=is_best_features, 
                                                                        metrics_list=metrics_list,
                                                                        filename=key, **setupkwargs)    
                        spec_pointwise_metrics_df = pd.concat([spec_pointwise_metrics_df, temp_df], axis=0)
                    if verbose:
                        print(f'metric computation of {key} completed') 
                    if save_path:
                        spec_pointwise_metrics_df.to_csv(save_path, index=None)
                spec_pointwise_metrics_df.reset_index(drop=True, inplace=True)
        
        else:
            spec_pointwise_metrics_df = pd.DataFrame([])
            for key in list(img_bank_dict.keys()):
                spec_img_arr = img_bank_dict[key]['image'][:]
                for idx in range(0, spec_img_arr.shape[0]):
                    if idx > max_realizations-1: 
                        break
                    spec_img = spec_img_arr[idx,0,:,:]
                    temp_df, _ = self.execute_pointwise_integration(spec_img, 
                                                                    is_hilbertize = is_hilbertize, 
                                                                    is_normalize_entropy = is_normalize_entropy,
                                                                    is_normalize_stat = is_normalize_stat, 
                                                                    is_best_features = is_best_features, 
                                                                    metrics_list = metrics_list,
                                                                    filename = key, 
                                                                    **setupkwargs)    
                    spec_pointwise_metrics_df = pd.concat([spec_pointwise_metrics_df, temp_df], axis=0)
                
                if verbose:
                    print(f'metric computation of {key} completed') 
                
                if save_path:
                    spec_pointwise_metrics_df.to_csv(save_path, index=None)
            
            spec_pointwise_metrics_df.reset_index(drop=True, inplace=True)

        return spec_pointwise_metrics_df  

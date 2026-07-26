import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import h5py

## For Google Colab
# sys.path.append('/content/drive/MyDrive/')

## For Local
sys.path.append('/path/to/my/all_projects') ## e.g. sys.path.append('/data/sswar_files/0_PUBLISHED_CODES')

from SPEC2VEC.utils.gisqa_compute_updated import GISQAPipeline

# Update the following paths to match your system when running from terminal
data_path = "time_complexity_analysis/data_noise_ref"
save_dir = 'time_complexity_analysis/codes_and_files'


#### Helper Functions
def generate_random_data(n_images, img_size=(256, 256)):
    all_imgs = [np.random.rand(*img_size) for _ in range(n_images)]
    all_ts = [np.linspace(0, 10, img_size[0]) for _ in range(n_images)]
    all_fs = [np.linspace(0, 50, img_size[1]) for _ in range(n_images)]
    all_names = [f"img_{i}" for i in range(n_images)]
    all_keys = ['noise_key'] * n_images
    return all_imgs, all_ts, all_fs, all_names, all_keys

def run_analysis(N_dim, N_list):
    results = []
    noisebnk_keys = ['noise_key']
    binary_keys = ['erdos_renyi_graph_0.4','erdos_renyi_graph_0.8','perlin_noise_0.4','perlin_noise_0.8']
    
    # We will use the common entp_columns from the notebook
    entp_columns = [
        'permutation_entropy_antropy', 
        'spectral_entropy_antropy',
        'svd_entropy_antropy', 
        'petrosian_fd_antropy', 
        'detrended_fluctuation_antropy', 
        'hjorth_mobility_antropy',
        'hjorth_complexity_antropy', 
        'higuchi_fd_antropy',
        'normalized_permutation_entropy_ordpy',
        'statistical_complexity_entropy_ordpy', 
        'fisher_shannon_ordpy',
        'global_node_entropy_ordpy', 
        'renyi_complexity_entropy_long_ordpy',
        'renyi_stat_complexity_long_ordpy',
        'tsallis_complexity_entropy_long_ordpy',
        'tsallis_stat_complexity_long_ordpy',
        'weighted_permutation_entropy_ordpy', 
        'missing_links_ordpy',
        '0_Absolute energy',
        '0_Average power',
        '0_Kurtosis', '0_Lempel-Ziv complexity', '0_Mean','0_Median','0_Skewness', '0_Standard deviation',
        '75th_percentile',
        '25th_percentile',
        'label']
    
    for ND, NL in zip(N_dim,N_list):
        noise_ref = [np.random.rand(*ND) for _ in range(1)]
        noisebnk_path = f'{data_path}/tc_noise_reference_{ND[0]}by{ND[1]}.h5'
        spec_dict = {}
        spec_dict['noise_key'] = {'image':np.expand_dims(noise_ref, axis=[0,1])}

        if not os.path.exists(noisebnk_path):
            with h5py.File(noisebnk_path, "w") as h5f:
                for key in spec_dict.keys():
                    grp = h5f.create_group(key)
                    grp.create_dataset('image', data=spec_dict[key]['image'])

        for N in NL:
            print(f"Running analysis for ND={ND}, N={N}...")
            all_imgs, all_ts, all_fs, all_names, all_keys = generate_random_data(n_images=N,img_size=ND)
            
            giq = GISQAPipeline()
            
            # Stat+Text Metrics
            start_time_stat = time.time()
            _ = giq.compute_pointwise_metrics_from_spec(
                all_names, all_imgs,
                is_hilbertize=1, is_normalize_stat=0, is_normalize_entropy=0,
                is_best_features=0, ordpydx=5, antropydx=5,
                metrics_list=entp_columns, hilbert_locs=None
            )
            end_time_stat = time.time()
            time_stat = end_time_stat - start_time_stat
            
            # Spatial Metrics
            start_time_spatial = time.time()
            for t_plot, f_plot, spec_img, filename, key in zip(all_ts, all_fs, all_imgs, all_names, all_keys):
                try:
                    _ = giq.execute_pairwise_pipleine(
                        t_plot=t_plot, f_plot=f_plot, spec_img=spec_img,
                        noisebnk_path=noisebnk_path, noise_cutoff=50,
                        noisebnk_keys=noisebnk_keys, binary_keys=binary_keys,
                        filename=filename, gmm_mean_threshold=0.025,
                        optimization_params={'type':'exponential', 'start':1, 'end':1, 'slope':1},
                        save_path=None, save_all_plots=False,
                        best_neighbour_key=key, min_zone_size_thrshld=2
                    )
                except:
                    continue
            end_time_spatial = time.time()
            time_spatial = end_time_spatial - start_time_spatial
            
            results.append({
                'N_dim': ND,
                'N': N,
                'Stat_Text_Time_Sec': time_stat,
                'Spatial_Time_Sec': time_spatial
            })
            print(f"Completed Dim={ND}, N={N}: Stat+Text Time = {time_stat:.2f}s, Spatial Time = {time_spatial:.2f}s")
        
    df = pd.DataFrame(results)
    df['total_time_sec'] = df['Spatial_Time_Sec']+df['Stat_Text_Time_Sec']

    # Save CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'time_complexity_results_diff_dims.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Save Plot
    plt.figure(figsize=(8, 5))
    for dim in N_dim:
        df_dim = df[df['N_dim'].apply(lambda x: tuple(x) == tuple(dim))]
        plt.plot(df_dim['N'], df_dim['total_time_sec'], marker='o', label=f'{dim[0]}x{dim[1]}')
    plt.xlabel('Number of Images (N)', fontsize=10, fontweight='bold')
    plt.ylabel('Total Execution Time (seconds)', fontsize=10, fontweight='bold')
    plt.title('Time Complexity of Spec2Vec for Different Image Dimensions', fontsize=11, fontweight='bold')
    plt.legend(title='Image Size')
    plt.grid(True)
    plot_path = os.path.join(save_dir, 'total_time_complexity_plot_diff_dims.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    plt.show()

    return df

#### Main
if __name__ == "__main__":
    N_dim = [(128,128),(256,256),(512,512)]
    N_list = [[50, 100, 500, 1000]]*len(N_dim)
    _ = run_analysis(N_dim, N_list)

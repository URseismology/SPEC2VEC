##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import ticker
from matplotlib.lines import Line2D
import matplotlib.markers as mmarkers
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from math import pi

from SPEC2VEC.utils.gisqa_helper import *
from SPEC2VEC.utils.gisqa_pointwise_metrics_updated import *
from SPEC2VEC.utils.gisqa_pairwise_metrics import *

class gisqa_plots:

    @staticmethod
    def plot_sizezone_variance_of_labeled_spec(glszm, isplot=1, issave=0, filename=None, **kwargs):
        tot_labels = glszm.P_glszm.shape[1]
        colors = ['#F5F5F5', '#228B22', '#00BFFF', '#F58231', '#911EB4', '#191970']
        colors = colors[1:tot_labels+1]
        cmap =  matplotlib.colors.ListedColormap(colors)

        plt.figure(figsize=(6,4))
        for i in range(0, tot_labels):
            color = cmap(i)
            plt.loglog(glszm.coefficients['jvector'], glszm.P_glszm[0,i,:],'o-', label=f'Level {i+1}', color=color)
        plt.ylabel('Number of Connected Components'); 
        plt.xlabel('# of Similar Neighbours')

        plt.legend()
    
        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.title(filename)
            plt.savefig(save_path)

        if isplot:
            None if issave else plt.title('Number of Connected Component By Size Zone', fontsize=10)
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_compare_spectrograms(t_plot:np.ndarray, f_plot:np.ndarray, spec_images:list, isplot:int=1, issave:int=0, filename:str=None, **kwargs):
        totalsubplots = len(spec_images)
        title_list = kwargs.get('title_list', None)
        assert len(title_list)==totalsubplots, 'number of titles does not match the number of images'

        plt.figure(figsize=(12,4))
        for i in range(0,totalsubplots):
            plt.subplot(1,totalsubplots,i+1); plt.pcolormesh(t_plot, f_plot, spec_images[i]); plt.title(title_list[i])

        if issave:
            assert filename,'Must pass a Filename to save figures'
            plt.suptitle(filename, fontsize=10)
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.savefig(save_path, dpi=300)
        
        if isplot:
            None if issave else plt.suptitle('Spectrogram Comparisons', fontsize=10)
            plt.tight_layout(); plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_labeled_spec(t_plot, f_plot, labeled_spec_image, isplot=1, issave=0, filename=None, **kwargs):
        colors = ['#F5F5F5', '#228B22', '#00BFFF', '#F58231', '#911EB4', '#191970']
        colors = colors if len(np.unique(labeled_spec_image)) == len(colors) else colors[:len(np.unique(labeled_spec_image))]
        cmap = kwargs.get('cmap', None)
        kde = kwargs.get('kde',1)
        bins= kwargs.get('bins',25)
        cmap = cmap if cmap else matplotlib.colors.ListedColormap(colors) 

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.pcolormesh(t_plot, f_plot, labeled_spec_image, cmap=cmap)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)'); fig.colorbar(im, ax=ax, orientation='vertical')
        if kde:
            freq_dist = []
            for fs in range(0,f_plot.shape[0]):
                freq_dist.append(np.sum(labeled_spec_image[fs,:]).item())
            freq_dist_df = pd.DataFrame(np.repeat(f_plot,freq_dist),columns=['freq_count'])
            divider = make_axes_locatable(ax)
            ax_histy = divider.append_axes("right", 1, pad=0.1, sharey=ax)
            ax_histy.yaxis.set_tick_params(labelleft=False)
            sns.histplot(data=freq_dist_df, y='freq_count', bins=25, kde=True, ax=ax_histy, color='gray', line_kws={'linewidth':'2'})
            ax_histy.lines[0].set_color('crimson'); ax_histy.set_xlabel('Pixel Count')
            
        if issave:
            assert filename,'Must pass a Filename to save figures'
            ax.set_title(filename, fontsize=10)
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.savefig(save_path, dpi=300)
        
        if isplot:
            None if issave else ax.set_title('Labeled Spectrogram', fontsize=10)
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_spec_aligned_distributions(realigned_histograms, original_histrograms, global_bins, additional_params, isplot=1, issave=0, filename=None, **kwargs):
        plt.figure(figsize=(12, 4))
        plt.subplot(1,2,1)
        plt.bar(global_bins[:-1], original_histrograms['hist1'], width=(global_bins[1] - global_bins[0]), alpha=0.5, label=additional_params['max_crosscorr_key'])
        plt.bar(global_bins[:-1], original_histrograms['hist2'], width=(global_bins[1] - global_bins[0]), alpha=0.5, label='good')
        plt.axvline(x=original_histrograms['noise_img_cutoff'], color='r', linestyle='--')
        plt.title('Histograms',fontsize=9); plt.legend(); plt.xlabel('Pixel Value'); plt.ylabel('Density')

        plt.subplot(1,2,2)
        plt.bar(global_bins[:-1], realigned_histograms['aligned_hist1'], width=(global_bins[1] - global_bins[0]), alpha=0.5, label=additional_params['max_crosscorr_key'])
        plt.bar(global_bins[:-1], realigned_histograms['aligned_hist2'], width=(global_bins[1] - global_bins[0]), alpha=0.5, label='good')
        plt.axvline(x=realigned_histograms['noise_img_cutoff_realigned'], color='r', linestyle='--')
        plt.title('Histograms Peak Aligned, area diff: '+str(additional_params['area_diff']),fontsize=9); plt.legend(); plt.xlabel('Pixel Value'); plt.ylabel('Density')
        plt.tight_layout()

        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.suptitle(filename)
            plt.savefig(save_path, dpi=300)

        if isplot:
            None if issave else plt.suptitle('Histogram Allignment Comparison', fontsize=10)
            plt.tight_layout(); plt.show()
        else:
            plt.close()

    @staticmethod
    def gmm_distribution_plot(gmm_intensities, gmm_nbins, gmm_fit_pdf_xaxis, gmm_fit_pdf):
        plt.figure(5,4)
        plt.hist(gmm_intensities, bins=gmm_nbins, density=True, alpha=0.6, color='g', label='Histogram')
        plt.plot(gmm_fit_pdf_xaxis, gmm_fit_pdf, '-r', label='GMM PDF', linewidth=1)
        plt.title('Distribution with GMM Fit')
        plt.show()
        
    @staticmethod
    def plot_spec_with_gmm_distribution(t_plot:np.ndarray, f_plot:np.ndarray, spec_image:np.ndarray, 
                                        isplot:int=0, issave:int=0, filename:str=None, **kwargs):
        """
        Plots a Spectrogram and its corresponding GMM distribution.

        Parameters
        ----------
        spec_image: Is a 2D Spectrogram image
        """
        gmm_nbins_plot = kwargs.get('gmm_nbins_plot',256)
        gmm_ncomp_plot = kwargs.get('gmm_ncomp_plot',2)
        gmm_normalized_plot = kwargs.get('gmm_normalized_plot',1)
        cmap = kwargs.get('cmap', 'gray')
        
        spec_image_shape = spec_image.shape
        if gmm_normalized_plot:
            spec_image = MinMaxScaler(feature_range=(-1, 1)).fit_transform(spec_image.reshape(-1,1)).reshape(spec_image_shape)
        plt.figure(figsize=(14,4))
        
        plt.subplot(1,3,1); 
        plt.pcolormesh(t_plot, f_plot, spec_image, cmap=cmap, shading='gouraud'); plt.colorbar(); plt.title('spectrogram', fontsize=10)
        plt.subplot(1,3,2); 
        plt.hist(spec_image.flatten(), bins=gmm_nbins_plot, density=True); plt.title('spec distribution', fontsize=10)
        plt.subplot(1,3,3); 
        pwfe = pairwise_feature_extractor()
        _, gmm_pdf, gmm_x, gmm_intensity_values = pwfe.gmm_optimal_func(spec_image,gmm_ncomp_plot,gmm_nbins_plot)
        plt.hist(gmm_intensity_values, bins=gmm_nbins_plot, density=True, alpha=0.6, color='g', label='Histogram')
        plt.plot(gmm_x, gmm_pdf, '-r', label='GMM PDF', linewidth=1) 
        plt.title('spec dist. with gmm fit', fontsize=10)

        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.suptitle(filename)
            plt.savefig(save_path, dpi=300)

        if isplot:
            None if issave else plt.suptitle('Spectrogram with GMM Distribution', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_distribution_cut_analysis(arguments_dict, isplot=1, issave=0, filename=None, **kwargs):
        
        plt.figure(figsize=(14,4))
        plt.subplot(1,3,1)
        plt.plot(arguments_dict['gmm_fit_pdf_xaxis'],arguments_dict['gmm_fit_pdf'],'-b'); 
        plt.title('Mean Locations',fontsize=10);plt.xlabel('Pixel Value'); plt.ylabel('Frequency');
        for i in range(0,len(arguments_dict['gmm_means'])):
            plt.axvline(x=arguments_dict['gmm_means'][i], color='r', linestyle='--')
        plt.subplot(1,3,2)
        plt.plot(arguments_dict['gmm_fit_pdf_xaxis'],arguments_dict['gmm_fit_pdf'],'-b'); 
        plt.title('Filtered Mean Locations',fontsize=10);plt.xlabel('Pixel Value'); plt.ylabel('Frequency');
        for i in range(0,len(arguments_dict['filtered_gmm_means'])):
            plt.axvline(x=arguments_dict['filtered_gmm_means'][i], color='r', linestyle='--')
        plt.subplot(1,3,3)
        plt.plot(arguments_dict['gmm_fit_pdf_xaxis'],arguments_dict['gmm_fit_pdf'],'-b'); 
        plt.title('Cut Locations',fontsize=10);plt.xlabel('Pixel Value'); plt.ylabel('Frequency');
        for i in range(0,len(arguments_dict['cut_locations'])):
            plt.axvline(x=arguments_dict['cut_locations'][i], color='r', linestyle='--')
        
        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.suptitle(filename)
            plt.savefig(save_path, dpi=300)

        if isplot:
            None if issave else plt.suptitle('GMM Mean Locations and Dist. Cut Locations', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        
    @staticmethod   
    def plot_distribution_diff_analysis(global_bins, arguments_dict, realigned_histograms, isplot=1, issave=0, filename=None, **kwargs):
        
        plt.figure(figsize=(14,4))
        plt.subplot(1,3,1)
        plt.bar(global_bins[:-1], arguments_dict['hist_diff'], width=(global_bins[1] - global_bins[0]), alpha=0.5)
        plt.axvline(x=realigned_histograms['noise_img_cutoff_realigned'], color='r', linestyle='--')
        plt.title('Histogram Diff', fontsize=10); plt.xlabel('Pixel Value'); plt.ylabel('Density Diff.')
        
        plt.subplot(1,3,2)
        plt.hist(arguments_dict['hist_diff_distribution'], bins=256, density=True);
        plt.axvline(x=realigned_histograms['noise_img_cutoff_realigned'], color='r', linestyle='--')
        plt.title('Histogram Diff Distribution', fontsize=10); plt.xlabel('Pixel Value'); plt.ylabel('Density')

        plt.subplot(1,3,3)
        plt.hist(arguments_dict['hist_diff_distribution_filtered'], bins=256, density=True, alpha=0.6, color='g') #arguments_dict['hist_diff_distribution_filtered']
        plt.plot(arguments_dict['gmm_fit_pdf_xaxis'], arguments_dict['gmm_fit_pdf'],'-r')
        plt.title('Filtered Histogram Diff Distribution', fontsize=10); plt.xlabel('Pixel Value'); plt.ylabel('Density')
       
        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pairwise')
            save_path = os.path.join(save_path,filename)
            plt.suptitle(filename)
            plt.savefig(save_path, dpi=300)

        if isplot:
            None if issave else plt.suptitle('Distribution Difference Analysis', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    @staticmethod   
    def plot_compare_metrics_pairs(noise_fos_df:pd.DataFrame, spec_fos_df:pd.DataFrame, target_cols:list, 
                           label_col:str='label', pair_normalize:bool=True, method='mean',
                           isplot=1, issave=0, filename:str=None,
                           scatter_palette='Dark2', hist_palette='nipy_spectral', **kwargs):
        
        assert target_cols != None or len(target_cols) >= 2, 'target columns can not be blank or have less than 2 features'
        
        labels_y = noise_fos_df[label_col]
        labels_x = spec_fos_df[label_col]

        if pair_normalize:
            y, x = HelperFunc.pair_normalize(noise_fos_df,spec_fos_df,target_cols)
            noise_fos_df = pd.DataFrame(y, columns=target_cols); noise_fos_df[label_col]=labels_y
            spec_fos_df = pd.DataFrame(x, columns=target_cols); spec_fos_df[label_col]=labels_x

        row_cols = len(target_cols)
        tot_subplots = row_cols ** 2
        tot_sigs = spec_fos_df.shape[0]
        cmap = matplotlib.colormaps[scatter_palette] #Other Colormaps: Accent, Set3, Set1, Dark2
        spec_feature_cols = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(tot_sigs)]
        spec_feature_marks = np.repeat('^', tot_sigs).tolist()
        spec_feature_legend = spec_fos_df.label.unique().tolist(); assert len(spec_feature_legend)==spec_fos_df.shape[0], 'integrity error, multiple rows for same spectrogram label'
        show_nearest_cluster= kwargs.get('show_nearest_cluster',False)
        pltcnt = 1

        fig_w = np.ceil((12/3)*row_cols).astype(int)
        fig_h = np.ceil((9/3)*row_cols).astype(int)

        fig, axes = plt.subplots(row_cols, row_cols, figsize=(fig_w, fig_h))
        for i_idx, i in enumerate(target_cols):
            for j_idx, j in enumerate(target_cols):
                ax = axes[i_idx, j_idx]
                if i == j:
                    sns.histplot(noise_fos_df, x=i, hue='label', legend=False,
                                element="poly", palette=hist_palette, ax=ax)
                    for row in range(tot_sigs):
                        ax.axvline(x=spec_fos_df.loc[row, i], color=spec_feature_cols[row], linestyle='--')
                    ax.set_xlabel(i, fontsize=7)
                    ax.set_ylabel('count', fontsize=7)
                    
                else:
                    sns.scatterplot(data=noise_fos_df, x=j, y=i, hue='label', legend=True,
                                    palette='Set3', ax=ax, s=75) # Other Palettes: hsv, nipy_spectral, terrain, gist_earth, Pastel1, gist_ncar
                    handles_label, labels_label = ax.get_legend_handles_labels()
                    ax.legend_.remove()
                    spec_feature_legend = spec_fos_df.label.unique().tolist()
                    for row in range(tot_sigs):
                        xx = spec_fos_df.loc[row, j]
                        yy = spec_fos_df.loc[row, i]
                        if show_nearest_cluster:
                            label = HelperFunc.calculate_2dintercluster_distnces(noise_fos_df, np.array([xx,yy]), features=[j,i])[0]
                            spec_feature_legend[row] = spec_feature_legend[row]+'-'+label
                        ax.scatter(xx, yy, marker=spec_feature_marks[row], s=100, c=spec_feature_cols[row], edgecolor='black')
                    ax.set_xlabel(j, fontsize=7)
                    ax.set_ylabel(i, fontsize=7)
                    if show_nearest_cluster:
                        spec_handles = [Line2D([0], [0], marker=spec_feature_marks[idx], color='w', 
                                        markerfacecolor=spec_feature_cols[idx], markersize=6, label=spec_feature_legend[idx])
                                        for idx in range(tot_sigs)]
                        ax.legend(handles=spec_handles, fontsize=7, loc='best', frameon=False)


        legend1=fig.legend(handles_label, labels_label, title="Noise Class", bbox_to_anchor=(0.9, 0.5), loc="upper left", fontsize=9)
        legend1.get_title().set_fontsize('9')
        if not show_nearest_cluster:
            spec_handles = [Line2D([0], [0], marker=spec_feature_marks[idx], color='w', markeredgecolor='k',
                            markerfacecolor=spec_feature_cols[idx], markersize=9, label=spec_feature_legend[idx])
                            for idx in range(tot_sigs)]
            fig.legend(handles=spec_handles, fontsize=9, bbox_to_anchor=(0.9, 0.9), loc="upper left")

        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pointwise')
            save_path = os.path.join(save_path,filename)
            plt.suptitle(filename)
            plt.savefig(save_path, dpi=300)
        if isplot:
            None if issave else plt.suptitle('First Order Stat Feature Pairs', fontsize=10)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
        else:
            plt.close()

        return HelperFunc.calculate_best_spectrogram_by_rmse(noise_fos_df,spec_fos_df,target_cols,method=method)

    @staticmethod
    def plot_meanfeaturediff_barplot(df:pd.DataFrame, isplot=1, issave=0, filename=None, feature_name=None, **kwargs):
        ax = df.plot(kind='bar', figsize=(10, 4), width=0.8)
        ax.tick_params(axis='both', labelsize=7)
        plt.ylabel('Feature Value Mean', fontsize=8)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=9)

        if issave:
            assert filename,'Must pass a Filename to save figures'
            save_path = kwargs.get('save_path','/home/urseismoadmin/Documents/PRJ_GIS_QA/figs/pointwise')
            save_path = os.path.join(save_path,filename)
            plt.title(filename, fontsize=9)
            plt.savefig(save_path)

        if isplot:
            title = feature_name+' Feature Value Comparison' if feature_name else 'Feature Value Comparison'
            None if issave else plt.title(title, fontsize=9)
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_multi_row_radar_chart(df:pd.DataFrame, title:str=None, column_lables:list=None, ax:matplotlib.axes.Axes=None, cmap:str=None, 
                                   legend=True):
        """
        Generates a radar chart with multiple rows overlaid, each with a different color and alpha.

        Args:
            df (pd.DataFrame): The input DataFrame.
                Expected to have rows representing entities and columns representing features.
                All feature columns should be max-normalized between 0 and 1.
            title (str): The title of the radar chart.
        """

        if df.empty:
            print("DataFrame is empty. Cannot plot radar chart.")
            return

        # Ensure all columns are numeric except for potential ID columns if present
        # For this example, we assume all columns are features to be plotted.
        feature_names = df.columns.tolist()
        num_features = len(feature_names)

        if not column_lables:
            column_lables = feature_names

        # Number of rows (entities) to plot
        num_rows = df.shape[0]

        # --- Radar Chart Setup ---
        
        # Calculate angle for each axis (equal spacing)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        # Complete the loop by adding the first angle again to close the radar shape
        angles_plot = angles + angles[:1]

        # Create a figure and a single polar subplot
        if not ax:
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

        # --- Color and Transparency Setup ---
        # Use a colormap to get distinct colors for each row
        cmap = cmap if cmap else 'Set2'
        colors = plt.cm.get_cmap(cmap, num_rows)
        alpha_value = 0.25 # Transparency for the filled area
        line_alpha_value = 0.8 # Transparency for the lines

        area_under_curve = {}
        # --- Plotting Loop: Overlay each row ---
        for i in range(num_rows):
            row_label = df.index[i] if df.index.name is None else df.index[i]
            values = df.iloc[i].tolist()
            values_plot = values + values[:1] # Complete the loop for plotting

            ax.plot(angles_plot, values_plot, color=colors(i), linewidth=1.5,
                    linestyle='solid', label=f'{row_label}', alpha=line_alpha_value)
            ax.fill(angles_plot, values_plot, color=colors(i), alpha=alpha_value)

            r = np.array(values)
            theta = np.array(angles)
            area_under_curve[row_label] = 0.5 * np.abs(np.sum(r * np.roll(r, -1) * np.sin(np.roll(theta, -1) - theta)))


        # --- Customization ---

        # Set the position of the first axis (usually top) and direction (clockwise)
        ax.set_theta_offset(np.pi / 2) # Rotate chart so 0 is at the top
        ax.set_theta_direction(-1)     # Go clockwise

        # Draw axis labels (feature names)
        ax.set_xticks(angles)
        ax.set_xticklabels(column_lables, fontsize=8, fontweight='bold')

        # Draw ylabels (radial ticks)
        # Positions for radial labels (0 to 1, with 0.2 intervals)
        ax.set_rlabel_position(0)
        ax.set_yticks(np.arange(0, 1.1, 0.2)) # Radial ticks from 0 to 1
        ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], color="gray", size=8)
        ax.set_ylim(0, 1) # Set radial limits from 0 to 1

        # Add title and legend
        if title:
            ax.set_title(title, fontsize=10, y=1.1)
        
        # Place legend outside the plot area
        if legend:
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=9)

        if not ax:
            plt.tight_layout() #plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for the legend
            #plt.show()

        return area_under_curve
        







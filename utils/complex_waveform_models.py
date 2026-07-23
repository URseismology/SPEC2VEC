##Authors: Sayan Kr. Swar, Tushar Mittal, Tolulope Olugboji
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import re
from PIL import Image
import numpy as np

from jax import jit
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import *
from jwave.utils import show_field, show_positive_field
from jwave.signal_processing import analytic_signal, apply_ramp, gaussian_window, smooth

import networkx as nx
import structify_net.zoo as zoo

import gstools as gst
from gstools import transform as tf
from gstools.random import MasterRNG


class Simulator:
  
    def __init__(self, domain=None, density_rho=1.0, sound_speed=1500.0, pml_size=5, num_air_grid=10, base_density=3000, defect_density=1000, air_speed_of_sound=343.0):
        """
        Initialize the wave propagation Simulator.

        Parameters:
        domain (Domain, optional): Spatial domain for the simulation. Defaults to a 256x256 grid.
        density_rho (float): Default density of the medium.
        sound_speed (float): Default sound speed in the medium.
        pml_size (int): Size of the Perfectly Matched Layer (absorbing boundaries).
        num_air_grid (int): Number of grid points at the boundaries to simulate an air layer.
        base_density (float): Base parameter for generating the medium structure.
        defect_density (float): Defect parameter for anomalies in the medium.
        air_speed_of_sound (float): Speed of sound in the boundary air layer.
        """

        if domain is None:
            self.domain = Domain((128*2, 128*2), (1, 1))
        else :
            self.domain = domain
        self.num_air_grid = num_air_grid
        self.base_density = base_density # m/s
        self.defect_density = defect_density # m/s
        self.air_speed_of_sound = air_speed_of_sound # m/s

        self.k = 20  # Each node is joined to k nearest neighbors in a ring topology (used only for small_world graph type)
        self.m = 10
        self.p_vals = 0.1 # The probability of rewiring each edge
        self.epsilon = .5 # Used by the structify_net graph generation functions and controls the "randomness" or deviation from the ideal structure of the specified graph type.
        self.size = self.domain.N[0]-2*self.num_air_grid # Number of pixels - images is size x size. Note that structify_net sometimes fails for values of the larger values of the size (or works best for some round numbers) -- so, for larger images, one can rescale or stack.
        self.graph_type = 'fractal_root'

        self.density_rho = density_rho
        self.sound_speed = sound_speed

        # We have to update the attenutation function based on user input
        attentuation = jnp.ones(self.domain.N)
        self.attentuation = FourierSeries(jnp.expand_dims(attentuation, -1), self.domain)
        # attenuation = jnp.expand_dims(attenuation.at[64:110, 125:220].set(100), -1)

        self.pml_size = pml_size
        self.time_axis = None
        self.sensors = None
        self.sources = None
        self.use_sensors = False
        self.use_sources_Time = False
        self.use_sources_p0 = False
        self.path = ''
        self.set_sound_flag = True 
        self.set_density_flag = False
        self.G_grid_array = None
        self.num_sensors = 48

        self.cov_model_name = 'Gaussian'
        self.cov_model_dim = 2
        self.cov_model_var = 1
        self.cov_model_len_scale = 15
        self.cov_model_angles = np.pi
        self.cov_model_transform_field = False

    def run_basic_setup(self, circle=False, use_p0=True, field_type='Graph', **kwargs):
      """
      Orchestrates the basic setup for the simulation, generating the structural field,
      setting medium properties, and configuring sources and sensors.

      Parameters:
      circle (bool): If True, sensors are arranged in a circular pattern; otherwise, they are placed along the edges.
      use_p0 (bool): If True, uses an initial pressure source; otherwise, uses time-varying pulse sources.
      field_type (str): Type of heterogeneity field to generate ('Graph' or 'Spatial').
      **kwargs: Additional optional parameters for controlling plotting, source locations, and sensor setups.
      """
      assert field_type in ['Graph','Spatial'], 'not a valid field type'
      plot_data = kwargs.get('plot_data', False) 
      plot_variogram = kwargs.get('plot_variogram', False) 
      plot_only_graph = kwargs.get('plot_only_graph', False) 
      save_summary_plot = kwargs.get('save_summary_plot', False)
      add_sensor_at_source = kwargs.get('add_sensor_at_source', False)
      
      self.field_type = field_type
      if self.field_type=='Graph':
        self.generate_graph(plot_me=plot_only_graph)
      else:
        transform_mode = kwargs.get('transform_mode', ['Binary'])
        seed_val = kwargs.get('seed_val', MasterRNG(None)())  
        _,_,_ = self.generate_spatial_corr_field(cov_model=self.cov_model_name,dim=self.cov_model_dim,
                          var=self.cov_model_var,len_scale=self.cov_model_len_scale,angles=self.cov_model_angles,
                          transform_field=self.cov_model_transform_field,
                          transform_mode=transform_mode,seed_val=seed_val,plot_field=plot_variogram)

      self.set_density()
      self.set_time_arr()

      if use_p0:
        self.source_type = "pressure"
        pressure_source_raddi = kwargs.get('pressure_source_raddi', 6) 
        pressurce_source_loc = kwargs.get('pressurce_source_loc', (62,62)) 
        self.generate_sources_p0(radius=pressure_source_raddi,center_x=pressurce_source_loc[0],center_y=pressurce_source_loc[1])
      else:
        self.source_type = "gaussian pulse"
        source_loc = kwargs.get('pulse_source_loc', ((62,62),(62,62))); assert len(source_loc)==2,'Please pass two different pulse sources in ((x1,y1),(x2,y2)) format. Both location can be of same value.'
        amp = kwargs.get('amp', 1);  freq = kwargs.get('freq', 50);  
        pulse_sigma = kwargs.get('pulse_sigma', 4e-2);  pulse_1_m = kwargs.get('pulse_1_m', .08);  pulse_2_m = kwargs.get('pulse_2_m', .52);  
        self.generate_sources_time(source_loc, amp, freq, pulse_sigma, pulse_1_m, pulse_2_m)

      if circle:
        self.sensor_allignment = "circular"
        self.circle_dim = kwargs.get('circle_dim', (125,125)) 
        self.generate_sensors_circle(num_sensors = self.num_sensors, circle_x=self.circle_dim[0],circle_y=self.circle_dim[1],add_sensor_at_source=add_sensor_at_source)
      else:
        self.sensor_allignment = "linear"
        self.sides = kwargs.get('sensor_locations', ['top', 'right', 'bottom', 'left']) 
        self.generate_sensors_edges(sides=self.sides,offset=self.num_air_grid+5,spacing_samples=6,add_sensor_at_source=add_sensor_at_source)
      
      if plot_data:
        self.plot_summary(save_me=save_summary_plot)
      
    def generate_graph(self, plot_me):
      """
      Generates a graph-based structural field and extracts its adjacency matrix to model medium heterogeneity.

      Parameters:
      plot_me (bool): If True, displays a visual representation of the generated grid array.
      """
      gpx, _ =  self._generate_graph(self.graph_type,self.size,epsilon=self.epsilon,p=self.p_vals,k=self.k,m=self.m)
      
      ## Get the adjacency matrix of the graph
      if self.G_grid_array is None:
        self.G_grid_array = nx.to_numpy_array(gpx)
      
      if plot_me:
        plt.figure(figsize=(10,10))
        plt.imshow(1-self.G_grid_array,cmap='grey')
        plt.colorbar()
        plt.show()
    
    def _generate_graph(self, graph_type, size, scores=False, verbose=False, **kwargs):
      """
      Generate a complex network graph of the specified type using NetworkX or structify_net.

      Parameters:
      graph_type (str): The type of graph to generate (e.g. 'erdos_renyi_graph, 'random', 'scale_free', 'small_world', 'fractal_root').
      size (int): The number of nodes in the graph (or base dimension).
      scores (bool): If True, calculates structure scores for structify models. Currently not being used in any of the modules thus defults to False.
      verbose (bool): If True, prints generation details.
      **kwargs: Additional arguments for specific graph types (e.g., 'p', 'k', 'm', 'epsilon').

      Returns:
      tuple: (G (networkx.Graph), df_scores_graph) where G is the generated graph.
      """
      p_vals = kwargs.get('p', 0.4)
      structufy_keys = ['ER', 'spatial', 'spatialWS', 'blocks_assortative', 'overlapping_communities', 'nestedness', 'maximal_stars', 'core_distance',
                        'fractal_leaves', 'fractal_root', 'fractal_hierarchy', 'fractal_star', 'perlin_noise', 'disconnected_cliques']

      if graph_type == 'erdos_renyi_graph':
          return nx.erdos_renyi_graph(int(size),p_vals),0 #p=1 -> all-to-all connectivity
      elif graph_type == 'random':
          return nx.gnm_random_graph(int(size), kwargs.get('num_edges', 500)),0
      elif graph_type == 'scale_free':
          return nx.barabasi_albert_graph(int(size), kwargs.get('m', 2)),0
      elif graph_type == 'small_world':
          return nx.watts_strogatz_graph(int(size), kwargs.get('k', 4), p_vals),0
      elif graph_type in structufy_keys:
          try:
              if verbose:
                  print(f'Size of graph : {size}, non-zero values : {int(p_vals*size**2.)}') 
              list_zoo = zoo.get_all_rank_models(n=size,m=int(p_vals*size**2.)) 
              using_size_diff = False
          except:
              size_use = np.max([50,size])
              list_zoo = zoo.get_all_rank_models(n=size_use,m=int(p_vals*size_use**2.))
              using_size_diff = True
              print('Here in Zoo',size,int(p_vals*size**2.))
          
          rank_model = list_zoo[graph_type]
          epsilon = kwargs.get('epsilon', 0.1)
          if scores :
              df_scores_graph = rank_model.scores(m=int(p_vals*size**2.),epsilons=epsilon,runs=10)
          else :
            df_scores_graph = None
          if using_size_diff :
              gpx =  rank_model.generate_graph(epsilon=epsilon,density=p_vals)
              # Get the adjacency matrix of the graph
              A = nx.to_numpy_array(gpx)
              n_avg = size
              # Calculate the block size for averaging
              block_size = int(np.ceil(A.shape[0] / n_avg))
              # Calculate the padded size
              padded_size = block_size * n_avg
              # Calculate the number of repetitions needed to reach the padded size
              repetitions = int(np.ceil(padded_size / A.shape[0]))
              # Repeat the original matrix to create a larger matrix
              A_repeated = np.tile(A, (repetitions, repetitions))
              # Trim the repeated matrix to the padded size
              A_padded = A_repeated[:padded_size, :padded_size]
              # Average the adjacency matrix locally to make it a n_avg x n_avg matrix
              A_avg = np.mean(np.mean(A_padded.reshape(n_avg, block_size, n_avg, block_size), axis=1), axis=2)
              return nx.from_numpy_array(A_avg),df_scores_graph
          else :
              return rank_model.generate_graph(epsilon=epsilon,density=p_vals),df_scores_graph
      else:
          raise ValueError(f"Invalid graph type: {graph_type}")

    def generate_spatial_corr_field(self,cov_model='Gaussian',dim=2,var=1,len_scale=5,angles=np.pi,transform_field=False,**kwargs):
      """
      Generates a spatially correlated random field using geostatistical models (gstools).

      Parameters:
      cov_model (str): The covariance model name (e.g., 'Gaussian', 'Exponential', 'Matern').
      dim (int): Dimensionality of the field (typically 2).
      var (float): Variance of the covariance model.
      len_scale (float): Length scale parameter determining spatial correlation range.
      angles (float): Anisotropy angle in radians.
      transform_field (bool): If True, applies transformations (e.g., Binary, LogNormal) to the generated field.
      **kwargs: Additional arguments such as 'seed_val', 'plot_field', and 'transform_mode'.

      Returns:
      tuple: (normalized_field_arr.T, field_arr.T, (x, y)) The structured grids of the generated field.
      """
      assert cov_model in ['Gaussian','Exponential','Matern','Spherical','Circular','Linear','Stable'], 'Undefined Covariance Model'
      seed_val = kwargs.get('seed_val', MasterRNG(None)())  
      plot_field = kwargs.get('plot_field', False)  

      x = y = range(self.size)
      self.graph_type = cov_model

      model_map = {
        'Gaussian': lambda dim, var, len_scale, angles: gst.Gaussian(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Exponential': lambda dim, var, len_scale, angles: gst.Exponential(dim=dim, var=var, len_scale=len_scale, angles=angles, anis=0.5),
        'Matern': lambda dim, var, len_scale, angles: gst.Matern(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Spherical': lambda dim, var, len_scale, angles: gst.Spherical(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Circular': lambda dim, var, len_scale, angles: gst.Circular(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Linear': lambda dim, var, len_scale, angles: gst.Linear(dim=dim, var=var, len_scale=len_scale, angles=angles),
        'Stable': lambda dim, var, len_scale, angles: gst.Stable(dim=dim, var=var, len_scale=len_scale, angles=angles)
      }

      model_to_call = model_map.get(cov_model, lambda dim, var, len_scale, angles: gst.Gaussian(dim=dim, var=var, len_scale=len_scale, angles=angles))
      model = model_to_call(dim, var, len_scale, angles)
      srf = gst.SRF(model, seed=seed_val)
      if not transform_field:
        field_arr = srf.structured([x, y])
      else:
        transform_mode = kwargs.get('transform_mode', ['Binary'])
        transfrom_map = {
        'Binary': lambda model: tf.binary(model),
        'ZinnHarvey': lambda model: tf.zinnharvey(model),
        'LogNormal': lambda model: tf.normal_to_lognormal(model),
        'ForceMoment': lambda model: tf.normal_force_moments(model)
        }
        srf.structured([x, y])
        for i in range(0,len(transform_mode)):
          transformed_call = transfrom_map.get(transform_mode[i], lambda model: tf.binary(model))
          field_arr = transformed_call(srf)

      min_val = np.min(field_arr)
      max_val = np.max(field_arr)
      #normalized_field_arr = 2 * ((field_arr - min_val) / (max_val - min_val)) - 1
      range_val = max_val - min_val; denominator = range_val if range_val != 0 else 1e-6;
      normalized_field_arr = (field_arr - min_val) / denominator
      
      if self.G_grid_array is None:
        self.G_grid_array = normalized_field_arr.T

      if plot_field:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.contourf(x, y, normalized_field_arr.T, levels=256); plt.title(f'{cov_model} Field')
        plt.colorbar()
        plt.title(f'spatially correlated field (normalized)')
        plt.subplot(1,2,2)
        plt.plot(model.variogram(np.linspace(0,100,1000)),label=f'{cov_model} Variogram')
        plt.legend()

      return normalized_field_arr.T, field_arr.T, (x, y)

    def set_density(self):
      """
      Configures the density or sound speed parameters of the simulation medium based on the generated field.
      Applies the field array to determine structural boundaries and medium attributes.
      """
      density_field = self._set_density_field(self.domain,self.num_air_grid,self.base_density,self.defect_density,self.air_speed_of_sound,self.G_grid_array)
      if self.set_sound_flag:
        self.sound_speed = density_field
      else:
        self.density_rho = density_field
      #print(f"Set density data for Sound : {self.set_sound_flag}, Density : {self.set_density_flag}")

    def _set_density_field(self, domain, num_air_grid, base_density, defect_density, air_speed_of_sound, G_grid_array, plot=False):
      """
      Internal function to construct the spatially varying parameter grid (density or sound speed) 
      using the structural field array and boundary conditions.

      Parameters:
      domain (Domain): The simulation domain.
      num_air_grid (int): The thickness of the surrounding air/boundary layer in grid points.
      base_density (float): Value assigned to the base matrix.
      defect_density (float): Value assigned to the structural anomalies/defects.
      air_speed_of_sound (float): Value assigned to the boundary layer.
      G_grid_array (np.ndarray): The 2D structural array defining the heterogeneity.
      plot (bool): If True, visualizes the resulting field.

      Returns:
      FourierSeries: A continuous differentiable field representation of the medium property.
      """
      ## Setting the density field from the graph data and other parameters
      density = jnp.zeros(domain.N)
      if self.field_type == 'Graph':
        G_grid_jax = jnp.array((1-G_grid_array)*base_density)
        density += density.at[num_air_grid:-num_air_grid, num_air_grid:-num_air_grid].set(G_grid_jax)
        G_grid_jax = jnp.array(G_grid_array*defect_density)
        density += density.at[num_air_grid:-num_air_grid, num_air_grid:-num_air_grid].set(G_grid_jax)
      else:
        G_grid_jax = jnp.array((1 - G_grid_array) * base_density + G_grid_array * defect_density) 
        #jnp.array(G_grid_array*defect_density) + base_density
        density += density.at[num_air_grid:-num_air_grid, num_air_grid:-num_air_grid].set(G_grid_jax)

      density = density.at[0:num_air_grid, :].set(air_speed_of_sound)
      density = density.at[:, 0:num_air_grid].set(air_speed_of_sound)
      density = density.at[-num_air_grid:, :].set(air_speed_of_sound)
      density = density.at[:, -num_air_grid:].set(air_speed_of_sound)
      density_field = FourierSeries(np.expand_dims(density, -1), domain)

      if plot:
        show_positive_field(density_field)
        _ = plt.title("Density")
      return density_field

    def set_time_arr(self):
      """
      Initializes the simulation time axis based on the physical properties (Medium) 
      and calculates the stable time step according to the CFL condition.
      """
      medium = Medium(domain=self.domain, sound_speed=self.sound_speed, density=self.density_rho, pml_size=self.pml_size, attenuation=self.attentuation)
      self.time_axis = TimeAxis.from_medium(medium, cfl=0.3)
      self.time_axis_arr = self.time_axis.to_array()
      self.medium = Medium
      #print(f'time_axis_arr Shape : {self.time_axis_arr.shape}')
    
    def generate_sources_p0(self,radius=6,center_x=62,center_y=62):
      """
      Configures an initial static pressure source in the domain.

      Parameters:
      radius (int): Radius of the circular pressure source.
      center_x (int): X-coordinate of the source center.
      center_y (int): Y-coordinate of the source center.
      """
      ### Pressure source properties
      self.radius = radius
      self.center_x = center_x
      self.center_y = center_y

      p0 = self._points_on_circle_press(self.radius,self.center_x,self.center_y,self.domain.N)
      self.p0 = p0
      self.use_sources_p0 = True
    
    def _points_on_circle_press(self,radius,center_x,center_y,N,amplt=1.,plot=False):
      """
      Generates a continuous spatial field representing a circular initial pressure distribution.

      Parameters:
      radius (int): Radius of the pressure source.
      center_x (int): X-coordinate of the center.
      center_y (int): Y-coordinate of the center.
      N (tuple): Dimensions of the domain grid.
      amplt (float): Amplitude of the pressure wave.
      plot (bool): If True, visualizes the initial pressure field.

      Returns:
      FourierSeries: A continuous differentiable field of the initial pressure.
        """
      # Defining the initial pressure
      p0 = circ_mask(N, radius, (center_x,center_y))
      p0 = amplt * jnp.expand_dims(p0, -1)
      p0 = FourierSeries(p0, self.domain)
      if plot:
        show_field(p0)
        plt.title("Initial pressure")
      return p0

    def generate_sources_time(self, source_loc=((62,62),(62,62)), amp=10, freq=50, pulse_sigma=4e-2, pulse_1_m = .08, pulse_2_m=.52):
        """
        Configures time-varying acoustic point sources (Gaussian pulses). By Deafult is configured with two source pulses.

        Parameters:
        source_loc (tuple of tuples): Coordinates for the sources ((x1,y1), (x2,y2)).
        amp (float): Amplitude of the sine wave.
        freq (float): Frequency of the sine wave.
        pulse_sigma (float): Standard deviation (width) of the Gaussian envelope.
        pulse_1_m (float): Center time for the first pulse.
        pulse_2_m (float): Center time for the second pulse.
        """
        self.radius = 1
        self.center_x = source_loc[0][0]
        self.center_y = source_loc[0][1]
        self.pulse_source_loc = source_loc

        t = np.arange(0, self.time_axis.t_end, self.time_axis.dt)
        s = np.sin(2 * np.pi * freq * t)
        s1 = gaussian_window(s, t, pulse_1_m, pulse_sigma)
        s2 = gaussian_window(s, t, pulse_2_m, pulse_sigma)
        self.sources =  Sources(
            positions=self.pulse_source_loc,
            signals=jnp.stack([amp*s1, amp*s2]),
            dt=self.time_axis.dt,
            domain=self.domain,
        )
        self.use_sources_Time = True
        #plt.plot(s1)
        #plt.plot(s2)

    def generate_sensors_circle(self, num_sensors, sensor_radius=100, circle_x=100, circle_y=100, add_sensor_at_source=False):
      """
      Places sensors in a circular array around a center point.

      Parameters:
      num_sensors (int): Total number of sensors to distribute on the circle.
      sensor_radius (int): Radius of the sensor array.
      circle_x (int): X-coordinate of the array center.
      circle_y (int): Y-coordinate of the array center.
      add_sensor_at_source (bool): If True, adds an additional sensor at the primary source location.
      """
      ## Sensors
      self.num_sensors = num_sensors
      self.sensor_radius = sensor_radius
      self.circle_x = circle_x
      self.circle_y = circle_y

      x, y = points_on_circle(num_sensors, sensor_radius, (circle_x, circle_y))
      print(num_sensors)
      if add_sensor_at_source:
        # Add One Sensor at Source with 1 Offset
        x = (self.radius+self.center_x,) + x
        y = (self.radius+self.center_y,) + y

      sensors_positions = (x, y)
      self.sensors = Sensors(positions=sensors_positions)
      #print("Sensors parameters: ",Sensors.__annotations__)
      self.use_sensors=True
      self.x = x
      self.y = y
      self.number_sensors = len(self.x)

    def generate_sensors_edges(self, arr=None, sides=['top', 'right', 'bottom', 'left'], offset=10, spacing_samples=1, add_sensor_at_source=False):
      """
      Places sensors along the specified edges of the domain grid.

      Parameters:
      arr (np.ndarray, optional): Array defining the domain layout.
      sides (list): Edges to populate with sensors ('top', 'bottom', 'left', 'right').
      offset (int): Padding from the true edge.
      spacing_samples (int): Spacing interval between adjacent sensors.
      add_sensor_at_source (bool): If True, adds an additional sensor at the primary source location.
      """
      if arr is None:
        arr = np.zeros(self.domain.N)
      # Example usage:
      points = self._get_edge_points(arr, offset, sides, spacing_samples=spacing_samples)
      x, y = zip(*points)
      if add_sensor_at_source:
        # Add One Sensor at Source with Offset
        x = (self.radius+self.center_x,) + x
        y = (self.radius+self.center_y,) + y

      sensors_positions = (x, y)
      self.sensors = Sensors(positions=sensors_positions)
      #print("Sensors parameters: ",Sensors.__annotations__)
      self.use_sensors=True
      self.x = x
      self.y = y
      self.number_sensors = len(self.x)
    
    def _get_edge_points(self, arr, offset, sides,spacing_samples=1):
      """
      Computes the discrete x-y coordinates for sensors placed along the edges.

      Parameters:
      arr (2D array): Input array representing the grid.
      offset (int): Offset distance from the outermost edge.
      sides (list of str): Which sides to include ('top', 'bottom', 'left', 'right').
      spacing_samples (int): Distance in grid points between sensors.

      Returns:
      list of tuples: Calculated (x, y) coordinates for all sensors.
      """
      rows, cols = arr.shape
      points = []

      if 'top' in sides:
          points.extend([(i, offset) for i in range(offset, cols - offset,spacing_samples)])
      if 'bottom' in sides:
          points.extend([(i, rows - offset - 1) for i in range(offset, cols - offset,spacing_samples)])
      if 'left' in sides:
          points.extend([(offset, i) for i in range(offset, rows - offset,spacing_samples)])
      if 'right' in sides:
          points.extend([(cols - offset - 1, i) for i in range(offset, rows - offset,spacing_samples)])

      return points

    def plot_summary(self,save_me):
      """
      Visualizes the initial state of the simulation, showing the source configuration
      (initial pressure field or time pulses) and the initial velocity field alongside sensor locations.

      Parameters:
      save_me (bool): If True, saves the summary plot to disk.
      """
      if self.use_sources_p0 == False and self.use_sources_Time == False:
        return ValueError('No source field set - use generate_sources_p0/time()')

      #fig, ax = plt.subplots(1,2, figsize=(15,10), dpi=100)
      fig = plt.figure(figsize=(10, 6), dpi=100)

      gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

      if self.use_sources_p0:
          ax0 = fig.add_subplot(gs[0])
          im1 = ax0.imshow(self.p0.on_grid, cmap="RdBu_r")
          cbar = fig.colorbar(im1, ax=ax0)
          cbar.ax.get_yaxis().labelpad = 5
          cbar.ax.set_ylabel('A.U.', rotation=270)

      elif self.use_sources_Time:
          gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[1, 3])
          ax0a = fig.add_subplot(gs_left[0])
          ax0b = fig.add_subplot(gs_left[1])

          ax0a.plot(self.sources.signals[0,:], label='pulse 1')
          ax0a.plot(self.sources.signals[1,:], label='pulse 2')
          ax0a.legend()
        
          xx = [point[0] for point in self.pulse_source_loc]
          yy = [point[1] for point in self.pulse_source_loc]
          plot_dim = self.size+self.pml_size+self.num_air_grid
          im1 = ax0b.imshow(jnp.zeros([plot_dim,plot_dim]), cmap="RdBu_r", vmin=0, vmax=1)
          cbar = fig.colorbar(im1, ax=ax0b)
          cbar.ax.get_yaxis().labelpad = 5
          cbar.ax.set_ylabel('A.U.', rotation=270)
          ax0b.scatter(xx,yy)
          ax0=ax0b
          del ax0b


      ax0.axis('off')
      ax0.set_title('Initial pressure')
      ax0.scatter(self.x, self.y, label="sensors", marker='.')
      ax0.legend(loc="upper right")

      ax1 = fig.add_subplot(gs[1])
      im1 = ax1.imshow(self.sound_speed.on_grid, cmap="RdBu_r")
      cbar = fig.colorbar(im1, ax=ax1)
      cbar.ax.get_yaxis().labelpad = 5
      cbar.ax.set_ylabel('A.U.', rotation=270)
      ax1.axis('off')
      ax1.set_title('Initial velocity field')
      ax1.scatter(self.x, self.y, label="sensors", marker='.')
      if save_me:
        if self.field_type == 'Graph':
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_p{self.p_vals}_eps{self.epsilon}_SummaryPlot.png')
        else:
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_var{self.cov_model_var}_scale{self.cov_model_len_scale}_SummaryPlot.png')
        plt.close()

      plt.tight_layout()
      plt.show()
    
    def plot_sensors_data(self,sensors_data,vmax=0.5,vmin=-0.5,scaling=0.1,save_me=False):
      """
      Visualizes the recorded acoustic waveforms (seismograms) captured at the sensor locations over time.

      Parameters:
      sensors_data (jnp.ndarray): The continuous sensor recordings returned by the simulation.
      vmax (float): Maximum value for the color scale.
      vmin (float): Minimum value for the color scale.
      scaling (float): Factor to scale the sensor data for visualization.
      save_me (bool): If True, saves the waveform plot to disk.
      """
      ### Sensor Case!
      sensors_data = sensors_data.squeeze()
      _field = FourierSeries(sensors_data.T, self.domain)
      if isinstance(_field, Field):
              _field_arr = _field.on_grid
      #show_field(_field/scaling, "Recorded acoustic signals",vmax=vmax)
      plt.figure(figsize=(6,6), dpi=100)
      plt.imshow(_field_arr/scaling,aspect='auto',cmap='RdBu_r',vmax=vmax,vmin=vmin)
      plt.colorbar()
      plt.title("Waveforms Recorded at Sensors")
      plt.xlabel("Time step")
      plt.ylabel("Sensor position")
      plt.axis("on")
      self.sensor_data_array = _field_arr
      if save_me:
        if self.field_type == 'Graph':
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_p{self.p_vals}_eps{self.epsilon}_Waveform.png')
        else:
          plt.savefig(self.path+f'/{self.field_type}_{self.graph_type}_var{self.cov_model_var}_scale{self.cov_model_len_scale}_Waveform.png')
        plt.close()
      plt.show()

    def run_simulation(self, use_p0=True):
        """
        Executes the numerical wave simulation using jwave.

        Parameters:
        use_p0 (bool): If True, uses the initial static pressure condition. Otherwise uses continuous time sources.

        Returns:
        jnp.ndarray: The continuous spatiotemporal pressure field or sensor recordings based on setup.
        """
        pressure = self._compiled_simulator(use_p0)
        return pressure

    def _compiled_simulator(self,use_p0=True):
        """
        Internal wrapper to initialize the Medium and run the full domain wave propagation simulation without sensors.

        Parameters:
        use_p0 (bool): If True, relies on p0 initial state.

        Returns:
        jnp.ndarray: The simulated pressure field at all grid points.
        """
        medium = Medium(domain=self.domain,sound_speed=self.sound_speed,density=self.density_rho,pml_size=self.pml_size,attenuation=self.attentuation)
        if use_p0 == True :
          return simulate_wave_propagation(medium, self.time_axis, p0=self.p0)
        else:
          return simulate_wave_propagation(medium, self.time_axis, sources=self.sources)

    def compiled_simulator_sensors(self):
        """
        Executes the simulation specifically configured for an initial pressure source (p0) 
        and extracts data only at defined sensor locations.

        Returns:
        jnp.ndarray: The recorded acoustic signals at the sensor coordinates.
        """
        ## called while using pressure pulse sources
        medium = Medium(domain=self.domain, sound_speed=self.sound_speed, density=self.density_rho, pml_size=self.pml_size, attenuation=self.attentuation)
        return simulate_wave_propagation(medium, self.time_axis, p0=self.p0, sensors=self.sensors)

    def compiled_simulator_sources(self):
        """
        Executes the simulation specifically configured for time-varying pulse sources 
        and extracts data only at defined sensor locations.

        Returns:
        jnp.ndarray: The recorded acoustic signals at the sensor coordinates.
        """
        ## called while using gaussian time pulse sources
        medium = Medium(domain=self.domain, sound_speed=self.sound_speed, density=self.density_rho, pml_size=self.pml_size, attenuation=self.attentuation)
        return simulate_wave_propagation(medium, self.time_axis, sources=self.sources, sensors=self.sensors)

    def plot_pressure_field(self, pressure, t):
        """
        Visualizes the spatial pressure field at a specific time step.

        Parameters:
        pressure (jnp.ndarray): The multi-dimensional simulated pressure field.
        t (int): The integer time step index to plot.
        """
        show_field(pressure[t])
        plt.title(f"Pressure field at t={self.time_axis.to_array()[t]}")

    def make_gif(self, readpath = '../results/4fwd_model/', savepath='../results/4fwd_model/', dur=50):
      """
      Compiles a sequence of generated image frames into an animated GIF.

      Parameters:
      readpath (str): Filepath pattern to glob the frame images (e.g., 'frames/*.png').
      savepath (str): Output filepath for the resulting GIF.
      dur (int): Frame duration in milliseconds.
      """
      file_list = glob.glob(readpath)
      sorted_files = sorted(file_list, key=lambda x: int(re.findall(r'(\d+)', os.path.basename(x))[0]))
      frames = [Image.open(image) for image in sorted_files]
      frame_one = frames[0]
      frame_one.save(savepath, format="GIF", append_images=frames[1:],
                    save_all=True, duration=dur, loop=0)

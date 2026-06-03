# -*- coding: utf-8 -*-
"""
Created on Sun May 31 08:32:13 2026

@author: tameem
"""

#%% Import libraries and modules 

import neo
import os
import re
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import plotly.express as px
import matplotlib.patches as mpatches
import scipy.stats
import plotly.graph_objs as go
import itertools
import scipy.stats as stats
import plotly.offline as pyo
import matplotlib.colors as mcolors
import mne
import ast
import plotly.graph_objects as go
import mpld3 
import nibabel as nib
from scipy.spatial.distance import cdist, pdist, squareform, mahalanobis
from scipy.stats import ttest_ind, ttest_1samp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import ConvexHull
from nibabel.affines import apply_affine
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.fftpack import fft
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy
from scipy.signal import welch
from scipy.special import rel_entr
from scipy.signal import resample_poly
from scipy.signal import resample
from tridesclous import DataIO
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter
from scipy.integrate import simps
from sklearn.cluster import KMeans, DBSCAN
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.fft import fft
from openTSNE import TSNE as oTSNE
from scipy.signal import butter, filtfilt
from scipy.stats import median_abs_deviation as mad
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.stats import circvar
from scipy.stats import rayleigh
from scipy.stats import mannwhitneyu
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots


#%% set preprocessing time and dataframe 

pre_time = 0.5 
post_time = 1.6
root_dir = r'D:\nigraPhys\spikes_neurons'
pd_header = ["patient_id", "hemisphere", "depth", "date", "s_freq", "rec_duration", "sample_idx", "class_idx", "wave"]

all_data = list()


#%%  Loop over and import patients (spikes from TDC)

for pt_id in  os.listdir(root_dir):
  
    recordings_dir = root_dir + '/' + pt_id
    for rec in  os.listdir(recordings_dir):
        
        hem, date, depth = rec.split("_")[5], rec.split("_")[2:5], rec.split("_")[-1]

        print("Patient:{}, recording:{}".format(pt_id, rec))
        
        rec_dir = recordings_dir + '/' + rec
        
        dataio = DataIO(dirname=rec_dir)
        
        n_left = int(pre_time * dataio.sample_rate * 0.001)
        n_right = int(post_time * dataio.sample_rate * 0.001) 

### set a data source
        
        ## load the spikes - segNum 0 - chanGrp 0 - cell#all.csv and get the array
        idxs = pd.read_csv(rec_dir+'/'+"export_catalogue_chan_grp_0"+ "/"+ "spikes - segNum 0 - chanGrp 0 - cell#all.csv", sep=',', header=None).values
        #spikes = os.path.koin
                
        #spikes = np.asarray(list(dataio.get_spikes(seg_num=0, chan_grp=0)))
        #idxs = np.array([[s[0],s[1]] for s in spikes])
        
        waves = dataio.get_some_waveforms(seg_num=0, chan_grp=0,
                                          n_left=-n_left, n_right=n_right,
                                          peak_sample_indexes=idxs[:,0])
        for s_idx in range(waves.shape[0]):
            all_data.append([pt_id, hem, depth,
                             '{}_{}_{}'.format(date[0],date[1],date[2]),
                             dataio.sample_rate,
                             dataio.get_duration_per_segments()[0],
                             idxs[s_idx,0],
                             idxs[s_idx,1], 
                             waves[s_idx,:,0]])

raw_df = pd.DataFrame(all_data,columns=pd_header)

###remove events with sampling frequencies other than 22K and 44K

sel_low_sampling  =  raw_df["s_freq"] > 10000
raw_df = raw_df[sel_low_sampling]


#%% import patient raw data (LFP)-raw files from Spike2 in .smr format

# Function to load LFP data from .smr file
def load_lfp_data(smr_file_path):
    reader = neo.io.Spike2IO(filename=smr_file_path)
    bl = reader.read_block()
    seg = bl.segments[0]
    analog_signals = seg.analogsignals

    # In case there are multiple channels in the .smr file, you can choose the desired channel here
    lfp_data = analog_signals[0].as_array()
    sample_rate = analog_signals[0].sampling_rate
    duration = seg.t_stop - seg.t_start

    return lfp_data, sample_rate, duration

# Set preprocessing time and DataFrame
pre_time = 0.5 
post_time = 1.6
pd_header = ["patient_id", "hemisphere", "depth", "date", "lfp_data", "sample_rate", "duration"]

# Let's define a list to keep data for DataFrame
all_data = list()

# Define the root directory where the .smr files are located
smr_files_root_dir = r'D:\nigraPhys\LFPs'

# Loop over and import LFP data for each patient and recording
for pt_id in os.listdir(smr_files_root_dir):
    recordings_dir = os.path.join(smr_files_root_dir, pt_id)

    for rec in os.listdir(recordings_dir):
        if rec.endswith(".smr"):
            # Use regular expression to extract patient ID, date, hemisphere, and depth from the filename
            regex_pattern = r'(\d+)_(\d+_\d+_\d+)_(\w+)_(\-?\d+(?:\.\d+)?)\.smr'
            match = re.match(regex_pattern, rec)
            
            if match:
                patient_id, date, hemisphere, depth = match.groups()

                print("Patient:{}, recording:{}".format(pt_id, rec))

                # Construct the corresponding .smr file path
                lfp_file_path = os.path.join(recordings_dir, rec)

                # Load the LFP data from the .smr file
                lfp_data, sample_rate, duration = load_lfp_data(lfp_file_path)

                # Append LFP data and other information to the all_data list
                all_data.append([patient_id, hemisphere, depth,
                                 date,
                                 lfp_data,
                                 sample_rate,
                                 duration])

# Create the DataFrame with LFP data
lfp_df = pd.DataFrame(all_data, columns=pd_header)

sel_low_sampling  =  lfp_df["sample_rate"] > 10000
lfp_df = lfp_df[sel_low_sampling]


#%% resample LFP

# Define the target sampling frequency
target_sampling_frequency = 22000

# Create a new column to store the resampled LFP data
lfp_df['lfp_data_resampled'] = None

# Loop through each recording in the DataFrame
for index, row in lfp_df.iterrows():
    # Calculate the resampling ratio
    resampling_ratio = row['sample_rate'] / target_sampling_frequency

    # Resample the LFP data using resample_poly
    resampled_data = resample_poly(row['lfp_data'], up=1, down=int(resampling_ratio))

    # Update the 'lfp_data_resampled' column with the resampled data
    lfp_df.at[index, 'lfp_data_resampled'] = resampled_data

print("Data resampling to {} Hz is complete.".format(target_sampling_frequency))


#%%  FIR filter bandpass for MUA

# Define the frequency range for filtering (300 Hz to 3000 Hz)
low_freq = 300.0
high_freq = 3000.0
target_sample_rate = 22000

# Function to apply filtering
def apply_filter(data, sfreq, low_freq, high_freq):
    # Reshape the data to (n_channels, n_samples)
    data = data.reshape(1, -1)

    raw = mne.io.RawArray(data, info=mne.create_info(ch_names=['lfp'], sfreq=sfreq, ch_types=['eeg']))
    raw.filter(low_freq, high_freq, fir_design='firwin')
    return raw.get_data()[0]

# Apply filtering and create a new column
lfp_df['lfp_data_second_filtered'] = lfp_df['lfp_data_resampled'].apply(
    lambda data: apply_filter(data, target_sample_rate, low_freq, high_freq))


#%% Adjust types of raw_df

raw_df["patient_id"] = pd.to_numeric(raw_df["patient_id"])
raw_df["depth"] = pd.to_numeric(raw_df["depth"])


#%% Load the coords and structures

mapper_df = pd.read_csv(r'D:\new code\MER_features_coords_mapper_with_names_distances.csv')
mapper_df = mapper_df.drop(columns=["Unnamed: 0"])


#%% Adjust types

mapper_df["x"] = pd.to_numeric(mapper_df["x"])
mapper_df["y"] = pd.to_numeric(mapper_df["y"])
mapper_df["z"] = pd.to_numeric(mapper_df["z"])
mapper_df["depth"] = pd.to_numeric(mapper_df["depth"])
mapper_df['OP_DATUM'] = pd.to_datetime(mapper_df['OP_DATUM']).dt.strftime('%d_%m_%Y')

date_to_patient = raw_df.groupby('date')['patient_id'].first().to_dict()

mapper_df['patient_id'] = mapper_df['OP_DATUM'].map(date_to_patient)
print(mapper_df)
mapper_df["patient_id"] = pd.to_numeric(mapper_df["patient_id"])


unique_dates_count1 = raw_df['date'].nunique()
unique_dates_count2 = mapper_df['OP_DATUM'].nunique()
print(f'The number of unique dates in raw_df is: {unique_dates_count1}')
print(f'The number of unique dates in mapper_df is: {unique_dates_count2}')


#%% Merge dataframes

df = pd.merge(raw_df , mapper_df, on=["patient_id", "hemisphere", "depth"], how="outer")

lfp_df['depth'] = pd.to_numeric(lfp_df['depth'], errors='coerce')

# Convert 'patient_id' and 'hemisphere' columns in both DataFrames to the same data type
lfp_df['patient_id'] = lfp_df['patient_id'].astype(str)
lfp_df['hemisphere'] = lfp_df['hemisphere'].astype(str)

mapper_df['patient_id'] = mapper_df['patient_id'].astype(str)
mapper_df['hemisphere'] = mapper_df['hemisphere'].astype(str)
mapper_df['depth'] = pd.to_numeric(mapper_df['depth'], errors='coerce')

# Merge lfp_df and mapper_df based on 'patient_id', 'hemisphere', and 'depth'
lfp_df = lfp_df.merge(mapper_df[['patient_id', 'hemisphere', 'depth', 'x', 'y', 'z', 'structure', 'structure_distances']], 
                      on=['patient_id', 'hemisphere', 'depth'], 
                      how='left')

print(lfp_df.dtypes)


#%% get time points

# Creating a new column 'rwave' with resampled data or original data based on the condition
df['rwave'] = df.apply(lambda row: row['wave'][::2] if row['s_freq'] == 44000 else row['wave'], axis=1)

# Optionally create a new column for the modified sampling frequency if resampled
df['new_s_freq'] = df['s_freq'].apply(lambda freq: 22000 if freq == 44000 else freq)

# Check the operation's result by printing the updated value counts
print(df['new_s_freq'].value_counts())

df['time_p'] = df['sample_idx'] / df["new_s_freq"]


#%% assign recordings to atlas structure

df['structure_distances'] = df['structure_distances'].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
lfp_df['structure_distances'] = lfp_df['structure_distances'].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))

# Create a function to find the closest structure and its distance for 'HyPD' and 'pauli' in a given row
def find_closest_structures_and_distances(row):
    closest_structure_hypd = None
    min_distance_hypd = float('inf')
    closest_structure_pauli = None
    min_distance_pauli = float('inf')
          

    for structure, distance in row['structure_distances'].items():
        if 'HyPD' in structure:
            if distance < min_distance_hypd:
                closest_structure_hypd = structure
                min_distance_hypd = distance
        elif 'pauli' in structure:
            if distance < min_distance_pauli:
                closest_structure_pauli = structure
                min_distance_pauli = distance
                
    return (closest_structure_hypd, min_distance_hypd, closest_structure_pauli, min_distance_pauli)

# Apply the function to each row to find the closest structures and their distances
df[['struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli']] = df.apply(find_closest_structures_and_distances, axis=1, result_type='expand')
lfp_df[['struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli']] = lfp_df.apply(find_closest_structures_and_distances, axis=1, result_type='expand')
    
print(df[['struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli']])
print(lfp_df[['struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli']])

unique_struct_HyPD = df['struct_HyPD'].unique()
unique_struct_pauli = df['struct_pauli'].unique()
print("Unique struct_HyPD:", unique_struct_HyPD)
print("Unique struct_pauli:", unique_struct_pauli)


unique_struct_HyPD_lfp_df = lfp_df['struct_HyPD'].unique()
unique_struct_pauli_lfp_df = lfp_df['struct_pauli'].unique()
print("Unique struct_HyPD in lfp_df:", unique_struct_HyPD_lfp_df)
print("Unique struct_pauli in lfp_df:", unique_struct_pauli_lfp_df)


lfp_df_min = lfp_df[['patient_id', 'hemisphere', 'depth', 'x', 'y', 'z','structure','structure_distances','struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli']]

#lfp_df_min.to_csv(r'C:\Users\tameem\Downloads\lfp_df_min.csv', index=False)

    
#%% cleanup 1: remove STN labels

unique_ptn_count_pre = lfp_df['patient_id'].nunique()
print(f'The number of unique patients in lfp_df is: {unique_ptn_count_pre}')

df = df[~(df['struct_HyPD'].str.contains('STN') | df['struct_HyPD'].isna())]
print(df['struct_HyPD'].unique())
print(df['struct_pauli'].unique())

df_atlas_struct = df.groupby(['patient_id', 'hemisphere', 'depth'])[['struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli', 'x', 'y', 'z']].first().reset_index()
lfp_df = lfp_df[~(lfp_df['struct_HyPD'].str.contains('STN') | lfp_df['struct_HyPD'].isna())]
print(lfp_df['struct_HyPD'].unique())
print(lfp_df['struct_pauli'].unique())

lfp_df_atlas_struct = lfp_df.groupby(['patient_id', 'hemisphere', 'depth'])[['struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli', 'x', 'y', 'z']].first().reset_index()
counts_HyPD = df_atlas_struct['struct_HyPD'].value_counts()
counts_pauli = df_atlas_struct['struct_pauli'].value_counts()

print("Counts for struct_HyPD:")
print(counts_HyPD)

print("Counts for struct_pauli:")
print(counts_pauli)

unique_ptn_count_post = lfp_df['patient_id'].nunique()
print(f'The number of unique patients in lfp_df is: {unique_ptn_count_post}')


#%% firing rate

# Calculate firing rate directly from the grouped DataFrame
df2 = df.groupby(["patient_id", "hemisphere", "depth", "class_idx"]).agg(
    sample_count=('sample_idx', 'size'),
    rec_duration=('rec_duration', 'first')
)
df2['firing'] = df2['sample_count'] / df2['rec_duration']

# Include additional columns in df2
df2 = pd.merge(
    df2.reset_index(),
    df[['patient_id', 'hemisphere', 'depth', 'rec_duration']].drop_duplicates(),
    on=["patient_id", "depth", "hemisphere"],
    how="outer"
)


#%% sort by cell and assign cell id

df_sorted = df.sort_values(by=['patient_id', 'depth', 'hemisphere', 'class_idx', 'time_p'], ascending=[True, False, True, True, True])

df_sorted['cell_id'] = df_sorted['class_idx'].astype(str) + '_' + df_sorted['patient_id'].astype(str) + '_' + df_sorted['hemisphere'].astype(str) + '_' + df_sorted['depth'].astype(str)


#%% ISI per spike per cell

df_grouped = df_sorted.groupby('cell_id')

df_isi = df_grouped['time_p'].diff()

df_sorted['isi'] = df_isi


#%% e-phys features calculations and summary

# Define functions to calculate physiological measures
def calculate_burst_index(isi):
    """Calculate burst index."""
    return sum(isi < isi.mean()) / len(isi)

def calculate_lv(isi):
    """Calculate local variation (LV) using the standard formula."""
    valid_isi = isi.dropna().to_numpy()  # Convert to NumPy array to ensure sequential indexing
    if len(valid_isi) < 2:  # LV requires at least two intervals
        return np.nan
    
    lv = np.sum([3 * (valid_isi[i+1] - valid_isi[i]) ** 2 / (valid_isi[i+1] + valid_isi[i]) ** 2
                 for i in range(len(valid_isi) - 1)]) / (len(valid_isi) - 1)
    return lv

def calculate_higher_order_lv(isi, order=2):
    """Calculate higher-order LV comparing every nth ISI."""
    valid_isi = isi.dropna().to_numpy()
    if len(valid_isi) < order + 1:
        return np.nan
    lv = np.sum([
        3 * (valid_isi[i+order] - valid_isi[i]) ** 2 / (valid_isi[i+order] + valid_isi[i]) ** 2
        for i in range(len(valid_isi) - order)
    ]) / (len(valid_isi) - order)
    return lv

def calculate_fano_factor(isi):
    """Calculate ISI-based Fano factor."""
    return isi.var() / isi.mean()

def calculate_renyi_entropy(isi, alpha=0.5):
    """Calculate Renyi entropy of ISI."""
    isi_clean = isi.dropna().to_numpy()
    if len(isi_clean) == 0:
        return np.nan
    hist, _ = np.histogram(isi_clean, bins='auto', density=True)
    hist += 1e-10  # Small constant to prevent log(0) errors
    renyi_entropy = 1 / (1 - alpha) * np.log(np.sum(hist ** alpha))
    return renyi_entropy


# Group by 'cell_id' and aggregate the physiological measures
df_summary = df_sorted.groupby('cell_id')['isi'].agg([
    ('mean_isi', 'mean'),
    ('burst_index', calculate_burst_index),
    ('fano_factor', calculate_fano_factor),
    ('lv', calculate_lv),
    ('higher_order_lv', calculate_higher_order_lv),
    ('renyi_entropy', lambda x: calculate_renyi_entropy(x, alpha=0.5)),
])


df_summary = df_summary.reset_index()  

df = pd.merge(df_sorted, df_summary, on=["cell_id"], how="outer")

#merged_df = pd.concat([df_summary, df2], axis=1)

print(df_summary)

df_summary.to_csv(r'D:\nigraPhys\df_summary_pub.csv', index=False)


#%% calculate average wave

grouped = df.groupby(["patient_id", "hemisphere", "depth","class_idx"], sort=False)
df4 = grouped["wave"].apply(np.mean,axis=0)

df4 = df4.to_frame()
df4 = df4.reset_index()

df5 = grouped["s_freq"].apply(np.mean,axis=0)
df6 = grouped["new_s_freq"].apply(np.mean,axis=0)

df5 = df5.to_frame()
df5 = df5.reset_index()
df6 = df6.to_frame()
df6 = df6.reset_index()
df4["s_freq"]= df5["s_freq"]
df4["new_s_freq"]= df6["new_s_freq"]


# Group by cell_id
grouped_time_p = df_sorted.groupby(["cell_id"], sort=False)

# Add 'time_p' associated with each cell as an array
df4["time_p"] = grouped_time_p["time_p"].apply(list).reset_index()["time_p"]

grouped_waves = df_sorted.groupby(["cell_id"], sort=False)

df4["waves"] = grouped_waves["rwave"].apply(list).reset_index()["rwave"]

print(df4)


#%% resample waves

def resample_wave(x,fs):
    if fs > 40000:
        indx = np.arange(0,len(x),2)
        output = x[indx]
    else:
        output = x
        
    return(output[0:41])

df4["resampled_wave"] = df4.apply(lambda x: resample_wave(x["wave"],x["s_freq"]),axis =1)
#df4_sorted["resampled_wave"] = df4.apply(lambda x: resample_wave(x["wave"],x["s_freq"]),axis =1)

df4["check_size"] = df4["resampled_wave"].apply(np.size)

def invert_only_positive(wave):
    abs_min = abs(np.min(wave))
    abs_max = abs(np.max(wave))
    if abs_max > abs_min:
        wave = -wave
        
    return(wave)

df4["resampled_wave"] = df4.apply(lambda x: invert_only_positive(x["resampled_wave"]),axis =1)


#%% calculate derivatives and add to df

def add_derivatives(df):
    """Adds columns for the first and second derivatives to a given dataframe"""
    first_derivatives = []
    second_derivatives = []
    for i, row in df4.iterrows():
        neuron = row['resampled_wave']
        first_derivative = np.gradient(neuron)
        second_derivative = np.gradient(first_derivative)
        first_derivatives.append(first_derivative)
        second_derivatives.append(second_derivative)
    df['first_derivative'] = first_derivatives
    df['second_derivative'] = second_derivatives
    return df

df4=add_derivatives(df4)


#%% z score waveforms

df4["z_wave"]= df4["resampled_wave"].apply(zscore)
df4["z_first_derivative"] = df4["first_derivative"].apply(zscore)
df4["z_second_derivative"] = df4["second_derivative"].apply(zscore)


#%% gaussian smooth 

df4["z_wave_s"] = df4.apply(lambda x: gaussian_filter(x['z_wave'], 2),axis =1)
df4["z_fd_s"] = df4.apply(lambda x: gaussian_filter(x['z_first_derivative'], 2),axis =1)
df4["z_sd_s"] = df4.apply(lambda x: gaussian_filter(x['z_second_derivative'], 2),axis =1)


#%% waveform measures

# Iterate over rows in df4
for i, row in df4.iterrows():
    
    # Extract waveform, first derivative and second derivative
    neuron = row['z_wave_s']
    first_derivative = row['z_fd_s']
    second_derivative = row['z_sd_s']

    # Find minimum peak and its index within the first 15 samples
    min_peak = neuron[:15].min()
    min_index = neuron.argmin()

    # If the minimum peak is not within the first 15 samples, skip this row
    if min_index > 14:
        continue

    # Find maximum peak and its index
    zero_crossing = False
    for j in range(min_index, len(first_derivative)):
        if first_derivative[j] < 0 and first_derivative[j-1] >= 0:
            if j - min_index >= 2:
                zero_crossing = True
                break
        elif abs(first_derivative[j]) <= 0.001:
            if j - min_index >= 2:
                zero_crossing = True
                break
        elif abs(first_derivative[j]) <= 0.0075:
            if j - min_index >= 2:
                zero_crossing = True
                break
        elif abs(first_derivative[j]) <= 0.01:
            if j - min_index >= 2:
                zero_crossing = True
                break

    if zero_crossing:
        max_index = np.argmax(neuron[min_index:j]) + min_index
    else:
        max_index = np.argmax(neuron[min_index:]) + min_index

    if neuron[max_index] == neuron[min_index]:
        # Find index of maximum value in the range starting from min_index+1
        max_index = np.argmax(neuron[min_index+1:]) + min_index + 1

    # Find inflection point
    zero_crossing = False
    for j in range(max_index, len(second_derivative)):
         if second_derivative[j] > 0 and second_derivative[j-1] <= 0:
             zero_crossing = True
             break

    if zero_crossing:
         inflection_index = j
    else:
         inflection_index = np.argmin(np.abs(second_derivative[max_index:])) + max_index

    # Compute various values
    duration = max_index - min_index
    peak_to_trough_amplitude = neuron[max_index] - neuron[min_index]
    hyperpolarization_time = inflection_index - max_index

    half_width_index = None
    for j in range(min_index, len(neuron)):
        if neuron[j] >= min_peak + peak_to_trough_amplitude / 2.0:
            half_width_index = j
            break

    half_width_time = half_width_index - min_index

    # Add computed values to the dataframe as new columns
    df4.at[i, 'min_peak'] = min_peak
    df4.at[i, 'min_index'] = min_index
    df4.at[i, 'max_peak'] = neuron[max_index]
    df4.at[i, 'max_index'] = max_index
    df4.at[i, 'peak_to_peak_duration'] = duration
    df4.at[i, 'peak_to_trough_amplitude'] = peak_to_trough_amplitude
    df4.at[i, 'hyperpolarization_time'] = hyperpolarization_time
    df4.at[i, 'half_width_time'] = half_width_time


sf= 22000
df4['peak_to_peak_duration'] = (df4['peak_to_peak_duration']/sf)*1000
df4['hyperpolarization_time'] = (df4['hyperpolarization_time']/sf)*1000
df4['half_width_time'] = (df4['half_width_time']/sf)*1000

# Add jitter to the data
jitter_amount = 0.05  # Adjust the jitter amount based on your preference
jitter_x = np.random.uniform(-jitter_amount, jitter_amount, size=len(df4))
jitter_y = np.random.uniform(-jitter_amount, jitter_amount, size=len(df4))
jittered_hyperpolarization_time = df4['hyperpolarization_time'] + jitter_x
jittered_peak_to_peak_duration = df4['peak_to_peak_duration'] + jitter_y

# Create the scatter plot with jittered data
plt.scatter(jittered_hyperpolarization_time, jittered_peak_to_peak_duration, s=10)
plt.xlabel('hyperpolarization_time')
plt.ylabel('Peak-to-Peak Duration (ms)')
plt.title('Hyperpolarization Duration vs Peak-to-Peak Duration')
plt.show()


#%% spike waveform visualization (Sanity check)

row = df4.iloc[10]
label = f"{row['patient_id']} {row['hemisphere']} {row['depth']} {row['class_idx']}"

neuron = row['z_wave_s']
first_derivative = row['z_fd_s']
second_derivative = row['z_sd_s']

time = range(len(neuron))
min_peak = neuron.min()
min_index = neuron.argmin()
max_peak = neuron[min_index]

# Find first zero crossing of first derivative after the minimum
zero_crossing = False
for i in range(min_index, len(first_derivative)):
    if first_derivative[i] < 0 and first_derivative[i-1] >= 0:
        if i - min_index >= 2:
            zero_crossing = True
            break
    elif abs(first_derivative[i]) <= 0.001:
        if i - min_index >= 2:
            zero_crossing = True
            break
    elif abs(first_derivative[i]) <= 0.0075:
        if i - min_index >= 2:
            zero_crossing = True
            break
    elif abs(first_derivative[i]) <= 0.01:
        if i - min_index >= 2:
            zero_crossing = True
            break

if zero_crossing:
    max_index = np.argmax(neuron[min_index:i]) + min_index
else:
    max_index = np.argmax(neuron[min_index:]) + min_index
    
if neuron[max_index] == neuron[min_index]:
    # Find index of maximum value in the range starting from min_index+1
    max_index = np.argmax(neuron[min_index+1:]) + min_index + 1

# Find next zero crossing of second derivative after the maximum
zero_crossing = False
for i in range(max_index, len(second_derivative)):
     if second_derivative[i] > 0 and second_derivative[i-1] <= 0:
         zero_crossing = True
         break

if zero_crossing:
     inflection_index = i
else:
     inflection_index = np.argmin(np.abs(second_derivative[max_index:])) + max_index

duration = max_index - min_index

hyperpolarization_time = inflection_index - max_index

peak_to_trough_amplitude = neuron[max_index] - neuron[min_index]

## spike HW

half_width_index = None
for i in range(min_index, len(neuron)):
    if neuron[i] >= min_peak + peak_to_trough_amplitude / 2.0:
        half_width_index = i
        break
half_width_time = half_width_index - min_index

fig = plt.figure()
plt.plot(neuron,'bo', label=label,)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.title('Z-scored Shape of Neuronal Cells')
plt.scatter(min_index, neuron[min_index], color='red')
plt.scatter(max_index, neuron[max_index], color='red')
plt.plot(time, second_derivative, label='Second Derivative')
plt.plot(time, first_derivative, label='First Derivative')
fill_array = np.zeros(len(time))
fill_array = np.ones(max_index-min_index+1)*neuron[min_index]
plt.fill_between(range(min_index,max_index+1),fill_array,color='gray', alpha=0.5)
plt.axhline(y=0, xmin=0, xmax=len(time)-1)
plt.text(min_index,min_peak,'peak_to_peak_duration: '+str(duration)+'ms',fontsize=9.5)
plt.fill_between(time[max_index:inflection_index+1],neuron[max_index:inflection_index+1],0, color='gray', alpha=0.5)
plt.text(inflection_index, 0.2, 'hyperpolarization: ' + str(hyperpolarization_time) + 'ms', fontsize=9.5)
plt.axvline(x=half_width_index, color='green', linestyle='--')
plt.text(half_width_index,min_peak,'Half Width: '+str(half_width_time)+'ms',fontsize=9.5)
plt.legend()
plt.show()


#%% conceptual spike and derivative visualization - Figure 1

# Step 1: Simulate a realistic action potential waveform
t = np.linspace(0, 1, 1000)  # Time vector (seconds)
waveform = -70 + 100 * np.exp(-100 * (t - 0.5)**2)  # Gaussian-like depolarization
waveform += -50 * np.exp(-50 * (t - 0.65)**2)  # Hyperpolarization undershoot

# Step 2: Compute first and second derivatives
first_derivative = np.gradient(waveform, t)
second_derivative = np.gradient(first_derivative, t)

# Step 3: Normalize each layer independently to avoid flattening
scale_waveform = max(np.abs(waveform))
scale_first_derivative = max(np.abs(first_derivative))
scale_second_derivative = max(np.abs(second_derivative))

normalized_waveform = waveform / scale_waveform
normalized_first_derivative = first_derivative / scale_first_derivative
normalized_second_derivative = second_derivative / scale_second_derivative

# Step 4: Identify key features
max_peak_idx = np.argmax(waveform)
min_peak_idx = np.argmin(waveform)
fd_min_idx = np.argmin(first_derivative)
sd_min_idx = np.argmin(second_derivative)
sd_max_idx = np.argmax(second_derivative)

# Step 5: Create the layout for the visualization
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Define depth spacing for better separation
depth_waveform = 0
depth_first_derivative = -10
depth_second_derivative = -20

# Plot the normalized original waveform at the front
ax.plot(t, normalized_waveform, zs=depth_waveform, zdir='y', color='blue', lw=3, label="Original Waveform")

# Plot the normalized first derivative at a deeper layer
ax.plot(t, normalized_first_derivative, zs=depth_first_derivative, zdir='y', color='orange', lw=3, label="First Derivative")

# Plot the normalized second derivative at the deepest layer
ax.plot(t, normalized_second_derivative, zs=depth_second_derivative, zdir='y', color='green', lw=3, label="Second Derivative")

# Highlighting key points with proper scaling
ax.scatter(t[max_peak_idx], depth_waveform, normalized_waveform[max_peak_idx], color='black', s=70, label="Max Peak")
ax.scatter(t[min_peak_idx], depth_waveform, normalized_waveform[min_peak_idx], color='red', s=70, label="Min Peak")
ax.scatter(t[fd_min_idx], depth_first_derivative, normalized_first_derivative[fd_min_idx], color='red', s=70, label="Min (1st Derivative)")
ax.scatter(t[sd_min_idx], depth_second_derivative, normalized_second_derivative[sd_min_idx], color='red', s=70, label="Min (2nd Derivative)")
ax.scatter(t[sd_max_idx], depth_second_derivative, normalized_second_derivative[sd_max_idx], color='black', s=70, label="Max (2nd Derivative)")

# Adjusting labels and ticks
ax.set_xlabel("Time (s)", labelpad=15)
ax.set_ylabel("Feature Layers", labelpad=15)
ax.set_zlabel("Normalized Amplitude", labelpad=15)

# Setting depth labels
ax.set_yticks([depth_waveform, depth_first_derivative, depth_second_derivative])
ax.set_yticklabels(["Original Waveform", "1st Derivative", "2nd Derivative"])

# Adding a legend for clarity
ax.legend(loc="upper left", bbox_to_anchor=(1.2, 1), fontsize=10)

# Refining perspective to ensure clarity without connecting lines
ax.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()


# Step 1: Simulate a realistic action potential waveform
t = np.linspace(0, 1, 1000)  # Time vector (seconds)
waveform = -70 + 100 * np.exp(-100 * (t - 0.5)**2)  # Gaussian-like depolarization
waveform += -50 * np.exp(-50 * (t - 0.65)**2)  # Hyperpolarization undershoot

# Step 2: Compute first and second derivatives
first_derivative = np.gradient(waveform, t)
second_derivative = np.gradient(first_derivative, t)

# Step 3: Compute FFT and energy spectrum
fft_values = fft(waveform)
frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))  # Frequency axis
energy_spectrum = np.abs(fft_values) ** 2  # Energy spectrum

# Crop frequency range (e.g., meaningful range: 0-100 Hz)
freq_limit = 100
valid_indices = frequencies >= 0  # Ignore negative frequencies
valid_indices &= frequencies <= freq_limit  # Limit to meaningful range

cropped_frequencies = frequencies[valid_indices]
cropped_energy_spectrum = energy_spectrum[valid_indices]

# Identify key features
max_peak = np.max(waveform)
min_peak = np.min(waveform)
fd_min = np.min(first_derivative)
sd_min = np.min(second_derivative)
sd_max = np.max(second_derivative)

# Indices of the key features
max_peak_idx = np.argmax(waveform)
min_peak_idx = np.argmin(waveform)
fd_min_idx = np.argmin(first_derivative)
sd_min_idx = np.argmin(second_derivative)
sd_max_idx = np.argmax(second_derivative)

# Step 4: Define GridSpec layout for better spacing
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(8, 8, figure=fig)

# Left Panel: Original waveform
ax_waveform = fig.add_subplot(gs[:, 0:3])  # Entire left column
ax_waveform.plot(t, waveform, color='saddlebrown', lw=5)
ax_waveform.set_title("Original Waveform", fontsize=25, weight='bold')
ax_waveform.set_xlabel("Time (ms)", fontsize=25, weight='bold')
ax_waveform.set_ylabel("Amplitude (mV)", fontsize=25, weight='bold')

# Right Panels: Distribute evenly in height
ax_peaks = fig.add_subplot(gs[0:2, 4:8])  # Peaks
ax_fd = fig.add_subplot(gs[2:4, 4:8])  # First Derivative
ax_sd = fig.add_subplot(gs[4:6, 4:8])  # Second Derivative
ax_energy = fig.add_subplot(gs[6:8, 4:8])  # Energy Spectrum

legend_font = {'size': 20, 'weight': 'bold'}

# Right Panel 1: First Derivative
ax_fd.plot(t, first_derivative, color='chocolate', lw=5, alpha=0.6)
ax_fd.scatter(t[fd_min_idx], fd_min, color='red', s=75, label="Min", zorder=7)
ax_fd.set_title("First Derivative", fontsize=25, weight='bold')
ax_fd.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop=legend_font)

# Right Panel 2: Second Derivative
ax_sd.plot(t, second_derivative, color='peru', lw=5, alpha=0.6)
ax_sd.scatter(t[sd_min_idx], sd_min, color='red', s=75, label="Min", zorder=7)
ax_sd.scatter(t[sd_max_idx], sd_max, color='black', s=75, label="Max", zorder=7)
ax_sd.set_title("Second Derivative", fontsize=25, weight='bold')
ax_sd.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop=legend_font)

# Right Panel 3: Peaks
ax_peaks.plot(t, waveform, color='saddlebrown', lw=5, alpha=0.6)
ax_peaks.scatter(t[min_peak_idx], min_peak, color='red', s=75, label="Min", zorder=7)
ax_peaks.scatter(t[max_peak_idx], max_peak, color='black', s=75, label="Max", zorder=7)
ax_peaks.set_title("Original Waveform Peaks", fontsize=25, weight='bold')
ax_peaks.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop=legend_font)

# Right Panel 4: Energy Spectrum (Log Scale)
ax_energy.plot(cropped_frequencies, cropped_energy_spectrum, color='sandybrown', lw=5, alpha=0.6, label="Energy Spectrum")
ax_energy.fill_between(cropped_frequencies, 0, cropped_energy_spectrum, color='sandybrown', alpha=0.4, label="Total Energy")
ax_energy.set_yscale('log')  # Log scale applied here
ax_energy.set_title("Energy Power Spectrum", fontsize=25, weight='bold')
ax_energy.set_xlabel("Frequency (Hz)", fontsize=25, weight='bold')
ax_energy.set_ylabel("Energy", fontsize=25, weight='bold')
ax_energy.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop=legend_font)

# Remove spines and grid for all plots, keep only x and y axes, and remove ticks
for ax in [ax_waveform, ax_fd, ax_sd, ax_peaks, ax_energy]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.grid(False)

plt.tight_layout()
plt.show()


# Step 1: Simulate a realistic action potential waveform
fs = 22000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs)  # Time vector (1 second at 22kHz)
waveform = -70 * np.ones_like(t)  # Baseline at -70 mV

# Add multiple Gaussian spikes with realistic amplitudes
spike_times = [0.1, 0.3, 0.5, 0.7, 0.9]  # Spike locations in seconds
spike_amplitudes = [50, 60, 70, 55, 65]  # Spike amplitudes in mV
for i, spike_time in enumerate(spike_times):
    waveform += spike_amplitudes[i] * np.exp(-200 * (t - spike_time)**2)

# Add realistic noise and subtle oscillations to simulate raw recording
np.random.seed(42)
base_noise = np.random.normal(0, 15, len(t))  # Noise amplitude matches your example
high_freq_noise = 1 * np.sin(2 * np.pi * 500 * t)  # Subtle 500 Hz oscillations
raw_data = waveform + base_noise + high_freq_noise  # Combine spike signal, noise, and oscillations

# Step 2: Apply bandpass filter (300Hz to 3000Hz)
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

lowcut = 300  # Low frequency cutoff (Hz)
highcut = 3000  # High frequency cutoff (Hz)
filtered_data = butter_bandpass_filter(raw_data, lowcut, highcut, fs)

# Detect spikes
spike_threshold = np.percentile(filtered_data, 97.5)  # Threshold for spike detection
spike_mask = filtered_data > spike_threshold  # Boolean mask for spike regions

# Step 3: Plot Raw and Filtered Data with spike highlighting
fig2 = plt.figure(figsize=(10, 6))  # Separate figure for clarity

# Plot raw data
raw_line, = plt.plot(t, raw_data, color='gray', lw=0.5, alpha=0.6, label="Raw Data")

# Plot the filtered trace with spike coloring
highlighted_lines = []
for start, end in zip(np.where(np.diff(spike_mask.astype(int)) == 1)[0], 
                      np.where(np.diff(spike_mask.astype(int)) == -1)[0]):
    highlighted_lines.append(
        plt.plot(t[start:end], filtered_data[start:end], color='saddlebrown', lw=3, label="_nolegend_")
    )

filtered_line, = plt.plot(t[~spike_mask], filtered_data[~spike_mask], color='black', lw=1.5, alpha=0.9, label="Filtered Data")

# Title and labels
plt.xlabel("Time (s)", fontsize=25, weight='bold')
plt.ylabel("Amplitude (mV)", fontsize=25, weight='bold')

# Remove ticks
plt.xticks([])
plt.yticks([])

# Remove upper and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Add legend with thicker symbols
plt.legend(
    [raw_line, filtered_line],
    ["Raw Data", "Filtered Data"],
    bbox_to_anchor=(1.0, 1.2),  # Manually adjust legend position (higher and to the right)

    loc="upper right",
    frameon=False,
    prop={'weight': 'bold', 'size': 20},
    handlelength=2,  # Increase handle length
    labelspacing=0.5,  # Space between labels
    handletextpad=1,  # Padding between handle and text
)

# Adjust legend line width
leg = plt.gca().get_legend()
for line in leg.legendHandles:
    line.set_linewidth(3)  # Make legend lines thicker

# Tight layout and show plot
plt.tight_layout()
plt.show()


#%% assign 'cell_id' and merge 

df2['cell_id'] = df2['class_idx'].astype(str) + '_' + df2['patient_id'].astype(str) + '_' + df2['hemisphere'].astype(str) + '_' + df2['depth'].astype(str)

df4['cell_id'] = df4['class_idx'].astype(str) + '_' + df4['patient_id'].astype(str) + '_' + df4['hemisphere'].astype(str) + '_' + df4['depth'].astype(str)

df4 = df4.merge(df2[['cell_id', 'firing']], on='cell_id', how='left')

df4 = df4.merge(df_summary[['cell_id', 'mean_isi','burst_index', 'fano_factor',
       'lv', 'higher_order_lv', 'renyi_entropy']], on='cell_id', how='left')

df4 = df4.merge(df_atlas_struct[['patient_id', 'hemisphere', 'depth', 'struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli', 'x', 'y', 'z']], 
                on=['patient_id', 'hemisphere', 'depth'], how='left')


print(df4)


#%% filtering/cleaning 2

# List of cell IDs to delete
cell_ids_to_delete = [
    '0_359_right_-4.0', '0_356_right_-3.0', '0_304_left_-4.5',
    '2_308_right_-8.0', '1_306_left_-1.5', '0_306_right_-6.5',
    '1_319_right_-4.2', '0_321_left_-3.5', '1_313_right_-3.0', '0_306_right_-4.5',
    '1_298_left_-6.0', '1_308_right_-7.0', '0_361_left_-2.92', '2_312_right_-4.5',
    '2_321_right_-4.0', '0_304_right_-7.0', '3_315_right_-3.0', '0_349_right_-4.0',
    '1_306_left_-4.7', '0_308_right_-5.5', '1_321_right_-6.0',  '0_359_right_-4.0',
    '1_307_left_-3.0', '1_349_left_-4.5','1_351_left_-5.0', '0_353_right_-2.6', '0_359_right_-4.22'
]


# Printing neurons being deleted
print("Neurons being deleted and their index in df4:")
for cell_id in cell_ids_to_delete:
    if cell_id in df4['cell_id'].values:  # Ensure the cell_id exists in df4
        # Find the index of the cell_id in df4
        index_number = df4.index[df4['cell_id'] == cell_id].tolist()
        firing_rate = df4.loc[df4['cell_id'] == cell_id, 'firing'].values[0]
        print(f"Cell ID: {cell_id}, Index in df4: {index_number}, Firing Rate: {firing_rate}")


# Drop rows from df4, df_sorted, and df_summary
df4 = df4[~df4['cell_id'].isin(cell_ids_to_delete)].reset_index(drop=True)
df_sorted = df_sorted[~df_sorted['cell_id'].isin(cell_ids_to_delete)]
df_summary = df_summary[~df_summary['cell_id'].isin(cell_ids_to_delete)]


#%% Higher-order analytical waveform feature calculation (for unsupervised waveform-derived clustering)

df4['energy_rms'] = df4['z_wave_s'].apply(lambda x: np.sqrt(np.mean(np.square(np.abs(x)))))

df4['energy_power_spectrum'] = df4['z_wave_s'].apply(lambda x: np.sum(np.abs(fft(x)) ** 2))

# First-derivative minimum
df4['fd_min'] = df4['z_fd_s'].apply(lambda x: min(x))

# Second-derivative minimum
df4['sd_min'] = df4['z_sd_s'].apply(lambda x: min(x))

# Second-derivative maximum
df4['sd_max'] = df4['z_sd_s'].apply(lambda x: max(x))

print(df4)


#%% elbow and kneedle measure to determine optimal K for unsupervised clustering

# Select features
data = df4[['energy_power_spectrum', 'fd_min', 'sd_min', 'sd_max', 'max_peak', 'min_peak']]

# Normalize the data
scaler = StandardScaler()  # You can also use MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# WCSS calculation
wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, n_init=100, random_state=100)
    kmeans.fit(normalized_data)
    wcss.append(kmeans.inertia_)

# Using the Kneedle algorithm to find the elbow
kneedle = KneeLocator(range(1, 10), wcss, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k: {optimal_k}')
plt.title('Elbow Method with Kneedle Algorithm')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()


#%% K-Means clustering all - Figure 3

# Define specific cluster colors
color_order = ['blue', 'red', 'green', 'cyan',  'magenta' ]

# Create a copy of df4 to preserve the original data
df_kmeans_all = df4.copy()

# Set up clustering
X_pre_imp = df4[['energy_power_spectrum', 'fd_min', 'sd_min', 'sd_max', 'max_peak', 'min_peak']]

# Handle missing values (NaN) with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X_pre_imp)

# Scale the data
X_scaled = StandardScaler().fit_transform(X)

# Perform clustering with K-means
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=224, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Append cluster labels to df_kmeans_all
df_kmeans_all['clustering_kmeans'] = labels_kmeans
df4['clustering_kmeans'] = labels_kmeans

# Calculate cluster distribution for K-means on the entire dataset
cluster_counts_kmeans = df_kmeans_all['clustering_kmeans'].value_counts()

# Sort the cluster percentages in ascending order
cluster_percentages = cluster_counts_kmeans / len(df_kmeans_all)
sorted_cluster_percentages = cluster_percentages.sort_values()

# Create a mapping between cluster labels and colors based on percentage
cluster_color_mapping = {
    label: color_order[i] for i, label in enumerate(sorted_cluster_percentages.index)
}

# Sort the cluster percentages in ascending order and reset the index
sorted_cluster_percentages = sorted_cluster_percentages.reset_index()

# Create a mapping between the sorted cluster labels and their corresponding names
cluster_name_mapping = {
    label: f'Cluster {i+1}' for i, label in enumerate(sorted_cluster_percentages['index'])
}

# Assign cluster names based on the mapping
df_kmeans_all['cluster_name'] = df_kmeans_all['clustering_kmeans'].map(cluster_name_mapping).astype(str)
df4['cluster_name'] = df_kmeans_all['clustering_kmeans'].map(cluster_name_mapping).astype(str)

# Assign cluster colors based on the original mapping
df_kmeans_all['cluster_color'] = df_kmeans_all['clustering_kmeans'].map(cluster_color_mapping)
df4['cluster_color'] = df_kmeans_all['clustering_kmeans'].map(cluster_color_mapping)

# Plot scatter plot for K-means with jitter and colored by percentage
fig, ax = plt.subplots(figsize=(10, 6))
jitter = 0.005

# Add jitter to the x and y data
x_jitter = df_kmeans_all['peak_to_peak_duration'] + np.random.uniform(-jitter, jitter, len(df4))
y_jitter = df_kmeans_all['firing'] + np.random.uniform(-jitter, jitter, len(df4))

scatter_kmeans = ax.scatter(
    x_jitter,
    y_jitter,
    c=df_kmeans_all['cluster_color'],
    label=[f'Cluster {label+1}' for label in df_kmeans_all['clustering_kmeans']],  # Updated labels
)
ax.set_title("K-means clustering with Jitter and Colored by Percentage")
ax.set_xlabel('peak_to_peak_duration')
ax.set_ylabel('firing')

# Create a legend mapping from label to color
legend_labels_kmeans = [f'Cluster {label+1}' for label in sorted_cluster_percentages.index]  # Updated labels
legend_elements_kmeans = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_order[i], markersize=10)
    for i in range(len(color_order))
]
ax.legend(legend_elements_kmeans, legend_labels_kmeans, loc='upper right', title='Cluster (K-means)')

plt.show()

# Pie Chart: Cluster Distribution
fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure for the pie chart

# Use sorted cluster percentages for values
cluster_percentages = sorted_cluster_percentages["clustering_kmeans"].values
cluster_labels = sorted_cluster_percentages["index"].map(cluster_name_mapping).values  # Use the same mapping logic
cluster_colors = sorted_cluster_percentages["index"].map(cluster_color_mapping).values  # Directly map colors

# Create the pie chart
wedges = ax.pie(
    cluster_percentages,
    colors=cluster_colors,
    startangle=90
)[0]  # Extract only the wedges (pie slices)

# Add a legend for the pie chart
ax.legend(
    wedges,  # Use the wedges for the legend
    cluster_labels,  # Use the properly sorted and mapped cluster names
    loc="best",  # Automatically find the best position for the legend
    prop={'size': 14,'weight': 'bold'}
)

plt.title("Cluster Distribution (K-means)", fontsize=14, weight="bold")
plt.tight_layout()
plt.show()


# Prepare data for the table
cluster_counts_by_name = df_kmeans_all['cluster_name'].value_counts()
cluster_percentages = cluster_counts_by_name / len(df_kmeans_all) * 100

# Combine counts and percentages into a DataFrame for the table
cluster_summary = cluster_counts_by_name.to_frame(name='Raw Count')
cluster_summary['Percentage'] = cluster_percentages
cluster_summary.reset_index(inplace=True)
cluster_summary.rename(columns={'index': 'Cluster'}, inplace=True)

# Ensure sorting by cluster number
cluster_summary['Cluster Number'] = cluster_summary['Cluster'].str.extract('(\d+)').astype(int)  # Extract numbers
cluster_summary.sort_values('Cluster Number', inplace=True)  # Sort by cluster number
cluster_summary.drop('Cluster Number', axis=1, inplace=True)  # Drop helper column

# Table Plot: Cluster Details
fig, ax_table = plt.subplots(figsize=(7, 7))  # Adjust size for the table
ax_table.axis('tight')
ax_table.axis('off')
table = ax_table.table(
    cellText=cluster_summary.values,
    colLabels=cluster_summary.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)  # Set font size
for key, cell in table.get_celld().items():
    cell.set_text_props(fontweight='bold')

# Show the table
plt.title('Cluster Details', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# Print unique values in the 'clustering_kmeans' column
unique_cluster_labels = df_kmeans_all['clustering_kmeans'].unique()
print("Unique Cluster Labels:")
print(unique_cluster_labels)

# Print unique values in the 'cluster_name' column
unique_cluster_names = df_kmeans_all['cluster_name'].unique()
print("\nUnique Cluster Names:")
print(unique_cluster_names)

# Print unique values in the 'cluster_color' column
unique_cluster_colors = df_kmeans_all['cluster_color'].unique()
print("\nUnique Cluster Colors:")
print(unique_cluster_colors)

# Calculate cluster distribution for K-means on the entire dataset
cluster_counts_by_name = df_kmeans_all['cluster_name'].value_counts()
print("Raw Counts of Clusters based on Renaming:")
print(cluster_counts_by_name)

df_kmeans_all['x'] = df_kmeans_all['x'].abs()

hemisphere_counts = df_kmeans_all['hemisphere'].value_counts()
print(hemisphere_counts)


#%% T-sne projection visualization

random_state=518

# Set up t-SNE for 2D and 3D projections for K-means clustering results

tsne_2d_kmeans = TSNE(n_components=2, perplexity=30, random_state=random_state)
tsne_3d_kmeans = TSNE(n_components=3, perplexity=30, random_state=random_state)

# Fit PCA and t-SNE to the normalized feature data for K-means clustering
X_tsne_2d_kmeans = tsne_2d_kmeans.fit_transform(X_scaled)
X_tsne_3d_kmeans = tsne_3d_kmeans.fit_transform(X_scaled)

# Get the cluster labels for each data point from K-means clustering
cluster_labels_kmeans = df_kmeans_all['clustering_kmeans']

# Create a list of colors for each cluster based on the renamed clusters
colors_kmeans = df_kmeans_all['cluster_color']

# --------------------------------- 2D PLOTS ----------------------------------

# 2D t-SNE projection
sorted_cluster_names = sorted(df_kmeans_all['cluster_name'].unique())

# 2D t-SNE projection
plt.figure(figsize=(9, 6))
for label in sorted_cluster_names:
    indices = df_kmeans_all['cluster_name'] == label
    plt.scatter(X_tsne_2d_kmeans[indices, 0], X_tsne_2d_kmeans[indices, 1], 
                c=df_kmeans_all.loc[indices, 'cluster_color'], label=label)
plt.title('t-SNE projection: KMeans clustering based on waveform features', fontsize=15, weight='bold')
plt.legend(loc='best', prop={'size': 15, 'weight': 'bold'})
plt.tight_layout()
plt.show()

# --------------------------------- 3D PLOTS ----------------------------------

# 3D t-SNE projection
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for label in sorted_cluster_names:
    indices = df_kmeans_all['cluster_name'] == label
    ax.scatter(X_tsne_3d_kmeans[indices, 0], X_tsne_3d_kmeans[indices, 1], 
               X_tsne_3d_kmeans[indices, 2], c=df_kmeans_all.loc[indices, 'cluster_color'], label=label)
ax.set_title('3D t-SNE projection: KMeans clustering based on waveform features', fontsize=15, weight='bold')
ax.legend(loc='best', prop={'size': 15, 'weight': 'bold'})

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')

if random_state == 21:
    ax.set_xlim(-20, 20)
    ax.set_ylim(-7, 7)
    ax.set_zlim(-11, 11)


plt.tight_layout()
plt.show()


#%% tsne with elbow inset - Figure 3

plt.figure(figsize=(9, 6))

# Main t-SNE plot
for label in sorted(df_kmeans_all['cluster_name'].unique()):
    indices = df_kmeans_all['cluster_name'] == label
    plt.scatter(X_tsne_2d_kmeans[indices, 0], X_tsne_2d_kmeans[indices, 1], 
                c=df_kmeans_all.loc[indices, 'cluster_color'], label=label)
plt.title('t-SNE projection: KMeans clustering based on waveform features', fontsize=15, weight='bold')

plt.xlabel('t-SNE 1', fontsize=12, weight='bold') 
plt.ylabel('t-SNE 2', fontsize=12, weight='bold')
plt.tick_params(axis='both', which='major', labelsize=10, width=2)
plt.setp(plt.gca().get_xticklabels(), fontweight='bold', fontsize=12) 
plt.setp(plt.gca().get_yticklabels(), fontweight='bold', fontsize=12)  

# Position the legend in the upper right to avoid overlap with the inset
plt.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})

# Create inset for Elbow plot in the top-left corner
inset_ax = plt.axes([0.17, 0.7, 0.245, 0.225])  # Adjust left, bottom, width, height for top-left placement
inset_ax.plot(range(1, 10), wcss, marker='o', linestyle='--')
inset_ax.axvline(x=optimal_k, color='red', linestyle='--', label='Optimal k')
inset_ax.set_xlabel('Clusters', fontsize=9, weight='bold')
inset_ax.set_ylabel('WCSS', fontsize=9, weight='bold')
inset_ax.tick_params(axis='both', which='major', labelsize=8)
plt.setp(inset_ax.get_xticklabels(), fontweight='bold', fontsize=9)
plt.setp(inset_ax.get_yticklabels(), fontweight='bold', fontsize=9)
inset_ax.legend(fontsize=8, prop={'weight': 'bold'},loc=(0.42, 0.8))

plt.tight_layout()
plt.show()


#%% Merge dataframes

df4[['patient_id', 'hemisphere', 'depth']] = df4[['patient_id', 'hemisphere', 'depth']].astype(str)
lfp_df[['patient_id', 'hemisphere', 'depth']] = lfp_df[['patient_id', 'hemisphere', 'depth']].astype(str)

df_spike_lfp = df4.copy()

df_spike_lfp = df_spike_lfp.merge(lfp_df[['patient_id', 'hemisphere', 'depth', 'lfp_data_resampled','lfp_data_second_filtered']], 
                on=['patient_id', 'hemisphere', 'depth'], 
                how='left')

print(df_spike_lfp)

df_spike_lfp['x'] = df_spike_lfp['x'].abs()


#%% Save dataframe to local computer

columns_to_save = ['patient_id', 'hemisphere', 'depth', 'cluster_name', 'cluster_color', 
                   'cell_id', 'firing', 'mean_isi', 'burst_index', 'fano_factor',
                   'lv', 'higher_order_lv', 'renyi_entropy', 'x', 'y', 'z']
filtered_df = df_spike_lfp[columns_to_save]

filtered_df.to_csv('clustered_cells_pub.csv', index=False)


#%% ISI distribution 

grouped = df_sorted.groupby(["cell_id"], sort=False)
isi_avg = grouped["isi"].apply(np.mean,axis=0).values
df4 = df4.assign(isi=isi_avg)


#%% Merge Dataframes

df_summary = pd.merge(df_summary, df_kmeans_all[['cell_id', 'clustering_kmeans', 'cluster_name', 'cluster_color','struct_HyPD', 'dist_HyPD', 'struct_pauli', 'dist_pauli', 'x', 'y', 'z', 'peak_to_peak_duration', 'hyperpolarization_time', 'peak_to_trough_amplitude', 'half_width_time', 'firing']],
                      on='cell_id', how='left')


#%% distribution of firing rates 

# Filter data for each cluster
cluster_1 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 1']['firing']
cluster_2 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 2']['firing']
cluster_3 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 3']['firing']
cluster_4 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 4']['firing']

class_colors = color_order

# Determine the overall range of firing rates
min_firing_rate = min(df_kmeans_all['firing'])
max_firing_rate = max(df_kmeans_all['firing'])

# Create density plots for firing rates with different bandwidth values
plt.figure(figsize=(7, 6))

for i, cluster_data in enumerate([cluster_1, cluster_2, cluster_3, cluster_4]):

    cluster_name = f'Cluster {i}'
    color = class_colors[i]

    # Experiment with different bandwidth values (e.g., bw=0.1, bw=0.5, bw=1.0)
    sns.kdeplot(cluster_data, color=color, label=cluster_name)  # Adjust the bandwidth value


font_labels = {'fontsize': 12, 'fontweight': 'bold'}

plt.xlabel('Firing Rate', fontdict=font_labels)
plt.ylabel('Density', fontdict=font_labels)
plt.xlim(0, max_firing_rate)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Perform Kruskal-Wallis test
kruskal_statistic, kruskal_pvalue = stats.kruskal(cluster_1, cluster_2, cluster_3, cluster_4)
print('Kruskal-Wallis Statistic:', kruskal_statistic)
print("Kruskal-Wallis Test - p-value:", kruskal_pvalue)

pairwise_comparisons = list(itertools.combinations(df_kmeans_all['cluster_name'].unique(), 2))
results = {}

for cluster1, cluster2 in pairwise_comparisons:
    test_statistic, p_value = stats.mannwhitneyu(df_kmeans_all[df_kmeans_all['cluster_name'] == cluster1]['firing'], df_kmeans_all[df_kmeans_all['cluster_name'] == cluster2]['firing'], alternative='two-sided')
    results[(cluster1, cluster2)] = {
        'Test Statistic': test_statistic,
        'P-value': p_value * len(pairwise_comparisons)  # Bonferroni correction
    }

# Convert the results to a DataFrame for easier visualization
posthoc_results = pd.DataFrame(results).T

plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(posthoc_results['P-value'].unstack().T, cmap='coolwarm', annot=True, fmt='.2g', cbar_kws={'label': 'P-value'})

# Customize x-axis and y-axis tick font sizes
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12, fontweight='bold')
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12, fontweight='bold')

# Customize color bar label font size and font weight
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6)  # Set color bar label size and style
cbar.ax.set_title(cbar.ax.get_title(), fontsize=12, fontweight='bold')  # Set color bar title font size and font weight

# Loop through text objects in the heatmap and set them to bold
for text in heatmap.texts:
    text.set_fontweight('bold')
    text.set_color('black')

plt.tight_layout()
plt.show()


#%% Distribution of Firing Rates 2.0

color_order = ['blue', 'red', 'green', 'cyan']

# Filter data for each cluster
cluster_1 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 1']['firing']
cluster_2 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 2']['firing']
cluster_3 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 3']['firing']
cluster_4 = df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 4']['firing']

# Combine the clusters into a single DataFrame for plotting
cluster_data = pd.DataFrame({
    'Firing Rate': pd.concat([cluster_1, cluster_2, cluster_3, cluster_4]),
    'Cluster': ['Cluster 1'] * len(cluster_1) + \
               ['Cluster 2'] * len(cluster_2) + \
               ['Cluster 3'] * len(cluster_3) + \
               ['Cluster 4'] * len(cluster_4)
})

# Create the histogram with polygon elements
plt.figure(figsize=(10, 6))
sns.histplot(
    data=cluster_data, 
    x='Firing Rate', 
    hue='Cluster', 
    element='poly',  # Use polygons instead of bars
    stat='count',
    common_norm=False, 
    bins=10,  # Adjust number of bins as necessary
    alpha=0.3,  # Set transparency for better visibility
    palette=color_order,  # Use the defined color palette
    legend=False  # Turn off automatic legend
)

# Set the labels and title
font_labels = {'fontsize': 12, 'fontweight': 'bold'}
plt.xlabel('Firing Rate (Hz)', fontdict=font_labels)
plt.ylabel('count', fontdict=font_labels)
plt.xlim(0, max(df_kmeans_all['firing']))  # Ensure x-axis starts from 0
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of Firing Rates by Cluster', fontsize=14, fontweight='bold')

# Manually add the legend
handles = [plt.Line2D([0], [0], color=color_order[i], lw=4) for i in range(len(color_order))]
plt.legend(handles, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], 
            fontsize=10, title_fontsize='12', loc='upper right', frameon=True)

plt.tight_layout()
plt.show()

# Perform Kruskal-Wallis test
kruskal_statistic, kruskal_pvalue = stats.kruskal(cluster_1, cluster_2, cluster_3, cluster_4)

print("Kruskal-Wallis Test - p-value:", kruskal_pvalue)


# Perform pairwise comparisons for post hoc tests with Bonferroni correction
pairwise_comparisons = list(itertools.combinations(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], 2))
results = {}

for cluster1, cluster2 in pairwise_comparisons:
    test_statistic, p_value = stats.mannwhitneyu(
        df_kmeans_all[df_kmeans_all['cluster_name'] == cluster1]['firing'],
        df_kmeans_all[df_kmeans_all['cluster_name'] == cluster2]['firing'],
        alternative='two-sided'
    )
    # Apply Bonferroni correction without a cap
    corrected_p_value = p_value * len(pairwise_comparisons)
    results[(cluster1, cluster2)] = corrected_p_value

# Create a DataFrame for the results to be visualized in a heatmap
# Limit y-axis to Cluster 2 to Cluster 4 and x-axis to Cluster 1 to Cluster 3
posthoc_results = pd.DataFrame(index=['Cluster 2', 'Cluster 3', 'Cluster 4'],  # Rows start from Cluster 2
                               columns=['Cluster 1', 'Cluster 2', 'Cluster 3'])  # Columns end at Cluster 3

# Populate the DataFrame with p-values for the specified lower triangle
for (cluster1, cluster2), p_value in results.items():
    if cluster1 in posthoc_results.columns and cluster2 in posthoc_results.index:
        posthoc_results.loc[cluster2, cluster1] = p_value  # Fill only specified lower triangle

# Convert values to numeric to ensure proper visualization
posthoc_results = posthoc_results.apply(pd.to_numeric)

# Plot the heatmap without a mask, as only necessary cells are populated
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    posthoc_results, 
    cmap='coolwarm', 
    annot=True, 
    fmt='.2g', 
    cbar_kws={'label': 'P-value'}
)

# Customize x-axis and y-axis tick font sizes and font weight
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12, fontweight='bold')
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12, fontweight='bold')

# Customize color bar label font size and font weight
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6)
cbar.ax.set_title(cbar.ax.get_title(), fontsize=12, fontweight='bold')

# Loop through text objects in the heatmap and set them to bold
for text in heatmap.texts:
    text.set_fontweight('bold')
    text.set_color('black')

plt.title('Post Hoc Analysis for Firing Rate', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% distribution of firing rates 3.0 (boxplots)

# Define colors for each cluster
color_map = {
    'Cluster 1': 'blue',
    'Cluster 2': 'red',
    'Cluster 3': 'green',
    'Cluster 4': 'cyan'
}

# Define pairwise comparisons
pairwise_comparisons = [
    ('Cluster 1', 'Cluster 2'),
    ('Cluster 1', 'Cluster 3'),
    ('Cluster 1', 'Cluster 4'),
    ('Cluster 2', 'Cluster 3'),
    ('Cluster 2', 'Cluster 4'),
    ('Cluster 3', 'Cluster 4')
]

# Create a figure with 1 row and subplots for each pair
fig, axes = plt.subplots(1, len(pairwise_comparisons), figsize=(20, 6), sharey=True)

for idx, (cluster1, cluster2) in enumerate(pairwise_comparisons):
    ax = axes[idx]  # Current subplot

    # Extract data for the two clusters being compared
    cluster1_data = df_kmeans_all[df_kmeans_all['cluster_name'] == cluster1]['firing']
    cluster2_data = df_kmeans_all[df_kmeans_all['cluster_name'] == cluster2]['firing']
    data = [cluster1_data, cluster2_data]

    # Create boxplot
    box = ax.boxplot(
        data,
        patch_artist=True,  # Enables coloring of the boxes
        widths=0.6,  # Box width
        showmeans=True,  # Show the mean
        meanline=True,  # Show the mean as a line
        meanprops={"color": "black", "linestyle": "--", "linewidth": 1.5},
        whiskerprops={"linewidth": 2},  # Default whisker props
        capprops={"linewidth": 2},  # Default cap props
        medianprops={"color": "black", "linewidth": 2}  # Median line
    )

    # Assign custom colors for boxes, whiskers, caps, and fliers
    cluster_colors = [color_map[cluster1], color_map[cluster2]]

    for i, color in enumerate(cluster_colors):
        # Box coloring
        box['boxes'][i].set_facecolor(color)
        box['boxes'][i].set_edgecolor(color)

        # Whisker coloring
        box['whiskers'][2 * i].set_color(color)  # Lower whisker
        box['whiskers'][2 * i].set_linewidth(2)
        box['whiskers'][2 * i + 1].set_color(color)  # Upper whisker
        box['whiskers'][2 * i + 1].set_linewidth(2)

        # Cap coloring
        box['caps'][2 * i].set_color(color)  # Lower cap
        box['caps'][2 * i].set_linewidth(2)
        box['caps'][2 * i + 1].set_color(color)  # Upper cap
        box['caps'][2 * i + 1].set_linewidth(2)

        # Flier (outlier) coloring
        box['fliers'][i].set_markerfacecolor(color)
        box['fliers'][i].set_markeredgecolor(color)

    # Annotate significance if p-value < 0.05
    p_value = stats.mannwhitneyu(cluster1_data, cluster2_data).pvalue
    if p_value < 0.05:
        y_max = max(max(cluster1_data), max(cluster2_data))
        ax.plot([1, 2], [y_max + 5, y_max + 5], lw=1.5, color="black")  # Line above boxes
        ax.text(1.5, y_max + 7, "***", ha="center", va="bottom", color="black", fontsize=14, fontweight='bold')

    # Set title and labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels([cluster1, cluster2], fontsize=12, fontweight='bold')

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust y-ticks font weight
    ax.tick_params(axis="y", labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    if idx == 0:  # Only the first subplot shows the y-axis label
        ax.set_ylabel("Firing Rate (Hz)", fontsize=12, fontweight='bold')

# Adjust layout and spacing
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)  # Add space between subplots
plt.show()


# Define colors for each cluster
color_map = {
    'Cluster 1': 'blue',
    'Cluster 2': 'red',
    'Cluster 3': 'green',
    'Cluster 4': 'cyan'
}

# Define pairwise comparisons
pairwise_comparisons = [
    ('Cluster 1', 'Cluster 2'),
    ('Cluster 1', 'Cluster 3'),
    ('Cluster 1', 'Cluster 4'),
    ('Cluster 2', 'Cluster 3'),
    ('Cluster 2', 'Cluster 4'),
    ('Cluster 3', 'Cluster 4')
]

# Create a figure with 2 rows and 3 columns of subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)

# Flatten the axes array for easier iteration
axes = axes.flatten()

for idx, (cluster1, cluster2) in enumerate(pairwise_comparisons):
    ax = axes[idx]  # Current subplot

    # Extract data for the two clusters being compared
    cluster1_data = df_kmeans_all[df_kmeans_all['cluster_name'] == cluster1]['firing']
    cluster2_data = df_kmeans_all[df_kmeans_all['cluster_name'] == cluster2]['firing']
    data = [cluster1_data, cluster2_data]

    # Create the boxplot
    box = ax.boxplot(
        data,
        patch_artist=True,  # Enables coloring of the boxes
        widths=0.6,  # Box width
        showmeans=True,  # Show the mean
        meanline=True,  # Mean is shown as a line
        meanprops={"color": "black", "linestyle": "--", "linewidth": 1.5},
        whiskerprops={"linewidth": 2},
        capprops={"linewidth": 2},
        boxprops={"linewidth": 2},
        medianprops={"color": "black", "linewidth": 2},
    )

    # Assign colors to boxes, whiskers, caps, and fliers
    cluster_colors = [color_map[cluster1], color_map[cluster2]]

    for i, color in enumerate(cluster_colors):
        # Color the boxes
        box['boxes'][i].set_facecolor(color)
        box['boxes'][i].set_edgecolor(color)

        # Color the whiskers
        box['whiskers'][2 * i].set_color(color)  # Lower whisker
        box['whiskers'][2 * i + 1].set_color(color)  # Upper whisker

        # Color the caps
        box['caps'][2 * i].set_color(color)  # Lower cap
        box['caps'][2 * i + 1].set_color(color)  # Upper cap

        # Color the fliers (outliers)
        if box['fliers']:
            box['fliers'][i].set_markerfacecolor(color)
            box['fliers'][i].set_markeredgecolor(color)

    # Annotate significance
    kruskal_stat, p_value = stats.mannwhitneyu(cluster1_data, cluster2_data)
    if p_value < 0.05:
        max_y = max(np.max(cluster1_data), np.max(cluster2_data))
        ax.plot([1, 2], [max_y + 5, max_y + 5], color="black", linewidth=1.5)  # Line between boxes
        ax.text(1.5, max_y + 10, "***", ha="center", fontsize=14, color="black", fontweight='bold')  # Asterisk for significance

    # Set subplot title and labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels([cluster1, cluster2], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)  # Adjust size
    for label in ax.get_yticklabels():
       label.set_fontweight('bold')
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if idx % 3 == 0:  # Only the first column shows the y-axis label
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')

# Adjust layout and spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space between rows and columns
plt.show()


#%% distribution of firing rates 4.0 (boxplots) - Figure 5

# Define colors for each cluster
color_map = {
    'Cluster 1': 'blue',
    'Cluster 2': 'red',
    'Cluster 3': 'green',
    'Cluster 4': 'cyan'
}

# Define pairwise comparisons
pairwise_comparisons = [
    ('Cluster 1', 'Cluster 2'),
    ('Cluster 1', 'Cluster 3'),
    ('Cluster 1', 'Cluster 4'),
    ('Cluster 2', 'Cluster 3'),
    ('Cluster 2', 'Cluster 4'),
    ('Cluster 3', 'Cluster 4')
]

# Create data for all clusters
clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
data = [df_kmeans_all[df_kmeans_all['cluster_name'] == cluster]['firing'] for cluster in clusters]

# Create a figure
plt.figure(figsize=(10, 6))

# Create a single boxplot for all clusters
box = plt.boxplot(
    data,
    patch_artist=True,
    widths=0.6,
    showmeans=True,
    meanline=True,
    meanprops={"color": "black", "linestyle": "--", "linewidth": 2},
    whiskerprops={"linewidth": 2},
    capprops={"linewidth": 2},
    medianprops={"color": "black", "linewidth": 2}
)

# Color the boxes, whiskers, and caps according to cluster
for i, patch in enumerate(box['boxes']):
    # Boxes
    patch.set_facecolor(color_map[clusters[i]])
    patch.set_edgecolor(color_map[clusters[i]])

    # Whiskers
    box['whiskers'][2 * i].set_color(color_map[clusters[i]])  # Lower whisker
    box['whiskers'][2 * i + 1].set_color(color_map[clusters[i]])  # Upper whisker
    box['whiskers'][2 * i].set_linewidth(2)
    box['whiskers'][2 * i + 1].set_linewidth(2)

    # Caps
    box['caps'][2 * i].set_color(color_map[clusters[i]])  # Lower cap
    box['caps'][2 * i + 1].set_color(color_map[clusters[i]])  # Upper cap
    box['caps'][2 * i].set_linewidth(2)
    box['caps'][2 * i + 1].set_linewidth(2)
    
    for i, flier in enumerate(box['fliers']):
        flier.set_markerfacecolor(color_map[clusters[i]])
        flier.set_markeredgecolor(color_map[clusters[i]])
        flier.set_markersize(8)

# Annotate only significant comparisons with connecting lines and asterisks
y_max = max([max(d) for d in data])  # Find the overall max y-value
line_offset = 7  # Vertical offset for each comparison line
asterisk_offset = -2  # Slightly above the line

for i, (cluster1, cluster2) in enumerate(pairwise_comparisons):
    idx1 = clusters.index(cluster1) + 1  # Boxplot index for cluster1
    idx2 = clusters.index(cluster2) + 1  # Boxplot index for cluster2

    # Retrieve the Bonferroni-corrected p-value
    corrected_p_value = results.get((cluster1, cluster2), None)

    # Skip non-significant comparisons
    if corrected_p_value is not None and corrected_p_value < 0.05:
        # Determine the y-position for the line
        y = y_max + (i + 1) * line_offset

        # Draw the connecting line
        plt.plot([idx1, idx2], [y, y], lw=1.5, color="black")  # Horizontal line

        # Annotate the asterisk directly above the line
        plt.text((idx1 + idx2) / 2, y + asterisk_offset, "***", ha="center", va="bottom", color="black", fontsize=14, fontweight='bold')

# Set x-axis labels
plt.xticks(ticks=np.arange(1, len(clusters) + 1), labels=clusters, fontsize=20, fontweight='bold')

# Set y-axis label
plt.ylabel("Firing Rate (Hz)", fontsize=20, fontweight="bold")

plt.gca().tick_params(axis='y', which='major', labelsize=20)
for label in plt.gca().get_yticklabels():
    label.set_fontweight('bold')
    
# Remove top and right spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Adjust layout
plt.tight_layout()
plt.show()

clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

print("Firing Rates:")
for cluster in clusters:
    data = df_kmeans_all[df_kmeans_all['cluster_name'] == cluster]['firing']
    cluster_mean = data.mean()
    cluster_std = data.std()
    print(f"{cluster}: mean = {cluster_mean:.2f}, std = {cluster_std:.2f}")
    
    

#%% Distribution of electrophysiological measures - Figure 4 & 5

import matplotlib.pyplot as plt
import seaborn as sns

# Define color palette for clusters
color_palette = ['blue', 'red', 'green', 'cyan']
clusters = [f'Cluster {i+1}' for i in range(len(color_palette))]

# Create custom legend handles
handles = [plt.Line2D([0], [0], color=color_palette[i], lw=4) for i in range(len(color_palette))]

electrophysiological_measures = ['mean_isi', 'burst_index', 'fano_factor', 'lv', 'higher_order_lv', 'renyi_entropy']

# Determine the overall range for each electrophysiological measure
measure_ranges = {measure: (df_summary[measure].min(), df_summary[measure].max()) for measure in electrophysiological_measures}

# Plot each electrophysiological measure using polygon histograms
for measure in electrophysiological_measures:
    plt.figure(figsize=(10, 6))

    # Loop through each cluster and plot its distribution with polygons
    for i, cluster in enumerate(clusters):
        cluster_data = df_summary[df_summary['cluster_name'] == cluster][measure]
        
        sns.histplot(
            cluster_data,
            element='poly',  # Use polygons instead of bars
            stat='count',  # Display counts rather than density
            bins=10,  # Adjust bins as needed
            alpha=0.3,  # Set transparency
            color=color_palette[i],  # Use color from palette
            label=cluster
        )

    title = measure.replace('_', ' ').capitalize()
    plt.title(f'{title} Distribution', fontsize=14, fontweight='bold')
    plt.xlabel(measure.capitalize(), fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.xlim(measure_ranges[measure])  # Set x-axis limits based on the min and max for each measure
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add custom legend to match second code block
    plt.legend(handles, clusters, fontsize=10, title_fontsize='12', loc='upper right', frameon=True)
    
    plt.tight_layout()
    
    # Save each plot
    #plt.savefig(f"D:/nigraPhys/figures/Nov_2/final_HQ_figures/{measure}_distribution.png")
    plt.show()
    #plt.close()

# Calculate Kruskal-Wallis test for each measure
kruskal_results = {}

for measure in electrophysiological_measures:
    kruskal_statistic, kruskal_pvalue = stats.kruskal(*[df_summary[df_summary['cluster_name'] == f'Cluster {i}'][measure] for i in range(1, 5)])
    kruskal_results[measure] = {'Kruskal-Wallis Test Statistic': kruskal_statistic, 'p-value': kruskal_pvalue}

    print(f"Kruskal-Wallis Test for {measure}:")
    print("Statistic:", kruskal_statistic)
    print("p-value:", kruskal_pvalue)
    print()

# Perform pairwise Mann-Whitney U tests with Bonferroni correction as post hoc test
for measure in electrophysiological_measures:
    pairwise_comparisons = list(itertools.combinations(range(1, 5), 2))
    results = {}

    for cluster1, cluster2 in pairwise_comparisons:
        test_statistic, p_value = stats.mannwhitneyu(df_summary[df_summary['cluster_name'] == f'Cluster {cluster1}'][measure],
                                                      df_summary[df_summary['cluster_name'] == f'Cluster {cluster2}'][measure],
                                                      alternative='two-sided')
        results[(f'Cluster {cluster1}', f'Cluster {cluster2}')] = {
            'Test Statistic': test_statistic,
            'P-value': p_value * len(pairwise_comparisons)  # Bonferroni correction
        }

    posthoc_results = pd.DataFrame(results).T

    # Plotting the heatmap for post hoc results
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(posthoc_results['P-value'].unstack().T, cmap='coolwarm', annot=True, fmt='.2g', cbar_kws={'label': 'P-value'})

    # Customize x-axis and y-axis tick font sizes
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12, fontweight='bold')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12, fontweight='bold')

    # Customize color bar label font size and font weight
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6)  # Set color bar label size and style
    cbar.ax.set_title(cbar.ax.get_title(), fontsize=12, fontweight='bold')  # Set color bar title font size and font weight

    # Loop through text objects in the heatmap and set them to bold
    for text in heatmap.texts:
        text.set_fontweight('bold')
        text.set_color('black')

    title = measure.replace('_', ' ').capitalize()
    plt.title(f'Post Hoc Analysis for {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    #plt.savefig(f"D:/nigraPhys/figures/Nov_2/final_HQ_figures/{measure}_posthoc_heatmap.png")  # Save heatmap plot
    plt.show()
    #plt.close()    
    


#%% distribution of electrophysiological measures and waveform features (boxplots) - Figure 4 & 5

import matplotlib.pyplot as plt
import numpy as np
import itertools

# Define colors for each cluster
color_map = {
    'Cluster 1': 'blue',
    'Cluster 2': 'red',
    'Cluster 3': 'green',
    'Cluster 4': 'cyan'
}

# Define measures
measures = ['mean_isi', 'burst_index', 'fano_factor', 'lv', 'higher_order_lv', 
            'renyi_entropy', 'peak_to_peak_duration', 'hyperpolarization_time',
            'peak_to_trough_amplitude']

# Define custom y-axis labels for specific measures
custom_labels = {
    'mean_isi': 'Mean Interspike Interval',
    'lv': 'Local Variability',
    'higher_order_lv': 'Higher Order Local Variability',
    'peak_to_peak_duration': 'Peak to Peak Duration (ms)',
    'hyperpolarization_time': 'Hyperpolarization Time (ms)'
}

# Loop through each measure
for measure in measures:
    clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    data = [df_summary[df_summary['cluster_name'] == cluster][measure] for cluster in clusters]

    # Perform pairwise Mann-Whitney U tests with Bonferroni correction
    pairwise_comparisons = list(itertools.combinations(range(1, 5), 2))
    results = {}
    for cluster1, cluster2 in pairwise_comparisons:
        test_statistic, p_value = mannwhitneyu(
            df_summary[df_summary['cluster_name'] == f'Cluster {cluster1}'][measure],
            df_summary[df_summary['cluster_name'] == f'Cluster {cluster2}'][measure],
            alternative='two-sided'
        )
        corrected_p_value = p_value * len(pairwise_comparisons)
        results[(f'Cluster {cluster1}', f'Cluster {cluster2}')] = corrected_p_value

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linestyle": "--", "linewidth": 2},
        whiskerprops={"linewidth": 2},
        capprops={"linewidth": 2},
        medianprops={"color": "black", "linewidth": 2}
    )

    # Color the boxes, whiskers, caps, and outliers according to cluster
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(color_map[clusters[i]])
        patch.set_edgecolor(color_map[clusters[i]])
        box['whiskers'][2 * i].set_color(color_map[clusters[i]])
        box['whiskers'][2 * i + 1].set_color(color_map[clusters[i]])
        box['caps'][2 * i].set_color(color_map[clusters[i]])
        box['caps'][2 * i + 1].set_color(color_map[clusters[i]])
        box['fliers'][i].set_markerfacecolor(color_map[clusters[i]])
        box['fliers'][i].set_markeredgecolor(color_map[clusters[i]])

    # Annotate significant comparisons and strong trends
    y_min = min([min(d) for d in data])
    y_max = max([max(d) for d in data])
    y_range = y_max - y_min

    line_offset = y_range * 0.05
    asterisk_offset = line_offset * -0.25
    dagger_offset = line_offset * 0.25  # Separate offset for the dagger symbol

    for i, (cluster1, cluster2) in enumerate(pairwise_comparisons):
        idx1 = cluster1
        idx2 = cluster2
        corrected_p_value = results.get((f'Cluster {cluster1}', f'Cluster {cluster2}'), None)

        # Skip if not significant or not a strong trend
        if corrected_p_value is None or corrected_p_value >= 0.1:
            continue

        # Determine the symbol and offset for significance or trend
        if corrected_p_value < 0.05:
            symbol = "***"
            offset = asterisk_offset
        elif corrected_p_value < 0.1:
            symbol = "†"
            offset = dagger_offset
        else:
            continue

        # Determine the y-position for the line and plot it
        y = y_max + (i + 1) * line_offset
        plt.plot([idx1, idx2], [y, y], lw=1.5, color="black")

        # Place the symbol above the line
        plt.text((idx1 + idx2) / 2, y + offset, symbol, ha="center", va="bottom", color="black", fontsize=14, fontweight='bold')

    # Set title and labels
    plt.xticks(ticks=np.arange(1, len(clusters) + 1), labels=clusters, fontsize=20, fontweight='bold')

    # Use custom labels if available, otherwise use the default measure name
    ylabel = custom_labels.get(measure, measure.replace("_", " ").capitalize())
    plt.ylabel(ylabel, fontsize=20, fontweight="bold")

    # Make y-ticks bold
    plt.gca().tick_params(axis='y', which='major', labelsize=20)
    for label in plt.gca().get_yticklabels():
        label.set_fontweight('bold')

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

for measure in measures:
    print(f"Measure: {measure}")
    for cluster in clusters:
        data = df_summary[df_summary['cluster_name'] == cluster][measure]
        cluster_mean = data.mean()
        cluster_std = data.std()
        print(f"{cluster}: mean = {cluster_mean:.2f}, std = {cluster_std:.2f}")
    print() 


#%% randomly plot sample from each cluster - Figure 5

# Select a random neuron from each cluster
random_neurons = df_summary.groupby('cluster_name').apply(lambda group: group.sample(n=1)).reset_index(drop=True)

# Define the time window for the 5-second segment
time_window = 5  # seconds

# Define the figure size and spacing
fig, axes = plt.subplots(nrows=len(random_neurons), figsize=(12, 6 * len(random_neurons)))
plt.subplots_adjust(hspace=0.5)

# Plot raster plots for each neuron
for ax, row in zip(axes, random_neurons.iterrows()):
    row = row[1]  # Get the actual row data
    cluster = row['cluster_name']
    neuron_id = row['cell_id']

    neuron_data = df_sorted[df_sorted['cell_id'] == neuron_id]
    spike_times = neuron_data['time_p']  # Assuming you have a 'time' column indicating spike times

    # Create a raster plot
    ax.scatter(spike_times, [0] * len(spike_times), marker='|', color='black')

    ax.set_title(f"Cluster {cluster} - Neuron {neuron_id} Raster Plot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron")

    # Set the x-axis limit for the 5-second segment
    ax.set_xlim(0, time_window)

    # Annotate with measured features
    features = ['mean_isi', 'burst_index', 'fano_factor', 'lv', 'higher_order_lv', 'renyi_entropy']
    
    annotation_text = "\n".join([f"{feat}: {row[feat]:.2f}" for feat in features])

    ax.annotate(annotation_text, xy=(1.01, 0.5), xycoords="axes fraction", ha="left", va="center", fontsize=10)

plt.show()


#%% waveforms shapes - Figure 4

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import stats
import pandas as pd

df4['time_s'] = df4.apply(lambda x: np.arange(0,41)/22,axis =1)
df4_exp = df4.explode(['resampled_wave','z_wave','z_wave_s','time_s'])

# Define the desired colors for each category
colors = {'Cluster 1': 'blue', 'Cluster 2': 'red', 'Cluster 3': 'green', 'Cluster 4': 'cyan'}

# Create the line plot with specified colors
fig = plt.figure()
sns.lineplot(x="time_s", y="z_wave_s", hue='cluster_name', data=df4_exp, palette=colors)

# Show the plot
plt.show()


# List of waveform features to analyze
waveform_features = ['peak_to_peak_duration', 'peak_to_trough_amplitude', 'hyperpolarization_time', 'half_width_time']

colors = {'Cluster 1': 'blue', 'Cluster 2': 'red', 'Cluster 3': 'green', 'Cluster 4': 'cyan'}

# Plot histogram with polygon elements for each waveform feature
for feature in waveform_features:
    plt.figure(figsize=(10, 6))

    for cluster_num in range(1, 5):  # Assuming there are 4 clusters
        cluster_data = df4[df4['cluster_name'] == f'Cluster {cluster_num}'][feature]
        color = colors[f'Cluster {cluster_num}']

        sns.histplot(
            cluster_data,
            bins=5,  # Adjust bins as needed
            stat='density',
            element='poly',  # Use polygon elements
            fill=True,
            color=color,
            alpha=0.3,
            label=f'Cluster {cluster_num}'  # Seaborn adds this to the legend
        )

    # Customize plot
    font_labels = {'fontsize': 12, 'fontweight': 'bold'}
    title = feature.replace('_', ' ').capitalize()  # Remove underscores from the title
    x_label = f'{title} (ms)' if feature != 'peak_to_trough_amplitude' else title  # Add unit conditionally
    plt.title(f'{title} Distribution', fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontdict=font_labels)  # Set x-axis label
    plt.ylabel('Density', fontdict=font_labels)

    # Manually create the legend with correct colors
    handles = [plt.Line2D([0], [0], color=colors[f'Cluster {i}'], lw=4) for i in range(1, 5)]
    legend_loc = 'upper left' if feature == 'peak_to_trough_amplitude' else 'upper right'  # Set legend position
    plt.legend(
        handles, 
        [f'Cluster {i}' for i in range(1, 5)], 
        title='Clusters', 
        fontsize=10, 
        title_fontsize=12, 
        loc=legend_loc,  # Adjust legend location dynamically
        frameon=True  # Optional: Add a box around the legend
    )

    plt.tight_layout()
    plt.show()
    
# Perform Kruskal-Wallis tests
kruskal_results = {}

for feature in waveform_features:
    kruskal_statistic, kruskal_pvalue = stats.kruskal(
        *[df4[df4['cluster_name'] == f'Cluster {i}'][feature] for i in range(1, 5)]
    )
    kruskal_results[feature] = {'Kruskal-Wallis Test Statistic': kruskal_statistic, 'p-value': kruskal_pvalue}

    print(f"Kruskal-Wallis Test for {feature}:")
    print("Statistic:", kruskal_statistic)
    print("p-value:", kruskal_pvalue)
    print()

    
# Perform pairwise Mann-Whitney U tests with Bonferroni correction
for feature in waveform_features:
    pairwise_comparisons = list(itertools.combinations(range(1, 5), 2))
    results = {}

    for cluster1, cluster2 in pairwise_comparisons:
        test_statistic, p_value = stats.mannwhitneyu(
            df4[df4['cluster_name'] == f'Cluster {cluster1}'][feature],
            df4[df4['cluster_name'] == f'Cluster {cluster2}'][feature],
            alternative='two-sided'
        )
        results[(f'Cluster {cluster1}', f'Cluster {cluster2}')] = {
            'Test Statistic': test_statistic,
            'P-value': p_value * len(pairwise_comparisons)  # Bonferroni correction
        }

    posthoc_results = pd.DataFrame(results).T
    
    posthoc_results = pd.DataFrame(results).T
    print(f"Post-hoc results for {feature}:")
    print(posthoc_results)

    # Extract P-values for heatmap
    p_values = posthoc_results['P-value'].unstack().T

    # Use TwoSlopeNorm to emphasize significant values
    norm = TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=0.1)  # Center around 0.05

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(
        p_values,
        cmap='coolwarm',
        annot=True,
        fmt='.2g',
        cbar_kws={'label': 'P-value'},
        norm=norm  # Use custom normalization
    )

    # Customize axis ticks
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12, fontweight='bold')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12, fontweight='bold')

    # Customize color bar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, width=2, length=6)
    cbar.ax.set_title('P-value', fontsize=12, fontweight='bold')

    # Set bold and black annotations
    for text in heatmap.texts:
        text.set_fontweight('bold')
        text.set_color('black')

    # Add title and save plot
    title = feature.replace('_', ' ').capitalize()
    plt.title(f'Post Hoc Analysis for {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
#%% waveform shapes 2.0 (one figure)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Define colors for each cluster
colors = {'Cluster 1': 'blue', 'Cluster 2': 'red', 'Cluster 3': 'green', 'Cluster 4': 'cyan'}
clusters = sorted(df4['cluster_name'].unique())

# Create a figure with two rows: top for individual clusters, bottom for overlaid waveform
fig = plt.figure(figsize=(12.5, 8))

# Use GridSpec to manually control the position of the subplots
gs = gridspec.GridSpec(2, len(clusters))  # 2 rows, len(clusters) columns

# Top Panel: Individual waveforms for each cluster in separate subplots arranged horizontally
for i, cluster in enumerate(clusters):
    ax = fig.add_subplot(gs[0, i])  # Creates subplots in the top row
    sns.lineplot(x="time_s", y="z_wave_s", data=df4_exp[df4_exp['cluster_name'] == cluster], 
                 color=colors[cluster], ax=ax)
    ax.set_title(f"{cluster} Waveform", fontsize=10, weight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

# Bottom Panel: Overlaid waveforms for all clusters (centered between the 2nd and 3rd columns)
# Place the overlaid plot in the center (between the 2nd and 3rd columns in the bottom row)
ax_overlay = fig.add_subplot(gs[1, 1:3])  # This spans the 2nd and 3rd columns in the second row
for cluster in clusters:
    sns.lineplot(x="time_s", y="z_wave_s", data=df4_exp[df4_exp['cluster_name'] == cluster], 
                 color=colors[cluster], label=cluster, ax=ax_overlay)
ax_overlay.set_title("All Clusters Overlaid", fontsize=10, weight='bold')
ax_overlay.set_xlabel("Time (s)")
ax_overlay.set_ylabel("Amplitude")
ax_overlay.legend(title='Cluster', fontsize=8)

# Adjust layout to prevent overlap and ensure everything fits
plt.tight_layout()
plt.show()

#%% waveform shapes 3.0 (separate figures) - Figure 4

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define colors for each cluster and fixed x-values for the vertical lines
colors = {'Cluster 1': 'blue', 'Cluster 2': 'red', 'Cluster 3': 'green', 'Cluster 4': 'cyan'}
clusters = sorted(df4['cluster_name'].unique())
vertical_lines = {'Cluster 1': 1.3635, 'Cluster 2': 0.90909, 'Cluster 3': 0.90909, 'Cluster 4': 1.0909}

# ---- Figure 1: Individual waveforms for each cluster ---- #
fig1 = plt.figure(figsize=(12, 6))  # Create a new figure for individual plots

for i, cluster in enumerate(clusters):
    ax = fig1.add_subplot(1, len(clusters), i + 1)  # Creates subplots in a single row
    sns.lineplot(
        x="time_s", y="z_wave_s", 
        data=df4_exp[df4_exp['cluster_name'] == cluster], 
        color=colors[cluster], ax=ax
    )
    ax.set_xlabel("Time (ms)", weight='bold',fontsize=20)
    ax.set_ylabel("Amplitude", weight='bold',fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Draw the vertical line
    ax.axvline(x=vertical_lines[cluster], color='grey', linestyle='--', linewidth=3)

plt.tight_layout()
plt.show() 

# ---- Figure 2: Overlaid waveforms for all clusters ---- #
fig2 = plt.figure(figsize=(12, 6))  # Create a new figure for the overlaid plot

ax_overlay = fig2.add_subplot(111)  # Single axis for overlaid plot

for cluster in clusters:
    sns.lineplot(
        x="time_s", y="z_wave_s", 
        data=df4_exp[df4_exp['cluster_name'] == cluster], 
        color=colors[cluster], ax=ax_overlay
    )

# Set titles and labels
ax_overlay.set_xlabel("Time (ms)", weight='bold', fontsize=20)
ax_overlay.set_ylabel("Amplitude", weight='bold', fontsize=20)

ax_overlay.tick_params(axis='both', which='major', labelsize=20)

# Make tick labels bold
for label in ax_overlay.get_xticklabels() + ax_overlay.get_yticklabels():
    label.set_fontweight('bold')

# Remove the top and right spines
ax_overlay.spines['top'].set_visible(False)
ax_overlay.spines['right'].set_visible(False)

plt.tight_layout()
plt.show() 


#%% distribution of recording depths in 3d space (new)

# Extract x, y, z columns from the DataFrame
x_data = lfp_df['x']
y_data = lfp_df['y']
z_data = lfp_df['z']

# Set up the figure and axes
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Initialize arrays to store histogram values and bins
histograms = []
bins = []

# Plot histograms for x, y, z and compute histograms
for i, (data, color, label) in enumerate(zip([x_data, y_data, z_data], ['blue', 'green', 'red'], ['x', 'y', 'z'])):
    ax = axes[i]
    hist, bin_edges, _ = ax.hist(data, bins=20, density=False, color=color, alpha=0.7, label=label)
    ax.set_title(f'Histogram of {label}')
    ax.set_xlabel(f'{label} values')
    ax.set_ylabel('Frequency')
    ax.legend()
    histograms.append(hist)
    bins.append(bin_edges)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Print histograms and bins
for i, label in enumerate(['x', 'y', 'z']):
    print(f"Histogram of {label}:")
    print("Histogram:", histograms[i])
    print("Bins:", bins[i])
    print("Size of bins:", np.diff(bins[i]))
    print()


# Extract x, y, z columns from the DataFrame
x_data = lfp_df['x']
y_data = lfp_df['y']
z_data = lfp_df['z']

# Set up the figure and axes
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Initialize arrays to store histogram values and bins
histograms = []
bins = []

# Define bin edges based on min and max values of data with bin size 1
for i, (data, color, label) in enumerate(zip([x_data, y_data, z_data], ['blue', 'green', 'red'], ['x', 'y', 'z'])):
    min_val = int(np.floor(data.min()))
    max_val = int(np.ceil(data.max()))
    bin_edges = np.arange(min_val, max_val + 1, 1)  # Ensure bin size is 1
    hist, _, _ = axes[i].hist(data, bins=bin_edges, density=False, color=color, alpha=0.7, label=label)
    axes[i].set_title(f'Histogram of {label}')
    axes[i].set_xlabel(f'{label} values')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    histograms.append(hist)
    bins.append(bin_edges)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

histograms = np.array(histograms)
bins = np.array(bins)

# Print histograms and bins
for i, label in enumerate(['x', 'y', 'z']):
    print(f"Histogram of {label}:")
    print("Histogram:", histograms[i])
    print("Bins:", bins[i])
    print("Size of bins:", np.diff(bins[i]))
    print()
    
 
normalized_cols = [col for col in df_kmeans_all.columns if 'normalized_count' in col]
df_kmeans_all.drop(columns=normalized_cols, inplace=True)


#%% plot dist of each cluster per recording bin

# In case of missing coordinates
#df_kmeans_all = df_kmeans_all.dropna(subset=['x', 'y', 'z'])

# Bin data and assign labels for each dimension
for i, label in enumerate(['x', 'y', 'z']):
    df_kmeans_all[f'{label}_bin'] = pd.cut(df_kmeans_all[label], bins=bins[i], labels=False)

# Create cluster name and color mappings
cluster_name_mapping = df_kmeans_all.set_index('clustering_kmeans')['cluster_name'].to_dict()
cluster_color_mapping = df_kmeans_all.set_index('clustering_kmeans')['cluster_color'].to_dict()

# Group data and compute counts for each dimension (X, Y, Z)
grouped_dimensions = {}

# Define all possible combinations for x dimension
all_x_bins = df_kmeans_all['x_bin'].unique()
all_clusters = df_kmeans_all['cluster_name'].unique()
all_x_combinations = [(x, cluster) for x in all_x_bins for cluster in all_clusters]

# Group data and compute counts for x dimension
grouped_x = df_kmeans_all.groupby(['x_bin', 'cluster_name']).size().reset_index(name='cell_count')

# Reindex to fill in missing combinations for x dimension
grouped_x = grouped_x.set_index(['x_bin', 'cluster_name']).reindex(all_x_combinations, fill_value=0).reset_index()
grouped_x.sort_values(by='x_bin', inplace=True)
grouped_x.reset_index(drop=True, inplace=True)

grouped_x.to_csv('grouped_data_x.csv', index=False)

# Print the results for x dimension
print("Grouped Data for Dimension: x")
print(grouped_x)
print("\n")

# Define all possible combinations for y dimension
all_y_bins = df_kmeans_all['y_bin'].unique()
all_y_combinations = [(y, cluster) for y in all_y_bins for cluster in all_clusters]

# Group data and compute counts for y dimension
grouped_y = df_kmeans_all.groupby(['y_bin', 'cluster_name']).size().reset_index(name='cell_count')

# Reindex to fill in missing combinations for y dimension
grouped_y = grouped_y.set_index(['y_bin', 'cluster_name']).reindex(all_y_combinations, fill_value=0).reset_index()
grouped_y.sort_values(by='y_bin', inplace=True)
grouped_y.reset_index(drop=True, inplace=True)

grouped_y.to_csv('grouped_data_y.csv', index=False)

# Print the results for y dimension
print("Grouped Data for Dimension: y")
print(grouped_y)
print("\n")

# Define all possible combinations for z dimension
all_z_bins = df_kmeans_all['z_bin'].unique()
all_z_combinations = [(z, cluster) for z in all_z_bins for cluster in all_clusters]

# Group data and compute counts for z dimension
grouped_z = df_kmeans_all.groupby(['z_bin', 'cluster_name']).size().reset_index(name='cell_count')

# Reindex to fill in missing combinations for z dimension
grouped_z = grouped_z.set_index(['z_bin', 'cluster_name']).reindex(all_z_combinations, fill_value=0).reset_index()
grouped_z.sort_values(by='z_bin', inplace=True)
grouped_z.reset_index(drop=True, inplace=True)

grouped_z.to_csv('grouped_data_z.csv', index=False)

# Print the results for z dimension
print("Grouped Data for Dimension: z")
print(grouped_z)
print("\n")

def bin_to_coordinate(bin_edges, bin_numbers):
    return [bin_edges[int(bin_number)] for bin_number in bin_numbers]

for dimension, grouped_data in grouped_dimensions.items():
    grouped_data[f'{dimension}_coordinate'] = bin_to_coordinate(bin_edges, grouped_data[f'{dimension}_bin'])

for i, label in enumerate(['x', 'y', 'z']):
    lfp_df[f'{label}_bin'] = pd.cut(lfp_df[label], bins=bins[i], labels=False)

total_counts = {}

total_counts_x = lfp_df.groupby(['x_bin']).size().reset_index(name='total_count_x')
total_counts['x'] = total_counts_x

total_counts_y = lfp_df.groupby(['y_bin']).size().reset_index(name='total_count_y')
total_counts['y'] = total_counts_y

total_counts_z = lfp_df.groupby(['z_bin']).size().reset_index(name='total_count_z')
total_counts['z'] = total_counts_z

grouped_x = pd.merge(grouped_x, total_counts_x, on='x_bin')
grouped_y = pd.merge(grouped_y, total_counts_y, on='y_bin')
grouped_z = pd.merge(grouped_z, total_counts_z, on='z_bin')

grouped_x['normalized_count'] = grouped_x['cell_count'] / grouped_x['total_count_x']
grouped_y['normalized_count'] = grouped_y['cell_count'] / grouped_y['total_count_y']
grouped_z['normalized_count'] = grouped_z['cell_count'] / grouped_z['total_count_z']

# Printing grouped data with normalized counts for X dimension
print("Grouped Data for Dimension: X with Normalized Count")
print(grouped_x)
print("\n")

# Printing grouped data with normalized counts for Y dimension
print("Grouped Data for Dimension: Y with Normalized Count")
print(grouped_y)
print("\n")

# Printing grouped data with normalized counts for Z dimension
print("Grouped Data for Dimension: Z with Normalized Count")
print(grouped_z)
print("\n")

cluster_color_mapping = {
    'Cluster 1': 'blue',
    'Cluster 2': 'red',
    'Cluster 3': 'green',
    'Cluster 4': 'cyan'
}
# Define bin edges for each dimension (X, Y, Z)
bin_edges_x = bins[0]  
bin_edges_y = bins[1] 
bin_edges_z = bins[2] 

# Define a function to plot line plots for each dimension
def plot_dimension_lines(grouped_data, dimension, bin_edges):
    # Convert bin numbers to actual coordinates
    grouped_data[f'{dimension}_coordinate'] = bin_to_coordinate(bin_edges, grouped_data[f'{dimension}_bin'])

    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    sns.lineplot(data=grouped_data, x=f'{dimension}_coordinate', y='normalized_count', hue='cluster_name', palette=cluster_color_mapping, ci=65)
    plt.title(f'Cluster Distribution in Dimension: {dimension}')
    plt.xlabel(f'{dimension} Coordinates')
    plt.ylabel('Normalized Count')
    plt.legend(title='Cluster')
    plt.show()

# Plot lines for each dimension
for plane, grouped_plane, bin_edges in zip(['x', 'y', 'z'], [grouped_x, grouped_y, grouped_z], [bin_edges_x, bin_edges_y, bin_edges_z]):
    plot_dimension_lines(grouped_plane, plane, bin_edges)
    

#%% weighted centroid and 3D dist of each cluster 

# Correctly access bin edges from the bins array for each dimension
bin_edges_x = bins[0]
bin_edges_y = bins[1]
bin_edges_z = bins[2]

# Calculate midpoints for each dimension
mid_points_x = bin_edges_x[:-1] + np.diff(bin_edges_x) / 2
mid_points_y = bin_edges_y[:-1] + np.diff(bin_edges_y) / 2
mid_points_z = bin_edges_z[:-1] + np.diff(bin_edges_z) / 2

# Map bin numbers to midpoints
bin_to_midpoint_map_x = dict(zip(range(len(mid_points_x)), mid_points_x))
bin_to_midpoint_map_y = dict(zip(range(len(mid_points_y)), mid_points_y))
bin_to_midpoint_map_z = dict(zip(range(len(mid_points_z)), mid_points_z))

# Assign midpoints to each bin in the DataFrames
grouped_x['x_midpoint'] = grouped_x['x_bin'].map(bin_to_midpoint_map_x)
grouped_y['y_midpoint'] = grouped_y['y_bin'].map(bin_to_midpoint_map_y)
grouped_z['z_midpoint'] = grouped_z['z_bin'].map(bin_to_midpoint_map_z)

# Calculate the weighted centroid for each cluster
weighted_centroid_x = grouped_x.groupby('cluster_name').apply(
    lambda df: np.sum(df['x_midpoint'] * df['normalized_count']) / np.sum(df['normalized_count'])
)
weighted_centroid_y = grouped_y.groupby('cluster_name').apply(
    lambda df: np.sum(df['y_midpoint'] * df['normalized_count']) / np.sum(df['normalized_count'])
)
weighted_centroid_z = grouped_z.groupby('cluster_name').apply(
    lambda df: np.sum(df['z_midpoint'] * df['normalized_count']) / np.sum(df['normalized_count'])
)

# Combine centroid coordinates into a single DataFrame
centroids = pd.DataFrame({
    'centroid_x': weighted_centroid_x,
    'centroid_y': weighted_centroid_y,
    'centroid_z': weighted_centroid_z
}).reset_index()

centroids.columns = ['cluster_name', 'centroid_x', 'centroid_y', 'centroid_z']
print(centroids)

color_order = ['blue', 'red', 'green', 'cyan']

# Filter data for each cluster
cluster_data = {
    'Cluster 1': df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 1'],
    'Cluster 2': df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 2'],
    'Cluster 3': df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 3'],
    'Cluster 4': df_kmeans_all[df_kmeans_all['cluster_name'] == 'Cluster 4']
}

# Function to calculate Mahalanobis distance between two points
def calculate_mahalanobis(centroid1, centroid2, cov_matrix):
    diff = centroid1 - centroid2
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return mahalanobis(centroid1, centroid2, inv_cov_matrix)

# Function to calculate Euclidean distance between two points
def calculate_euclidean(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

# Calculate and print distances
distances = []
for pair in combinations(cluster_data.keys(), 2):
    centroid1 = centroids[centroids['cluster_name'] == pair[0]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    centroid2 = centroids[centroids['cluster_name'] == pair[1]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values

    # Combine points from both clusters to calculate the pooled covariance matrix
    points1 = cluster_data[pair[0]][['x', 'y', 'z']].values
    points2 = cluster_data[pair[1]][['x', 'y', 'z']].values
    combined_points = np.vstack((points1, points2))
    cov_matrix = np.cov(combined_points, rowvar=False)
    
    # Calculate Mahalanobis distance
    mahalanobis_distance = calculate_mahalanobis(centroid1, centroid2, cov_matrix)
    
    # Calculate Euclidean distance
    euclidean_distance = calculate_euclidean(centroid1, centroid2)
    
    distances.append((pair[0], pair[1], euclidean_distance, mahalanobis_distance))

# Print distances
print("\nDistances between centroids:")
for pair in distances:
    print(f"{pair[0]} - {pair[1]}: Euclidean = {pair[2]:.2f}, Mahalanobis = {pair[3]:.2f}")

fig_mahalanobis = go.Figure()

# Add centroids as scatter points
for idx, cluster_name in enumerate(cluster_data.keys()):
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    fig_mahalanobis.add_trace(go.Scatter3d(
        x=centroid['centroid_x'],
        y=centroid['centroid_y'],
        z=centroid['centroid_z'],
        mode='markers+text',
        marker=dict(size=8, color=color_order[idx]),
        text=cluster_name,
        textposition="top center"
    ))

# Add lines connecting centroids of each pair of clusters with Mahalanobis distances
for pair in distances:
    centroid1 = centroids[centroids['cluster_name'] == pair[0]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    centroid2 = centroids[centroids['cluster_name'] == pair[1]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values

    # Add line connecting centroids
    fig_mahalanobis.add_trace(go.Scatter3d(
        x=[centroid1[0], centroid2[0]],
        y=[centroid1[1], centroid2[1]],
        z=[centroid1[2], centroid2[2]],
        mode='lines',
        line=dict(color='gray', width=2),
    ))

    # Annotate distance
    midpoint = [(centroid1[0] + centroid2[0]) / 2, (centroid1[1] + centroid2[1]) / 2, (centroid1[2] + centroid2[2]) / 2]
    fig_mahalanobis.add_trace(go.Scatter3d(
        x=[midpoint[0]],
        y=[midpoint[1]],
        z=[midpoint[2]],
        mode='text',
        text=[f'{pair[3]:.2f}'],
        textposition='top center',
        showlegend=False
    ))

# Set layout properties
fig_mahalanobis.update_layout(
    title='3D Visualization of Cluster Centroids with Mahalanobis Distances',
    scene=dict(
        xaxis_title='Centroid X',
        yaxis_title='Centroid Y',
        zaxis_title='Centroid Z'
    )
)

# Save the plot as HTML using Plotly
fig_mahalanobis.write_html('D:/new code/mahalanobis_distances.html')

# Show the plot
fig_mahalanobis.show()

# Create a 3D plot for Euclidean distances using Plotly
fig_euclidean = go.Figure()

# Add centroids as scatter points
for idx, cluster_name in enumerate(cluster_data.keys()):
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    fig_euclidean.add_trace(go.Scatter3d(
        x=centroid['centroid_x'],
        y=centroid['centroid_y'],
        z=centroid['centroid_z'],
        mode='markers+text',
        marker=dict(size=8, color=color_order[idx]),
        text=cluster_name,
        textposition="top center"
    ))

# Add lines connecting centroids of each pair of clusters with Euclidean distances
for pair in distances:
    centroid1 = centroids[centroids['cluster_name'] == pair[0]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    centroid2 = centroids[centroids['cluster_name'] == pair[1]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values

    # Add line connecting centroids
    fig_euclidean.add_trace(go.Scatter3d(
        x=[centroid1[0], centroid2[0]],
        y=[centroid1[1], centroid2[1]],
        z=[centroid1[2], centroid2[2]],
        mode='lines',
        line=dict(color='gray', width=2),
    ))

    # Annotate distance
    midpoint = [(centroid1[0] + centroid2[0]) / 2, (centroid1[1] + centroid2[1]) / 2, (centroid1[2] + centroid2[2]) / 2]
    fig_euclidean.add_trace(go.Scatter3d(
        x=[midpoint[0]],
        y=[midpoint[1]],
        z=[midpoint[2]],
        mode='text',
        text=[f'{pair[2]:.2f}'],
        textposition='top center',
        showlegend=False
    ))

# Set layout properties
fig_euclidean.update_layout(
    title='3D Visualization of Cluster Centroids with Euclidean Distances',
    scene=dict(
        xaxis_title='Centroid X',
        yaxis_title='Centroid Y',
        zaxis_title='Centroid Z'
    )
)

# Save the plot as HTML using Plotly
fig_euclidean.write_html('D:/new code/euclidean_distances.html')

# Show the plot
fig_euclidean.show()

# Function to plot an ellipsoid
def plot_ellipsoid(centroid, cov_matrix, n_std=2, color='blue'):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate the radii of the ellipsoid
    radii = np.sqrt(eigenvalues) * n_std

    # Generate the points for the ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate the points
    for i in range(len(x)):
        for j in range(len(x[i])):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eigenvectors) + centroid

    return x, y, z

fig = go.Figure()

# Plot centroids and ellipsoids for each cluster
for idx, cluster_name in enumerate(cluster_data.keys()):
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    centroid_coords = centroid.iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    
    # Scatter plot for the centroid
    fig.add_trace(go.Scatter3d(
        x=[centroid_coords[0]], 
        y=[centroid_coords[1]], 
        z=[centroid_coords[2]], 
        mode='markers', 
        marker=dict(size=8, color=color_order[idx]), 
        name=cluster_name
    ))

    # Get data points for the current cluster
    cluster_points = cluster_data[cluster_name][['x', 'y', 'z']]
    
    # Calculate covariance matrix for the cluster
    cov_matrix = np.cov(cluster_points, rowvar=False)
    
    # Generate ellipsoid points
    x, y, z = plot_ellipsoid(centroid_coords, cov_matrix, n_std=2, color=color_order[idx])
    
    # Add ellipsoid surface to the plot
    fig.add_trace(go.Surface(
        x=x, 
        y=y, 
        z=z, 
        opacity=0.5, 
        colorscale=[[0, color_order[idx]], [1, color_order[idx]]],
        showscale=False
    ))

# Updating layout
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ),
    title='Cluster Centroids with Ellipsoids'
)

# Save the figure as an HTML file
fig.write_html('3d_cluster_ellipsoids.html')


# Create a 3D plot for Mahalanobis distances
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot centroids with specific colors
for idx, cluster_name in enumerate(cluster_data.keys()):
    cluster_points = cluster_data[cluster_name]
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    ax.scatter(centroid['centroid_x'], centroid['centroid_y'], centroid['centroid_z'], color=color_order[idx], s=100, label=cluster_name)

# Plot lines connecting centroids of each pair of clusters with Mahalanobis distances
for pair in distances:
    centroid1 = centroids[centroids['cluster_name'] == pair[0]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    centroid2 = centroids[centroids['cluster_name'] == pair[1]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    
    # Plot line connecting centroids
    ax.plot([centroid1[0], centroid2[0]],
            [centroid1[1], centroid2[1]],
            [centroid1[2], centroid2[2]],
            color='gray', linestyle='--')

    # Calculate midpoint
    midpoint_x = (centroid1[0] + centroid2[0]) / 2
    midpoint_y = (centroid1[1] + centroid2[1]) / 2
    midpoint_z = (centroid1[2] + centroid2[2]) / 2

    # Annotate distance
    ax.text(midpoint_x, midpoint_y, midpoint_z, f'{pair[3]:.2f}', color='black', fontsize=10)

# Labeling each point with its cluster name
for i, txt in enumerate(centroids['cluster_name']):
    ax.text(centroids['centroid_x'].iloc[i], centroids['centroid_y'].iloc[i], centroids['centroid_z'].iloc[i], txt)

# Adding labels and title
ax.set_xlabel('Centroid X')
ax.set_ylabel('Centroid Y')
ax.set_zlabel('Centroid Z')
ax.set_title('3D Visualization of Cluster Centroids with Mahalanobis Distances and Lines')
ax.legend()

# Show the plot
plt.show()

# Create a 3D plot for Euclidean distances
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot centroids with specific colors
for idx, cluster_name in enumerate(cluster_data.keys()):
    cluster_points = cluster_data[cluster_name]
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    ax.scatter(centroid['centroid_x'], centroid['centroid_y'], centroid['centroid_z'], color=color_order[idx], s=100, label=cluster_name)

# Plot lines connecting centroids of each pair of clusters with Euclidean distances
for pair in distances:
    centroid1 = centroids[centroids['cluster_name'] == pair[0]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    centroid2 = centroids[centroids['cluster_name'] == pair[1]].iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values
    
    # Plot line connecting centroids
    ax.plot([centroid1[0], centroid2[0]],
            [centroid1[1], centroid2[1]],
            [centroid1[2], centroid2[2]],
            color='gray', linestyle='--')

    # Calculate midpoint
    midpoint_x = (centroid1[0] + centroid2[0]) / 2
    midpoint_y = (centroid1[1] + centroid2[1]) / 2
    midpoint_z = (centroid1[2] + centroid2[2]) / 2

    # Annotate distance
    ax.text(midpoint_x, midpoint_y, midpoint_z, f'{pair[2]:.2f}', color='black', fontsize=10)

# Labeling each point with its cluster name
for i, txt in enumerate(centroids['cluster_name']):
    ax.text(centroids['centroid_x'].iloc[i], centroids['centroid_y'].iloc[i], centroids['centroid_z'].iloc[i], txt)

# Adding labels and title
ax.set_xlabel('Centroid X')
ax.set_ylabel('Centroid Y')
ax.set_zlabel('Centroid Z')
ax.set_title('3D Visualization of Cluster Centroids with Euclidean Distances and Lines')
ax.legend()

# Show the plot
plt.show()

# Function to plot an ellipsoid
def plot_ellipsoid2(ax, centroid, cov_matrix, n_std=1, color='blue'):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate the radii of the ellipsoid
    radii = np.sqrt(eigenvalues) * n_std

    # Generate the points for the ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate the points
    for i in range(len(x)):
        for j in range(len(x[i])):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eigenvectors) + centroid

    # Add the ellipsoid to the axis
    ax.plot_surface(x, y, z, color=color, alpha=0.3)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot centroids and ellipsoids for each cluster
for idx, cluster_name in enumerate(cluster_data.keys()):
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    ax.scatter(centroid['centroid_x'], centroid['centroid_y'], centroid['centroid_z'], color=color_order[idx], s=100, label=cluster_name)

    # Get data points for the current cluster
    cluster_points = cluster_data[cluster_name][['x', 'y', 'z']]
    # Calculate covariance matrix for the cluster
    cov_matrix = np.cov(cluster_points, rowvar=False)
    # Plot ellipsoid for the current cluster
    plot_ellipsoid2(ax, centroid.iloc[0][['centroid_x', 'centroid_y', 'centroid_z']].values, cov_matrix, n_std=2, color=color_order[idx])

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cluster Centroids with Ellipsoids')
ax.legend()
plt.show()


#%% Weighted Distribution of X, Y, Z Coords per Cluster - Figure 6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import kruskal

# Make isolated copies of grouped_x, grouped_y, grouped_z
grouped_x_temp = grouped_x.copy()
grouped_y_temp = grouped_y.copy()
grouped_z_temp = grouped_z.copy()

# --- Statistical Calculations ---
# Function to calculate weighted mean and standard deviation
def weighted_stats(df, value_col, weight_col):
    mean = np.sum(df[value_col] * df[weight_col]) / np.sum(df[weight_col])
    variance = np.sum(df[weight_col] * (df[value_col] - mean)**2) / np.sum(df[weight_col])
    std_dev = np.sqrt(variance)
    return mean, std_dev

# Create a DataFrame for weighted stats
weighted_summary_temp = []
for dim, grouped_data in zip(['x', 'y', 'z'], [grouped_x_temp, grouped_y_temp, grouped_z_temp]):
    for cluster in grouped_data['cluster_name'].unique():
        cluster_data_local = grouped_data[grouped_data['cluster_name'] == cluster]
        mean, std_dev = weighted_stats(cluster_data_local, f'{dim}_midpoint', 'normalized_count')
        weighted_summary_temp.append({'cluster_name': cluster, 'dimension': dim, 'mean': mean, 'std_dev': std_dev})

weighted_summary_temp = pd.DataFrame(weighted_summary_temp)

# Define the desired order for clusters
desired_order_temp = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

# Set the cluster_name column as a categorical type with the desired order
weighted_summary_temp['cluster_name'] = pd.Categorical(
    weighted_summary_temp['cluster_name'], 
    categories=desired_order_temp, 
    ordered=True
)

# Sort the DataFrame by cluster_name and dimension
weighted_summary_temp = weighted_summary_temp.sort_values(by=['dimension', 'cluster_name']).reset_index(drop=True)

# Display the sorted DataFrame
print(weighted_summary_temp)

# Perform Kruskal-Wallis test for each dimension
for dim, grouped_data in zip(['x', 'y', 'z'], [grouped_x_temp, grouped_y_temp, grouped_z_temp]):
    # Collect normalized counts for each cluster
    cluster_counts = [
        grouped_data[grouped_data['cluster_name'] == cluster]['normalized_count'].values
        for cluster in grouped_data['cluster_name'].unique()
    ]
    
    # Perform Kruskal-Wallis test
    kruskal_stat, kruskal_p = kruskal(*cluster_counts)
    print(f"Kruskal-Wallis for {dim}: H-statistic = {kruskal_stat:.3f}, p-value = {kruskal_p:.3e}")

# Define cluster colors
cluster_colors_temp = {
    'Cluster 1': 'blue',
    'Cluster 2': 'red',
    'Cluster 3': 'green',
    'Cluster 4': 'cyan'
}

# Function to dynamically retrieve significant pairs from Dunn's test
def get_significant_pairs_temp(data, alpha=0.05):
    dunn_results = sp.posthoc_dunn(
        data, 
        val_col='normalized_count', 
        group_col='cluster_name', 
        p_adjust='bonferroni'
    )
    print("Dunn's test results:\n", dunn_results)  # Print Dunn's test results
    
    significant_pairs = []
    for i, cluster1 in enumerate(dunn_results.index):
        for j, cluster2 in enumerate(dunn_results.columns):
            if i < j and dunn_results.loc[cluster1, cluster2] < alpha:
                significant_pairs.append((cluster1, cluster2, dunn_results.loc[cluster1, cluster2]))
    return significant_pairs

# Function to add significance annotations
def add_significance_annotations_temp(ax, pairs, y_start, y_step, cluster_names):
    for i, (cluster1, cluster2, p_value) in enumerate(pairs):
        # Get x-axis positions of the clusters
        x1 = cluster_names.index(cluster1)
        x2 = cluster_names.index(cluster2)
        y = y_start + i * y_step  # Increment the height for each pair
        
        # Draw the connecting line without caps
        ax.plot([x1, x1, x2, x2], [y, y, y, y], lw=1.5, color='black')

        # Add appropriate asterisk based on p-value
        if p_value < 0.001:
            annotation = '***'
        elif p_value < 0.01:
            annotation = '**'
        elif p_value < 0.05:
            annotation = '*'
        else:
            annotation = 'ns'  # Not significant
        ax.text(
            (x1 + x2) / 2, y + -0.05, annotation,
            ha='center', va='bottom', fontsize=14, color='black', weight='bold'
        )

# Create a weighted dataset for violin plots
def expand_data_for_violin_temp(df, value_col, weight_col):
    expanded_data = []
    for _, row in df.iterrows():
        # Repeat the value proportional to its weight (scaled for clarity)
        repetitions = int(row[weight_col] * 1000)  # Scale to avoid excessive data
        expanded_data.extend([row[value_col]] * repetitions)
    return expanded_data

# Plotting with weighted violin plots
for dim, label in zip(['x', 'y', 'z'], ['X', 'Y', 'Z']):
    grouped_data_temp = grouped_x_temp if dim == 'x' else grouped_y_temp if dim == 'y' else grouped_z_temp
    dim_data = grouped_data_temp[['cluster_name', f'{dim}_midpoint', 'normalized_count']].copy()
    dim_data.rename(columns={f'{dim}_midpoint': 'midpoint'}, inplace=True)

    # Expand data based on normalized counts
    weighted_data_temp = []
    for cluster_name in desired_order_temp:
        cluster_subset = dim_data[dim_data['cluster_name'] == cluster_name]
        expanded_points = expand_data_for_violin_temp(cluster_subset, 'midpoint', 'normalized_count')
        weighted_data_temp.extend([(cluster_name, point) for point in expanded_points])
    
    weighted_df_temp = pd.DataFrame(weighted_data_temp, columns=['cluster_name', 'midpoint'])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Add violin plot for weighted data
    sns.violinplot(
        data=weighted_df_temp,
        x='cluster_name', y='midpoint',
        palette=cluster_colors_temp, ax=ax, cut=0, linewidth=1, inner=None
    )

    # Overlay weighted mean and error bars
    dim_summary = weighted_summary_temp[weighted_summary_temp['dimension'] == dim]
    for i, cluster_name in enumerate(desired_order_temp):
        cluster_mean = dim_summary.loc[dim_summary['cluster_name'] == cluster_name, 'mean'].values[0]
        cluster_std = dim_summary.loc[dim_summary['cluster_name'] == cluster_name, 'std_dev'].values[0]
        ax.errorbar(
            i, cluster_mean, yerr=cluster_std,
            fmt='o', color='black', capsize=8, elinewidth=2, label=None
        )

    # Adjust the starting height of significance lines
    y_start = max(weighted_df_temp['midpoint']) + 0.5  # Start above violins and error bars
    y_step = 0.5  # Distance between each significance line

    # Add significance annotations
    significant_pairs = get_significant_pairs_temp(grouped_data_temp[['cluster_name', 'normalized_count']])
    add_significance_annotations_temp(
        ax=ax,
        pairs=significant_pairs,
        y_start=y_start,
        y_step=y_step,
        cluster_names=desired_order_temp
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Customize ticks (both x and y)
    ax.tick_params(axis='x', labelsize=20, labelrotation=0)  # Adjust x-ticks
    ax.tick_params(axis='y', labelsize=20)  # Adjust y-ticks

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')  # Bold x-ticks
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')  # Bold y-ticks

    # Final plot details
    ax.set_xlabel(None)
    ax.set_ylabel(f'{label} Coordinate', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

  
#%% Spatial organization of SN units in MNI space (Simple)

# Define path to your NIfTI structures
ATLAS_PATH = r"D:\nigraPhys\3D_visualizations_neural_classes\3D_visualizations_neural_classes\data\atlases\PAULI"
structures = ['lh_SNc_pauli.nii.gz', 'lh_SNr_pauli.nii.gz', 'rh_SNc_pauli.nii.gz', 'rh_SNr_pauli.nii.gz']

# Assign colors to each structure for better visibility
structure_colors = ['black', 'magenta', 'black', 'magenta']

def load_and_extract_surface(nii_file, threshold=0.3):
    # Load NIfTI structure and extract surface points
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Threshold to create a binary mask
    mask = data > threshold
    idxs = np.argwhere(mask)

    # Convert voxel coordinates to real-world coordinates
    xyz = apply_affine(img.affine, idxs)
    return xyz

# Load the SNr and SNc structures
structure_points = []
for structure in structures:
    nii_file = os.path.join(ATLAS_PATH, structure)
    points = load_and_extract_surface(nii_file)
    structure_points.append(points)

# Create a 3D plot for Mahalanobis distances and structures
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot centroids with specific colors
for idx, cluster_name in enumerate(cluster_data.keys()):
    cluster_points = cluster_data[cluster_name]
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    ax.scatter(centroid['centroid_x'], centroid['centroid_y'], centroid['centroid_z'], color=color_order[idx], s=175, label=cluster_name)



    # Calculate midpoint
    midpoint_x = (centroid1[0] + centroid2[0]) / 2
    midpoint_y = (centroid1[1] + centroid2[1]) / 2
    midpoint_z = (centroid1[2] + centroid2[2]) / 2

   

# Plot the extracted NIfTI structure points with assigned colors
for idx, points in enumerate(structure_points):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=structure_colors[idx], alpha=0.8, label=structures[idx].split('_')[1])  # Plot points of the surface with unique colors and labels

# Adding labels and title
ax.set_xlabel('Centroid X')
ax.set_ylabel('Centroid Y')
ax.set_zlabel('Centroid Z')
ax.set_title('3D Visualization of Cluster Centroids with NIfTI Structures')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Show the plot
plt.show()


#%% Spatial organization of SN units in MNI space - Figure 6

# Define path to your NIfTI structures
ATLAS_PATH = r"D:\nigraPhys\3D_visualizations_neural_classes\3D_visualizations_neural_classes\data\atlases\PAULI"
structures = ['lh_SNc_pauli.nii.gz', 'lh_SNr_pauli.nii.gz', 'rh_SNc_pauli.nii.gz', 'rh_SNr_pauli.nii.gz']

# Assign colors to each structure for distinguishability
structure_colors = ['black', 'magenta', 'black', 'magenta'] 

def load_and_extract_surface(nii_file, threshold=0.3):
    # Load NIfTI structure and extract surface points
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Threshold to create a binary mask
    mask = data > threshold
    idxs = np.argwhere(mask)

    # Convert voxel coordinates to real-world coordinates
    xyz = apply_affine(img.affine, idxs)
    return xyz

# Load the SNr and SNc structures into structure_points
structure_points = []
for structure in structures:
    nii_file = os.path.join(ATLAS_PATH, structure)
    points = load_and_extract_surface(nii_file)
    structure_points.append(points)

# Now you can create the 3D plot as in your original code
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Assuming cluster_data and color_order are already defined
# Plot cells for each cluster
for idx, cluster_name in enumerate(cluster_data.keys()):
    cluster_points = cluster_data[cluster_name]
    ax.scatter(cluster_points['x'], cluster_points['y'], cluster_points['z'], color='brown', s=20, label=None)

# Plot the extracted NIfTI structure points with assigned colors
legend_labels = set()

# Plot the extracted NIfTI structure points with assigned colors
for idx, points in enumerate(structure_points):
    label = structures[idx].split('_')[1]  # Extract label (e.g., "SNc" or "SNr")
    if label not in legend_labels:  # Check if the label is already added
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=structure_colors[idx], alpha=0.8, label=label)
        legend_labels.add(label)  # Mark this label as added
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=structure_colors[idx], alpha=0.8)  # No label added

# Adding labels and title
ax.set_xlabel('X', fontsize=14, fontweight='bold')
ax.set_ylabel('Y', fontsize=14, fontweight='bold')
ax.set_zlabel('Z', fontsize=14, fontweight='bold')
ax.set_title('3D Visualization of Cells Overlaid with SNr and SNc Structures')
 # Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=12)  # Increase tick label size
for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    label.set_fontweight('bold')  # Make tick labels bold

legend = ax.legend(
        bbox_to_anchor=(0.85, 0.9),
        loc='upper left',
        borderaxespad=0.,
        fontsize=12,
        scatterpoints=3,  # Increases the size of the legend markers
        markerscale=2,  # Scale factor for the legend markers
    )
for text in legend.get_texts():
    text.set_fontweight('bold')
    
plt.show()


# Define path to your NIfTI structures
ATLAS_PATH = r"D:\nigraPhys\3D_visualizations_neural_classes\3D_visualizations_neural_classes\data\atlases\PAULI"
structures = ['lh_SNc_pauli.nii.gz', 'lh_SNr_pauli.nii.gz', 'rh_SNc_pauli.nii.gz', 'rh_SNr_pauli.nii.gz']

# Assign colors to each structure for distinguishability
structure_colors = ['black', 'magenta', 'black', 'magenta'] 

def load_and_extract_surface(nii_file, threshold=0.3):
    # Load NIfTI structure and extract surface points
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Threshold to create a binary mask
    mask = data > threshold
    idxs = np.argwhere(mask)

    # Convert voxel coordinates to real-world coordinates
    xyz = apply_affine(img.affine, idxs)
    return xyz

# Load the SNr and SNc structures into structure_points
structure_points = []
for structure in structures:
    nii_file = os.path.join(ATLAS_PATH, structure)
    points = load_and_extract_surface(nii_file)
    structure_points.append(points)

# Iterate over each cluster and create a separate plot for each
for idx, cluster_name in enumerate(cluster_data.keys()):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cells for the specific cluster
    cluster_points = cluster_data[cluster_name]
    ax.scatter(cluster_points['x'], cluster_points['y'], cluster_points['z'], color=color_order[idx], s=20, label=f'{cluster_name} Cells')

    # Plot centroid for the specific cluster
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    ax.scatter(centroid['centroid_x'], centroid['centroid_y'], centroid['centroid_z'], color=color_order[idx], s=175, edgecolor='k', linewidths=2.5, label=f'{cluster_name} Centroid')

    # Plot the extracted NIfTI structure points with assigned colors
    for i, points in enumerate(structure_points):
        # Only plot the points without adding labels for structures
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=structure_colors[i], alpha=0.8)

    # Adding labels and title
    ax.set_xlabel('X',fontsize=20, fontweight='bold')
    ax.set_ylabel('Y',fontsize=20, fontweight='bold')
    ax.set_zlabel('Z',fontsize=20, fontweight='bold')
    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=20)  # Increase tick label size
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
       label.set_fontweight('bold')  # Make tick labels bold
    ax.set_title(f'3D Visualization of {cluster_name} with Centroid Overlaid with SNr and SNc Structures')

    # Add legend for other elements (excluding structures)
    legend = ax.legend(bbox_to_anchor=(0.75, 0.9), loc='upper left', fontsize=20, borderaxespad=0.)
    for text in legend.get_texts():
       text.set_fontweight('bold') 
    plt.show()
    

#%% Spatial organization of SN units in MNI space 2.0 - Figure 6

# Define path to your NIfTI structures
ATLAS_PATH = r"D:\nigraPhys\3D_visualizations_neural_classes\3D_visualizations_neural_classes\data\atlases\PAULI"
structures = ['lh_SNc_pauli.nii.gz', 'lh_SNr_pauli.nii.gz', 'rh_SNc_pauli.nii.gz', 'rh_SNr_pauli.nii.gz']

# Assign colors to each structure for distinguishability
structure_colors = ['black', 'magenta', 'black', 'magenta'] 

def load_and_extract_surface(nii_file, threshold=0.3):
    # Load NIfTI structure and extract surface points
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Threshold to create a binary mask
    mask = data > threshold
    idxs = np.argwhere(mask)

    # Convert voxel coordinates to real-world coordinates
    xyz = apply_affine(img.affine, idxs)
    return xyz

# Load the SNr and SNc structures into structure_points
structure_points = []
for structure in structures:
    nii_file = os.path.join(ATLAS_PATH, structure)
    points = load_and_extract_surface(nii_file)
    structure_points.append(points)

# Create a single figure for multiple subplots
fig = plt.figure(figsize=(15, 15))

# Define subplot grid dimensions
n_rows, n_cols = 2, 2

# Iterate over each cluster and create a subplot for each
for idx, cluster_name in enumerate(cluster_data.keys()):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

    # Plot cells for the specific cluster
    cluster_points = cluster_data[cluster_name]
    ax.scatter(cluster_points['x'], cluster_points['y'], cluster_points['z'], color=color_order[idx], s=20, label=f'{cluster_name} Cells')

    # Plot centroid for the specific cluster
    centroid = centroids[centroids['cluster_name'] == cluster_name]
    ax.scatter(centroid['centroid_x'], centroid['centroid_y'], centroid['centroid_z'], color=color_order[idx], s=175, edgecolor='k', linewidths=3, label=f'{cluster_name} Centroid')

    # Initialize a set to track added structure labels
    legend_labels = set()

    # Plot the extracted NIfTI structure points with assigned colors
    for i, points in enumerate(structure_points):
        label = structures[i].split('_')[1]  # Extract label (e.g., "SNc" or "SNr")
        if label not in legend_labels:  # Check if the label is already added
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=structure_colors[i], alpha=0.8, label=label)
            legend_labels.add(label)  # Mark this label as added
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=structure_colors[i], alpha=0.8)  # No label added

    # Adding labels and title for each subplot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{cluster_name} with Centroid and Structures')
    ax.legend(loc='best', borderaxespad=0.)

# Adjust layout and display the figure
plt.tight_layout()
plt.show()


#%% Count of cell/cluster in SNc and SNr 

# Define cluster colors
colors = {'Cluster 1': 'blue', 'Cluster 2': 'red', 'Cluster 3': 'green', 'Cluster 4': 'cyan'}

# Define structure colors
structure_colors = {'SNc': 'black', 'SNr': 'magenta'}

# Combine hemispheres for struct_pauli
df_kmeans_all['combined_struct'] = df_kmeans_all['struct_pauli'].str.extract(r'_(SN[rc])_')[0]

# Group by combined structure and cluster_name, then count occurrences
combined_structure_counts = (
    df_kmeans_all.groupby(['combined_struct', 'cluster_name'])
    .size()
    .reset_index(name='cell_count')
)

# Function to load 3D structure mesh points for the right hemisphere only
def load_right_hemisphere_mesh(structure_name, structure_points):
    if structure_name == 'SNc':
        mesh_points = structure_points[2]  # Right hemisphere SNc
    elif structure_name == 'SNr':
        mesh_points = structure_points[0]  # Right hemisphere SNr
    else:
        mesh_points = None

    if mesh_points is not None:
        # Ensure positive X-axis values for SNr
        if structure_name == 'SNr' and np.any(mesh_points[:, 0] < 0):
            mesh_points[:, 0] = -mesh_points[:, 0]  # Flip X-axis for SNr

    return mesh_points

# Iterate over each combined structure and create a pie chart with inset
unique_structures = combined_structure_counts['combined_struct'].unique()

for structure in unique_structures:
    # Filter data for the current structure
    structure_data = combined_structure_counts[
        combined_structure_counts['combined_struct'] == structure
    ]

    # Plot a pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        structure_data['cell_count'],
        labels=structure_data['cluster_name'],
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors[cluster] for cluster in structure_data['cluster_name']],
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    ax.set_title(f'Cluster Distribution in Combined Structure: {structure}')

    # Add an inset for the mesh visualization
    inset_ax = fig.add_axes([0.58, 0.75, 0.25, 0.25], projection='3d')  # Custom position and size for inset

    # Load the mesh points for the right hemisphere of the current structure
    mesh_points = load_right_hemisphere_mesh(structure, structure_points)
    if mesh_points is not None:
        inset_ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], 
                         s=1, color=structure_colors[structure], alpha=0.8)


        # Calculate and plot convex hull (border)
        try:
            hull = ConvexHull(mesh_points)
            for simplex in hull.simplices:
                inset_ax.plot(
                    mesh_points[simplex, 0], 
                    mesh_points[simplex, 1], 
                    mesh_points[simplex, 2], 
                    color=structure_colors[structure], linewidth=2.5
                )
        except Exception as e:
            print(f"Error calculating convex hull for {structure}: {e}")
            
                  
        # Add axis labels
        inset_ax.set_xlabel('X-axis', fontsize=8)
        inset_ax.set_ylabel('Y-axis', fontsize=8)
        inset_ax.set_zlabel('Z-axis', fontsize=8)

        # Set an appropriate viewing angle
        inset_ax.view_init(elev=10, azim=-65)  # Adjust view for clear visualization
        fig.text(
        0.71, 0.95,  # Adjust these coordinates to align the title above the inset
        f'{structure}',  # Text to display (e.g., SNc or SNr)
        fontsize=12, weight='bold', ha='center', va='bottom', color='black'
)   
    else:
        print(f"No valid mesh points for structure: {structure}")

    plt.show()
    

#%% Stats on cluster counts in SNc and SNr - Figure 7

import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# Extract cluster counts for SNc and SNr
combined_structure_counts = (
    df_kmeans_all.groupby(['combined_struct', 'cluster_name'])
    .size()
    .reset_index(name='cell_count')
)

# Pivot the table to get cluster counts for SNc and SNr
pivot_counts = combined_structure_counts.pivot(index='cluster_name', columns='combined_struct', values='cell_count').fillna(0)

# Calculate proportions for SNc and SNr
pivot_counts['SNc_Proportion'] = pivot_counts['SNc'] / pivot_counts['SNc'].sum()
pivot_counts['SNr_Proportion'] = pivot_counts['SNr'] / pivot_counts['SNr'].sum()

# Display the proportions
print("Proportions of Clusters in SNc and SNr:\n", pivot_counts[['SNc_Proportion', 'SNr_Proportion']])

# Chi-Square Test: Test overall distribution differences
contingency_table = pivot_counts[['SNc', 'SNr']].to_numpy()
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:\n- Chi2: {chi2}\n- P-value: {p}\n- Degrees of Freedom: {dof}")

if p < 0.05:
    print("The overall cluster distributions between SNc and SNr are significantly different.")
else:
    print("No significant difference in overall cluster distributions between SNc and SNr.")

# Z-Test for Proportions for each cluster
print("\nZ-Test Results for Individual Clusters:")
for cluster in pivot_counts.index:
    # Count of cells for this cluster in SNc and SNr
    count = [pivot_counts.loc[cluster, 'SNc'], pivot_counts.loc[cluster, 'SNr']]
    # Total cell count in SNc and SNr
    nobs = [pivot_counts['SNc'].sum(), pivot_counts['SNr'].sum()]

    # Perform Z-Test for proportions
    zstat, pval = proportions_ztest(count, nobs)
    print(f"- {cluster}: Z={zstat:.2f}, P={pval:.4f}")

    if pval < 0.05:
        print(f"  Significant difference in proportions for {cluster} between SNc and SNr.")
    else:
        print(f"  No significant difference in proportions for {cluster} between SNc and SNr.")


#%% Plot cluster proportions in SNc and SNr - Figure 7

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Data for plotting
clusters = pivot_counts.index
snc_proportions = pivot_counts['SNc_Proportion']
snr_proportions = pivot_counts['SNr_Proportion']

# Bar positions
x = np.arange(len(clusters))  # Position for bars
width = 0.35  # Width of the bars

# Perform Z-tests dynamically and collect significant clusters
significant_pairs = []
for cluster in clusters:
    # Count of cells for this cluster in SNc and SNr
    count = [pivot_counts.loc[cluster, 'SNc'], pivot_counts.loc[cluster, 'SNr']]
    # Total cell count in SNc and SNr
    nobs = [pivot_counts['SNc'].sum(), pivot_counts['SNr'].sum()]
    
    # Perform Z-Test for proportions
    zstat, pval = proportions_ztest(count, nobs)
    if pval < 0.05:
        significant_pairs.append(cluster)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
bars_snc = ax.bar(
    x - width / 2, snc_proportions, width, 
    label='SNc', color='black', alpha=0.8
)
bars_snr = ax.bar(
    x + width / 2, snr_proportions, width, 
    label='SNr', color='magenta', alpha=0.8
)

# Add labels, title, and legend
ax.set_ylabel('Proportion', fontsize=20,weight='bold')
#ax.set_title('Proportion of Clusters in SNc vs SNr', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(clusters, fontsize=20, weight='bold')
ax.legend(
    loc='upper left',
    fontsize=None,  # Leave this as None when using `prop`
    prop={'weight': 'bold', 'size': 20},  # Specify both weight and size here
    frameon=False  # Optional: Remove legend border
)
ax.tick_params(axis='y', labelsize=20, which='major')  # Adjust size
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Add proportion values above bars
for bars in [bars_snc, bars_snr]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}', 
            xy=(bar.get_x() + bar.get_width() / 2, height), 
            xytext=(0, 3),  # Offset text position
            textcoords="offset points",
            ha='center', va='bottom', fontsize=20, fontweight='bold'
        )

# Add asterisk for significant pairs
for i, cluster in enumerate(clusters):
    if cluster in significant_pairs:
        # Calculate the maximum height of the SNc and SNr bars for the cluster
        snc_height = snc_proportions[i]
        snr_height = snr_proportions[i]
        max_height = max(snc_height, snr_height)
        
        # Add a line and asterisk above the pair
        ax.plot(
            [x[i] - width / 2, x[i] + width / 2],  # Line spans both bars
            [max_height + 0.05, max_height + 0.05],  # Line height slightly above the taller bar
            color='black', linewidth=2
        )
        ax.annotate(
            '*', 
            xy=(x[i], max_height + 0.0475),  # Position the asterisk above the line
            ha='center', va='bottom', fontsize=16, color='red', weight='bold'
        )

plt.tight_layout()
plt.show()


#%% Stats on clusters location and distribution (Mahalanobis and Euclidean)

# Regularization value for the covariance matrix
regularization_value = 1e-5

# Function to compute Mahalanobis distance with regularization
def mahalanobis_regularized(u, v, cov):
    cov += np.eye(cov.shape[0]) * regularization_value  # Regularization
    inv_cov = np.linalg.inv(cov)
    return mahalanobis(u, v, inv_cov)

# Compute pairwise distances between different clusters (Euclidean and Mahalanobis)
inter_cluster_distances_list = []
mahalanobis_distances_list = []

for i in range(len(centroids)):
    for j in range(i + 1, len(centroids)):
        cluster_data_i = df_kmeans_all[df_kmeans_all['cluster_name'] == centroids.iloc[i]['cluster_name']]
        cluster_data_j = df_kmeans_all[df_kmeans_all['cluster_name'] == centroids.iloc[j]['cluster_name']]
        # Euclidean distances
        inter_cluster_distances_list.append(cdist(cluster_data_i[['x', 'y', 'z']], cluster_data_j[['x', 'y', 'z']]))
        
        # Mahalanobis distances
        combined_points = np.vstack((cluster_data_i[['x', 'y', 'z']].values, cluster_data_j[['x', 'y', 'z']].values))
        cov_matrix = np.cov(combined_points, rowvar=False)
        for point_i in cluster_data_i[['x', 'y', 'z']].values:
            for point_j in cluster_data_j[['x', 'y', 'z']].values:
                mahalanobis_distances_list.append(mahalanobis_regularized(point_i, point_j, cov_matrix))

# Compute pairwise distances within each cluster
intra_cluster_distances_list = []
intra_mahalanobis_distances_list = []

for centroid in centroids.itertuples(index=False):
    cluster_data = df_kmeans_all[df_kmeans_all['cluster_name'] == centroid.cluster_name]
    centroid_coords = centroid[1:4]  # (centroid_x, centroid_y, centroid_z)
    # Euclidean distances
    intra_cluster_distances_list.append(cdist(cluster_data[['x', 'y', 'z']], [centroid_coords]))
    
    # Mahalanobis distances
    points = cluster_data[['x', 'y', 'z']].values
    cov_matrix = np.cov(points, rowvar=False)
    for point in points:
        intra_mahalanobis_distances_list.append(mahalanobis_regularized(point, centroid_coords, cov_matrix))

# Reshape arrays to have a single row
inter_cluster_distances_list_flat = [distances.flatten() for distances in inter_cluster_distances_list]
intra_cluster_distances_list_flat = [distances.flatten() for distances in intra_cluster_distances_list]

# Concatenate the flattened arrays
inter_cluster_distances = np.concatenate(inter_cluster_distances_list_flat)
intra_cluster_distances = np.concatenate(intra_cluster_distances_list_flat)

# Convert Mahalanobis distances to arrays
inter_mahalanobis_distances = np.array(mahalanobis_distances_list)
intra_mahalanobis_distances = np.array(intra_mahalanobis_distances_list)

# Perform statistical comparison (e.g., t-test)
t_statistic_euclidean, p_value_euclidean = ttest_ind(intra_cluster_distances, inter_cluster_distances)
t_statistic_mahalanobis, p_value_mahalanobis = ttest_ind(intra_mahalanobis_distances, inter_mahalanobis_distances)

# Print the results
print("Euclidean Distances T-Statistic:", t_statistic_euclidean)
print("Euclidean Distances P-Value:", p_value_euclidean)
print("Mahalanobis Distances T-Statistic:", t_statistic_mahalanobis)
print("Mahalanobis Distances P-Value:", p_value_mahalanobis)

# Calculate mean and standard deviation for intra-cluster and inter-cluster distances
mean_intra_distance = np.mean(intra_cluster_distances)
mean_inter_distance = np.mean(inter_cluster_distances)
std_intra_distance = np.std(intra_cluster_distances)
std_inter_distance = np.std(inter_cluster_distances)

mean_intra_mahalanobis_distance = np.mean(intra_mahalanobis_distances)
mean_inter_mahalanobis_distance = np.mean(inter_mahalanobis_distances)
std_intra_mahalanobis_distance = np.std(intra_mahalanobis_distances)
std_inter_mahalanobis_distance = np.std(inter_mahalanobis_distances)

print("Mean Intra-Cluster Euclidean Distance:", mean_intra_distance)
print("Mean Inter-Cluster Euclidean Distance:", mean_inter_distance)
print("Standard Deviation of Intra-Cluster Euclidean Distance:", std_intra_distance)
print("Standard Deviation of Inter-Cluster Euclidean Distance:", std_inter_distance)

print("Mean Intra-Cluster Mahalanobis Distance:", mean_intra_mahalanobis_distance)
print("Mean Inter-Cluster Mahalanobis Distance:", mean_inter_mahalanobis_distance)
print("Standard Deviation of Intra-Cluster Mahalanobis Distance:", std_intra_mahalanobis_distance)
print("Standard Deviation of Inter-Cluster Mahalanobis Distance:", std_inter_mahalanobis_distance)

# Perform Welch's t-test
t_statistic_welch_euclidean, p_value_welch_euclidean = ttest_ind(intra_cluster_distances, inter_cluster_distances, equal_var=False)
t_statistic_welch_mahalanobis, p_value_welch_mahalanobis = ttest_ind(intra_mahalanobis_distances, inter_mahalanobis_distances, equal_var=False)

print("Welch's t-test Statistic (Euclidean):", t_statistic_welch_euclidean)
print("Welch's t-test P-Value (Euclidean):", p_value_welch_euclidean)
print("Welch's t-test Statistic (Mahalanobis):", t_statistic_welch_mahalanobis)
print("Welch's t-test P-Value (Mahalanobis):", p_value_welch_mahalanobis)

# Calculate pairwise distances between centroids (Euclidean)
centroid_coordinates = centroids[['centroid_x', 'centroid_y', 'centroid_z']]
distance_matrix_euclidean = pdist(centroid_coordinates, metric='euclidean')
distance_matrix_euclidean_square = squareform(distance_matrix_euclidean)  # Convert to a square matrix format for easier handling

# Print the distance matrix for visualization
print("Pairwise Centroid Distance Matrix (Euclidean):\n", distance_matrix_euclidean_square)

# Calculate mean and standard deviation of the distances for analysis
mean_centroid_distance_euclidean = np.mean(distance_matrix_euclidean)
std_centroid_distance_euclidean = np.std(distance_matrix_euclidean)

# Perform a one-sample t-test against the null hypothesis that the mean distance is zero (no separation)
t_statistic_centroid_euclidean, p_value_centroid_euclidean = ttest_1samp(distance_matrix_euclidean, 0)

print("Mean Centroid Distance (Euclidean):", mean_centroid_distance_euclidean)
print("Standard Deviation of Centroid Distance (Euclidean):", std_centroid_distance_euclidean)
print("T-Statistic (Euclidean):", t_statistic_centroid_euclidean)
print("P-Value (Euclidean):", p_value_centroid_euclidean)

# Calculate pairwise distances between centroids (Mahalanobis)
distance_matrix_mahalanobis = []
for i in range(len(centroid_coordinates)):
    for j in range(i + 1, len(centroid_coordinates)):
        combined_points = np.vstack((centroid_coordinates.iloc[i].values, centroid_coordinates.iloc[j].values))
        cov_matrix = np.cov(combined_points, rowvar=False)
        dist = mahalanobis_regularized(centroid_coordinates.iloc[i].values, centroid_coordinates.iloc[j].values, cov_matrix)
        distance_matrix_mahalanobis.append(dist)
distance_matrix_mahalanobis = np.array(distance_matrix_mahalanobis)

# Calculate mean and standard deviation of the distances for analysis
mean_centroid_distance_mahalanobis = np.mean(distance_matrix_mahalanobis)
std_centroid_distance_mahalanobis = np.std(distance_matrix_mahalanobis)

# Perform a one-sample t-test against the null hypothesis that the mean distance is zero (no separation)
t_statistic_centroid_mahalanobis, p_value_centroid_mahalanobis = ttest_1samp(distance_matrix_mahalanobis, 0)

print("Mean Centroid Distance (Mahalanobis):", mean_centroid_distance_mahalanobis)
print("Standard Deviation of Centroid Distance (Mahalanobis):", std_centroid_distance_mahalanobis)
print("T-Statistic (Mahalanobis):", t_statistic_centroid_mahalanobis)
print("P-Value (Mahalanobis):", p_value_centroid_mahalanobis)


from scipy.stats import mannwhitneyu

u_stat, p_value_u = mannwhitneyu(intra_mahalanobis_distances, inter_mahalanobis_distances, alternative='less')
print("Mann-Whitney U Test Statistic:", u_stat)
print("Mann-Whitney U Test P-Value:", p_value_u)


#%% Stats plots Intra vs Inter cluster distances - Figure 6


# Create DataFrames for plotting
distance_data_euclidean = pd.DataFrame({
    'Distance Type': ['Intra-Cluster'] * len(intra_cluster_distances) + ['Inter-Cluster'] * len(inter_cluster_distances),
    'Distance': np.concatenate([intra_cluster_distances, inter_cluster_distances])
})

distance_data_mahalanobis = pd.DataFrame({
    'Distance Type': ['Intra-Cluster'] * len(intra_mahalanobis_distances) + ['Inter-Cluster'] * len(inter_mahalanobis_distances),
    'Distance': np.concatenate([intra_mahalanobis_distances, inter_mahalanobis_distances])
})

# Plotting Euclidean distances
plt.figure(figsize=(10, 6))
sns.boxplot(x='Distance Type', y='Distance', data=distance_data_euclidean)
plt.title('Comparison of Intra-Cluster and Inter-Cluster Distances (Euclidean)')
plt.ylabel('Distance')
plt.xlabel('Cluster Type')

# Annotate p-value on Euclidean boxplot
plt.text(0.5, max(distance_data_euclidean['Distance']) * 1.05, f'p = {p_value_euclidean:.3e}',
         ha='center', va='bottom', color='black', fontsize=12)

plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Distance Type', y='Distance', data=distance_data_euclidean, inner='quartile', palette='muted')
plt.title('Violin Plot of Intra-Cluster vs. Inter-Cluster Distances (Euclidean)')
plt.ylabel('Distance')
plt.xlabel('Cluster Type')

# Annotate p-value on Euclidean violin plot
plt.text(0.5, max(distance_data_euclidean['Distance']) * 1.05, f'p = {p_value_euclidean:.3e}',
         ha='center', va='bottom', color='black', fontsize=12)

plt.show()

# Plotting Mahalanobis distances
plt.figure(figsize=(10, 6))
sns.boxplot(x='Distance Type', y='Distance', data=distance_data_mahalanobis)
plt.title('Comparison of Intra-Cluster and Inter-Cluster Distances (Mahalanobis)')
plt.ylabel('Distance')
plt.xlabel('Cluster Type')

# Annotate p-value on Mahalanobis boxplot
plt.text(0.5, max(distance_data_mahalanobis['Distance']) * 1.05, f'p = {p_value_mahalanobis:.3e}',
         ha='center', va='bottom', color='black', fontsize=12)

plt.show()

# Create the violin plot
plt.figure(figsize=(10, 8))
sns.violinplot(
    x='Distance Type', y='Distance', data=distance_data_mahalanobis,
    inner='quartile', palette=['grey', 'grey']
)

# Customize labels, title, and font weight
plt.title('Mahalanobis Distances', fontsize=20, weight='bold')
plt.ylabel('Distance', fontsize=20, weight='bold')
plt.xlabel('')  # Remove x-axis label

# Adjust tick labels
plt.xticks(fontsize=20, weight='bold')
plt.yticks(fontsize=20, weight='bold')

# Add a significance line and asterisk
max_distance = distance_data_mahalanobis['Distance'].max()
x_positions = [0, 1]  # x-coordinates for the violins
y, h, col = max_distance * 1.05, -0.05, 'black'  # Line height and color
plt.plot([x_positions[0], x_positions[1]], [y, y], color=col, linewidth=2)  # Line
plt.text(
    np.mean(x_positions), y + h, '***',  # Asterisk position
    ha='center', va='bottom', color='red', fontsize=16, weight='bold'
)

plt.tight_layout()
plt.show()


# Create a DataFrame for the centroid distances
centroid_distance_df = pd.DataFrame(distance_matrix_euclidean_square, columns=centroids['cluster_name'], index=centroids['cluster_name'])

plt.figure(figsize=(8, 6))
sns.heatmap(centroid_distance_df, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Distance'})
plt.title('Pairwise Centroid Distance Matrix (Euclidean)')
plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
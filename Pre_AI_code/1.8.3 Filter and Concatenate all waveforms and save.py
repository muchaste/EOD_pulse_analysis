# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:38:50 2024

@author: Admin
"""

import os
import pandas as pd
import glob
import tkinter as tk
from tkinter import filedialog

# Load the CSV file
# Set directories
root = tk.Tk()
root.withdraw()
directory = filedialog.askdirectory(title = "Select Folder with Logger Files")

# Initialize lists to store the data and metadata
all_waveforms = []
all_metadata = []

# Iterate through all CSV files with "_h2t_waveforms" in the filename
for file_path in glob.glob(os.path.join(directory, '*_h2t_waveforms.csv')):
    # Read the waveform CSV file
    waveforms_df = pd.read_csv(file_path)
    
    # Corresponding metadata file
    metadata_file = file_path.replace('_h2t_waveforms.csv', '_eod_table.csv')
    
    if os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file)
        
        # Merge waveforms and metadata
        combined_df = waveforms_df.merge(metadata_df, left_index=True, right_index=True)
        
        # Append to the combined list
        all_waveforms.append(combined_df.drop(columns=metadata_df.columns))
        all_metadata.append(metadata_df)

# Concatenate all data into a single DataFrame
combined_waveforms = pd.concat(all_waveforms, ignore_index=True)
combined_metadata = pd.concat(all_metadata, ignore_index=True)

# Filter out unwanted waveforms
filtered_indices = combined_metadata[
    (~combined_metadata['species'].str.startswith('X')) &
    (~combined_metadata['fish_id'].str.contains('baby|asym|clip', case=False))
].index

combined_waveforms = combined_waveforms.loc[filtered_indices].reset_index(drop=True)
combined_metadata = combined_metadata.loc[filtered_indices].reset_index(drop=True)

# Select the highest amplitude waveforms per ID
filtered_waveforms = []
filtered_metadata = []

for unique_id in combined_metadata['fish_id'].unique():
    id_indices = combined_metadata[combined_metadata['fish_id'] == unique_id].index
    id_waveforms = combined_waveforms.loc[id_indices]
    id_metadata = combined_metadata.loc[id_indices]
    
    if len(id_indices) > 50:
        top_waveforms = id_metadata.nlargest(50, 'amplitude').index
    else:
        top_waveforms = id_metadata.nlargest(int(len(id_indices) * 0.7), 'amplitude').index
        print(str(unique_id) + ": no 50 pulses, used the 70% highest amp pulses: "+str(int(len(id_indices) * 0.7)))
    
    filtered_waveforms.append(id_waveforms.loc[top_waveforms])
    filtered_metadata.append(id_metadata.loc[top_waveforms])

final_waveforms = pd.concat(filtered_waveforms, ignore_index=True)
final_metadata = pd.concat(filtered_metadata, ignore_index=True)

# Save the filtered waveforms and metadata to CSV files
final_waveforms.to_csv(directory+'/filtered_waveforms.csv', index=False)
final_metadata.to_csv(directory + '/filtered_metadata.csv', index=False)

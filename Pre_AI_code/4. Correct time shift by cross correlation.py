"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
import audioio as aio
from scipy.signal import correlate
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
import datetime as dt


# If the logger lost the time, 
if timecol[0].year == 2019:
    print("Logger seems to have lost the time. Select another file with correct timestamp for the first recording")
    anchor_file = filedialog.askopenfilename()
    anchor_timestamp = pd.to_datetime(anchor_file.split('/')[-1].split('-')[-1].split('.')[0], format='%Y%m%dT%H%M%S')
    anchor_data,rate = aio.load_audio(anchor_file)
    first_file, _ = aio.load_audio(filelist[0])
    
    # calculate offset
    # Compute cross-correlation
    correlation = correlate(anchor_data[:,0], first_file[:,0])
    
    # Plot the cross-correlation
    plt.figure(figsize=(10, 5))
    plt.plot(correlation)
    plt.title('Cross-Correlation between Recordings')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.savefig('%s\\cross_correlation.png' % (output_path))
    plt.show(block=False)
    # Find the index of the peak
    offset_index = np.argmax(correlation) - (len(first_file[:,0]) - 1)
    offset_seconds = offset_index/rate
    first_timestamp = anchor_timestamp + dt.timedelta(seconds = offset_seconds)
    first_timedelta = first_timestamp - timecol[0]
    timecol = timecol + first_timedelta
    
    # Create a DataFrame with the offset information
    df = pd.DataFrame({'Offset_samples': [offset_index], 'Offset_seconds': [offset_seconds],\
                       'File': [filelist[0]], 'Anchor': [anchor_file],\
                           'New_first_timestamp':[first_timestamp]})
    # Save the DataFrame to a CSV file
    df.to_csv('%s\\cross_correlation_offset.csv' %(output_path), index=False)
    
    del anchor_data, first_file, correlation


file_set = pd.DataFrame({'timestamp': timecol, 'filename': filelist})
file_set = file_set.sort_values(by=['timestamp'], ignore_index=True)
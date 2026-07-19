# Program To Read video 
# and Extract Frames 
import cv2 
import numpy as np
import tkinter
from tkinter import filedialog
import glob
import pandas as pd
import re
import os


# LED pixelcoordinates
x_s = 15
x_e = 40
y_s = 338
y_e = 364

# Video settings
fps = 40

# Time windows
frames = 60*fps

# Pick a directory
root = tkinter.Tk()
root.withdraw()

root_dir = filedialog.askdirectory(title = "Select root folder with data")
# find vid_files recursively
vid_files = sorted(glob.glob(os.path.join(root_dir, '**', '*.avi'), recursive=True))
log_files = sorted(glob.glob(os.path.join(root_dir, '**', 'log_*.txt'), recursive=True))

# vidpath = filedialog.askdirectory(title = "Select folder with video files")
# output_path = filedialog.askdirectory(title = 'Select folder for output of results')
# logpath = filedialog.askdirectory(title = "Select folder with log .txt files")
# vidlist = glob.glob(vidpath+'/*.avi')          # List all .avi files in data directory
# loglist = glob.glob(logpath+'/*[0-9].txt')          # List all .txt files in data directory

# fish_id = open(loglist[0], "r").readlines(30)[1][9:-1]

logtext = open(log_files[0], "r").readlines()
fish_id = logtext[1].split(':')[1][1:-1]

#%%
# Preallocate empty lists
file_basenames = []
firstframe = []
lastframe = []
blinkduration = []
log_start = []
blink_start = []
    
for fname in vid_files:
    # Path to video file 
    vidObj = cv2.VideoCapture(fname) 
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))
    n_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Used as counter variable 
    count = 0
      
    # checks whether frames were extracted 
    success = 1
    
    # empty lists
    brightness = []
    frame = []
    
    while success and count < frames: 
      
        success, image = vidObj.read() 
        crop_img = image[y_s:y_e, x_s:x_e]   
        brightness.append(np.mean(crop_img))
        frame.append(count+1)
        count += 1
            
            
    threshold = (np.median(brightness) + max(brightness))/2
    
    blinkframes = np.argwhere(brightness > threshold)
    firstframe.append(min(blinkframes)[0])
    lastframe.append(max(blinkframes)[0])
    blinkduration.append(len(blinkframes))
    
    # New: read time from .txt log file
    logfile = [i for i in log_files if re.split("\\\\", fname)[-1][0:-8] in i][0]
    logtext = open(logfile, "r").readlines()
    time_start_log = pd.to_datetime(logtext[0], format='%Y\\%m\\%d ; %H:%M:%S.%f\n')
    time_start_blink = time_start_log + pd.to_timedelta(firstframe[-1]/fps, unit='s')
    log_start.append(time_start_log)
    blink_start.append(time_start_blink)
    # get the basename of the video file without the path and extension
    file_basenames.append(os.path.splitext(os.path.basename(fname))[0])
    print(fname)


    

df = pd.DataFrame(
    {'videoname':vid_files, 'file_basenames': file_basenames, 'firstframe': firstframe, 'lastframe': lastframe, \
     'blinkduration': blinkduration, 'log_start': log_start, 'blink_start': blink_start, 'id': fish_id, 'fps': fps}
    )

# Sort df by log_start
df = df.sort_values(by='log_start')

csv_name = os.path.join(root_dir, fish_id+"_ledblinks_python.csv")
df.to_csv(csv_name, index = False)

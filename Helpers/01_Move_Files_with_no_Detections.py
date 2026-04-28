import os
import shutil
import configparser
import glob
import tkinter as tk
from tkinter import filedialog
import pandas as pd

root = tk.Tk()
root.withdraw()

config_filepath = filedialog.askopenfilename(
    title="Select a parameter config file", filetypes=[("CFG files", "*.cfg")]
)
if not config_filepath:
    raise ValueError("No config file selected")

config = configparser.ConfigParser()
config.read(config_filepath)

input_path = config['Paths']['input_path']
output_path = config['Paths']['output_path']
print(f"Input:  {input_path}")
print(f"Output: {output_path}")

summary_path = os.path.join(output_path, "all_event_summaries.csv")
if not os.path.exists(summary_path):
    raise FileNotFoundError(f"all_event_summaries.csv not found in {output_path}")

summaries = pd.read_csv(summary_path)

referenced_basenames = set()
for cell in summaries['file_names'].dropna():
    for entry in cell.split(','):
        referenced_basenames.add(os.path.basename(entry.strip()))

wav_files = glob.glob(os.path.join(input_path, "*.wav"))
wav_basenames = {os.path.basename(f): f for f in wav_files}

no_detection_basenames = set(wav_basenames.keys()) - referenced_basenames

if not no_detection_basenames:
    print("No files to move — all WAV files have detections.")
else:
    dest_folder = os.path.join(input_path, "no_detections")
    os.makedirs(dest_folder, exist_ok=True)
    for basename in sorted(no_detection_basenames):
        src = wav_basenames[basename]
        dst = os.path.join(dest_folder, basename)
        shutil.move(src, dst)
        print(f"  Moved: {basename}")
    print(f"\nMoved {len(no_detection_basenames)} file(s) to '{dest_folder}'")
    print(f"{len(wav_files) - len(no_detection_basenames)} file(s) remain in input folder")


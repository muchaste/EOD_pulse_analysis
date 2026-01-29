#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EOD Pulse Detection Diagnostic Tool
===================================

Simple diagnostic tool based on Script 03 logic with adjustable parameters.
This is essentially Script 03 with a GUI for parameter adjustment and visualization.

Author: AI Assistant & User
Date: September 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import audioio as aio
import json
import pickle
from scipy import signal


# Import from Script 03
import thunderfish.pulses as pulses
from pulse_functions import (extract_pulse_snippets, 
                             bandpass_filter,
                             filter_waveforms, 
                             filter_waveforms_with_classifier,
                             unify_across_channels,
                             normalize_waveforms)

# ML-related imports for classifier functionality
try:
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML classification will be disabled.")
    ML_AVAILABLE = False

class PulseDiagnosticTool:
    def __init__(self, root):
        self.root = root
        self.root.title("EOD Pulse Detection Diagnostic Tool")
        self.root.geometry("1600x1200")
        
        # Data storage
        self.raw_data = None
        self.sample_rate = None
        self.calibration_factors = None
        self.calibrated_data = None
        self.data_source = 'multich_linear'  # 'multich_linear', '1ch_diff'
        self.file_name = None
        
        # File metadata for reloading
        self.source_file_path = None
        self.source_file_type = None  # 'audio', 'precalibrated', '1ch_diff', 'ni_source'
        self.calibration_file_path = None
        self.loaded_time_window = {'start_sec': 0.0, 'end_sec': 0.0}  # Currently loaded window
        self.full_data = None  # Store full data for subsetting (if reasonable size)
        
        # Default parameters (exactly from Script 03)
        self.parameters = {
            'thresh': 1,
            'min_rel_slope_diff': 0.25,
            'min_width_us': 30,
            'max_width_us': 1000,
            'width_fac_detection': 7.0,
            'interp_factor': 3,
            'amplitude_ratio_min': 0.2,
            'amplitude_ratio_max': 4.0,
            'save_filtered_out': False,
            'peak_fft_freq_min': 50,
            'peak_fft_freq_max': 10000,
            'return_diff': True,
            'length': 2000
        }
        
        # Time window parameters (0,0 means use full file)
        self.time_window = {'start_sec': 0.0, 'end_sec': 0.0}
        
        # Analysis parameters for control recording style analysis
        self.clip_threshold = tk.DoubleVar(value=20.0)  # Peak-trough amplitude threshold
        self.top_percent = tk.DoubleVar(value=100.0)   # Top x% of pulses to analyze
        
        # Storage for extracted pulse data (for analysis)
        self.current_pulse_data = None
        
        # ML configuration variables
        self.use_ml_filtering = tk.BooleanVar(value=False)
        self.classifier_path = tk.StringVar(value="")
        self.fish_probability_threshold = tk.DoubleVar(value=0.5)
        self.loaded_classifier = None
        self.loaded_scaler = None
        self.classifier_name = None

        # Bandpass configuration variables
        self.use_bandpass_filter = tk.BooleanVar(value=False)
        self.bandpass_lowcut = tk.DoubleVar(value=100.0)
        self.bandpass_highcut = tk.DoubleVar(value=6000.0)
        self.bandpass_order = tk.IntVar(value=4)

        
        self.setup_gui()
        
    def setup_gui(self):
        # Main panes
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=0)
        
        # Right panel for plot
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self.setup_control_panel(left_frame)
        self.setup_plot_panel(right_frame)
        
    def setup_control_panel(self, parent):
        # File selection
        file_frame = ttk.LabelFrame(parent, text="Files", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Browse Audio File", command=self.select_audio_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Browse Calibration File", command=self.select_calibration_file).pack(fill=tk.X, pady=2)
        
        # Separator
        ttk.Separator(file_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Load Pre-Calibrated Data", command=self.load_precalibrated_data).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load 1 Ch. Diff Data", command=self.load_single_ch_diff_data).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Data from NI Source", command=self.load_ni_source_data).pack(fill=tk.X, pady=2)
        
        self.file_info = ttk.Label(file_frame, text="No files loaded", wraplength=350)
        self.file_info.pack(fill=tk.X, pady=5)
        
        # Time window selection
        time_frame = ttk.LabelFrame(parent, text="Time Window (seconds)", padding=10)
        time_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row for start and end time inputs
        time_inputs_frame = ttk.Frame(time_frame)
        time_inputs_frame.pack(fill=tk.X, pady=2)
        
        # Start time (left side)
        start_frame = ttk.Frame(time_inputs_frame)
        start_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Label(start_frame, text="Start:", width=5).pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value="0.0")
        start_entry = ttk.Entry(start_frame, textvariable=self.start_time_var, width=10)
        start_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        start_entry.bind('<Return>', lambda e: self.load_data())
        
        # End time (right side)
        end_frame = ttk.Frame(time_inputs_frame)
        end_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Label(end_frame, text="End:", width=5).pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value="0.0")
        end_entry = ttk.Entry(end_frame, textvariable=self.end_time_var, width=10)
        end_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        end_entry.bind('<Return>', lambda e: self.load_data())
        
        # Load Data button spanning full width below
        ttk.Button(time_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=(5, 2))
        
        ttk.Label(time_frame, text="(0,0 = use full recording)", font=('Arial', 8)).pack(pady=2)       
        
        # Parameters (scrollable, arranged in two columns)
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding=5)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create two-column layout for parameters
        param_columns_frame = ttk.Frame(param_frame)
        param_columns_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column
        left_param_frame = ttk.Frame(param_columns_frame)
        left_param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right column
        right_param_frame = ttk.Frame(param_columns_frame)
        right_param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Parameter inputs split into two columns
        self.param_vars = {}
        param_items = list(self.parameters.items())
        mid_point = (len(param_items) + 1) // 2
        
        # Left column parameters
        for i, (param, value) in enumerate(param_items[:mid_point]):
            row_frame = ttk.Frame(left_param_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(row_frame, text=param, width=18).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(row_frame, textvariable=var, width=8)
            entry.pack(side=tk.RIGHT)
            entry.bind('<Return>', lambda e: self.detect_pulses())
            self.param_vars[param] = var
        
        # Right column parameters
        for i, (param, value) in enumerate(param_items[mid_point:]):
            row_frame = ttk.Frame(right_param_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(row_frame, text=param, width=18).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(row_frame, textvariable=var, width=8)
            entry.pack(side=tk.RIGHT)
            entry.bind('<Return>', lambda e: self.detect_pulses())
            self.param_vars[param] = var
        
        # Control buttons and settings
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create two-column layout for controls
        controls_columns_frame = ttk.Frame(button_frame)
        controls_columns_frame.pack(fill=tk.X, pady=2)
        
        # Left column for main controls
        left_controls = ttk.Frame(controls_columns_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right column for additional controls
        right_controls = ttk.Frame(controls_columns_frame)
        right_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # LEFT COLUMN: Plot mode and ML controls
        # Plotting mode selection
        plot_mode_frame = ttk.LabelFrame(left_controls, text="Plot Mode", padding=5)
        plot_mode_frame.pack(fill=tk.X, pady=2)
        
        self.plot_mode = tk.StringVar(value="differential")
        ttk.Radiobutton(plot_mode_frame, text="Differential", variable=self.plot_mode, 
                       value="differential").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(plot_mode_frame, text="Single-Ended", variable=self.plot_mode, 
                       value="single_ended").pack(side=tk.LEFT, padx=5)
        
        # ML Classification controls (compact)
        ml_frame = ttk.LabelFrame(left_controls, text="ML Classification", padding=5)
        ml_frame.pack(fill=tk.X, pady=2)
        
        # ML enable and threshold in one row
        ml_row1 = ttk.Frame(ml_frame)
        ml_row1.pack(fill=tk.X, pady=1)
        
        self.ml_checkbox = ttk.Checkbutton(
            ml_row1, 
            text="Enable ML", 
            variable=self.use_ml_filtering,
            command=self.on_ml_toggle
        )
        self.ml_checkbox.pack(side=tk.LEFT)
        
        ttk.Label(ml_row1, text="Threshold:", width=8).pack(side=tk.LEFT, padx=(10,0))
        threshold_entry = ttk.Entry(ml_row1, textvariable=self.fish_probability_threshold, width=6)
        threshold_entry.pack(side=tk.LEFT, padx=2)
        
        # Classifier selection in second row
        ml_row2 = ttk.Frame(ml_frame)
        ml_row2.pack(fill=tk.X, pady=1)
        
        ttk.Label(ml_row2, text="Classifier:", width=8).pack(side=tk.LEFT)
        self.classifier_label = ttk.Label(ml_row2, text="None", foreground="gray", width=12)
        self.classifier_label.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(ml_row2, text="Browse", command=self.select_classifier_file, width=8).pack(side=tk.RIGHT)
        
        # RIGHT COLUMN: Analysis parameters
        # Analysis parameters frame (compact)
        analysis_frame = ttk.LabelFrame(right_controls, text="Pulse Analysis", padding=5)
        analysis_frame.pack(fill=tk.X, pady=2)
        
        # Clip threshold and top percent in compact layout
        analysis_row1 = ttk.Frame(analysis_frame)
        analysis_row1.pack(fill=tk.X, pady=1)
        
        ttk.Label(analysis_row1, text="Clip:", width=5).pack(side=tk.LEFT)
        clip_entry = ttk.Entry(analysis_row1, textvariable=self.clip_threshold, width=6)
        clip_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(analysis_row1, text="Top%:", width=5).pack(side=tk.LEFT, padx=(10,0))
        percent_entry = ttk.Entry(analysis_row1, textvariable=self.top_percent, width=6)
        percent_entry.pack(side=tk.LEFT, padx=2)

        # Bandpass filter control
        bp_frame = ttk.LabelFrame(right_controls, text="Bandpass Filter", padding=5)
        bp_frame.pack(fill=tk.X, pady=2)

        # BP enable
        bp_enable_checkbox = ttk.Checkbutton(bp_frame, text="Enable Bandpass", variable=self.use_bandpass_filter)
        bp_enable_checkbox.pack(side=tk.LEFT, padx=5, pady=2)

        # BP parameters in one row
        bp_row = ttk.Frame(bp_frame)
        bp_row.pack(fill=tk.X, pady=1)
        ttk.Label(bp_row, text="Lowcut (Hz):", width=10).pack(side=tk.LEFT)
        lowcut_entry = ttk.Entry(bp_row, textvariable=self.bandpass_lowcut, width=8)
        lowcut_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(bp_row, text="Highcut (Hz):", width=10).pack(side=tk.LEFT, padx=(10,0))
        highcut_entry = ttk.Entry(bp_row, textvariable=self.bandpass_highcut, width=8)
        highcut_entry.pack(side=tk.LEFT, padx=2)

        # Order in second row
        bp_row2 = ttk.Frame(bp_frame)
        bp_row2.pack(fill=tk.X, pady=1)
        ttk.Label(bp_row2, text="Order:", width=6).pack(side=tk.LEFT, padx=(10,0))
        order_entry = ttk.Entry(bp_row2, textvariable=self.bandpass_order, width=4)
        order_entry.pack(side=tk.LEFT, padx=2)
        
        # Main action buttons
        buttons_row = ttk.Frame(button_frame)
        buttons_row.pack(fill=tk.X, pady=5)
        
        ttk.Button(buttons_row, text="Detect Pulses", command=self.detect_pulses).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(buttons_row, text="Analyze Pulses", command=self.analyze_pulses).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Secondary buttons
        buttons_row2 = ttk.Frame(button_frame)
        buttons_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(buttons_row2, text="Export Parameters", command=self.export_parameters).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(buttons_row2, text="Import Parameters", command=self.import_parameters).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Save audio button
        buttons_row3 = ttk.Frame(button_frame)
        buttons_row3.pack(fill=tk.X, pady=2)
        
        ttk.Button(buttons_row3, text="Save as Audio", command=self.save_as_audio).pack(fill=tk.X, padx=2)
        
        # Initially disable ML controls if sklearn not available
        if not ML_AVAILABLE:
            self.ml_checkbox.configure(state="disabled")
            self.classifier_label.configure(text="sklearn N/A")
        
        # Results summary (reduced height)
        self.results_text = tk.Text(parent, height=6, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, padx=5, pady=5)
        
    def setup_plot_panel(self, parent):
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        
        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.ax.text(0.5, 0.5, 'Load audio and calibration files to begin', 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=14, alpha=0.5)
        self.ax.set_title('EOD Pulse Detection Diagnostic')
        self.canvas.draw()
        
    def select_audio_file(self):
        """Select audio file - stores path only, doesn't load data"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            self.source_file_path = file_path
            self.source_file_type = 'audio'
            self.file_name = os.path.basename(file_path)
            self.update_file_info()
            print(f"Selected audio file: {file_path}")
            print("Click 'Load Data' to load the audio file")
    
    def select_calibration_file(self):
        """Select calibration file - stores path only, doesn't load data"""
        file_path = filedialog.askopenfilename(
            title="Select Calibration File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.calibration_file_path = file_path
            self.update_file_info()
            print(f"Selected calibration file: {file_path}")
            print("Click 'Load Data' to load with calibration")
    
    def load_precalibrated_data(self):
        """Select pre-calibrated audio file - stores path only, doesn't load data"""
        file_path = filedialog.askopenfilename(
            title="Select Pre-Calibrated Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            self.source_file_path = file_path
            self.source_file_type = 'precalibrated'
            self.file_name = os.path.basename(file_path)
            self.calibration_file_path = None  # Clear any calibration file
            self.update_file_info()
            print(f"Selected pre-calibrated audio: {file_path}")
            print("Click 'Load Data' to load the pre-calibrated audio")
    
    def load_single_ch_diff_data(self):
        """Select single-channel differential file - stores path only, doesn't load data"""
        file_path = filedialog.askopenfilename(
            title="Select Single-Channel Differential Recording",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            self.source_file_path = file_path
            self.source_file_type = '1ch_diff'
            self.file_name = os.path.basename(file_path)
            self.calibration_file_path = None  # Clear any calibration file
            self.update_file_info()
            print(f"Selected single-channel differential data: {file_path}")
            print("Click 'Load Data' to load the differential recording")
    
    def load_ni_source_data(self):
        """Select NI source file - stores path only, doesn't load data"""
        file_path = filedialog.askopenfilename(
            title="Select NI Log File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.source_file_path = file_path
            self.source_file_type = 'ni_source'
            self.file_name = os.path.basename(file_path).replace('log_', '').replace('.txt', '')
            self.calibration_file_path = None  # Clear any calibration file
            self.update_file_info()
            print(f"Selected NI source data: {file_path}")
            print("Click 'Load Data' to load from NI source")
    
    def load_ni_source_data_old(self):
        """OLD VERSION - Load data from NI source (.bin file with .txt logfile)"""
        file_path = filedialog.askopenfilename(
            title="Select NI Log File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Extract filename from path
                self.file_name = os.path.basename(file_path).replace('log_', '').replace('.txt', '')
                
                # Parse log file
                with open(file_path, 'r') as file:
                    log_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in file.readlines()}

                # Get base filepath for data files
                base_filepath = file_path.split('log_')[0] + file_path.split('log_')[-1].split('.')[0]
                feather_filepath = base_filepath + '.feather'
                parquet_filepath = base_filepath + '.parquet'
                bin_filepath = base_filepath + '.bin'

                # Extract parameters
                if "Sample_Rate" in log_data:
                    sample_rate = int(log_data["Sample_Rate"])
                elif "Sample Rate" in log_data:
                    sample_rate = int(log_data["Sample Rate"])

                # Get time window from GUI
                start_time = float(self.start_time_var.get()) if self.start_time_var.get().strip() else 0.0
                end_time = float(self.end_time_var.get()) if self.end_time_var.get().strip() else 0.0
                
                start_sample = int(start_time * sample_rate)
                end_sample = None if end_time == 0 else int(end_time * sample_rate)

                # Determine number of channels from log
                if "N_Input_Channels" in log_data:
                    n_channels = int(log_data["N_Input_Channels"])
                elif "Input_Channels" in log_data:
                    n_channels = len(log_data["Input_Channels"].split(","))
                elif "Number of Input Channels" in log_data:
                    n_channels = int(log_data["Number of Input Channels"])
                else:
                    n_channels = 1  # fallback

                # # For NI source, only accept single-channel differential data
                # if n_channels > 1:
                #     messagebox.showerror("Error", f"NI source with {n_channels} channels not supported.\nOnly single-channel differential data is supported for NI source.")
                #     return

                n_cols = n_channels + 1  # time_ms + channels

                # Load data from available file format
                data = None
                if os.path.exists(bin_filepath):
                    # Load from .bin file
                    if end_sample is not None:
                        n_samples = end_sample - start_sample
                        offset = start_sample * n_cols * 8  # float64 = 8 bytes
                        with open(bin_filepath, 'rb') as f:
                            f.seek(offset)
                            raw_data = np.fromfile(f, dtype='f8', count=n_samples * n_cols)
                    else:
                        offset = start_sample * n_cols * 8
                        with open(bin_filepath, 'rb') as f:
                            f.seek(offset)
                            raw_data = np.fromfile(f, dtype='f8')
                    reshaped = raw_data.reshape(-1, n_cols)
                    columns = ['time_ms'] + [f'ch{i+1}' for i in range(n_channels)]
                    data = pd.DataFrame(reshaped, columns=columns)

                elif os.path.exists(feather_filepath):
                    try:
                        import pyarrow.feather as feather
                        table = feather.read_table(feather_filepath)
                        if end_sample is None:
                            table = table.slice(start_sample)
                        else:
                            table = table.slice(start_sample, end_sample - start_sample)
                        data = table.to_pandas()
                    except ImportError:
                        raise ImportError("pyarrow not available for .feather file reading")

                elif os.path.exists(parquet_filepath):
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(parquet_filepath)
                        if end_sample is None:
                            table = table.slice(start_sample)
                        else:
                            table = table.slice(start_sample, end_sample - start_sample)
                        data = table.to_pandas()
                    except ImportError:
                        raise ImportError("pyarrow not available for .parquet file reading")

                else:
                    raise FileNotFoundError("Expected data file (.bin, .feather, or .parquet) not found.")

                # Convert to audio format (remove time column, keep only channels)
                channel_columns = [col for col in data.columns if col.startswith('ch')]
                audio_data = data[channel_columns].values
                
                # Set data
                self.raw_data = audio_data
                self.sample_rate = sample_rate
                self.calibrated_data = audio_data.copy()  # NI data doesn't need calibration
                
                # Store metadata for reloading
                self.source_file_path = file_path
                self.source_file_type = 'ni_source'
                self.calibration_factors = None
                self.calibration_file_path = None
                self.full_data = None  # NI source: don't cache, reload on demand
                
                actual_end = (len(audio_data) / sample_rate) + start_time if end_time == 0 else end_time
                self.loaded_time_window = {'start_sec': start_time, 'end_sec': actual_end}
                
                # Set data source to single-channel differential (NI source assumption)
                self.data_source = '1ch_diff'
                
                self.update_file_info()
                print(f"Loaded NI source data: {file_path}")
                print(f"Shape: {self.calibrated_data.shape}, Sample rate: {self.sample_rate}")
                print(f"Channels: {n_channels}, Time window: {start_time}-{end_time}s")
                
                # Plot the data immediately
                self.data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load NI source data:\n{str(e)}")
                print(f"Full error: {e}")
    
    def load_data(self):
        """Unified data loading function - loads data based on selected file type and time window"""
        if self.source_file_path is None:
            messagebox.showwarning("Warning", "Please select a data file first")
            return
        
        try:
            # Get time window parameters
            try:
                start_sec = float(self.start_time_var.get().strip()) if self.start_time_var.get().strip() else 0.0
                end_sec = float(self.end_time_var.get().strip()) if self.end_time_var.get().strip() else 0.0
            except ValueError:
                messagebox.showerror("Error", "Invalid time window values")
                return
            
            print(f"Loading data of type: {self.source_file_type}")
            
            # Load based on file type
            if self.source_file_type == 'ni_source':
                success = self._load_ni_source_data(start_sec, end_sec)
            else:
                # For audio files (audio, precalibrated, 1ch_diff)
                success = self._load_audio_data(start_sec, end_sec)
            
            if success:
                messagebox.showinfo("Success", f"Data loaded successfully\nTime window: {start_sec:.2f}s - {self.loaded_time_window['end_sec']:.2f}s")
                self.plot_data()
            else:
                messagebox.showerror("Error", "Failed to load data")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data:\n{str(e)}")
            print(f"Full error: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_audio_data(self, start_sec, end_sec):
        """Load audio data (handles audio, precalibrated, and 1ch_diff types)"""
        try:
            # Load full audio file
            audio_data, sample_rate = aio.load_audio(self.source_file_path)
            
            self.raw_data = audio_data
            self.sample_rate = sample_rate
            
            # Cache full data if reasonable size
            data_size_gb = audio_data.nbytes / (1024**3)
            if data_size_gb < 10:
                self.full_data = audio_data.copy()
                print(f"Cached full data in memory ({data_size_gb:.2f} GB)")
            else:
                self.full_data = None
                print(f"Data too large to cache ({data_size_gb:.2f} GB)")
            
            # Apply calibration if needed (for 'audio' type with calibration file)
            if self.source_file_type == 'audio' and self.calibration_file_path is not None:
                # Load calibration factors
                self.calibration_factors = np.array(pd.read_csv(self.calibration_file_path))
                print(f"Loaded calibration: {self.calibration_file_path}")
                
                # Apply calibration
                self.calibrated_data = audio_data.copy()
                n_channels = self.calibrated_data.shape[1]
                for i in range(n_channels):
                    self.calibrated_data[:, i] *= self.calibration_factors[i, 1]
                
                # Also calibrate full_data cache
                if self.full_data is not None:
                    for i in range(n_channels):
                        self.full_data[:, i] *= self.calibration_factors[i, 1]
                
                self.data_source = 'multich_linear'
                print("Applied calibration to audio data")
                
            else:
                # No calibration needed (precalibrated or 1ch_diff)
                self.calibrated_data = audio_data.copy()
                self.calibration_factors = None
                
                # Set data source based on type
                if self.source_file_type == '1ch_diff':
                    self.data_source = '1ch_diff'
                else:
                    self.data_source = 'multich_linear'
            
            # Subset to requested time window
            total_duration = len(self.calibrated_data) / self.sample_rate
            if end_sec <= 0 or end_sec > total_duration:
                end_sec = total_duration
            
            if start_sec > 0 or end_sec < total_duration:
                start_idx = int(start_sec * self.sample_rate)
                end_idx = int(end_sec * self.sample_rate)
                self.calibrated_data = self.calibrated_data[start_idx:end_idx, :]
            
            self.loaded_time_window = {'start_sec': start_sec, 'end_sec': end_sec}
            
            self.update_file_info()
            print(f"Loaded audio data: {self.source_file_path}")
            print(f"Shape: {self.calibrated_data.shape}, Sample rate: {self.sample_rate}")
            print(f"Time window: {start_sec:.2f}s - {end_sec:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"Error loading audio data: {e}")
            messagebox.showerror("Error", f"Failed to load audio data:\n{str(e)}")
            return False
    
    def _load_ni_source_data(self, start_sec, end_sec):
        """Load NI source data for specified time window"""
        try:
            file_path = self.source_file_path
            
            # Parse log file
            with open(file_path, 'r') as file:
                log_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in file.readlines()}

            # Get base filepath for data files
            base_filepath = file_path.split('log_')[0] + file_path.split('log_')[-1].split('.')[0]
            feather_filepath = base_filepath + '.feather'
            parquet_filepath = base_filepath + '.parquet'
            bin_filepath = base_filepath + '.bin'

            # Extract parameters
            if "Sample_Rate" in log_data:
                sample_rate = int(log_data["Sample_Rate"])
            elif "Sample Rate" in log_data:
                sample_rate = int(log_data["Sample Rate"])

            self.sample_rate = sample_rate
            
            start_sample = int(start_sec * sample_rate)
            end_sample = None if end_sec == 0 else int(end_sec * sample_rate)

            # Determine number of channels from log
            if "N_Input_Channels" in log_data:
                n_channels = int(log_data["N_Input_Channels"])
            elif "Input_Channels" in log_data:
                n_channels = len(log_data["Input_Channels"].split(","))
            elif "Number of Input Channels" in log_data:
                n_channels = int(log_data["Number of Input Channels"])
            else:
                n_channels = 1

            n_cols = n_channels + 1  # time_ms + channels

            # Load data from available file format
            data = None
            if os.path.exists(bin_filepath):
                if end_sample is not None:
                    n_samples = end_sample - start_sample
                    offset = start_sample * n_cols * 8
                    with open(bin_filepath, 'rb') as f:
                        f.seek(offset)
                        raw_data = np.fromfile(f, dtype='f8', count=n_samples * n_cols)
                else:
                    offset = start_sample * n_cols * 8
                    with open(bin_filepath, 'rb') as f:
                        f.seek(offset)
                        raw_data = np.fromfile(f, dtype='f8')
                reshaped = raw_data.reshape(-1, n_cols)
                columns = ['time_ms'] + [f'ch{i+1}' for i in range(n_channels)]
                data = pd.DataFrame(reshaped, columns=columns)

            elif os.path.exists(feather_filepath):
                try:
                    import pyarrow.feather as feather
                    table = feather.read_table(feather_filepath)
                    if end_sample is None:
                        table = table.slice(start_sample)
                    else:
                        table = table.slice(start_sample, end_sample - start_sample)
                    data = table.to_pandas()
                except ImportError:
                    raise ImportError("pyarrow not available for .feather file reading")

            elif os.path.exists(parquet_filepath):
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(parquet_filepath)
                    if end_sample is None:
                        table = table.slice(start_sample)
                    else:
                        table = table.slice(start_sample, end_sample - start_sample)
                    data = table.to_pandas()
                except ImportError:
                    raise ImportError("pyarrow not available for .parquet file reading")

            else:
                raise FileNotFoundError("Expected data file (.bin, .feather, or .parquet) not found.")

            # Convert to audio format
            channel_columns = [col for col in data.columns if col.startswith('ch')]
            audio_data = data[channel_columns].values
            
            self.raw_data = audio_data
            self.calibrated_data = audio_data.copy()
            self.calibration_factors = None
            self.full_data = None  # NI source: don't cache, reload on demand
            
            actual_end = (len(audio_data) / sample_rate) + start_sec if end_sec == 0 else end_sec
            self.loaded_time_window = {'start_sec': start_sec, 'end_sec': actual_end}
            
            self.data_source = '1ch_diff'
            
            self.update_file_info()
            print(f"Loaded NI source data: {file_path}")
            print(f"Shape: {self.calibrated_data.shape}, Sample rate: {self.sample_rate}")
            print(f"Time window: {start_sec:.2f}s - {actual_end:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"Error loading NI source: {e}")
            messagebox.showerror("Error", f"Failed to load NI source:\n{str(e)}")
            return False
    
    def update_file_info(self):
        info = ""
        
        # Show selected file info
        if self.source_file_path is not None:
            info += f"Selected: {self.file_name}\n"
            
            # Show file type
            if self.source_file_type == 'audio':
                info += "Type: Audio (needs calibration)\n"
                if self.calibration_file_path is not None:
                    cal_filename = os.path.basename(self.calibration_file_path)
                    info += f"Calibration: {cal_filename}\n"
                else:
                    info += "Calibration: Not selected\n"
            elif self.source_file_type == 'precalibrated':
                info += "Type: Pre-calibrated audio\n"
            elif self.source_file_type == '1ch_diff':
                info += "Type: Single-channel differential\n"
            elif self.source_file_type == 'ni_source':
                info += "Type: NI source (binary)\n"
        
        # Show loaded data info if available
        if self.raw_data is not None:
            info += f"\nLoaded: {self.raw_data.shape} @ {self.sample_rate} Hz\n"
            
            if self.loaded_time_window is not None:
                start = self.loaded_time_window['start_sec']
                end = self.loaded_time_window['end_sec']
                info += f"Time window: {start:.2f}s - {end:.2f}s\n"
            
            # Show data source type
            if self.data_source == 'multich_linear':
                source_desc = "Multi-channel linear"
            elif self.data_source == '1ch_diff':
                source_desc = "Single-channel differential"
            else:
                source_desc = self.data_source
            info += f"Source: {source_desc}"
        elif self.source_file_path is not None:
            info += "\n[Click 'Load Data' to load]"
        
        self.file_info.config(text=info)
    
    def on_ml_toggle(self):
        """Handle ML checkbox toggle"""
        if self.use_ml_filtering.get() and not self.loaded_classifier:
            # If enabling ML but no classifier loaded, prompt to load one
            self.select_classifier_file()
        
        # Update GUI state
        self.update_ml_status()
    
    def select_classifier_file(self):
        """Select and load ML classifier file"""
        if not ML_AVAILABLE:
            messagebox.showerror("Error", "scikit-learn is not available. Cannot load ML classifier.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Trained Classifier File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_classifier(file_path)
    
    def load_classifier(self, file_path):
        """Load the ML classifier from file"""
        try:
            print(f"Loading ML classifier from: {file_path}")
            with open(file_path, 'rb') as f:
                classifier_data = pickle.load(f)
            
            self.loaded_classifier = classifier_data['classifier']
            self.loaded_scaler = classifier_data['scaler']
            self.classifier_name = classifier_data['classifier_name']
            training_accuracy = classifier_data['accuracy']
            
            self.classifier_path.set(file_path)
            
            print(f"Loaded classifier: {self.classifier_name}")
            print(f"Training accuracy: {training_accuracy:.3f}")
            
            self.update_ml_status()
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"Loaded classifier: {self.classifier_name}\n"
                              f"Training accuracy: {training_accuracy:.3f}")
            
        except Exception as e:
            print(f"Error loading classifier: {e}")
            messagebox.showerror("Error", f"Failed to load classifier:\n{str(e)}")
            self.loaded_classifier = None
            self.loaded_scaler = None
            self.classifier_name = None
            self.use_ml_filtering.set(False)
            self.update_ml_status()
    
    def update_ml_status(self):
        """Update ML status display"""
        if self.loaded_classifier and self.loaded_scaler:
            # Show classifier name (abbreviated for compact display)
            display_name = self.classifier_name[:12] + "..." if self.classifier_name and len(self.classifier_name) > 12 else self.classifier_name
            if not display_name:
                display_name = "Loaded"
            self.classifier_label.configure(text=display_name, foreground="darkgreen")
        else:
            # Show no classifier
            if ML_AVAILABLE:
                self.classifier_label.configure(text="None", foreground="gray")
            else:
                self.classifier_label.configure(text="sklearn N/A", foreground="red")
                self.use_ml_filtering.set(False)
    
    def calibrate_data(self):
        """Apply calibration exactly like Script 03"""
        if self.raw_data is None or self.calibration_factors is None:
            return
            
        self.calibrated_data = self.raw_data.copy()
        n_channels = self.calibrated_data.shape[1]
        
        # Apply calibration factors (Script 03 logic)
        for i in range(n_channels):
            self.calibrated_data[:, i] *= self.calibration_factors[i, 1]
        
        # Also update full_data cache if it exists
        if self.full_data is not None:
            for i in range(n_channels):
                self.full_data[:, i] *= self.calibration_factors[i, 1]
        
        # Set data source to multi-channel linear (calibrated data)
        self.data_source = 'multich_linear'
            
        print("Data calibrated successfully")
        
        # Plot the calibrated data
        self.plot_data()
    
    def plot_data(self):
        """Plot the loaded data without pulse detection"""
        if self.calibrated_data is None:
            return
            
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get time window parameters
            try:
                start_sec = float(self.start_time_var.get().strip()) if self.start_time_var.get().strip() else 0.0
                # end_sec = float(self.end_time_var.get().strip()) if self.end_time_var.get().strip() else 0.0
            except ValueError:
                start_sec = 0.0
                # end_sec = 0.0
            
            # Get data for plotting
            data = self.calibrated_data
            rate = self.sample_rate
            
            # Handle single-channel differential data
            if self.data_source == '1ch_diff':
                # For single-channel differential, use only the first column
                data = data[:, 0:1]  # Keep as 2D array with shape (samples, 1)
                n_channels = 1
            else:   
                n_channels = data.shape[1]
            
            # # Apply time windowing if specified
            # if start_sec > 0 or end_sec > 0:
            #     total_duration = len(data) / rate
            #     if end_sec <= 0:
            #         end_sec = total_duration
                    
            #     if start_sec < 0:
            #         start_sec = 0
                    
            #     if start_sec >= end_sec:
            #         start_sec = 0
            #         end_sec = total_duration
                    
            #     if end_sec > total_duration:
            #         end_sec = total_duration
                
            #     start_idx = int(start_sec * rate)
            #     end_idx = int(end_sec * rate)
            #     data = data[start_idx:end_idx, :]
            # else:
            #     start_sec = 0
            #     end_sec = len(data) / rate
            
            # Plot the data using the current plot mode
            if self.plot_mode.get() == "single_ended":
                self.plot_data_single_ended(data, rate, start_sec, n_channels)
            else:
                self.plot_data_differential(data, rate, start_sec, n_channels)
                
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error plotting data: {e}")
    
    def plot_data_differential(self, data, rate, start_sec, n_channels):
        """Plot differential signals without pulse markers"""
        
        # Handle different data source types
        if self.data_source == '1ch_diff':
            # For single-channel differential, data is already differential
            data_diff = data[:, 0]
            
            # Downsample for plotting
            step = max(1, len(data_diff) // 1500000)
            x_coords = np.arange(0, len(data_diff), step)
            
            # Plot differential signal
            self.ax.plot((x_coords / rate) + start_sec, 
                        data_diff[::step], 
                        linewidth=0.5, label='Diff Ch')
            
            self.ax.set_ylim(bottom=np.min(data_diff)*1.2, top=np.max(data_diff)*1.2)
            source_desc = "Single-Ch Diff"
        else:
            data_diff = np.diff(data, axis=1)
            # Downsample for plotting
            step = max(1, len(data_diff[:,0]) // 1500000)
            x_coords = np.arange(0, len(data_diff[:,0]), step)

            offset_diff = np.max(np.abs(data_diff)) * 1.2

            for ch in range(n_channels - 1):  # Differential channels
                # Create differential signal for this channel pair
                # data_diff = np.diff(data[:, i:i+2], axis=1).flatten()
                
                # # Downsample for plotting
                # step = max(1, len(data_diff[:,i]) // 1500000)
                # x_coords = np.arange(0, len(data_diff), step)
                
                # Plot differential signal
                self.ax.plot((x_coords / rate) + start_sec, 
                           data_diff[::step, ch] + ((ch + 0.5) * offset_diff), 
                           linewidth=0.5, label=f'Ch{ch}-{ch+1}')
            
            # self.ax.set_ylim(bottom=np.min(data_diff)*1.2, top=(n_channels-1.5)*offset_diff)
            self.ax.set_ylim(bottom=-0.5*offset_diff, top=(n_channels-0.5)*offset_diff)
            source_desc = "Multi-Ch Diff"
        
        # Format plot
        title = f'EOD Data ({source_desc}) - Ready for Pulse Detection'
        if self.file_name:
            title = f'{self.file_name} - {title}'
        self.ax.set_title(title)
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage')
        self.ax.grid(True, alpha=0.3)
    
    def plot_data_single_ended(self, data, rate, start_sec, n_channels):
        """Plot single-ended signals without pulse markers"""
        
        # Handle single-channel differential data
        if self.data_source == '1ch_diff':
            # For single-channel differential data, show a message that single-ended view is not applicable
            self.ax.text(0.5, 0.5, 'Single-Ended view not applicable for\nSingle-Channel Differential data.\n\nPlease use Differential view mode.', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=16, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            self.ax.set_title('Single-Ended View Not Applicable')
            return
        
        # Calculate offset for stacking channels
        offset_se = np.max(abs(data)) * 1.2
        
        # Downsample for plotting
        step = max(1, len(data) // 1500000)
        x_coords = np.arange(0, len(data), step)
        
        # Plot all single-ended channels
        for ch in range(n_channels):
            self.ax.plot((x_coords / rate) + start_sec, 
                        data[::step, ch] + ch * offset_se, 
                        linewidth=0.5, label=f'SE Ch{ch}')
        
        # Format plot
        self.ax.set_ylim(bottom=-0.5*offset_se, top=(n_channels-0.5)*offset_se)
        title = f'EOD Data (Single-Ended) - Ready for Pulse Detection'
        if self.file_name:
            title = f'{self.file_name} - {title}'
        self.ax.set_title(title)
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage')
        self.ax.grid(True, alpha=0.3)
        
    def update_parameters(self):
        """Update parameters from GUI inputs"""
        try:
            for param, var in self.param_vars.items():
                value_str = var.get().strip()
                
                if param in ['save_filtered_out', 'return_diff']:
                    self.parameters[param] = value_str.lower() in ['true', '1', 'yes', 'on']
                elif param in ['interp_factor', 'min_width_us', 'max_width_us', 'peak_fft_freq_min', 'peak_fft_freq_max', 'length']:
                    self.parameters[param] = int(float(value_str))
                else:
                    self.parameters[param] = float(value_str)
                    
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Invalid parameter value: {str(e)}")
            return False
        return True
    
    def detect_pulses(self):
        """Main pulse detection function - separate from plotting"""
        if self.calibrated_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        if not self.update_parameters():
            return
            
        try:
            # Get time window parameters
            try:
                start_sec = float(self.start_time_var.get().strip())
                end_sec = float(self.end_time_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Invalid time window values")
                return
            
            # Get data for detection
            data = self.calibrated_data
            rate = self.sample_rate
            
            # Handle single-channel differential data
            if self.data_source == '1ch_diff':
                # For single-channel differential, use only the first column
                data = data[:, 0:1]  # Keep as 2D array with shape (samples, 1)
                n_channels = 1
            else:   
                n_channels = data.shape[1]

            # Adjust time window if needed
            if end_sec <= 0:
                end_sec = len(data) / rate
            if start_sec < 0:
                start_sec = 0
            
            # # Apply time windowing if specified
            # if start_sec > 0 or end_sec > 0:
            #     total_duration = len(data) / rate
            #     if end_sec <= 0:
            #         end_sec = total_duration
                    
            #     if start_sec < 0:
            #         start_sec = 0
                    
            #     if start_sec >= end_sec:
            #         messagebox.showerror("Error", "Start time must be less than end time")
            #         return
                    
            #     if end_sec > total_duration:
            #         end_sec = total_duration
            #         messagebox.showwarning("Warning", f"End time adjusted to file duration: {end_sec:.2f}s")
                
            #     start_idx = int(start_sec * rate)
            #     end_idx = int(end_sec * rate)
            #     data = data[start_idx:end_idx, :]
                
            #     print(f"Using time window: {start_sec:.2f}s to {end_sec:.2f}s ({len(data)} samples)")
            # else:
            #     start_sec = 0
            #     end_sec = len(data) / rate
            #     print(f"Using full recording: {end_sec:.2f}s ({len(data)} samples)")
            
            print(f"Detecting pulses on {n_channels} channels...")
            
            # Detect pulses on each channel
            all_peaks = []
            all_troughs = []
            all_widths = []
            
            for i in range(n_channels):
                if self.use_bandpass_filter.get():
                    # Apply bandpass filter before detection
                    ch_data = bandpass_filter(
                        data[:, i], rate, 
                        lowcut=float(self.bandpass_lowcut.get()), 
                        highcut=float(self.bandpass_highcut.get()), 
                        order=int(self.bandpass_order.get())
                    )
                else:
                    ch_data = data[:, i]

                ch_peaks, ch_troughs, _, ch_pulse_widths = pulses.detect_pulses(
                    ch_data, 
                    rate,
                    thresh=self.parameters['thresh'], 
                    min_rel_slope_diff=self.parameters['min_rel_slope_diff'],
                    min_width=self.parameters['min_width_us'] / 1e6,
                    max_width=self.parameters['max_width_us'] / 1e6,
                    width_fac=self.parameters['width_fac_detection'],
                    verbose=0,
                    return_data=False
                )
                all_peaks.append(ch_peaks)
                all_troughs.append(ch_troughs)
                all_widths.append(ch_pulse_widths)
            
            if self.data_source == 'multich_linear':
                # Unify pulses across channels
                unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
                    all_peaks, all_troughs, all_widths)
                
                print(f"    Detected {len(unique_midpoints)} unified pulses across channels")
                # Create unique pulses list
                unique_pulses = list(zip(unique_midpoints, unique_peaks, unique_troughs, unique_widths))
            else:
                # For single-channel differential, just use detected pulses from the single channel
                unique_pulses = list(zip(
                    [(p + t) // 2 for p, t in zip(all_peaks[0], all_troughs[0])],
                    all_peaks[0],
                    all_troughs[0],
                    all_widths[0]
                ))
                
                print(f"    Detected {len(unique_pulses)} pulses on single-channel differential data")
            
            # Process pulses even if none detected - for analysis purposes
            if len(unique_pulses) > 0:
                # Extract pulse snippets
                midpoints = np.array([p[0] for p in unique_pulses])
                peaks = np.array([p[1] for p in unique_pulses])
                troughs = np.array([p[2] for p in unique_pulses])
                widths = np.array([p[3] for p in unique_pulses])
                
                # Extract waveforms
                (
                    eod_waveforms, eod_amps, eod_widths, eod_chan, is_differential,
                    snippet_p1_idc, snippet_p2_idc, final_p1_idc, final_p2_idc, 
                    pulse_orientation, amplitude_ratios, fft_peak_freqs, pulse_locations
                ) = extract_pulse_snippets(
                    data, peaks, troughs, rate = rate, length = self.parameters['length'], 
                    source=self.data_source, return_differential=self.parameters['return_diff']
                )

                print(f"    Filtering for differential pulses...{len(unique_pulses)} total, {np.sum(is_differential)} differential")

                # Filter clipped and low-quality pulses
                clip_thresh = self.clip_threshold.get()
                top_pct = self.top_percent.get()
                
                # # Filter kept pulses by clipping threshold
                # kept_waveforms = eod_waveforms[keep_indices]
                # kept_amps = eod_amps[keep_indices]
                # kept_widths = eod_widths[keep_indices]
                # kept_ratios = amplitude_ratios[keep_indices]
                # kept_freqs = fft_peak_freqs[keep_indices]
                
                # Index non-clipped pulses (peak-trough amplitude < threshold)
                non_clipped_mask = eod_amps < clip_thresh
                if not np.any(non_clipped_mask):
                    messagebox.showwarning("Warning", f"No pulses below clipping threshold ({clip_thresh})")
                    return

                # Convert boolean mask to integer indices for indexing
                non_clipped_indices = np.where(non_clipped_mask)[0]
                
                # Convert to numpy arrays if needed for proper indexing
                eod_waveforms = np.array(eod_waveforms)[non_clipped_indices]
                eod_amps = np.array(eod_amps)[non_clipped_indices]
                eod_widths = np.array(eod_widths)[non_clipped_indices]
                eod_chan = np.array(eod_chan)[non_clipped_indices]
                is_differential = np.array(is_differential)[non_clipped_indices]
                amplitude_ratios = np.array(amplitude_ratios)[non_clipped_indices]
                fft_peak_freqs = np.array(fft_peak_freqs)[non_clipped_indices]
                final_p1_idc = np.array(final_p1_idc)[non_clipped_indices]
                final_p2_idc = np.array(final_p2_idc)[non_clipped_indices]
                snippet_p1_idc = np.array(snippet_p1_idc)[non_clipped_indices]
                snippet_p2_idc = np.array(snippet_p2_idc)[non_clipped_indices]
                pulse_orientation = np.array(pulse_orientation)[non_clipped_indices]
                pulse_locations = np.array(pulse_locations)[non_clipped_indices]

                # Select top x% by amplitude
                n_top = max(1, int(len(eod_amps) * top_pct / 100.0))
                top_indices = np.argsort(eod_amps)[-n_top:]

                eod_waveforms = eod_waveforms[top_indices]
                eod_amps = eod_amps[top_indices]
                eod_widths = eod_widths[top_indices]
                eod_chan = eod_chan[top_indices]
                is_differential = is_differential[top_indices]
                amplitude_ratios = amplitude_ratios[top_indices]
                fft_peak_freqs = fft_peak_freqs[top_indices]
                final_p1_idc = final_p1_idc[top_indices]
                final_p2_idc = final_p2_idc[top_indices]
                snippet_p1_idc = np.array(snippet_p1_idc)[top_indices]
                snippet_p2_idc = np.array(snippet_p2_idc)[top_indices]
                pulse_orientation = np.array(pulse_orientation)[top_indices]
                pulse_locations = pulse_locations[top_indices]

                print(f"Analyzing {len(eod_waveforms)} pulses (top {top_pct}% below clip threshold {clip_thresh})")

                # Apply filtering (with optional ML enhancement)
                if (self.use_ml_filtering.get() and 
                    self.loaded_classifier is not None and 
                    self.loaded_scaler is not None):
                    # Use enhanced ML-based filtering
                    print(f"    Using ML-enhanced filtering with {self.classifier_name}")
                    keep_indices = filter_waveforms_with_classifier(
                        eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate,
                        classifier=self.loaded_classifier,
                        scaler=self.loaded_scaler,
                        dur_min=self.parameters['min_width_us'], 
                        dur_max=self.parameters['max_width_us'],
                        pp_r_min=self.parameters['amplitude_ratio_min'], 
                        pp_r_max=self.parameters['amplitude_ratio_max'],
                        fft_freq_min=self.parameters['peak_fft_freq_min'], 
                        fft_freq_max=self.parameters['peak_fft_freq_max'],
                        fish_probability_threshold=self.fish_probability_threshold.get(),
                        use_basic_filtering=True
                    )
                else:
                    # Use basic threshold filtering
                    keep_indices = filter_waveforms(
                        eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate,
                        dur_min=self.parameters['min_width_us'], 
                        dur_max=self.parameters['max_width_us'],
                        pp_r_min=self.parameters['amplitude_ratio_min'], 
                        pp_r_max=self.parameters['amplitude_ratio_max'],
                        fft_freq_min=self.parameters['peak_fft_freq_min'], 
                        fft_freq_max=self.parameters['peak_fft_freq_max']
                    )
                
                # Ensure indices are 1D arrays (flatten in case they're not)
                keep_indices = np.array(keep_indices).flatten()
                all_indices = np.arange(len(eod_waveforms))
                filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
                
                print(f"    Filtering result: {len(keep_indices)} kept, {len(filtered_out_indices)} filtered out of {len(eod_waveforms)} total")
                
                # Split data into kept and filtered_out sets
                kept_data = {
                    'eod_waveforms': eod_waveforms[keep_indices],
                    'eod_amps': eod_amps[keep_indices], 
                    'eod_widths': eod_widths[keep_indices],
                    'amplitude_ratios': amplitude_ratios[keep_indices],
                    'fft_peak_freqs': fft_peak_freqs[keep_indices],
                    'final_p1_idc': final_p1_idc[keep_indices],
                    'final_p2_idc': final_p2_idc[keep_indices],
                    'snippet_p1_idc': snippet_p1_idc[keep_indices],
                    'snippet_p2_idc': snippet_p2_idc[keep_indices],
                    'pulse_orientation': pulse_orientation[keep_indices],
                    'eod_chan': eod_chan[keep_indices],
                    'pulse_location': pulse_locations[keep_indices]
                }
                
                filtered_out_data = {
                    'eod_chan': eod_chan[filtered_out_indices],
                    'final_p1_idc': final_p1_idc[filtered_out_indices],
                    'final_p2_idc': final_p2_idc[filtered_out_indices]
                }
                
                # Store pulse data for analysis (only kept pulses)
                self.current_pulse_data = {
                    'eod_waveforms': kept_data['eod_waveforms'],
                    'eod_amps': kept_data['eod_amps'], 
                    'eod_widths': kept_data['eod_widths'],
                    'amplitude_ratios': kept_data['amplitude_ratios'],
                    'fft_peak_freqs': kept_data['fft_peak_freqs'],
                    'sample_rate': rate,
                    'final_p1_idc': kept_data['final_p1_idc'],
                    'final_p2_idc': kept_data['final_p2_idc'],
                    'snippet_p1_idc': kept_data['snippet_p1_idc'],
                    'snippet_p2_idc': kept_data['snippet_p2_idc'],
                    'pulse_orientation': kept_data['pulse_orientation'],
                    'eod_chan': kept_data['eod_chan'],
                    'pulse_location': kept_data['pulse_location']
                }
                
                # Clear existing markers from previous detections
                # Find and remove all scatter plot markers (red, blue, grey circles)
                for artist in self.ax.get_children():
                    if hasattr(artist, 'get_marker') and artist.get_marker() == 'o':
                        artist.remove()
                
                # Plot with pulse markers (don't clear, just add markers)
                if self.plot_mode.get() == "single_ended":
                    self.add_pulse_markers_single_ended(data, rate, start_sec, n_channels, 
                                         kept_data, filtered_out_data)
                else:
                    self.add_pulse_markers_differential(data, rate, start_sec, n_channels, 
                                         kept_data, filtered_out_data)
                
                # Update results summary
                self.update_results_summary(len(unique_pulses), len(keep_indices), len(filtered_out_indices), start_sec, end_sec)
                
            else:
                # No pulses detected - just update summary with custom message
                print("    No pulses detected")
                
                # Update results summary with zero detections and custom message
                self.update_results_summary_no_pulses()
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error during pulse detection:\n{str(e)}")
            print(f"Full error: {e}")
            import traceback
            traceback.print_exc()

    def add_pulse_markers_differential(self, data, rate, start_sec, n_channels, kept_data, filtered_out_data):
        """Add pulse markers to existing differential plot without re-plotting the data"""
        
        # Handle different data source types
        if self.data_source == '1ch_diff':
            # For single-channel differential, data is already differential
            data_diff = data[:, 0]
            
            # Plot kept pulses (red=p1, blue=p2)
            for idx in range(len(kept_data['eod_chan'])):
                p1_idx = int(kept_data['final_p1_idc'][idx])
                p2_idx = int(kept_data['final_p2_idc'][idx])

                if p1_idx < len(data_diff):
                    self.ax.plot((p1_idx / rate) + start_sec, 
                                data_diff[p1_idx], 
                                'o', markersize=4, color='red')
                if p2_idx < len(data_diff):
                    self.ax.plot((p2_idx / rate) + start_sec, 
                                data_diff[p2_idx], 
                                'o', markersize=4, color='blue')

            # Plot filtered out pulses (grey)
            for idx in range(len(filtered_out_data['eod_chan'])):
                p1_idx = int(filtered_out_data['final_p1_idc'][idx])
                p2_idx = int(filtered_out_data['final_p2_idc'][idx])
                if p1_idx < len(data_diff):
                    self.ax.plot((p1_idx / rate) + start_sec, 
                                data_diff[p1_idx], 
                                'o', markersize=4, color='grey', alpha=0.6)
                if p2_idx < len(data_diff):
                    self.ax.plot((p2_idx / rate) + start_sec, 
                                data_diff[p2_idx], 
                                'o', markersize=4, color='grey', alpha=0.6)
                                            
        else:
            # Create differential data for multi-channel linear
            data_diff = np.diff(data, axis=1)
            # Calculate offset
            offset_diff = np.max(np.abs(data_diff)) * 1.2

            # Multi-channel linear data - create differential pairs
            for ch in range(n_channels - 1):  # Differential channels
                # Create differential signal for this channel pair
                # data_diff = np.diff(data[:, ch:ch+2], axis=1).flatten()
                
                # Plot kept pulses (red=p1, blue=p2) on this channel
                kept_ch_mask = (kept_data['eod_chan'] == ch)
                for idx in np.where(kept_ch_mask)[0]:
                    p1_idx = int(kept_data['final_p1_idc'][idx])
                    p2_idx = int(kept_data['final_p2_idc'][idx])
                    pulse_location = kept_data['pulse_location'][idx]
                    
                    if p1_idx < len(data_diff):
                        self.ax.plot((p1_idx / rate) + start_sec, 
                                   data_diff[p1_idx, ch] + ((ch + 0.5) * offset_diff), 
                                   'o', markersize=4, color='red')
                        
                        # Pulse location visualization
                        time_coord = (p1_idx / rate) + start_sec
                        # Draw thin line from channel offset to pulse_location offset
                        self.ax.plot([time_coord, time_coord], 
                                   [(ch + 0.5) * offset_diff, pulse_location * offset_diff], 
                                   'k-', linewidth=0.5, alpha=0.6)
                        # Mark pulse_location with small black marker
                        self.ax.plot(time_coord, 
                                   pulse_location * offset_diff, 
                                   'ko', markersize=2, alpha=0.8)
                    if p2_idx < len(data_diff):
                        self.ax.plot((p2_idx / rate) + start_sec, 
                                   data_diff[p2_idx, ch] + ((ch + 0.5) * offset_diff), 
                                   'o', markersize=4, color='blue')
                    
                    
                
                # Plot filtered out pulses (grey) on this channel
                filtered_ch_mask = (filtered_out_data['eod_chan'] == ch)
                for idx in np.where(filtered_ch_mask)[0]:
                    p1_idx = int(filtered_out_data['final_p1_idc'][idx])
                    p2_idx = int(filtered_out_data['final_p2_idc'][idx])
                    
                    if p1_idx < len(data_diff):
                        self.ax.plot((p1_idx / rate) + start_sec, 
                                   data_diff[p1_idx, ch] + ((ch + 0.5) * offset_diff), 
                                   'o', markersize=4, color='grey', alpha=0.6)
                    if p2_idx < len(data_diff):
                        self.ax.plot((p2_idx / rate) + start_sec, 
                                   data_diff[p2_idx, ch] + ((ch + 0.5) * offset_diff), 
                                   'o', markersize=4, color='grey', alpha=0.6)
        
        # Update title to show detection results
        n_kept = len(kept_data['eod_chan'])
        n_filtered = len(filtered_out_data['eod_chan'])
        total_detected = n_kept + n_filtered
        source_desc = "Single-Ch Diff" if self.data_source == '1ch_diff' else "Multi-Ch Diff"
        title = f'EOD Data ({source_desc}) - {n_kept} kept, {n_filtered} filtered out of {total_detected} detected'
        if self.file_name:
            title = f'{self.file_name} - {title}'
        self.ax.set_title(title)
    
    def add_pulse_markers_single_ended(self, data, rate, start_sec, n_channels, kept_data, filtered_out_data):
        """Add pulse markers to existing single-ended plot without re-plotting the data"""
        
        # Handle single-channel differential data
        if self.data_source == '1ch_diff':
            # Update title to show this view is not applicable
            self.ax.set_title('Single-Ended view not applicable for Single-Channel Differential data')
            return
        
        # Calculate offset for stacking channels (same as original plotting)
        offset_se = np.max(abs(data)) * 1.2
        
        # Mark peaks and troughs on the appropriate single-ended channels
        for diff_ch in range(n_channels - 1):
            se_ch1 = diff_ch      # First single-ended channel
            se_ch2 = diff_ch + 1  # Second single-ended channel
            
            # Plot kept pulses on both single-ended channels that form this differential pair
            kept_ch_mask = (kept_data['eod_chan'] == diff_ch)
            for idx in np.where(kept_ch_mask)[0]:
                p1_idx = int(kept_data['final_p1_idc'][idx])
                p2_idx = int(kept_data['final_p2_idc'][idx])
                
                # Mark on first single-ended channel (se_ch1)
                if p1_idx < len(data):
                    self.ax.plot((p1_idx / rate) + start_sec, 
                               data[p1_idx, se_ch1] + se_ch1 * offset_se, 
                               'o', markersize=5, color='red')
                if p2_idx < len(data):
                    self.ax.plot((p2_idx / rate) + start_sec, 
                               data[p2_idx, se_ch1] + se_ch1 * offset_se, 
                               'o', markersize=5, color='blue')
                
                # Mark on second single-ended channel (se_ch2)
                if p1_idx < len(data):
                    self.ax.plot((p1_idx / rate) + start_sec, 
                               data[p1_idx, se_ch2] + se_ch2 * offset_se, 
                               'o', markersize=5, color='red')
                if p2_idx < len(data):
                    self.ax.plot((p2_idx / rate) + start_sec, 
                               data[p2_idx, se_ch2] + se_ch2 * offset_se, 
                               'o', markersize=5, color='blue')
            
            # Plot filtered out pulses (grey) on both single-ended channels
            filtered_ch_mask = (filtered_out_data['eod_chan'] == diff_ch)
            for idx in np.where(filtered_ch_mask)[0]:
                p1_idx = int(filtered_out_data['final_p1_idc'][idx])
                p2_idx = int(filtered_out_data['final_p2_idc'][idx])
                
                # Mark on first single-ended channel (se_ch1)
                if p1_idx < len(data):
                    self.ax.plot((p1_idx / rate) + start_sec, 
                               data[p1_idx, se_ch1] + se_ch1 * offset_se, 
                               'o', markersize=5, color='grey', alpha=0.6)
                if p2_idx < len(data):
                    self.ax.plot((p2_idx / rate) + start_sec, 
                               data[p2_idx, se_ch1] + se_ch1 * offset_se, 
                               'o', markersize=5, color='grey', alpha=0.6)
                
                # Mark on second single-ended channel (se_ch2)
                if p1_idx < len(data):
                    self.ax.plot((p1_idx / rate) + start_sec, 
                               data[p1_idx, se_ch2] + se_ch2 * offset_se, 
                               'o', markersize=5, color='grey', alpha=0.6)
                if p2_idx < len(data):
                    self.ax.plot((p2_idx / rate) + start_sec, 
                               data[p2_idx, se_ch2] + se_ch2 * offset_se, 
                               'o', markersize=5, color='grey', alpha=0.6)
        
        # Update title to show detection results
        n_kept = len(kept_data['eod_chan'])
        n_filtered = len(filtered_out_data['eod_chan'])
        total_detected = n_kept + n_filtered
        title = f'EOD Data (Single-Ended) - {n_kept} kept, {n_filtered} filtered out of {total_detected} detected'
        if self.file_name:
            title = f'{self.file_name} - {title}'
        self.ax.set_title(title)

    def update_results_summary_no_pulses(self):
        """Update the results summary when no pulses are detected"""
        summary = f"""Detection Results:
==================
No pulses detected

Suggestion: Modify detection parameters or load different data

Current Parameters:
==================
"""
        for param, value in self.parameters.items():
            summary += f"{param}: {value}\n"
            
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, summary)
    
    def update_results_summary(self, total, kept, filtered_out, start_sec, end_sec):
        """Update results summary"""
        duration = end_sec - start_sec
        summary = f"""Detection Results:
==================
Time window: {start_sec:.2f}s - {end_sec:.2f}s ({duration:.2f}s)
Total detected: {total}
Kept after filtering: {kept}
Filtered out: {filtered_out}
Keep rate: {kept/total*100:.1f}%
Pulse rate: {kept/duration:.1f} pulses/sec

Current Parameters:
==================
"""
        for param, value in self.parameters.items():
            summary += f"{param}: {value}\n"
            
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, summary)
    
    def export_parameters(self):
        """Export current parameters to JSON"""
        file_path = filedialog.asksaveasfilename(
            title="Export Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if not self.update_parameters():
                    return
                    
                with open(file_path, 'w') as f:
                    json.dump(self.parameters, f, indent=2)
                messagebox.showinfo("Success", f"Parameters exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export parameters:\n{str(e)}")
    
    def import_parameters(self):
        """Import parameters from JSON"""
        file_path = filedialog.askopenfilename(
            title="Import Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    imported_params = json.load(f)
                
                # Update parameters and GUI
                self.parameters.update(imported_params)
                for param, value in self.parameters.items():
                    if param in self.param_vars:
                        self.param_vars[param].set(str(value))
                
                messagebox.showinfo("Success", f"Parameters imported from:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import parameters:\n{str(e)}")
    
    def save_as_audio(self):
        """Save the current calibrated data as an audio file"""
        if self.calibrated_data is None:
            messagebox.showwarning("Warning", "No data loaded to save")
            return
        
        # Open file dialog to select save location and filename
        file_path = filedialog.asksaveasfilename(
            title="Save Audio File",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Get time window parameters
                try:
                    start_sec = float(self.start_time_var.get().strip()) if self.start_time_var.get().strip() else 0.0
                    end_sec = float(self.end_time_var.get().strip()) if self.end_time_var.get().strip() else 0.0
                except ValueError:
                    start_sec = 0.0
                    end_sec = 0.0
                
                # Get data to save
                data_to_save = self.calibrated_data
                
                # # Apply time windowing if specified
                # if start_sec > 0 or end_sec > 0:
                #     total_duration = len(data_to_save) / self.sample_rate
                #     if end_sec <= 0:
                #         end_sec = total_duration
                    
                #     if start_sec < 0:
                #         start_sec = 0
                    
                #     if start_sec >= end_sec:
                #         start_sec = 0
                #         end_sec = total_duration
                    
                #     if end_sec > total_duration:
                #         end_sec = total_duration
                    
                #     start_idx = int(start_sec * self.sample_rate)
                #     end_idx = int(end_sec * self.sample_rate)
                #     data_to_save = data_to_save[start_idx:end_idx, :]
                
                # Save using audioio
                aio.write_audio(file_path, data_to_save, self.sample_rate)
                
                duration = len(data_to_save) / self.sample_rate
                messagebox.showinfo("Success", 
                    f"Audio saved successfully:\n{file_path}\n\n"
                    f"Duration: {duration:.2f}s\n"
                    f"Channels: {data_to_save.shape[1]}\n"
                    f"Sample rate: {self.sample_rate} Hz")
                
                print(f"Saved audio to: {file_path}")
                print(f"Shape: {data_to_save.shape}, Sample rate: {self.sample_rate}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save audio file:\n{str(e)}")
                print(f"Full error: {e}")
    
    def analyze_pulses(self):
        """Perform in-depth analysis of detected pulses with control recording style plots"""
        if self.current_pulse_data is None:
            messagebox.showwarning("Warning", "Please run pulse detection first")
            return
            
        try:
            # Extract stored data (only kept pulses)
            eod_waveforms = self.current_pulse_data['eod_waveforms']
            eod_amps = self.current_pulse_data['eod_amps']
            eod_widths = self.current_pulse_data['eod_widths']
            amplitude_ratios = self.current_pulse_data['amplitude_ratios']
            fft_peak_freqs = self.current_pulse_data['fft_peak_freqs']
            rate = self.current_pulse_data['sample_rate']
            snippet_p1_idc = self.current_pulse_data['snippet_p1_idc']
            snippet_p2_idc = self.current_pulse_data['snippet_p2_idc']
            pulse_orientation = self.current_pulse_data['pulse_orientation']

            if len(eod_waveforms) == 0:
                messagebox.showwarning("Warning", "No pulses passed filtering")
                return
            
            # Create analysis window
            self.create_analysis_window(eod_waveforms, eod_amps, eod_widths,
                                      amplitude_ratios, fft_peak_freqs, rate,
                                      snippet_p1_idc, snippet_p2_idc, pulse_orientation)

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during pulse analysis:\n{str(e)}")
            print(f"Full error: {e}")
            import traceback
            traceback.print_exc()
    
    def create_analysis_window(self, waveforms, amps, widths, ratios, freqs, rate, snippet_p1_idc, snippet_p2_idc, pulse_orientation):
        """Create separate window with detailed pulse analysis"""
        analysis_window = tk.Toplevel(self.root)
        window_title = "Pulse Analysis Results"
        if self.file_name:
            window_title = f"{self.file_name} - {window_title}"
        analysis_window.title(window_title)
        analysis_window.geometry("1400x1000")
        
        # Create main figure with subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 9))
        
        # Create canvas in analysis window FIRST
        canvas = FigureCanvasTkAgg(fig, analysis_window)
        
        # Create toolbar for the analysis window (connected to analysis canvas)
        toolbar_frame = ttk.Frame(analysis_window)
        toolbar_frame.pack(fill=tk.X)
        
        # Add label to distinguish this toolbar
        ttk.Label(toolbar_frame, text="Analysis Plot Tools:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(5, 10))
        
        analysis_toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        analysis_toolbar.update()
        
        # Pack canvas after toolbar
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Normalize waveforms for overlay plot with normalize_waveforms function
        normalized_waveforms = np.array(normalize_waveforms(waveforms, snippet_p1_idc, snippet_p2_idc))
        
        # Time axis for waveforms (centered around 0)
        n_samples = normalized_waveforms.shape[1]
        time_axis = (np.arange(n_samples) - n_samples//2) / rate * 1000  # in ms
        
        # 1. Overlay plot of normalized waveforms
        for i, wf in enumerate(normalized_waveforms):
            ax1.plot(time_axis, wf, 'b-', alpha=0.3, linewidth=0.5)
        
        # Mean waveform in red
        mean_waveform = np.mean(normalized_waveforms, axis=0)
        ax1.plot(time_axis, mean_waveform, 'r-', linewidth=2, label=f'Mean (n={len(waveforms)})')
        ax1.set_title('Normalized Waveforms Overlay')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Normalized Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Amplitude ratio histogram (moved from plot 3)
        ax2.hist(ratios, bins=20, alpha=0.7, edgecolor='black')
        mean_amp_ratio = np.mean(ratios)
        ax2.axvline(mean_amp_ratio, color='r', linestyle='dashed', linewidth=1)
        ax2.set_title(f'Amplitude Ratios\nMean: {mean_amp_ratio:.3f} ± {np.std(ratios):.3f}')
        ax2.set_xlabel('Peak/Trough Ratio')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Duration histogram (moved from plot 4)
        # durations_ms = widths * 1000  # Convert to milliseconds
        ax3.hist(widths, bins=20, alpha=0.7, edgecolor='black')
        mean_width = np.mean(widths)
        ax3.axvline(mean_width, color='r', linestyle='dashed', linewidth=1)
        ax3.set_title(f'Pulse Durations\nMean: {mean_width:.1f} ± {np.std(widths):.1f} us')
        ax3.set_xlabel('Duration (us)')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. FFT frequency histogram (moved from plot 5)
        ax4.hist(freqs, bins=20, alpha=0.7, edgecolor='black')
        mean_freq = np.mean(freqs)
        ax4.axvline(mean_freq, color='r', linestyle='dashed', linewidth=1)
        ax4.set_title(f'Peak FFT Frequencies\nMean: {mean_freq:.0f} ± {np.std(freqs):.0f} Hz')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # 5. Empty plot (placeholder)
        ax5.axis('off')
        
        # 6. Power Spectral Density plot (0-20000 Hz)
        # Calculate PSD using Welch's method for better frequency resolution
        
        # Calculate PSD for each waveform and average
        psd_list = []
        for wf in normalized_waveforms:
            freqs_psd, psd = signal.welch(wf, fs=rate, nperseg=min(len(wf), 1024))
            psd_list.append(psd)
        
        # Average PSD across all waveforms
        mean_psd = np.mean(psd_list, axis=0)

        # Plot only up to 100 kHz
        freq_mask = freqs_psd <= 100000
        for psd in psd_list:
            ax6.loglog(freqs_psd[freq_mask], psd[freq_mask], 'b-', alpha=0.3, linewidth=0.5)
        ax6.loglog(freqs_psd[freq_mask], mean_psd[freq_mask], 'r-', linewidth=1)
        ax6.set_title('Power Spectral Density (0-100 kHz)')
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Power Spectral Density')
        ax6.set_xlim(0, 100000)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add export controls
        export_frame = ttk.Frame(analysis_window)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Checkbox for waveform export (checked by default)
        export_waveforms_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_frame, text="Export Waveforms", variable=export_waveforms_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(export_frame, text="Export Analysis Results", 
                  command=lambda: self.export_analysis_results(amps, widths, ratios, freqs, mean_waveform, time_axis, 
                                                               waveforms, export_waveforms_var.get())).pack(side=tk.LEFT)
        
        # Add summary text
        summary_text = f"""Analysis Summary:
Analyzed Pulses: {len(waveforms)}
Clip Threshold: {self.clip_threshold.get():.4f}
Top Percent: {self.top_percent.get():.1f}%

Statistics:
- Amplitude: {np.mean(amps):.4f} ± {np.std(amps):.4f}
- Duration: {np.mean(widths):.1f} ± {np.std(widths):.1f} us
- Ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}
- Frequency: {np.mean(freqs):.0f} ± {np.std(freqs):.0f} Hz"""
        
        summary_label = ttk.Label(export_frame, text=summary_text, font=('Courier', 9))
        summary_label.pack(side=tk.RIGHT, padx=10)
        
        canvas.draw()
    
    def export_analysis_results(self, amps, widths, ratios, freqs, mean_waveform, time_axis, 
                               waveforms=None, export_waveforms=False):
        """Export analysis results to CSV files and optionally waveforms"""
        file_path = filedialog.asksaveasfilename(
            title="Export Analysis Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Export statistics
                stats_df = pd.DataFrame({
                    'amplitude': amps,
                    'duration_us': widths,
                    'amplitude_ratio': ratios,
                    'peak_fft_freq_hz': freqs
                })
                stats_df.to_csv(file_path, index=False)
                
                # Export mean waveform
                base_path = file_path.rsplit('.', 1)[0]
                mean_wf_path = base_path + '_mean_waveform.csv'
                mean_df = pd.DataFrame({
                    'time_ms': time_axis,
                    'normalized_amplitude': mean_waveform
                })
                mean_df.to_csv(mean_wf_path, index=False)
                
                exported_files = [file_path, mean_wf_path]
                
                # Export waveforms if requested
                if export_waveforms and waveforms is not None:
                    waveforms_path = base_path + '_waveforms.csv'
                    
                    # Create DataFrame with all waveforms
                    # Each column is one waveform, rows are time samples
                    waveforms_df = pd.DataFrame(waveforms.T)  # Transpose so each column is a waveform
                    waveforms_df.columns = [f'waveform_{i}' for i in range(len(waveforms))]
                    
                    # Add time axis as first column
                    time_ms = np.arange(len(waveforms[0])) / self.sample_rate * 1000
                    waveforms_df.insert(0, 'time_ms', time_ms)
                    
                    # Save to CSV
                    waveforms_df.to_csv(waveforms_path, index=False)
                    
                    exported_files.append(waveforms_path)
                    print(f"Exported {len(waveforms)} waveforms to: {waveforms_path}")
                
                file_list = "\n".join(exported_files)
                messagebox.showinfo("Success", f"Analysis results exported:\n{file_list}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export analysis results:\n{str(e)}")
                print(f"Full error: {e}")


def main():
    root = tk.Tk()
    app = PulseDiagnosticTool(root)
    root.mainloop()



if __name__ == "__main__":
    main()

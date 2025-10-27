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

# Import from Script 03
import thunderfish.pulses as pulses
from eod_functions import extract_pulse_snippets, filter_waveforms, filter_waveforms_with_classifier, save_variable_length_waveforms

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
        self.data_source = 'multich_linear'  # 'multich_linear' or '1ch_diff'
        
        # Default parameters (exactly from Script 03)
        self.parameters = {
            'thresh': 0.004,
            'min_rel_slope_diff': 0.25,
            'min_width_us': 30,
            'max_width_us': 1000,
            'width_fac_detection': 7.0,
            'interp_factor': 3,
            'amplitude_ratio_min': 0.2,
            'amplitude_ratio_max': 4.0,
            'save_filtered_out': False,
            'noise_removal': False,
            'peak_fft_freq_min': 50,
            'peak_fft_freq_max': 10000,
            'return_diff': True
        }
        
        # Time window parameters (0,0 means use full file)
        self.time_window = {'start_sec': 0.0, 'end_sec': 0.0}
        
        # ML configuration variables
        self.use_ml_filtering = tk.BooleanVar(value=False)
        self.classifier_path = tk.StringVar(value="")
        self.fish_probability_threshold = tk.DoubleVar(value=0.5)
        self.loaded_classifier = None
        self.loaded_scaler = None
        self.classifier_name = None
        
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
        
        self.file_info = ttk.Label(file_frame, text="No files loaded", wraplength=350)
        self.file_info.pack(fill=tk.X, pady=5)
        
        # Time window selection
        time_frame = ttk.LabelFrame(parent, text="Time Window (seconds)", padding=10)
        time_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Start time
        start_frame = ttk.Frame(time_frame)
        start_frame.pack(fill=tk.X, pady=2)
        ttk.Label(start_frame, text="Start:", width=8).pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value="0.0")
        start_entry = ttk.Entry(start_frame, textvariable=self.start_time_var, width=10)
        start_entry.pack(side=tk.LEFT, padx=5)
        start_entry.bind('<Return>', lambda e: self.detect_and_plot())
        
        # End time  
        end_frame = ttk.Frame(time_frame)
        end_frame.pack(fill=tk.X, pady=2)
        ttk.Label(end_frame, text="End:", width=8).pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value="0.0")
        end_entry = ttk.Entry(end_frame, textvariable=self.end_time_var, width=10)
        end_entry.pack(side=tk.LEFT, padx=5)
        end_entry.bind('<Return>', lambda e: self.detect_and_plot())
        
        ttk.Label(time_frame, text="(0,0 = use full recording)", font=('Arial', 8)).pack(pady=2)
        
        # Parameters (scrollable)
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding=5)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollable parameter area
        canvas = tk.Canvas(param_frame, height=300)
        scrollbar = ttk.Scrollbar(param_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Parameter inputs
        self.param_vars = {}
        for param, value in self.parameters.items():
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=param, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(row_frame, textvariable=var, width=10)
            entry.pack(side=tk.RIGHT)
            entry.bind('<Return>', lambda e: self.detect_and_plot())
            self.param_vars[param] = var
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Plotting mode selection
        plot_mode_frame = ttk.LabelFrame(button_frame, text="Plot Mode", padding=5)
        plot_mode_frame.pack(fill=tk.X, pady=2)
        
        self.plot_mode = tk.StringVar(value="differential")
        ttk.Radiobutton(plot_mode_frame, text="Differential", variable=self.plot_mode, 
                       value="differential").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(plot_mode_frame, text="Single-Ended", variable=self.plot_mode, 
                       value="single_ended").pack(side=tk.LEFT, padx=5)
        
        # ML Classification controls
        ml_frame = ttk.LabelFrame(button_frame, text="ML Fish vs Noise Classification", padding=5)
        ml_frame.pack(fill=tk.X, pady=2)
        
        # Enable/disable ML filtering
        ml_enable_frame = ttk.Frame(ml_frame)
        ml_enable_frame.pack(fill=tk.X, pady=2)
        
        self.ml_checkbox = ttk.Checkbutton(
            ml_enable_frame, 
            text="Enable ML-based noise filtering", 
            variable=self.use_ml_filtering,
            command=self.on_ml_toggle
        )
        self.ml_checkbox.pack(side=tk.LEFT)
        
        # Classifier file selection
        classifier_frame = ttk.Frame(ml_frame)
        classifier_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(classifier_frame, text="Classifier:", width=12).pack(side=tk.LEFT)
        self.classifier_label = ttk.Label(classifier_frame, text="No classifier loaded", 
                                         foreground="gray", wraplength=200)
        self.classifier_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(classifier_frame, text="Browse...", 
                  command=self.select_classifier_file, width=10).pack(side=tk.RIGHT)
        
        # Threshold setting
        threshold_frame = ttk.Frame(ml_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(threshold_frame, text="Fish threshold:", width=12).pack(side=tk.LEFT)
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.fish_probability_threshold, width=8)
        threshold_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(threshold_frame, text="(0.1-0.9)", foreground="gray").pack(side=tk.LEFT)
        
        # Initially disable ML controls if sklearn not available
        if not ML_AVAILABLE:
            self.ml_checkbox.configure(state="disabled")
            self.classifier_label.configure(text="sklearn not available")
        
        ttk.Button(button_frame, text="Detect & Plot", command=self.detect_and_plot).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Export Parameters", command=self.export_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Import Parameters", command=self.import_parameters).pack(fill=tk.X, pady=2)
        
        # Results summary
        self.results_text = tk.Text(parent, height=8, wrap=tk.WORD)
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
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.raw_data, self.sample_rate = aio.load_audio(file_path)
                # Extra bit: multiply the higher channels by -1
                # for ch in range(self.raw_data.shape[1]):
                #     if ch > 3:
                #         self.raw_data[:, ch] *= -1
                self.update_file_info()
                print(f"Loaded audio: {file_path}")
                print(f"Shape: {self.raw_data.shape}, Sample rate: {self.sample_rate}")
                
                if self.calibration_factors is not None:
                    self.calibrate_data()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file:\n{str(e)}")
    
    def select_calibration_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Calibration File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.calibration_factors = np.array(pd.read_csv(file_path))
                self.update_file_info()
                print(f"Loaded calibration: {file_path}")
                print(f"Calibration factors shape: {self.calibration_factors.shape}")
                
                if self.raw_data is not None:
                    self.calibrate_data()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load calibration file:\n{str(e)}")
    
    def load_precalibrated_data(self):
        """Load pre-calibrated audio data directly without requiring a calibration file"""
        file_path = filedialog.askopenfilename(
            title="Select Pre-Calibrated Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load audio data
                audio_data, sample_rate = aio.load_audio(file_path)

                # # Extra bit: multiply the higher channels by -1
                # for ch in range(audio_data.shape[1]):
                #     if ch > 3:
                #         audio_data[:, ch] *= -1

                # Set both raw and calibrated data to the same values
                # This bypasses the need for calibration factors
                self.raw_data = audio_data
                self.sample_rate = sample_rate
                self.calibrated_data = audio_data.copy()  # Already calibrated
                
                # Clear calibration factors since we don't need them
                self.calibration_factors = None
                
                # Set data source to multi-channel linear (default for pre-calibrated data)
                self.data_source = 'multich_linear'
                
                self.update_file_info()
                print(f"Loaded pre-calibrated audio: {file_path}")
                print(f"Shape: {self.calibrated_data.shape}, Sample rate: {self.sample_rate}")
                print("Data loaded as pre-calibrated - no calibration file needed")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pre-calibrated audio file:\n{str(e)}")
    
    def load_single_ch_diff_data(self):
        """Load single-channel differential data (like control recordings)"""
        file_path = filedialog.askopenfilename(
            title="Select Single-Channel Differential Recording",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load audio data
                audio_data, sample_rate = aio.load_audio(file_path)
                
                # Set both raw and calibrated data to the same values
                # Single-channel differential data doesn't need calibration
                self.raw_data = audio_data
                self.sample_rate = sample_rate
                self.calibrated_data = audio_data.copy()
                
                # Clear calibration factors since we don't need them
                self.calibration_factors = None
                
                # Set data source to single-channel differential
                self.data_source = '1ch_diff'
                
                self.update_file_info()
                print(f"Loaded single-channel differential data: {file_path}")
                print(f"Shape: {self.calibrated_data.shape}, Sample rate: {self.sample_rate}")
                print("Data source set to single-channel differential")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load single-channel differential data:\n{str(e)}")
    
    def update_file_info(self):
        info = ""
        if self.raw_data is not None:
            info += f"Audio: {self.raw_data.shape} @ {self.sample_rate} Hz\n"
        
        if self.calibration_factors is not None:
            info += f"Calibration: {self.calibration_factors.shape}\n"
        elif self.calibrated_data is not None and self.raw_data is not None:
            info += "Pre-calibrated data (no calibration file needed)\n"
        
        # Show data source type
        if self.calibrated_data is not None:
            source_desc = "Multi-channel linear" if self.data_source == 'multich_linear' else "Single-channel differential"
            info += f"Source: {source_desc}"
        
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
            # Show classifier name
            display_name = self.classifier_name if self.classifier_name else "Loaded"
            self.classifier_label.configure(text=display_name, foreground="darkgreen")
        else:
            # Show no classifier
            if ML_AVAILABLE:
                self.classifier_label.configure(text="No classifier loaded", foreground="gray")
            else:
                self.classifier_label.configure(text="sklearn not available", foreground="red")
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
        
        # Set data source to multi-channel linear (calibrated data)
        self.data_source = 'multich_linear'
            
        print("Data calibrated successfully")
        
    def update_parameters(self):
        """Update parameters from GUI inputs"""
        try:
            for param, var in self.param_vars.items():
                value_str = var.get().strip()
                
                if param in ['save_filtered_out', 'noise_removal', 'return_diff']:
                    self.parameters[param] = value_str.lower() in ['true', '1', 'yes', 'on']
                elif param in ['interp_factor', 'min_width_us', 'max_width_us', 'peak_fft_freq_min', 'peak_fft_freq_max']:
                    self.parameters[param] = int(float(value_str))
                else:
                    self.parameters[param] = float(value_str)
                    
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Invalid parameter value: {str(e)}")
            return False
        return True
    
    def detect_and_plot(self):
        """Main detection and plotting function - exactly like Script 03"""
        if self.calibrated_data is None:
            messagebox.showwarning("Warning", "Please load audio data first (either with calibration file or pre-calibrated data)")
            return
            
        if not self.update_parameters():
            return
            
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get time window parameters
            try:
                start_sec = float(self.start_time_var.get().strip())
                end_sec = float(self.end_time_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Invalid time window values")
                return
            
            # Script 03 exact logic starts here
            data = self.calibrated_data
            rate = self.sample_rate
            
            # Handle single-channel differential data
            if self.data_source == '1ch_diff':
                # For single-channel differential, use only the first column
                data = data[:, 0:1]  # Keep as 2D array with shape (samples, 1)
                n_channels = 1
            else:   
                n_channels = data.shape[1]
            
            # Apply time windowing if specified
            if start_sec > 0 or end_sec > 0:
                total_duration = len(data) / rate
                if end_sec <= 0:
                    end_sec = total_duration
                    
                if start_sec < 0:
                    start_sec = 0
                    
                if start_sec >= end_sec:
                    messagebox.showerror("Error", "Start time must be less than end time")
                    return
                    
                if end_sec > total_duration:
                    end_sec = total_duration
                    messagebox.showwarning("Warning", f"End time adjusted to file duration: {end_sec:.2f}s")
                
                start_idx = int(start_sec * rate)
                end_idx = int(end_sec * rate)
                data = data[start_idx:end_idx, :]
                
                print(f"Using time window: {start_sec:.2f}s to {end_sec:.2f}s ({len(data)} samples)")
            else:
                start_sec = 0
                end_sec = len(data) / rate
                print(f"Using full recording: {end_sec:.2f}s ({len(data)} samples)")
            
            print(f"Detecting pulses on {n_channels} channels...")
            
            # Detect pulses on each channel (Script 03 logic)
            all_peaks = []
            all_troughs = []
            all_widths = []
            
            for i in range(n_channels):
                ch_peaks, ch_troughs, _, ch_pulse_widths = pulses.detect_pulses(
                    data[:, i], 
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
            
            # Unify pulses across channels (Script 03 logic)
            unified_pulses = []
            for ch_peaks, ch_troughs, ch_widths in zip(all_peaks, all_troughs, all_widths):
                n_pairs = min(len(ch_peaks), len(ch_troughs))
                for j in range(n_pairs):
                    mp = (ch_peaks[j] + ch_troughs[j]) // 2
                    unified_pulses.append((mp, ch_peaks[j], ch_troughs[j], ch_widths[j]))
            
            # Sort and remove duplicates (Script 03 logic)
            unified_pulses.sort(key=lambda x: x[0])
            unique_pulses = []
            
            if unified_pulses:
                last_mp = unified_pulses[0][0]
                unique_pulses.append(unified_pulses[0])
                for event in unified_pulses[1:]:
                    if event[0] - last_mp > 0.001 * rate:  # 1ms minimum separation
                        unique_pulses.append(event)
                        last_mp = event[0]
            
            print(f"    Unified to {len(unique_pulses)} unique pulses")
            
            if len(unique_pulses) == 0:
                self.ax.text(0.5, 0.5, 'No pulses detected\nTry lowering threshold', 
                           transform=self.ax.transAxes, ha='center', va='center',
                           fontsize=14, color='red')
                self.canvas.draw()
                return
            
            # Extract pulse snippets (Script 03 logic)
            midpoints = np.array([p[0] for p in unique_pulses])
            peaks = np.array([p[1] for p in unique_pulses])
            troughs = np.array([p[2] for p in unique_pulses])
            widths = np.array([p[3] for p in unique_pulses])
            
            # Convert parameters to DataFrame format
            params_df = pd.DataFrame({key: [value] for key, value in self.parameters.items()})

            # Extract waveforms
            (
                eod_waveforms, eod_amps, eod_widths, ch_amps, ch_cor_coeffs, eod_chan, is_differential,
                snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc,
                final_peak_idc, final_trough_idc, final_midpoint_idc,
                pulse_orientation, amplitude_ratios, waveform_lengths, fft_peak_freqs
            ) = extract_pulse_snippets(
                data, midpoints, peaks, troughs, thresh=self.parameters['thresh'], 
                width_factor=self.parameters['width_fac_detection'], interp_factor=self.parameters['interp_factor'], rate=rate,
                source=self.data_source, return_differential=self.parameters['return_diff']
            )
            # extraction_results = extract_pulse_snippets(
            #     data, params_df, rate, midpoints, peaks, troughs, widths
            # )
            
            # # Unpack results
            # (eod_waveforms, eod_amps, eod_widths, ch_amps, ch_cor_coeffs, eod_chan, 
            #  is_differential, snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc,
            #  final_peak_idc, final_trough_idc, final_midpoint_idc, pulse_orientation,
            #  amplitude_ratios, waveform_lengths, fft_peak_freqs) = extraction_results
            
            print(f"    Filtering for differential pulses...{len(unique_pulses)} total, {np.sum(is_differential)} differential")
            
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
            
            keep_indices = np.array(keep_indices)
            all_indices = np.arange(len(unique_pulses))
            filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
            
            # Choose plotting mode
            if self.plot_mode.get() == "single_ended":
                self.plot_single_ended(data, rate, start_sec, n_channels, 
                                     keep_indices, filtered_out_indices, 
                                     eod_chan, final_peak_idc, final_trough_idc)
            else:
                self.plot_differential(data, rate, start_sec, n_channels, 
                                     keep_indices, filtered_out_indices, 
                                     eod_chan, final_peak_idc, final_trough_idc)
            
            # Update results summary
            self.update_results_summary(len(unique_pulses), len(keep_indices.flatten()), len(filtered_out_indices), start_sec, end_sec)
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error during pulse detection:\n{str(e)}")
            print(f"Full error: {e}")
            import traceback
            traceback.print_exc()
        
        self.canvas.draw()
    
    def plot_differential(self, data, rate, start_sec, n_channels, keep_indices, filtered_out_indices, 
                         eod_chan, final_peak_idc, final_trough_idc):
        """Plot differential signals with peaks and troughs marked"""
        # Plot exactly like Script 03 differential detection plot
        plot_duration = min(60.0, len(data) / rate)  # Max 60 seconds
        plot_samples = int(plot_duration * rate)
        plot_data = data[:plot_samples]
        
        # Script 03 plotting logic
        offset_diff = np.max(abs(plot_data))
        
        # Handle different data source types
        if self.data_source == '1ch_diff':
            # For single-channel differential, data is already differential
            data_diff = plot_data[:, 0]
            
            # Downsample for plotting
            step = max(1, len(data_diff) // 15000000)
            x_coords = np.arange(0, len(data_diff), step)
            
            # Plot differential signal
            self.ax.plot((x_coords / rate) + start_sec, 
                        data_diff[::step], 
                        linewidth=0.5, label='Diff Ch')
            
            # Plot pulses

            # if all(final_peak_idc < len(data_diff)):
            #     self.ax.plot((final_peak_idc / rate) + start_sec,
            #                 data_diff[final_peak_idc],
            #                 'o', markersize=2, color='red')
            # if all(final_trough_idc < len(data_diff)):
            #     self.ax.plot((final_trough_idc / rate) + start_sec, 
            #                 data_diff[final_trough_idc], 
            #                 'o', markersize=2, color='blue')
                
            # Plot kept pulses (red=peaks, blue=troughs)
            if len(keep_indices) > 0:
                valid_keep_indices = keep_indices[keep_indices < len(eod_chan)]
                for idx in valid_keep_indices:
                    peak_idx = int(final_peak_idc[idx])
                    trough_idx = int(final_trough_idc[idx])
                    if peak_idx < len(data_diff):
                        self.ax.plot((peak_idx / rate) + start_sec, 
                                    data_diff[peak_idx], 
                                    'o', markersize=2, color='red')
                    if trough_idx < len(data_diff):
                        self.ax.plot((trough_idx / rate) + start_sec, 
                                    data_diff[trough_idx], 
                                    'o', markersize=2, color='blue')
            
            # Plot filtered out pulses (grey)
            if len(filtered_out_indices) > 0:
                valid_filtered_indices = filtered_out_indices[filtered_out_indices < len(eod_chan)]
                for idx in valid_filtered_indices:
                    peak_idx = int(final_peak_idc[idx])
                    trough_idx = int(final_trough_idc[idx])
                    if peak_idx < len(data_diff):
                        self.ax.plot((peak_idx / rate) + start_sec, 
                                    data_diff[peak_idx], 
                                    'o', markersize=2, color='grey', alpha=0.6)
                    if trough_idx < len(data_diff):
                        self.ax.plot((trough_idx / rate) + start_sec, 
                                    data_diff[trough_idx], 
                                    'o', markersize=2, color='grey', alpha=0.6)
                                            
        else:
            # Multi-channel linear data - create differential pairs
            for i in range(n_channels - 1):  # Differential channels
                # Create differential signal for this channel pair
                data_diff = np.diff(plot_data[:, i:i+2], axis=1).flatten()
                
                # Downsample for plotting
                step = max(1, len(data_diff) // 15000000)
                x_coords = np.arange(0, len(data_diff), step)
                
                # Plot differential signal
                self.ax.plot((x_coords / rate) + start_sec, 
                           data_diff[::step] + i * offset_diff, 
                           linewidth=0.5, label=f'Ch{i}-{i+1}')
                
                # Plot kept pulses (red=peaks, blue=troughs) - Script 03 style
                if len(keep_indices) > 0:
                    # Filter out indices that are out of bounds
                    valid_keep_indices = keep_indices[keep_indices < len(eod_chan)]
                    if len(valid_keep_indices) > 0:
                        kept_ch_mask = (eod_chan[valid_keep_indices] == i)
                        if np.any(kept_ch_mask):
                            kept_ch_indices = valid_keep_indices[kept_ch_mask]
                            
                            # Plot peaks and troughs
                            for idx in kept_ch_indices:
                                peak_idx = int(final_peak_idc[idx])
                                trough_idx = int(final_trough_idc[idx])
                                
                                if peak_idx < len(data_diff):
                                    self.ax.plot((peak_idx / rate) + start_sec, 
                                               data_diff[peak_idx] + i * offset_diff, 
                                               'o', markersize=2, color='red')
                                if trough_idx < len(data_diff):
                                    self.ax.plot((trough_idx / rate) + start_sec, 
                                               data_diff[trough_idx] + i * offset_diff, 
                                               'o', markersize=2, color='blue')
                
                # Plot filtered out pulses (grey) - Script 03 style
                if len(filtered_out_indices) > 0:
                    # Filter out indices that are out of bounds
                    valid_filtered_indices = filtered_out_indices[filtered_out_indices < len(eod_chan)]
                    if len(valid_filtered_indices) > 0:
                        filtered_ch_mask = (eod_chan[valid_filtered_indices] == i)
                        if np.any(filtered_ch_mask):
                            filtered_ch_indices = valid_filtered_indices[filtered_ch_mask]
                            
                            for idx in filtered_ch_indices:
                                peak_idx = int(final_peak_idc[idx])
                                trough_idx = int(final_trough_idc[idx])
                                
                                if peak_idx < len(data_diff):
                                    self.ax.plot((peak_idx / rate) + start_sec, 
                                               data_diff[peak_idx] + i * offset_diff, 
                                               'o', markersize=2, color='grey', alpha=0.6)
                                if trough_idx < len(data_diff):
                                    self.ax.plot((trough_idx / rate) + start_sec, 
                                               data_diff[trough_idx] + i * offset_diff, 
                                               'o', markersize=2, color='grey', alpha=0.6)
        
        # Format plot (Script 03 style)
        if self.data_source == '1ch_diff':
            # Single-channel differential - simpler y-axis
            # self.ax.set_ylim(bottom=None, top=0.5*offset_diff)
            self.ax.set_ylim(bottom=np.min(data_diff)*1.2, top=np.max(data_diff)*1.2)
            source_desc = "Single-Ch Diff"
        else:
            # Multi-channel linear - multiple differential pairs
            # self.ax.set_ylim(bottom=None, top=(n_channels-1.5)*offset_diff)
            self.ax.set_ylim(bottom=np.min(plot_data)*1.2, top=(n_channels-1.5)*offset_diff)
            source_desc = "Multi-Ch Diff"
        
        self.ax.set_title(f'EOD Pulse Detection ({source_desc}) - Red=Peaks, Blue=Troughs, Grey=Filtered Out\n'
                        f'{len(keep_indices.flatten())} kept, {len(filtered_out_indices)} filtered out')
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage')
        self.ax.grid(True, alpha=0.3)
    
    def plot_single_ended(self, data, rate, start_sec, n_channels, keep_indices, filtered_out_indices,
                         eod_chan, final_peak_idc, final_trough_idc):
        """Plot single-ended signals with peaks and troughs marked on the appropriate channels"""
        
        # Handle single-channel differential data
        if self.data_source == '1ch_diff':
            # For single-channel differential data, show a message that single-ended view is not applicable
            self.ax.text(0.5, 0.5, 'Single-Ended view not applicable for\nSingle-Channel Differential data.\n\nPlease use Differential view mode.', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=16, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            self.ax.set_title('Single-Ended View Not Applicable')
            return
        
        # Plot single-ended channels (multi-channel linear data)
        plot_duration = min(60.0, len(data) / rate)  # Max 60 seconds
        plot_samples = int(plot_duration * rate)
        plot_data = data[:plot_samples]
        
        # Calculate offset for stacking channels
        offset_se = np.max(abs(plot_data)) * 1.2
        
        # Downsample for plotting
        step = max(1, len(plot_data) // 15000000)
        x_coords = np.arange(0, len(plot_data), step)
        
        # Plot all single-ended channels
        for ch in range(n_channels):
            self.ax.plot((x_coords / rate) + start_sec, 
                        plot_data[::step, ch] + ch * offset_se, 
                        linewidth=0.5, label=f'SE Ch{ch}')
        
        # Now mark peaks and troughs on the appropriate single-ended channels
        # For differential channel i: use single-ended channels i and i+1
        for diff_ch in range(n_channels - 1):
            se_ch1 = diff_ch      # First single-ended channel
            se_ch2 = diff_ch + 1  # Second single-ended channel
            
            # Plot kept pulses on both single-ended channels that form this differential pair
            if len(keep_indices) > 0:
                valid_keep_indices = keep_indices[keep_indices < len(eod_chan)]
                if len(valid_keep_indices) > 0:
                    kept_ch_mask = (eod_chan[valid_keep_indices] == diff_ch)
                    if np.any(kept_ch_mask):
                        kept_ch_indices = valid_keep_indices[kept_ch_mask]
                        
                        for idx in kept_ch_indices:
                            peak_idx = int(final_peak_idc[idx])
                            trough_idx = int(final_trough_idc[idx])
                            
                            # Mark on first single-ended channel (se_ch1)
                            if peak_idx < len(plot_data):
                                self.ax.plot((peak_idx / rate) + start_sec, 
                                           plot_data[peak_idx, se_ch1] + se_ch1 * offset_se, 
                                           'o', markersize=3, color='red')
                            if trough_idx < len(plot_data):
                                self.ax.plot((trough_idx / rate) + start_sec, 
                                           plot_data[trough_idx, se_ch1] + se_ch1 * offset_se, 
                                           'o', markersize=3, color='blue')
                            
                            # Mark on second single-ended channel (se_ch2)
                            if peak_idx < len(plot_data):
                                self.ax.plot((peak_idx / rate) + start_sec, 
                                           plot_data[peak_idx, se_ch2] + se_ch2 * offset_se, 
                                           'o', markersize=3, color='red')
                            if trough_idx < len(plot_data):
                                self.ax.plot((trough_idx / rate) + start_sec, 
                                           plot_data[trough_idx, se_ch2] + se_ch2 * offset_se, 
                                           'o', markersize=3, color='blue')
            
            # Plot filtered out pulses (grey) on both single-ended channels
            if len(filtered_out_indices) > 0:
                valid_filtered_indices = filtered_out_indices[filtered_out_indices < len(eod_chan)]
                if len(valid_filtered_indices) > 0:
                    filtered_ch_mask = (eod_chan[valid_filtered_indices] == diff_ch)
                    if np.any(filtered_ch_mask):
                        filtered_ch_indices = valid_filtered_indices[filtered_ch_mask]
                        
                        for idx in filtered_ch_indices:
                            peak_idx = int(final_peak_idc[idx])
                            trough_idx = int(final_trough_idc[idx])
                            
                            # Mark on first single-ended channel (se_ch1)
                            if peak_idx < len(plot_data):
                                self.ax.plot((peak_idx / rate) + start_sec, 
                                           plot_data[peak_idx, se_ch1] + se_ch1 * offset_se, 
                                           'o', markersize=3, color='grey', alpha=0.6)
                            if trough_idx < len(plot_data):
                                self.ax.plot((trough_idx / rate) + start_sec, 
                                           plot_data[trough_idx, se_ch1] + se_ch1 * offset_se, 
                                           'o', markersize=3, color='grey', alpha=0.6)
                            
                            # Mark on second single-ended channel (se_ch2)
                            if peak_idx < len(plot_data):
                                self.ax.plot((peak_idx / rate) + start_sec, 
                                           plot_data[peak_idx, se_ch2] + se_ch2 * offset_se, 
                                           'o', markersize=3, color='grey', alpha=0.6)
                            if trough_idx < len(plot_data):
                                self.ax.plot((trough_idx / rate) + start_sec, 
                                           plot_data[trough_idx, se_ch2] + se_ch2 * offset_se, 
                                           'o', markersize=3, color='grey', alpha=0.6)
        
        # Format plot
        self.ax.set_ylim(bottom=-0.5*offset_se, top=(n_channels-0.5)*offset_se)
        self.ax.set_title(f'EOD Pulse Detection (Single-Ended) - Red=Peaks, Blue=Troughs, Grey=Filtered Out\n'
                        f'{len(keep_indices.flatten())} kept, {len(filtered_out_indices)} filtered out\n'
                        f'Diff Ch0=SE Ch0-Ch1, Diff Ch1=SE Ch1-Ch2, etc.')
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage')
        self.ax.grid(True, alpha=0.3)
    
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


def main():
    root = tk.Tk()
    app = PulseDiagnosticTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()

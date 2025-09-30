#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Event Annotation Tool for EOD Analysis
==================================================

This script provides an interactive GUI for manually annotating extracted events
to create training data for improved filtering algorithms.

Features:
- Load events from Script 04 output
- Display event plots with audio playback capability
- Manual annotation interface (Clear Fish / Clear Noise / Ambiguous)
- Extract comprehensive features for each event
- Save annotated dataset for training filtering models

Author: AI Assistant & User
Date: September 2025
"""

import os
import sys
import gc
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons, Slider
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from datetime import datetime
import json

# Audio processing
try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("WARNING: Audio playback not available (install soundfile and sounddevice)")

# Import EOD functions for loading variable-length waveforms
try:
    from eod_functions import load_variable_length_waveforms
except ImportError:
    print("Warning: Could not import eod_functions - waveform loading may not work")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Annotation categories
ANNOTATION_CATEGORIES = {
    'clear_fish': {'label': 'Clear Fish', 'color': 'green', 'key': '1'},
    'clear_noise': {'label': 'Clear Noise', 'color': 'red', 'key': '2'},
    'ambiguous': {'label': 'Ambiguous', 'color': 'orange', 'key': '3'},
    'unannotated': {'label': 'Unannotated', 'color': 'gray', 'key': '0'}
}

class EventAnnotationTool:
    def __init__(self):
        self.events_folder = None
        self.output_folder = None
        self.event_files = []
        self.current_event_idx = 0
        self.annotations = {}
        self.fish_counts = {}  # Store fish count information for clear_fish annotations
        self.event_features = {}
        
        # Fish count settings
        self.fish_count_enabled = False
        self.waiting_for_fish_count = False
        self.pending_event_id = None
        
        # GUI components
        self.root = None
        self.fig = None
        self.ax = None
        self.current_annotation = 'unannotated'
        
        # Audio components
        self.current_audio = None
        self.sample_rate = 96000  # Default sample rate
        self.audio_playback_mode = 'differential_mix'  # Options: 'differential_mix', 'stereo_channels', 'mono_mix'
        self.selected_channels = [0, 1]  # Default channels for stereo playback
        
        # Event data
        self.current_event_data = None
        self.current_event_summary = None
        self.current_eod_table = None
        
    def setup_gui(self):
        """Initialize the main GUI window"""
        self.root = tk.Tk()
        self.root.title("EOD Event Annotation Tool")
        self.root.geometry("1400x900")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Folder selection
        folder_frame = ttk.Frame(control_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(folder_frame, text="Select Events Folder", 
                  command=self.select_events_folder).pack(side=tk.LEFT, padx=(0, 10))
        self.folder_label = ttk.Label(folder_frame, text="No folder selected")
        self.folder_label.pack(side=tk.LEFT)
        
        # Navigation controls
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="◀◀ Previous", 
                  command=self.previous_event).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next ▶▶", 
                  command=self.next_event).pack(side=tk.LEFT, padx=(0, 10))
        
        self.event_label = ttk.Label(nav_frame, text="Event: - / -")
        self.event_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Go to event controls
        goto_frame = ttk.Frame(nav_frame)
        goto_frame.pack(side=tk.LEFT, padx=(10, 10))
        
        ttk.Label(goto_frame, text="Go to:").pack(side=tk.LEFT, padx=(0, 2))
        self.goto_var = tk.StringVar()
        self.goto_entry = ttk.Entry(goto_frame, textvariable=self.goto_var, width=6)
        self.goto_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(goto_frame, text="Go", 
                  command=self.go_to_event).pack(side=tk.LEFT)
        
        # Bind Enter key to go to event
        self.goto_entry.bind('<Return>', lambda e: self.go_to_event())
        
        # Progress bar
        self.progress = ttk.Progressbar(nav_frame, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Annotation controls
        annotation_frame = ttk.LabelFrame(control_frame, text="Annotation", padding=10)
        annotation_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Annotation buttons
        button_frame = ttk.Frame(annotation_frame)
        button_frame.pack(fill=tk.X)
        
        for category, info in ANNOTATION_CATEGORIES.items():
            if category != 'unannotated':
                btn = ttk.Button(button_frame, 
                               text=f"{info['label']} [{info['key']}]",
                               command=lambda cat=category: self.annotate_event(cat))
                btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Fish count control
        fish_count_frame = ttk.Frame(annotation_frame)
        fish_count_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.fish_count_var = tk.BooleanVar()
        fish_count_checkbox = ttk.Checkbutton(fish_count_frame, 
                                             text="Enable Fish Count Input (for Clear Fish)",
                                             variable=self.fish_count_var,
                                             command=self.toggle_fish_count)
        fish_count_checkbox.pack(side=tk.LEFT)
        
        # Fish count status label
        self.fish_count_status = ttk.Label(fish_count_frame, text="", foreground='blue')
        self.fish_count_status.pack(side=tk.LEFT, padx=(20, 0))
        
        # Audio controls (if available)
        if AUDIO_AVAILABLE:
            audio_frame = ttk.Frame(annotation_frame)
            audio_frame.pack(fill=tk.X, pady=(10, 0))
            
            # Audio playback controls
            playback_frame = ttk.Frame(audio_frame)
            playback_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Button(playback_frame, text="▶ Play Audio", 
                      command=self.play_audio).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(playback_frame, text="⏹ Stop Audio", 
                      command=self.stop_audio).pack(side=tk.LEFT, padx=(0, 10))
            
            # Playback speed control
            ttk.Label(playback_frame, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
            self.speed_var = tk.DoubleVar(value=1.0)
            speed_scale = ttk.Scale(playback_frame, from_=0.1, to=3.0, 
                                  variable=self.speed_var, length=100)
            speed_scale.pack(side=tk.LEFT, padx=(0, 10))
            self.speed_label = ttk.Label(playback_frame, text="1.0x")
            self.speed_label.pack(side=tk.LEFT)
            self.speed_var.trace('w', self.update_speed_label)
            
            # Audio mode selection
            mode_frame = ttk.Frame(audio_frame)
            mode_frame.pack(fill=tk.X)
            
            ttk.Label(mode_frame, text="Audio Mode:").pack(side=tk.LEFT, padx=(0, 5))
            self.audio_mode_var = tk.StringVar(value='differential_mix')
            audio_modes = [
                ('Differential Mix', 'differential_mix'),
                ('Stereo (Ch 0-1)', 'stereo_channels'),
                ('Mono Mix', 'mono_mix')
            ]
            
            for label, mode in audio_modes:
                ttk.Radiobutton(mode_frame, text=label, variable=self.audio_mode_var, 
                               value=mode, command=self.update_audio_mode).pack(side=tk.LEFT, padx=(0, 10))
            
            # Channel selection for stereo mode
            ttk.Label(mode_frame, text="Channels:").pack(side=tk.LEFT, padx=(20, 5))
            self.ch1_var = tk.IntVar(value=0)
            self.ch2_var = tk.IntVar(value=1)
            
            ttk.Spinbox(mode_frame, from_=0, to=7, width=3, textvariable=self.ch1_var).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Label(mode_frame, text="-").pack(side=tk.LEFT)
            ttk.Spinbox(mode_frame, from_=0, to=7, width=3, textvariable=self.ch2_var).pack(side=tk.LEFT, padx=(2, 0))
        
        # Save/Load controls
        save_frame = ttk.Frame(annotation_frame)
        save_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(save_frame, text="💾 Save Annotations", 
                  command=self.save_annotations).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(save_frame, text="📁 Load Annotations", 
                  command=self.load_annotations).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(save_frame, text="📊 Export Dataset", 
                  command=self.export_dataset).pack(side=tk.LEFT, padx=(0, 10))
        
        # Event info panel
        info_frame = ttk.LabelFrame(main_frame, text="Event Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create text widget for event info
        self.info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Event Visualization", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib canvas will be added here when an event is loaded
        self.plot_frame = plot_frame
        
        # Bind keyboard shortcuts (with focus checking to avoid conflicts with entry field)
        self.root.bind('<Key-1>', self.handle_key_1)
        self.root.bind('<Key-2>', self.handle_key_2)
        self.root.bind('<Key-3>', self.handle_key_3)
        self.root.bind('<Key-u>', self.handle_key_u)  # For 'unclear' fish count
        self.root.bind('<Left>', lambda e: self.previous_event())
        self.root.bind('<Right>', lambda e: self.next_event())
        self.root.bind('<space>', lambda e: self.play_audio())
        
        self.root.focus_set()  # Enable keyboard shortcuts
        
    def select_events_folder(self):
        """Select folder containing event files from Script 04"""
        folder = filedialog.askdirectory(
            title="Select Events Folder (Script 04 output)",
            initialdir=os.getcwd()
        )
        
        if folder:
            self.events_folder = Path(folder)
            self.folder_label.config(text=f"Folder: {self.events_folder.name}")
            self.load_event_files()
    
    def load_event_files(self):
        """Load all event files from the selected folder"""
        if not self.events_folder:
            return
            
        # Find all event audio files
        audio_files = list(self.events_folder.glob("event_*.wav"))
        
        if not audio_files:
            messagebox.showerror("Error", "No event audio files found in the selected folder!")
            return
        
        # Sort by event ID
        self.event_files = sorted(audio_files, key=lambda x: int(x.stem.split('_')[1]))
        
        print(f"Found {len(self.event_files)} event files")
        
        # Initialize annotations and fish count dictionaries
        self.annotations = {f.stem: 'unannotated' for f in self.event_files}
        self.fish_counts = {}  # Will be populated as needed
        
        # Update progress bar
        self.progress['maximum'] = len(self.event_files)
        
        # Load first event
        self.current_event_idx = 0
        self.load_current_event()
    
    def load_current_event(self):
        """Load and display the current event"""
        if not self.event_files:
            return
            
        # Clean up memory from previous event before loading new one
        self.cleanup_previous_event()
            
        event_file = self.event_files[self.current_event_idx]
        event_id = event_file.stem
        
        try:
            # Load audio file
            self.load_event_audio(event_file)
            
            # Load EOD table if available - try multiple filename patterns
            eod_file_patterns = [
                f"{event_id}_eod_table.csv",  # Full event ID with timestamp
                f"event_{event_id.split('_')[1]}_eod_table.csv"  # Just event number
            ]
            
            self.current_eod_table = None
            for pattern in eod_file_patterns:
                eod_file = self.events_folder / pattern
                if eod_file.exists():
                    self.current_eod_table = pd.read_csv(eod_file)
                    break
            
            # Load event summary info (try to find from main summary file)
            self.load_event_summary_info()
            
            # Extract features for this event
            self.extract_event_features()
            
            # Update display
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load event {event_id}:\n{str(e)}")
    
    def cleanup_previous_event(self):
        """Clean up memory from previous event to prevent RAM buildup"""
        # Clear large audio arrays
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            del self.current_audio
            self.current_audio = None
        
        # Clear EOD table to free DataFrame memory
        if hasattr(self, 'current_eod_table') and self.current_eod_table is not None:
            del self.current_eod_table
            self.current_eod_table = None
        
        # Clear event summary
        if hasattr(self, 'current_event_summary') and self.current_event_summary is not None:
            del self.current_event_summary
            self.current_event_summary = None
        
        # Clean up matplotlib resources (this is already done in update_plot, but ensure it's thorough)
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            del self.fig
            self.fig = None
        
        if hasattr(self, 'ax') and self.ax is not None:
            del self.ax
            self.ax = None
        
        # Force garbage collection to free memory immediately
        gc.collect()
        
        print(f"Memory cleanup completed for previous event")
    
    def load_event_audio(self, audio_file):
        """Load audio data for the current event with memory optimization"""
        
        # Clean up any existing audio data first
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            del self.current_audio
            self.current_audio = None
            gc.collect()
        
        if AUDIO_AVAILABLE:
            try:
                self.current_audio, self.sample_rate = sf.read(str(audio_file))
                print(f"Loaded audio: {self.current_audio.shape}, {self.current_audio.dtype}, {self.current_audio.nbytes / 1024 / 1024:.1f} MB")
            except Exception as e:
                self.current_audio = None
                print(f"Failed to load audio with soundfile: {e}")
        else:
            # Load with scipy as fallback
            try:
                import scipy.io.wavfile as wav
                self.sample_rate, audio_data = wav.read(str(audio_file))
                # Convert to float and normalize
                if audio_data.dtype == np.int16:
                    self.current_audio = audio_data.astype(np.float32) / 32768.0
                else:
                    self.current_audio = audio_data.astype(np.float32)
                    
                # Clean up temporary array
                del audio_data
                gc.collect()
                print(f"Loaded audio (scipy): {self.current_audio.shape}, {self.current_audio.dtype}, {self.current_audio.nbytes / 1024 / 1024:.1f} MB")
            except ImportError:
                self.current_audio = None
                print("No audio libraries available")
            except Exception as e:
                self.current_audio = None
                print(f"Failed to load audio with scipy: {e}")
    
    def load_event_summary_info(self):
        """Load summary information for the current event"""
        # Try to load from main summary file
        summary_file = self.events_folder / "session_events_summary.csv"
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            
            # Extract event number from filename - handle different patterns
            event_filename = self.event_files[self.current_event_idx].stem
            
            # Try different ways to extract event ID
            event_num = None
            
            # Pattern 1: event_000_20231011T171511 -> 0
            if '_' in event_filename:
                parts = event_filename.split('_')
                if len(parts) >= 2:
                    try:
                        event_num = int(parts[1])  # Take second part (000 -> 0)
                    except ValueError:
                        pass
            
            # Pattern 2: event_0 -> 0 (fallback)
            if event_num is None:
                try:
                    event_num = int(event_filename.replace('event_', '').split('_')[0])
                except ValueError:
                    event_num = self.current_event_idx  # Use index as fallback
            
            print(f"Looking for event_id {event_num} in summary (from filename: {event_filename})")
            
            # Look for matching event in summary
            event_row = summary_df[summary_df['event_id'] == event_num]
            if len(event_row) > 0:
                self.current_event_summary = event_row.iloc[0]
                print(f"Found summary info for event {event_num}")
            else:
                self.current_event_summary = None
                print(f"No summary found for event_id {event_num}")
                print(f"Available event_ids in summary: {sorted(summary_df['event_id'].unique())}")
        else:
            self.current_event_summary = None
            print("No session_events_summary.csv file found")
    
    def extract_event_features(self):
        """Extract comprehensive features for the current event"""
        event_id = self.event_files[self.current_event_idx].stem
        features = {}
        
        # Basic audio features
        if self.current_audio is not None:
            features['duration_seconds'] = len(self.current_audio) / self.sample_rate
            features['n_channels'] = self.current_audio.shape[1] if len(self.current_audio.shape) > 1 else 1
            features['audio_rms'] = np.sqrt(np.mean(self.current_audio**2))
            features['audio_max_amplitude'] = np.max(np.abs(self.current_audio))
        
        # EOD-based features
        if self.current_eod_table is not None and len(self.current_eod_table) > 0:
            eods = self.current_eod_table
            
            # Basic counts
            features['n_eods'] = len(eods)
            features['n_channels_with_eods'] = eods['eod_channel'].nunique() if 'eod_channel' in eods.columns else 0
            
            # Amplitude features
            if 'eod_amplitude' in eods.columns:
                features['mean_eod_amplitude'] = eods['eod_amplitude'].mean()
                features['std_eod_amplitude'] = eods['eod_amplitude'].std()
                features['max_eod_amplitude'] = eods['eod_amplitude'].max()
                features['amplitude_cv'] = features['std_eod_amplitude'] / features['mean_eod_amplitude'] if features['mean_eod_amplitude'] > 0 else 0
            
            # Temporal features
            if 'timestamp_dt' in eods.columns:
                eods['timestamp_dt'] = pd.to_datetime(eods['timestamp_dt'])
                time_diffs = eods.sort_values('timestamp_dt')['timestamp_dt'].diff().dt.total_seconds().dropna()
                if len(time_diffs) > 0:
                    features['mean_ipi'] = time_diffs.mean()
                    features['std_ipi'] = time_diffs.std()
                    features['median_ipi'] = time_diffs.median()
                    features['ipi_cv'] = features['std_ipi'] / features['mean_ipi'] if features['mean_ipi'] > 0 else 0
                    features['min_ipi'] = time_diffs.min()
                    features['max_ipi'] = time_diffs.max()
            
            # Waveform features
            if 'eod_width_us' in eods.columns:
                features['mean_pulse_width'] = eods['eod_width_us'].mean()
                features['std_pulse_width'] = eods['eod_width_us'].std()
            
            if 'eod_amplitude_ratio' in eods.columns:
                features['mean_amplitude_ratio'] = eods['eod_amplitude_ratio'].mean()
                features['std_amplitude_ratio'] = eods['eod_amplitude_ratio'].std()
            
            # Frequency features
            if 'fft_freq_max' in eods.columns:
                features['mean_peak_frequency'] = eods['fft_freq_max'].mean()
                features['std_peak_frequency'] = eods['fft_freq_max'].std()
            
            # Channel distribution
            if 'eod_channel' in eods.columns:
                channel_counts = eods['eod_channel'].value_counts()
                features['primary_channel'] = channel_counts.index[0]
                features['channel_concentration'] = channel_counts.iloc[0] / len(eods)  # Fraction on primary channel
                features['channel_span'] = eods['eod_channel'].max() - eods['eod_channel'].min() + 1
        
        # Summary-based features
        if self.current_event_summary is not None:
            for col in ['duration_seconds', 'n_eods', 'n_channels', 'mean_ipi_seconds', 
                       'median_ipi_seconds', 'mean_amplitude', 'max_amplitude']:
                if col in self.current_event_summary:
                    features[f'summary_{col}'] = self.current_event_summary[col]
        
        # Store features
        self.event_features[event_id] = features
        
        return features
    
    def update_display(self):
        """Update all display elements"""
        event_file = self.event_files[self.current_event_idx]
        event_id = event_file.stem
        
        # Update event label and progress
        self.event_label.config(text=f"Event: {self.current_event_idx + 1} / {len(self.event_files)} ({event_id})")
        self.progress['value'] = self.current_event_idx + 1
        
        # Update event info text
        self.update_info_text()
        
        # Update plot
        self.update_plot()
        
        # Update window title with annotation status
        current_annotation = self.annotations.get(event_id, 'unannotated')
        annotation_info = ANNOTATION_CATEGORIES[current_annotation]
        self.root.title(f"EOD Event Annotation Tool - {event_id} [{annotation_info['label']}]")
    
    def update_info_text(self):
        """Update the event information text display"""
        event_id = self.event_files[self.current_event_idx].stem
        
        # Clear existing text
        self.info_text.delete(1.0, tk.END)
        
        # Event ID and annotation status
        current_annotation = self.annotations.get(event_id, 'unannotated')
        annotation_info = ANNOTATION_CATEGORIES[current_annotation]
        
        info_text = f"Event ID: {event_id}\n"
        info_text += f"Annotation: {annotation_info['label']}\n"
        
        # Add fish count information if available and applicable
        if current_annotation == 'clear_fish' and event_id in self.fish_counts:
            fish_count = self.fish_counts[event_id]
            info_text += f"Fish Count: {fish_count}\n"
        
        # Features
        if event_id in self.event_features:
            features = self.event_features[event_id]
            info_text += f"Duration: {features.get('duration_seconds', 'N/A'):.1f}s, "
            info_text += f"EODs: {features.get('n_eods', 'N/A')}, "
            info_text += f"Channels: {features.get('n_channels_with_eods', 'N/A')}\n"
            
            if 'mean_ipi' in features:
                info_text += f"Mean IPI: {features['mean_ipi']:.3f}s, "
                info_text += f"IPI CV: {features['ipi_cv']:.2f}, "
                info_text += f"Amplitude CV: {features.get('amplitude_cv', 'N/A'):.2f}\n"
        
        self.info_text.insert(1.0, info_text)
    
    def update_plot(self):
        """Update the matplotlib plot with memory optimization"""
        # Clear existing plot and toolbar - thorough cleanup
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
            del self.canvas
        if hasattr(self, 'toolbar'):
            self.toolbar.destroy()
            del self.toolbar
        if hasattr(self, 'toolbar_frame'):
            self.toolbar_frame.destroy()
            del self.toolbar_frame
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # Close the matplotlib figure to free memory
            del self.fig
            self.fig = None
        if hasattr(self, 'ax'):
            del self.ax
            self.ax = None
        
        # Clear any remaining widgets in plot_frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Force garbage collection after cleanup
        gc.collect()
        
        if self.current_audio is None:
            return
        
        # Create matplotlib figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Plot differential signals with aggressive downsampling for very large files
        n_channels = self.current_audio.shape[1] if len(self.current_audio.shape) > 1 else 1
        # audio_size_mb = self.current_audio.nbytes / 1024 / 1024
        
        # # Adaptive downsampling based on file size
        # if audio_size_mb > 100:  # > 100MB
        #     max_points = 10000
        # elif audio_size_mb > 50:  # > 50MB
        #     max_points = 25000
        # elif audio_size_mb > 10:  # > 10MB
        #     max_points = 100000
        # else:
        #     max_points = 500000
        
        # print(f"Audio size: {audio_size_mb:.1f} MB, using max {max_points} points for display")
        
        offset_multiplier = np.max(np.abs(self.current_audio)) * 1.5
        
        if n_channels > 1:
            for i in range(n_channels - 1):
                # Calculate differential signal
                diff_signal = np.diff(self.current_audio[:, i:i+2], axis=1).flatten()

                # Downsample for display if too long
                # max_points = 500000  # Max points to display
                if len(diff_signal) > 50000000:
                    max_points = 25000000
                elif len(diff_signal) > 10000000:
                    max_points = round(len(diff_signal) // 2)
                else:
                    max_points = len(diff_signal)

                # Downsample for display
                step = max(1, len(diff_signal) // max_points)
                
                x_coords = np.arange(0, len(diff_signal), step)
                y_coords = diff_signal[::step] + i * offset_multiplier
                
                self.ax.plot(x_coords, y_coords, linewidth=0.5, label=f'Ch{i}-{i+1}', alpha=0.8)
                
                # Clean up temporary arrays
                del diff_signal, x_coords, y_coords
                
                # Plot EOD markers (limit to avoid memory issues)
                if self.current_eod_table is not None and len(self.current_eod_table) > 0:
                    ch_eods = self.current_eod_table[self.current_eod_table['eod_channel'] == i]
                    
                    if len(ch_eods) > 0 and 'peak_idx_event_snippet' in ch_eods.columns:
                        # Limit number of markers for very large datasets
                        if len(ch_eods) > 1000:
                            # Sample every nth marker to avoid too many points
                            marker_step = len(ch_eods) // 1000
                            ch_eods = ch_eods.iloc[::marker_step]
                        
                        peak_indices = ch_eods['peak_idx_event_snippet'].values
                        
                        # Recompute diff_signal just for markers (more memory efficient)
                        diff_signal_for_markers = np.diff(self.current_audio[:, i:i+2], axis=1).flatten()
                        valid_peaks = (peak_indices >= 0) & (peak_indices < len(diff_signal_for_markers)) & (~np.isnan(peak_indices))
                        
                        if np.any(valid_peaks):
                            valid_peak_samples = peak_indices[valid_peaks].astype(int)
                            peak_values = diff_signal_for_markers[valid_peak_samples] + i * offset_multiplier
                            
                            self.ax.plot(valid_peak_samples, peak_values, 'ro', markersize=2, alpha=0.7,
                                       label='Peaks' if i == 0 else "")
                        
                        # Clean up
                        del diff_signal_for_markers
        else:
            # Single channel - just plot it
            audio_data = self.current_audio.flatten() if len(self.current_audio.shape) > 1 else self.current_audio
            step = max(1, len(audio_data) // max_points)
            
            x_coords = np.arange(0, len(audio_data), step)
            y_coords = audio_data[::step]
            
            self.ax.plot(x_coords, y_coords, linewidth=0.5, label='Audio', alpha=0.8)
            
            # Clean up temporary arrays
            del x_coords, y_coords
        
        # Formatting
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Amplitude (stacked by channel)' if n_channels > 1 else 'Amplitude')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # Color-code by annotation
        event_id = self.event_files[self.current_event_idx].stem
        current_annotation = self.annotations.get(event_id, 'unannotated')
        bg_color = ANNOTATION_CATEGORIES[current_annotation]['color']
        
        self.ax.set_title(f'{event_id} - {ANNOTATION_CATEGORIES[current_annotation]["label"]}', 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.3))
        
        # Create canvas first (but don't pack yet)
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        
        # Add navigation toolbar for zoom/pan (pack first so it appears at top)
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(fill=tk.X, side=tk.TOP)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        # Now pack the canvas (it will appear below the toolbar)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas.draw()
        
        # Force garbage collection after plotting
        gc.collect()
    
    def annotate_event(self, category):
        """Annotate the current event with the given category"""
        if not self.event_files:
            return
            
        event_id = self.event_files[self.current_event_idx].stem
        self.annotations[event_id] = category
        
        print(f"Annotated {event_id} as {ANNOTATION_CATEGORIES[category]['label']}")
        
        # Check if fish count input is enabled and this is a clear_fish annotation
        if category == 'clear_fish' and self.fish_count_enabled:
            # Set up for fish count input
            self.waiting_for_fish_count = True
            self.pending_event_id = event_id
            self.fish_count_status.config(text="Waiting for fish count: [1], [2], or [u]nclear")
            
            # Update only the info text and window title, not the plot
            self.update_info_text()
            annotation_info = ANNOTATION_CATEGORIES[category]
            self.root.title(f"EOD Event Annotation Tool - {event_id} [{annotation_info['label']}] - Waiting for fish count")
        else:
            # Normal annotation - update display and auto-advance
            self.update_display()
            
            # Auto-advance to next event
            if self.current_event_idx < len(self.event_files) - 1:
                self.next_event()
    
    def next_event(self):
        """Navigate to next event"""
        self.clear_fish_count_waiting_state()
        if self.current_event_idx < len(self.event_files) - 1:
            self.current_event_idx += 1
            self.load_current_event()
    
    def previous_event(self):
        """Navigate to previous event"""
        self.clear_fish_count_waiting_state()
        if self.current_event_idx > 0:
            self.current_event_idx -= 1
            self.load_current_event()
    
    def clear_fish_count_waiting_state(self):
        """Clear fish count waiting state when navigating"""
        if self.waiting_for_fish_count:
            self.waiting_for_fish_count = False
            self.pending_event_id = None
            if self.fish_count_enabled:
                self.fish_count_status.config(text="Fish count input enabled")
    
    def go_to_event(self):
        """Navigate to specific event by number"""
        if not self.event_files:
            return
        
        try:
            # Get the event number from the entry field
            event_num_str = self.goto_var.get().strip()
            if not event_num_str:
                return
            
            event_num = int(event_num_str)
            
            # Convert to 0-based index (user inputs 1-based numbers)
            target_idx = event_num - 1
            
            # Validate range
            if target_idx < 0 or target_idx >= len(self.event_files):
                messagebox.showerror("Error", 
                                   f"Event number must be between 1 and {len(self.event_files)}")
                return
            
            # Clear fish count waiting state and navigate to the event
            self.clear_fish_count_waiting_state()
            self.current_event_idx = target_idx
            self.load_current_event()
            
            # Clear the entry field
            self.goto_var.set("")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid event number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to navigate to event: {str(e)}")
    
    def handle_key_1(self, event):
        """Handle key '1' - annotate as clear fish or set fish count to 1"""
        if self.root.focus_get() != self.goto_entry:
            if self.waiting_for_fish_count:
                self.set_fish_count('1')
            else:
                self.annotate_event('clear_fish')
    
    def handle_key_2(self, event):
        """Handle key '2' - annotate as clear noise or set fish count to 2"""
        if self.root.focus_get() != self.goto_entry:
            if self.waiting_for_fish_count:
                self.set_fish_count('2')
            else:
                self.annotate_event('clear_noise')
    
    def handle_key_3(self, event):
        """Handle key '3' - annotate as ambiguous"""
        if self.root.focus_get() != self.goto_entry:
            if not self.waiting_for_fish_count:  # Don't handle '3' during fish count input
                self.annotate_event('ambiguous')
    
    def handle_key_u(self, event):
        """Handle key 'u' - set fish count to unclear"""
        if self.root.focus_get() != self.goto_entry:
            if self.waiting_for_fish_count:
                self.set_fish_count('unclear')
    
    def set_fish_count(self, count):
        """Set fish count for the pending clear_fish annotation"""
        if not self.waiting_for_fish_count or not self.pending_event_id:
            return
        
        # Store the fish count
        self.fish_counts[self.pending_event_id] = count
        
        # Show a brief popup notification (without sound)
        self.show_fish_count_popup(count)
        
        # Clear the waiting state
        self.waiting_for_fish_count = False
        self.pending_event_id = None
        self.fish_count_status.config(text="Fish count input enabled")
        
        # Update display with fish count information
        self.update_display()
        
        # Auto-advance to next event
        if self.current_event_idx < len(self.event_files) - 1:
            self.next_event()
    
    def show_fish_count_popup(self, count):
        """Show a brief popup notification for fish count"""
        popup = tk.Toplevel(self.root)
        popup.title("Fish Count Set")
        popup.geometry("200x80")
        popup.transient(self.root)
        popup.grab_set()
        
        # Center the popup
        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Create message
        msg = f"Fish count: {count}"
        label = ttk.Label(popup, text=msg, font=('Arial', 12))
        label.pack(expand=True)
        
        # Auto-close after 1.5 seconds
        popup.after(1500, popup.destroy)
    
    def toggle_fish_count(self):
        """Toggle fish count input functionality"""
        self.fish_count_enabled = self.fish_count_var.get()
        
        if self.fish_count_enabled:
            self.fish_count_status.config(text="Fish count input enabled")
        else:
            self.fish_count_status.config(text="")
            # Clear any pending fish count state
            self.waiting_for_fish_count = False
            self.pending_event_id = None
    
    def update_audio_mode(self):
        """Update audio playback mode"""
        self.audio_playback_mode = self.audio_mode_var.get()
    
    def prepare_audio_for_playback(self):
        """Prepare audio data for stereo playback based on selected mode"""
        if self.current_audio is None:
            return None
            
        # Ensure we have multi-channel data
        if len(self.current_audio.shape) == 1:
            # Single channel audio - just duplicate for stereo
            return np.column_stack([self.current_audio, self.current_audio])
            
        n_channels = self.current_audio.shape[1]
        
        if self.audio_playback_mode == 'differential_mix':
            # Create differential signals and mix them
            if n_channels >= 2:
                # Calculate all differential signals
                diff_signals = []
                for i in range(n_channels - 1):
                    diff_sig = self.current_audio[:, i] - self.current_audio[:, i+1]
                    diff_signals.append(diff_sig)
                
                # Mix all differential signals
                mixed_signal = np.mean(diff_signals, axis=0)
                
                # Create stereo by duplicating the mixed signal
                playback_audio = np.column_stack([mixed_signal, mixed_signal])
            else:
                # Single channel - duplicate
                playback_audio = np.column_stack([self.current_audio[:, 0], self.current_audio[:, 0]])
                
        elif self.audio_playback_mode == 'stereo_channels':
            # Use selected channels for stereo playback
            ch1 = max(0, min(self.ch1_var.get(), n_channels - 1))
            ch2 = max(0, min(self.ch2_var.get(), n_channels - 1))
            
            left_channel = self.current_audio[:, ch1]
            right_channel = self.current_audio[:, ch2]
            
            playback_audio = np.column_stack([left_channel, right_channel])
            
        elif self.audio_playback_mode == 'mono_mix':
            # Mix all channels to mono, then duplicate for stereo
            mixed_signal = np.mean(self.current_audio, axis=1)
            playback_audio = np.column_stack([mixed_signal, mixed_signal])
        
        else:
            # Default: use first two channels or duplicate first channel
            if n_channels >= 2:
                playback_audio = self.current_audio[:, :2]
            else:
                playback_audio = np.column_stack([self.current_audio[:, 0], self.current_audio[:, 0]])
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(playback_audio))
        if max_val > 0:
            playback_audio = playback_audio / max_val * 0.8  # Leave some headroom
        
        return playback_audio
        
    def play_audio(self):
        """Play audio for current event"""
        if not AUDIO_AVAILABLE or self.current_audio is None:
            return
        
        try:
            # Prepare audio for stereo playback
            playback_audio = self.prepare_audio_for_playback()
            if playback_audio is None:
                return
            
            speed = self.speed_var.get() if hasattr(self, 'speed_var') else 1.0
            playback_rate = int(self.sample_rate * speed)
            
            sd.play(playback_audio, samplerate=playback_rate)
            
        except Exception as e:
            # Try fallback - just use first channel as mono
            try:
                if self.current_audio is not None and len(self.current_audio.shape) > 1:
                    fallback_audio = self.current_audio[:, 0]
                    sd.play(fallback_audio, samplerate=int(self.sample_rate))
            except Exception as e2:
                pass
    
    def stop_audio(self):
        """Stop audio playback"""
        if AUDIO_AVAILABLE:
            sd.stop()
    
    def update_speed_label(self, *args):
        """Update speed label"""
        speed = self.speed_var.get()
        self.speed_label.config(text=f"{speed:.1f}x")
    
    def save_annotations(self):
        """Save annotations to file"""
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save!")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Save Annotations",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            # Prepare data to save
            save_data = {
                'annotations': self.annotations,
                'fish_counts': self.fish_counts,
                'features': self.event_features,
                'metadata': {
                    'events_folder': str(self.events_folder),
                    'n_events': len(self.event_files),
                    'annotation_counts': {cat: sum(1 for a in self.annotations.values() if a == cat) 
                                        for cat in ANNOTATION_CATEGORIES.keys()},
                    'fish_count_counts': {count: sum(1 for c in self.fish_counts.values() if c == count) 
                                        for count in ['1', '2', 'unclear']},
                    'created_date': datetime.now().isoformat()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            messagebox.showinfo("Success", f"Annotations saved to {filename}")
    
    def load_annotations(self):
        """Load annotations from file"""
        filename = filedialog.askopenfilename(
            title="Load Annotations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    load_data = json.load(f)
                
                self.annotations.update(load_data.get('annotations', {}))
                self.fish_counts.update(load_data.get('fish_counts', {}))
                self.event_features.update(load_data.get('features', {}))
                
                # Update display
                self.update_display()
                
                messagebox.showinfo("Success", f"Annotations loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations:\n{str(e)}")
    
    def export_dataset(self):
        """Export annotated dataset for training"""
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to export!")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Export Training Dataset",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Create dataset DataFrame
                dataset_rows = []
                
                for event_file in self.event_files:
                    event_id = event_file.stem
                    annotation = self.annotations.get(event_id, 'unannotated')
                    features = self.event_features.get(event_id, {})
                    fish_count = self.fish_counts.get(event_id, None)
                    
                    if annotation != 'unannotated':  # Only include annotated events
                        row = {
                            'event_id': event_id,
                            'annotation': annotation,
                            'fish_count': fish_count if annotation == 'clear_fish' else None,
                            'file_path': str(event_file)
                        }
                        row.update(features)
                        dataset_rows.append(row)
                
                if not dataset_rows:
                    messagebox.showwarning("Warning", "No annotated events to export!")
                    return
                
                dataset_df = pd.DataFrame(dataset_rows)
                dataset_df.to_csv(filename, index=False)
                
                # Also save summary statistics
                summary_file = filename.replace('.csv', '_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write("Dataset Summary\n")
                    f.write("===============\n\n")
                    f.write(f"Total annotated events: {len(dataset_df)}\n")
                    f.write(f"Annotation breakdown:\n")
                    for category in ANNOTATION_CATEGORIES.keys():
                        if category != 'unannotated':
                            count = len(dataset_df[dataset_df['annotation'] == category])
                            f.write(f"  {ANNOTATION_CATEGORIES[category]['label']}: {count}\n")
                    
                    # Fish count breakdown
                    clear_fish_events = dataset_df[dataset_df['annotation'] == 'clear_fish']
                    if len(clear_fish_events) > 0:
                        f.write(f"\nFish count breakdown (Clear Fish events only):\n")
                        fish_count_stats = clear_fish_events['fish_count'].value_counts()
                        for count in ['1', '2', 'unclear']:
                            count_val = fish_count_stats.get(count, 0)
                            f.write(f"  {count} fish: {count_val}\n")
                        no_count = len(clear_fish_events[clear_fish_events['fish_count'].isna()])
                        if no_count > 0:
                            f.write(f"  No fish count: {no_count}\n")
                    
                    f.write(f"\nFeatures included: {len([c for c in dataset_df.columns if c not in ['event_id', 'annotation', 'fish_count', 'file_path']])}\n")
                    f.write(f"Export date: {datetime.now().isoformat()}\n")
                
                messagebox.showinfo("Success", 
                                  f"Dataset exported successfully!\n\n"
                                  f"Dataset: {filename}\n"
                                  f"Summary: {summary_file}\n"
                                  f"Annotated events: {len(dataset_df)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export dataset:\n{str(e)}")
    
    def run(self):
        """Start the annotation tool"""
        self.setup_gui()
        print("Interactive Event Annotation Tool")
        print("=================================")
        print("Keyboard shortcuts:")
        print("  1 - Clear Fish")
        print("  2 - Clear Noise") 
        print("  3 - Ambiguous")
        print("  ← → - Navigate events")
        print("  Space - Play audio")
        print("\nSelect an events folder to begin annotation...")
        
        # Add cleanup when window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing - clean up memory"""
        print("Cleaning up resources...")
        
        # Stop any audio playback
        if AUDIO_AVAILABLE:
            try:
                sd.stop()
            except:
                pass
        
        # Clean up current event resources
        self.cleanup_previous_event()
        
        # Clear event files list and other large data structures
        if hasattr(self, 'event_files'):
            del self.event_files
        if hasattr(self, 'annotations'):
            del self.annotations
        if hasattr(self, 'event_features'):
            del self.event_features
        
        # Final garbage collection
        gc.collect()
        
        print("Cleanup completed")
        self.root.destroy()

def main():
    """Main function"""
    tool = EventAnnotationTool()
    tool.run()

if __name__ == "__main__":
    main()

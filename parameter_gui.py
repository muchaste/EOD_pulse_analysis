"""
Parameter Configuration GUI for EOD Pulse Analysis

This module provides a comprehensive GUI for configuring analysis parameters,
file paths, and machine learning settings for EOD pulse extraction scripts.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import configparser
import os


class ParameterConfigGUI:
    """
    Comprehensive GUI for configuring all analysis parameters, paths, and ML settings.
    """
    
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("EOD Pulse Extraction - Parameter Configuration")
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Create canvas and scrollbar (wider for landscape layout)
        canvas = tk.Canvas(main_frame, height=1000, width=700)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        main_frame.rowconfigure(0, weight=1)
        
        # Store parameter widgets
        self.param_vars = {}
        self.path_vars = {}
        self.ml_vars = {}
        
        current_row = 0
        
        # ===== CONFIG FILE MANAGEMENT =====
        config_frame = ttk.LabelFrame(scrollable_frame, text="Configuration File", padding="10")
        config_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        ttk.Button(config_frame, text="Load Config", command=self.load_config).grid(row=0, column=0, padx=5)
        ttk.Button(config_frame, text="Save Config", command=self.save_config).grid(row=0, column=1, padx=5)
        
        # ===== PATH SETTINGS =====
        path_frame = ttk.LabelFrame(scrollable_frame, text="File and Folder Paths", padding="10")
        path_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Input path
        ttk.Label(path_frame, text="Input Folder (WAV files):").grid(row=0, column=0, sticky=tk.W)
        self.path_vars['input_path'] = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.path_vars['input_path'], width=60).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_folder('input_path')).grid(row=0, column=2)
        
        # Calibration file
        ttk.Label(path_frame, text="Calibration File:").grid(row=1, column=0, sticky=tk.W)
        self.path_vars['cal_file'] = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.path_vars['cal_file'], width=60).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_file('cal_file', 
                  [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=1, column=2)
        
        # Output path
        ttk.Label(path_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W)
        self.path_vars['output_path'] = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.path_vars['output_path'], width=60).grid(row=2, column=1, padx=5)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_folder('output_path')).grid(row=2, column=2)
        
        # Source parameter (special case - dropdown)
        ttk.Label(path_frame, text="Data Source:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.param_vars['source'] = tk.StringVar(value='multich_linear')
        source_combo = ttk.Combobox(path_frame, textvariable=self.param_vars['source'], 
                                    values=['multich_linear', '1ch_diff'], state='readonly', width=15)
        source_combo.grid(row=3, column=1, sticky=tk.W, padx=5)

        # ===== ML FILTERING SETTINGS =====
        ml_frame = ttk.LabelFrame(scrollable_frame, text="Machine Learning Filtering", padding="10")
        ml_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        self.ml_vars['use_ml_filtering'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(ml_frame, text="Use ML-based Fish vs Noise Classification", 
                       variable=self.ml_vars['use_ml_filtering'],
                       command=self.toggle_ml_fields).grid(row=0, column=0, columnspan=3, sticky=tk.W)
        
        ttk.Label(ml_frame, text="Classifier File:").grid(row=1, column=0, sticky=tk.W)
        self.ml_vars['classifier_path'] = tk.StringVar()
        self.ml_classifier_entry = ttk.Entry(ml_frame, textvariable=self.ml_vars['classifier_path'], 
                                             width=60, state='disabled')
        self.ml_classifier_entry.grid(row=1, column=1, padx=5)
        self.ml_classifier_btn = ttk.Button(ml_frame, text="Browse", state='disabled',
                                           command=lambda: self.browse_file('classifier_path',
                                           [("Pickle files", "*.pkl"), ("All files", "*.*")], ml=True))
        self.ml_classifier_btn.grid(row=1, column=2)
        
        ttk.Label(ml_frame, text="Fish Probability Threshold (0.1-0.9):").grid(row=2, column=0, sticky=tk.W)
        self.ml_vars['fish_probability_threshold'] = tk.DoubleVar(value=0.5)
        self.ml_threshold_entry = ttk.Entry(ml_frame, textvariable=self.ml_vars['fish_probability_threshold'], 
                                            width=10, state='disabled')
        self.ml_threshold_entry.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # ===== PULSE EXTRACTION PARAMETERS (Detection + Filtering + Options) =====
        pulse_frame = ttk.LabelFrame(scrollable_frame, text="Pulse Extraction", padding="10")
        pulse_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Left column 
        # Bandpass Filter Option
        self.param_vars['enable_bandpass_filter'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(pulse_frame, text="Bandpass Filter (Detection Only)", 
                       variable=self.param_vars['enable_bandpass_filter']).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        detection_params = [
            ('thresh', 'Detection Threshold:', 0.02, float),
            ('bandpass_low_cutoff', 'BP Low Cutoff (Hz):', 100, float),
            ('bandpass_high_cutoff', 'BP High Cutoff (Hz):', 6000, float),
            ('min_rel_slope_diff', 'Min Relative Slope Difference:', 0.25, float),
            ('min_width_us', 'Min Pulse Width (μs):', 30, float),
            ('max_width_us', 'Max Pulse Width (μs):', 1000, float),
            ('width_fac_detection', 'Width Factor for Detection:', 7.0, float),
            ('duplicate_samples', 'Duplicate Removal Samples:', 5, int)
        ]
        
        for i, (key, label, default, dtype) in enumerate(detection_params):
            ttk.Label(pulse_frame, text=label).grid(row=i+1, column=0, sticky=tk.W, pady=2)
            if dtype == float:
                self.param_vars[key] = tk.DoubleVar(value=default)
            else:
                self.param_vars[key] = tk.IntVar(value=default)
            ttk.Entry(pulse_frame, textvariable=self.param_vars[key], width=12).grid(
                row=i+1, column=1, sticky=tk.W, padx=5)
        
        # Add vertical separator
        ttk.Separator(pulse_frame, orient='vertical').grid(row=0, column=2, rowspan=15, sticky='ns', padx=10)
        
        # Right column
        
        filter_params = [
            ('window_length_extraction_us', 'Waveform Window Length (μs):', 4000, float),
            ('interp_factor', 'Interpolation Factor:', 3, int),
            ('amplitude_ratio_min', 'Min Amplitude Ratio:', 0.2, float),
            ('amplitude_ratio_max', 'Max Amplitude Ratio:', 4.0, float),
            ('peak_fft_freq_min', 'Min FFT Peak Frequency (Hz):', 100, float),
            ('peak_fft_freq_max', 'Max FFT Peak Frequency (Hz):', 10000, float),
        ]
        
        for i, (key, label, default, dtype) in enumerate(filter_params):
            ttk.Label(pulse_frame, text=label).grid(row=i, column=3, sticky=tk.W, pady=2)
            if dtype == float:
                self.param_vars[key] = tk.DoubleVar(value=default)
            else:
                self.param_vars[key] = tk.IntVar(value=default)
            ttk.Entry(pulse_frame, textvariable=self.param_vars[key], width=12).grid(
                row=i, column=4, sticky=tk.W, padx=5)
        
        # Waveform extraction method dropdown
        next_row = len(filter_params)
        ttk.Label(pulse_frame, text="Waveform Extraction:").grid(row=next_row, column=3, sticky=tk.W, pady=2)
        self.param_vars['waveform_extraction'] = tk.StringVar(value='Differential')
        waveform_combo = ttk.Combobox(pulse_frame, textvariable=self.param_vars['waveform_extraction'], 
                                      values=['Differential', 'PCA'], state='readonly', width=12)
        waveform_combo.grid(row=next_row, column=4, sticky=tk.W, padx=5)
        
        # Return only Head-to-Tail checkbox
        self.param_vars['return_diff'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(pulse_frame, text="Return only Head-to-Tail", 
                       variable=self.param_vars['return_diff']).grid(
            row=next_row+1, column=3, columnspan=2, sticky=tk.W, pady=2)
        
        # ===== EVENT CREATION PARAMETERS =====
        event_frame = ttk.LabelFrame(scrollable_frame, text="Event Creation Parameters", padding="10")
        event_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        # Create Events checkbox at the top
        self.param_vars['create_events'] = tk.BooleanVar(value=True)
        self.create_events_checkbox = ttk.Checkbutton(
            event_frame, text="Create Events from EODs", 
            variable=self.param_vars['create_events'],
            command=self.toggle_event_fields)
        self.create_events_checkbox.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Left column - Numeric parameters
        event_params = [
            ('min_amplitude', 'Min Event Amplitude:', 0.04, float),
            ('max_ipi_seconds', 'Max Inter-Pulse Interval (s):', 5.0, float),
            ('min_eods_premerge', 'Min EODs Pre-Merge:', 5, int),
            ('max_merge_gap_seconds', 'Max Merge Gap (s):', 0.5, float),
            ('min_eods_postmerge', 'Min EODs Post-Merge:', 30, int),
            ('margin', 'Event Time Margin (s):', 1.0, float),
        ]
        
        self.event_param_widgets = []  # Store widgets for enable/disable
        
        for i, (key, label, default, dtype) in enumerate(event_params):
            lbl = ttk.Label(event_frame, text=label)
            lbl.grid(row=i+1, column=0, sticky=tk.W, pady=2)
            if dtype == float:
                self.param_vars[key] = tk.DoubleVar(value=default)
            else:
                self.param_vars[key] = tk.IntVar(value=default)
            entry = ttk.Entry(event_frame, textvariable=self.param_vars[key], width=12)
            entry.grid(row=i+1, column=1, sticky=tk.W, padx=5)
            self.event_param_widgets.extend([lbl, entry])
        
        # Add vertical separator
        ttk.Separator(event_frame, orient='vertical').grid(row=0, column=2, rowspan=7, sticky='ns', padx=10)
        
        # Right column - Boolean options
        ttk.Label(event_frame, text="Event Processing Options", font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=3, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        event_boolean_params = [
            ('merge_events', 'Merge Channel Events', True),
            ('pre_merge_filtering', 'Filter Before Merging', True),
            ('post_merge_filtering', 'Filter After Merging', True),
        ]
        
        for i, (key, label, default) in enumerate(event_boolean_params):
            self.param_vars[key] = tk.BooleanVar(value=default)
            cb = ttk.Checkbutton(event_frame, text=label, variable=self.param_vars[key])
            cb.grid(row=i+1, column=3, columnspan=2, sticky=tk.W, pady=2)
            self.event_param_widgets.append(cb)
        
        # ===== PROCESSING OPTIONS =====
        options_frame = ttk.LabelFrame(scrollable_frame, text="Processing Options", padding="10")
        options_frame.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        current_row += 1
        
        boolean_params = [
            ('save_filtered_out', 'Save Filtered-Out Pulses', False),
            ('create_plots', 'Create Diagnostic Plots', True),
        ]
        
        for i, (key, label, default) in enumerate(boolean_params):
            self.param_vars[key] = tk.BooleanVar(value=default)
            ttk.Checkbutton(options_frame, text=label, variable=self.param_vars[key]).grid(
                row=i, column=0, sticky=tk.W, pady=2)
        
        # # Filtering method
        # ttk.Label(options_frame, text="Filtering Method:").grid(row=len(boolean_params)+1, column=0, sticky=tk.W, pady=2)
        # self.param_vars['filtering_method'] = tk.StringVar(value='basic')
        # filter_combo = ttk.Combobox(options_frame, textvariable=self.param_vars['filtering_method'], 
        #                             values=['basic', 'ml_enhanced'], state='readonly', width=15)
        # filter_combo.grid(row=len(boolean_params)+1, column=1, sticky=tk.W, padx=5)
        
        # ===== ACTION BUTTONS =====
        button_frame = ttk.Frame(scrollable_frame, padding="10")
        button_frame.grid(row=current_row, column=0, columnspan=3, pady=10)
        current_row += 1
        
        ttk.Button(button_frame, text="Start Processing", command=self.on_ok, 
                  style='Accent.TButton').grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).grid(row=0, column=1, padx=5)
        
        # Result flag
        self.result = None
        
        # Initialize event fields state
        self.toggle_event_fields()
        
    def toggle_ml_fields(self):
        """Enable/disable ML-related fields based on checkbox state."""
        if self.ml_vars['use_ml_filtering'].get():
            self.ml_classifier_entry.config(state='normal')
            self.ml_classifier_btn.config(state='normal')
            self.ml_threshold_entry.config(state='normal')
        else:
            self.ml_classifier_entry.config(state='disabled')
            self.ml_classifier_btn.config(state='disabled')
            self.ml_threshold_entry.config(state='disabled')
    
    def toggle_event_fields(self):
        """Enable/disable event-related fields based on create_events checkbox state."""
        state = 'normal' if self.param_vars['create_events'].get() else 'disabled'
        for widget in self.event_param_widgets:
            widget.config(state=state)
    
    def browse_folder(self, var_name):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(title=f"Select {var_name.replace('_', ' ').title()}")
        if folder:
            self.path_vars[var_name].set(folder)
    
    def browse_file(self, var_name, filetypes, ml=False):
        """Open file selection dialog."""
        initialdir = None
        if ml and self.path_vars.get('output_path'):
            initialdir = self.path_vars['output_path'].get()
        
        filename = filedialog.askopenfilename(
            title=f"Select {var_name.replace('_', ' ').title()}",
            filetypes=filetypes,
            initialdir=initialdir
        )
        if filename:
            if ml:
                self.ml_vars[var_name].set(filename)
            else:
                self.path_vars[var_name].set(filename)
    
    def save_config(self):
        """Save current configuration to a .cfg file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".cfg",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        config = configparser.ConfigParser()
        
        # Paths section
        config['Paths'] = {k: v.get() for k, v in self.path_vars.items()}
        
        # ML section
        config['MachineLearning'] = {
            'use_ml_filtering': str(self.ml_vars['use_ml_filtering'].get()),
            'classifier_path': self.ml_vars['classifier_path'].get(),
            'fish_probability_threshold': str(self.ml_vars['fish_probability_threshold'].get())
        }
        
        # Parameters section
        config['Parameters'] = {}
        for key, var in self.param_vars.items():
            if isinstance(var, tk.BooleanVar):
                config['Parameters'][key] = str(var.get())
            else:
                config['Parameters'][key] = str(var.get())
        
        try:
            with open(filename, 'w') as configfile:
                config.write(configfile)
            messagebox.showinfo("Success", f"Configuration saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
    
    def load_config(self):
        """Load configuration from a .cfg file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        config = configparser.ConfigParser()
        
        try:
            config.read(filename)
            
            # Load paths
            if 'Paths' in config:
                for key in self.path_vars:
                    if key in config['Paths']:
                        self.path_vars[key].set(config['Paths'][key])
            
            # Load ML settings
            if 'MachineLearning' in config:
                if 'use_ml_filtering' in config['MachineLearning']:
                    self.ml_vars['use_ml_filtering'].set(
                        config['MachineLearning'].getboolean('use_ml_filtering'))
                if 'classifier_path' in config['MachineLearning']:
                    self.ml_vars['classifier_path'].set(config['MachineLearning']['classifier_path'])
                if 'fish_probability_threshold' in config['MachineLearning']:
                    self.ml_vars['fish_probability_threshold'].set(
                        config['MachineLearning'].getfloat('fish_probability_threshold'))
                self.toggle_ml_fields()  # Update field states
            
            # Load parameters
            if 'Parameters' in config:
                for key, var in self.param_vars.items():
                    if key in config['Parameters']:
                        value_str = config['Parameters'][key]
                        if isinstance(var, tk.BooleanVar):
                            var.set(value_str.lower() in ('true', '1', 'yes'))
                        elif isinstance(var, tk.IntVar):
                            var.set(int(float(value_str)))  # Handle potential float strings
                        elif isinstance(var, tk.DoubleVar):
                            var.set(float(value_str))
                        else:  # StringVar
                            var.set(value_str)
            
            messagebox.showinfo("Success", f"Configuration loaded from:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
    
    def validate_inputs(self):
        """Validate all user inputs before proceeding."""
        errors = []
        
        # Check required paths
        if not self.path_vars['input_path'].get():
            errors.append("Input folder path is required")
        elif not os.path.exists(self.path_vars['input_path'].get()):
            errors.append("Input folder does not exist")
        
        if not self.path_vars['cal_file'].get():
            errors.append("Calibration file path is required")
        elif not os.path.exists(self.path_vars['cal_file'].get()):
            errors.append("Calibration file does not exist")
        
        if not self.path_vars['output_path'].get():
            errors.append("Output folder path is required")
        
        # Check ML settings if enabled
        if self.ml_vars['use_ml_filtering'].get():
            if not self.ml_vars['classifier_path'].get():
                errors.append("Classifier file is required when ML filtering is enabled")
            elif not os.path.exists(self.ml_vars['classifier_path'].get()):
                errors.append("Classifier file does not exist")
            
            threshold = self.ml_vars['fish_probability_threshold'].get()
            if threshold < 0.1 or threshold > 0.9:
                errors.append("Fish probability threshold must be between 0.1 and 0.9")
        
        # Check numeric parameters are valid
        try:
            if self.param_vars['thresh'].get() <= 0:
                errors.append("Detection threshold must be positive")
            if self.param_vars['min_width_us'].get() >= self.param_vars['max_width_us'].get():
                errors.append("Min pulse width must be less than max pulse width")
            
            if self.param_vars['enable_bandpass_filter'].get():
                if self.param_vars['bandpass_low_cutoff'].get() >= self.param_vars['bandpass_high_cutoff'].get():
                    errors.append("Bandpass low cutoff must be less than high cutoff")
                if self.param_vars['bandpass_low_cutoff'].get() <= 0:
                     errors.append("Bandpass cutoff must be positive")
                     
        except tk.TclError:
            errors.append("One or more numeric parameters have invalid values")
        
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False
        
        return True
    
    def on_ok(self):
        """Handle OK button click."""
        if not self.validate_inputs():
            return
        
        # Gather all configuration
        self.result = {
            'paths': {k: v.get() for k, v in self.path_vars.items()},
            'ml_settings': {k: v.get() for k, v in self.ml_vars.items()},
            'parameters': {}
        }
        
        # Convert parameter values to appropriate types
        for key, var in self.param_vars.items():
            self.result['parameters'][key] = var.get()
        
        self.parent.quit()
        self.parent.destroy()
    
    def on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.parent.quit()
        self.parent.destroy()

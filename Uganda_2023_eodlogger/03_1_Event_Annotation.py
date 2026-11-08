# -*- coding: utf-8 -*-
"""
03_1_Event_Annotation.py

This script provides an interactive GUI for manually annotating extracted events
to create training data for improved filtering algorithms.

Features:
- Load events from Script 04 output
- Display event plots with audio playback capability
- Manual annotation interface (Clear Fish / Clear Noise / Ambiguous)
- Optional tracking folder (04_1_Track_Fish output): pulse markers coloured by fish_id, IPI and
  location panels per fish, fish-level corrections (merge / drop fish), fish count 1-9 or
  'a' = accept the (corrected) tracker count
- Optional annotation sample CSV (03_3): shows only the sampled events of the opened folder and
  sets the tracking folder; 'g' flags a reappearing single fish (trail_gaps); '[' ']' step through
  30 s windows of long events, 'w' shows the whole event
- Pulse-level corrections: lasso on the IPI / location / amplitude panel selects pulses, a digit
  assigns them to that fish ('n' = new fish, Esc clears); stored per pulse (p1_idx_channel -> fish)
- Extract comprehensive features for each event
- Save annotated dataset for training filtering models

Authors: Stefan Mucha with Claude Sonnet 4.6
Date: September 2025
"""

import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MplPath
from matplotlib.widgets import LassoSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

FISH_COLORS = plt.get_cmap('tab10').colors + plt.get_cmap('Set2').colors
UNASSIGNED_COLOR = (0.6, 0.6, 0.6)

class EventAnnotationTool:
    def __init__(self):
        self.events_folder = None
        self.tracking_folder = None
        self.output_folder = None
        self.event_files = []
        self.current_event_idx = 0
        self.annotations = {}
        self.fish_counts = {}  # Store fish count information for clear_fish annotations
        self.event_features = {}
        # Tracking overlay: fish_id per pulse of the current event (from 04_1 *_tracked.csv),
        # fish-level corrections per event {'merge': [[keep, absorbed], ...], 'drop': [fish, ...]}
        self.current_fish_ids = None
        self.fish_corrections = {}
        self.tracker_results = {}
        # Annotation sample (03_3 CSV): restrict the event list to the sampled events of the opened folder
        self.sample_df = None
        self.sample_label = None
        # Flags per event (e.g. trail_gaps = one fish that leaves and reappears)
        self.event_flags = {}
        # Window stepper for long events: None = whole event, else (start_sample, end_sample)
        self.view_window_s = 30.0
        self.view_window = None
        # Pulse-level corrections {event: {"<p1_idx>_<eod_channel>": fish}}, lasso selection state
        self.pulse_corrections = {}
        self.lasso_selected = set()
        self.lasso_selectors = []
        self.panel_points = {}
        self.keep_xlim = None
        
        # Fish count settings
        self.fish_count_enabled = True   # fish count asked after every 'clear fish' (checkbox can switch it off)
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
        """Initialize the main GUI window (three compact control rows, the plot takes the rest)"""
        self.root = tk.Tk()
        self.root.title("EOD Event Annotation Tool")
        self.root.geometry("1400x900")
        try:
            self.root.state('zoomed')
        except tk.TclError:
            pass
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        
        # Row 1: folders and sample
        folder_frame = ttk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(folder_frame, text="Events Folder", command=self.select_events_folder).pack(side=tk.LEFT, padx=(0, 4))
        self.folder_label = ttk.Label(folder_frame, text="No folder selected")
        self.folder_label.pack(side=tk.LEFT)
        ttk.Button(folder_frame, text="Tracking Folder", command=self.select_tracking_folder).pack(side=tk.LEFT, padx=(16, 4))
        self.tracking_label = ttk.Label(folder_frame, text="No tracking folder")
        self.tracking_label.pack(side=tk.LEFT)
        ttk.Button(folder_frame, text="Sample CSV (03_3)", command=self.load_sample_csv).pack(side=tk.LEFT, padx=(16, 4))
        self.sample_label = ttk.Label(folder_frame, text="No sample filter")
        self.sample_label.pack(side=tk.LEFT)
        ttk.Button(folder_frame, text="Export Dataset", command=self.export_dataset).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(folder_frame, text="Load Annotations", command=self.load_annotations).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(folder_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.RIGHT, padx=(4, 0))
        
        # Row 2: navigation, annotation buttons, fish count
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(nav_frame, text="<", width=3, command=self.previous_event).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text=">", width=3, command=self.next_event).pack(side=tk.LEFT, padx=(2, 6))
        self.event_label = ttk.Label(nav_frame, text="Event: - / -")
        self.event_label.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(nav_frame, text="Go to:").pack(side=tk.LEFT)
        self.goto_var = tk.StringVar()
        self.goto_entry = ttk.Entry(nav_frame, textvariable=self.goto_var, width=5)
        self.goto_entry.pack(side=tk.LEFT, padx=(2, 2))
        self.goto_entry.bind('<Return>', lambda e: self.go_to_event())
        ttk.Button(nav_frame, text="Go", width=3, command=self.go_to_event).pack(side=tk.LEFT, padx=(0, 12))
        for category, info in ANNOTATION_CATEGORIES.items():
            if category != 'unannotated':
                ttk.Button(nav_frame, text=f"{info['label']} [{info['key']}]",
                           command=lambda cat=category: self.annotate_event(cat)).pack(side=tk.LEFT, padx=(0, 4))
        self.fish_count_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(nav_frame, text="Fish count input", variable=self.fish_count_var,
                        command=self.toggle_fish_count).pack(side=tk.LEFT, padx=(12, 4))
        self.fish_count_status = ttk.Label(nav_frame, text="", foreground='blue')
        self.fish_count_status.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(nav_frame, text="Help [h]", command=self.show_help).pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(nav_frame, mode='determinate', length=160)
        self.progress.pack(side=tk.RIGHT, padx=(10, 10))
        
        # Row 3: tracker corrections and audio
        correction_frame = ttk.Frame(main_frame)
        correction_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(correction_frame, text="Correction (m A B | d A | r):").pack(side=tk.LEFT, padx=(0, 4))
        self.correction_var = tk.StringVar()
        self.correction_entry = ttk.Entry(correction_frame, textvariable=self.correction_var, width=10)
        self.correction_entry.pack(side=tk.LEFT, padx=(0, 2))
        self.correction_entry.bind('<Return>', lambda e: self.apply_fish_correction())
        ttk.Button(correction_frame, text="Apply", width=6, command=self.apply_fish_correction).pack(side=tk.LEFT)
        self.correction_status = ttk.Label(correction_frame, text="", foreground='blue')
        self.correction_status.pack(side=tk.LEFT, padx=(8, 0))
        if AUDIO_AVAILABLE:
            self.ch1_var = tk.IntVar(value=0)
            self.ch2_var = tk.IntVar(value=1)
            ttk.Spinbox(correction_frame, from_=0, to=7, width=2, textvariable=self.ch2_var).pack(side=tk.RIGHT)
            ttk.Label(correction_frame, text="-").pack(side=tk.RIGHT)
            ttk.Spinbox(correction_frame, from_=0, to=7, width=2, textvariable=self.ch1_var).pack(side=tk.RIGHT)
            ttk.Label(correction_frame, text="Ch:").pack(side=tk.RIGHT, padx=(8, 2))
            self.audio_mode_var = tk.StringVar(value='differential_mix')
            for label, mode in [('Mono', 'mono_mix'), ('Stereo', 'stereo_channels'), ('Diff mix', 'differential_mix')]:
                ttk.Radiobutton(correction_frame, text=label, variable=self.audio_mode_var,
                                value=mode, command=self.update_audio_mode).pack(side=tk.RIGHT, padx=(0, 4))
            self.speed_label = ttk.Label(correction_frame, text="1.0x")
            self.speed_label.pack(side=tk.RIGHT, padx=(0, 8))
            self.speed_var = tk.DoubleVar(value=1.0)
            ttk.Scale(correction_frame, from_=0.1, to=3.0, variable=self.speed_var, length=80).pack(side=tk.RIGHT)
            self.speed_var.trace('w', self.update_speed_label)
            ttk.Label(correction_frame, text="Speed:").pack(side=tk.RIGHT, padx=(8, 2))
            ttk.Button(correction_frame, text="Stop", width=5, command=self.stop_audio).pack(side=tk.RIGHT)
            ttk.Button(correction_frame, text="Play audio", command=self.play_audio).pack(side=tk.RIGHT, padx=(8, 2))
        
        # Event info: three lines, no scrollbar (the convention sits in the help popup)
        self.info_text = tk.Text(main_frame, height=3, wrap=tk.NONE, relief=tk.FLAT, background=self.root.cget('background'))
        self.info_text.pack(fill=tk.X, pady=(0, 2))
        
        # Plot frame (takes all remaining space)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib canvas will be added here when an event is loaded
        self.plot_frame = plot_frame
        
        # Bind keyboard shortcuts (with focus checking to avoid conflicts with entry field)
        self.root.bind('<Key-1>', self.handle_key_1)
        self.root.bind('<Key-2>', self.handle_key_2)
        self.root.bind('<Key-3>', self.handle_key_3)
        for digit in [0, 4, 5, 6, 7, 8, 9]:
            self.root.bind(f'<Key-{digit}>', self.handle_key_digit)  # fish counts 4-9 / lasso assignment 0-9
        self.root.bind('<Key-n>', self.handle_key_n)  # lasso selection -> new fish
        self.root.bind('<Escape>', lambda e: self.clear_selection())
        self.root.bind('<Key-a>', self.handle_key_a)  # accept the (corrected) tracker fish count
        self.root.bind('<Key-u>', self.handle_key_u)  # For 'unclear' fish count
        self.root.bind('<Key-g>', self.handle_key_g)  # toggle the trail_gaps flag (fish leaves and reappears)
        self.root.bind('<bracketleft>', lambda e: self.step_window(-1))
        self.root.bind('<bracketright>', lambda e: self.step_window(+1))
        self.root.bind('<Key-w>', lambda e: self.step_window(0))  # whole event
        self.root.bind('<Key-h>', lambda e: None if self.keyboard_target_is_entry() else self.show_help())
        self.root.bind('<Key-l>', lambda e: None if self.keyboard_target_is_entry() else self.lasso_mode())
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

    def select_tracking_folder(self):
        """Select the 04_1_Track_Fish output folder whose *_tracked.csv files carry fish_id per pulse"""
        folder = filedialog.askdirectory(
            title="Select Tracking Folder (04_1_Track_Fish output)",
            initialdir=str(self.events_folder) if self.events_folder else os.getcwd()
        )
        if folder:
            self.tracking_folder = Path(folder)
            n_tracked = len(list(self.tracking_folder.glob("*_tracked.csv")))
            self.tracking_label.config(text=f"Tracking: {self.tracking_folder.name} ({n_tracked} tracked events)")
            if self.event_files:
                self.load_current_event()

    def keyboard_target_is_entry(self):
        focus = self.root.focus_get()
        return focus == self.goto_entry or focus == self.correction_entry

    def load_sample_csv(self):
        """Load an annotation sample (03_3 output); the event list is then restricted to the sampled events
        of the opened folder and the tracking folder is set from the CSV."""
        filename = filedialog.askopenfilename(title="Load annotation sample CSV (03_3)",
                                              filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filename:
            return
        self.sample_df = pd.read_csv(filename)
        self.sample_label.config(text=f"Sample: {os.path.basename(filename)} ({len(self.sample_df)} events)")
        if self.events_folder:
            self.load_event_files()

    def apply_sample_filter(self):
        """Keep only the events of the opened folder that are in the loaded sample; set the tracking folder."""
        if self.sample_df is None or not self.events_folder:
            return
        folder_norm = os.path.normcase(os.path.normpath(str(self.events_folder)))
        rows = self.sample_df[[os.path.normcase(os.path.normpath(str(p))) == folder_norm for p in self.sample_df['extraction_folder']]]
        if len(rows) == 0:
            messagebox.showwarning("Sample", "No sampled events in this folder")
            return
        stems = set(rows['base_stem'])
        self.event_files = [e for e in self.event_files if e['base_stem'] in stems]
        self.sample_strata = dict(zip(rows['base_stem'], rows['stratum']))
        track_folder = Path(str(rows['track_folder'].iloc[0]))
        if track_folder.exists():
            self.tracking_folder = track_folder
            n_tracked = len(list(self.tracking_folder.glob("*_tracked.csv")))
            self.tracking_label.config(text=f"Tracking: {self.tracking_folder.name} ({n_tracked} tracked events)")
        print(f"Sample filter: {len(self.event_files)} of the sampled events found in this folder")

    def step_window(self, direction):
        """Show a view_window_s long window of the current event (direction -1 / +1), or the whole event (0)."""
        if self.current_audio is None:
            return
        n_samples = len(self.current_audio)
        window_samples = int(self.view_window_s * self.sample_rate)
        if direction == 0 or n_samples <= window_samples:
            self.view_window = None
        else:
            start = 0 if self.view_window is None else self.view_window[0]
            if self.view_window is not None:
                start = max(0, min(n_samples - window_samples, start + direction * window_samples))
            self.view_window = (start, min(n_samples, start + window_samples))
        self.update_plot()

    def lasso_mode(self):
        """Leave the toolbar's zoom/pan mode so that left-drag on the lower panels is a lasso again."""
        toolbar = getattr(self, 'toolbar', None)
        if toolbar is None:
            return
        if toolbar.mode == 'zoom rect':
            toolbar.zoom()
        elif toolbar.mode == 'pan/zoom':
            toolbar.pan()
        self.correction_status.config(text="lasso mode (toolbar zoom/pan off)")

    def handle_key_g(self, event):
        """Key 'g' - toggle the trail_gaps flag of the current event (one fish that leaves and reappears)."""
        if self.keyboard_target_is_entry() or not self.event_files:
            return
        event_id = self.event_files[self.current_event_idx]['base_stem']
        flags = self.event_flags.setdefault(event_id, {})
        flags['trail_gaps'] = not flags.get('trail_gaps', False)
        self.update_info_text()

    def load_tracking(self, event_id):
        """Attach the tracker's fish_id to the current EOD table (matched on p1_idx and eod_channel);
        pulses without a tracked row get -1 (unassigned). Stored corrections are applied afterwards."""
        self.current_fish_ids = None
        if self.tracking_folder is None or self.current_eod_table is None:
            return
        tracked_file = self.tracking_folder / f"{event_id}_tracked.csv"
        if not tracked_file.exists():
            return
        tracked = pd.read_csv(tracked_file, usecols=['p1_idx', 'eod_channel', 'fish_id'])
        tracked = tracked.drop_duplicates(subset=['p1_idx', 'eod_channel'])
        merged = self.current_eod_table[['p1_idx', 'eod_channel']].merge(tracked, on=['p1_idx', 'eod_channel'], how='left')
        fish_ids = merged['fish_id'].fillna(-1).astype(int).values
        self.tracker_results[event_id] = {
            'tracker_n_fish': int(len(set(fish_ids[fish_ids >= 0]))),
            'tracker_pulses_per_fish': {int(f): int((fish_ids == f).sum()) for f in sorted(set(fish_ids[fish_ids >= 0]))},
            'n_unassigned': int((fish_ids < 0).sum()),
        }
        self.current_fish_ids = self.apply_corrections_to_ids(fish_ids, self.fish_corrections.get(event_id, {}))
        pulse_corr = self.pulse_corrections.get(event_id, {})
        if pulse_corr:
            keys = [f"{p}_{c}" for p, c in zip(self.current_eod_table['p1_idx'].values, self.current_eod_table['eod_channel'].values)]
            for i, key in enumerate(keys):
                if key in pulse_corr:
                    self.current_fish_ids[i] = int(pulse_corr[key])
        self.tracker_results[event_id]['corrected_n_fish'] = int(len(set(self.current_fish_ids[self.current_fish_ids >= 0])))
        self.tracker_results[event_id]['n_pulse_corrections'] = len(pulse_corr)

    def apply_corrections_to_ids(self, fish_ids, corrections):
        fish_ids = fish_ids.copy()
        for keep, absorbed in corrections.get('merge', []):
            fish_ids[fish_ids == absorbed] = keep
        for dropped in corrections.get('drop', []):
            fish_ids[fish_ids == dropped] = -1
        return fish_ids

    def apply_fish_correction(self):
        """Parse the correction entry: 'm A B' merges fish B into A, 'd A' drops fish A, 'r' resets."""
        if self.current_fish_ids is None:
            self.correction_status.config(text="No tracking data for this event")
            return
        event_id = self.event_files[self.current_event_idx]['base_stem']
        tokens = self.correction_var.get().strip().lower().split()
        corrections = self.fish_corrections.setdefault(event_id, {'merge': [], 'drop': []})
        if len(tokens) == 3 and tokens[0] == 'm':
            corrections['merge'].append([int(tokens[1]), int(tokens[2])])
        elif len(tokens) == 2 and tokens[0] == 'd':
            corrections['drop'].append(int(tokens[1]))
        elif tokens == ['r']:
            self.fish_corrections.pop(event_id, None)
            self.pulse_corrections.pop(event_id, None)
            self.lasso_selected = set()
        else:
            self.correction_status.config(text="Use: m A B | d A | r")
            return
        self.correction_var.set("")
        self.load_tracking(event_id)
        self.correction_status.config(text=f"Corrections: {self.fish_corrections.get(event_id, {})}, pulse-level: {len(self.pulse_corrections.get(event_id, {}))}")
        self.keep_xlim = self.ax.get_xlim() if self.ax is not None else None
        self.update_display()
        self.root.focus_set()

    def on_lasso(self, ax_panel, vertices):
        """Lasso callback: pulses inside the drawn path on this panel join the selection; a digit then
        assigns the selection to that fish (n = new fish), Escape clears it."""
        if ax_panel not in self.panel_points or len(vertices) < 3:
            return
        x, y = self.panel_points[ax_panel]
        if ax_panel.get_yscale() == 'log':
            # path vertices are in data coordinates; compare in log space to match the display
            path = MplPath([(vx, np.log10(vy)) for vx, vy in vertices if vy > 0])
            inside = path.contains_points(np.column_stack([x, np.log10(np.where(y > 0, y, np.nan))]))
        else:
            path = MplPath(vertices)
            inside = path.contains_points(np.column_stack([x, y]))
        inside = inside & ~np.isnan(y)
        self.lasso_selected |= set(np.where(inside)[0].tolist())
        self.correction_status.config(text=f"{len(self.lasso_selected)} pulses selected: press fish number (0-9), n = new fish, Esc = clear")
        self.keep_xlim = self.ax.get_xlim()
        self.update_plot()

    def assign_selection_to_fish(self, fish):
        """Store the lasso selection as pulse-level corrections for the current event and re-render."""
        if not self.lasso_selected or self.current_eod_table is None:
            return
        event_id = self.event_files[self.current_event_idx]['base_stem']
        if fish == 'n':
            fish = int(self.current_fish_ids.max()) + 1 if self.current_fish_ids is not None and len(self.current_fish_ids) else 0
        corr = self.pulse_corrections.setdefault(event_id, {})
        p1 = self.current_eod_table['p1_idx'].values
        ch = self.current_eod_table['eod_channel'].values
        for i in self.lasso_selected:
            corr[f"{p1[i]}_{ch[i]}"] = int(fish)
        n_assigned = len(self.lasso_selected)
        self.lasso_selected = set()
        self.load_tracking(event_id)
        self.correction_status.config(text=f"{n_assigned} pulses -> fish {fish}; pulse-level corrections: {len(corr)}")
        self.keep_xlim = self.ax.get_xlim()
        self.update_display()

    def clear_selection(self):
        if self.lasso_selected:
            self.lasso_selected = set()
            self.correction_status.config(text="selection cleared")
            self.keep_xlim = self.ax.get_xlim()
            self.update_plot()

    def current_fish_count_from_tracker(self):
        if self.current_fish_ids is None:
            return None
        return int(len(set(self.current_fish_ids[self.current_fish_ids >= 0])))
    
    def load_event_files(self):
        """Load all event files from the selected folder"""
        if not self.events_folder:
            return
            
        # Find all event audio files
        # Supports both whole-event files and split part files (_part_N suffix)
        audio_files = list(self.events_folder.glob("*_event_*.wav"))
        
        if not audio_files:
            messagebox.showerror("Error", "No event audio files found in the selected folder!")
            return
        
        # Group files by base stem (strip _part_N suffix if present)
        events_dict = {}
        for f in audio_files:
            stem = f.stem
            if '_part_' in stem:
                base_stem = stem[:stem.rfind('_part_')]
            else:
                base_stem = stem
            if base_stem not in events_dict:
                events_dict[base_stem] = []
            events_dict[base_stem].append(f)
        
        # Extract event number for sorting
        def extract_event_num(base_stem):
            parts = base_stem.split('_event_')
            if len(parts) == 2:
                try:
                    return int(parts[1].split('_')[0])
                except ValueError:
                    return 0
            return 0
        
        # Build sorted list of event entry dicts
        self.event_files = []
        for base_stem in sorted(events_dict.keys(), key=extract_event_num):
            parts_list = sorted(events_dict[base_stem], key=lambda p: p.stem)
            self.event_files.append({
                'base_stem': base_stem,
                'event_num': extract_event_num(base_stem),
                'audio_parts': parts_list,
                'is_split': len(parts_list) > 1,
            })
        
        print(f"Found {len(self.event_files)} event files")
        self.sample_strata = {}
        self.apply_sample_filter()
        
        # Initialize annotations and fish count dictionaries
        self.annotations = {e['base_stem']: 'unannotated' for e in self.event_files}
        self.fish_counts = {}  # Will be populated as needed
        self.event_flags = {}
        
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
        self.view_window = None
        self.lasso_selected = set()
        self.keep_xlim = None
            
        event_file = self.event_files[self.current_event_idx]
        event_id = event_file['base_stem']
        
        try:
            # Load audio file
            self.load_event_audio(event_file)
            
            # Load EOD table if available
            # Old format: loggerID_loggerTimestamp_event_###_eventTimestamp.csv
            # Old format: loggerID_loggerTimestamp_event_###_eod_table.csv
            eod_file = self.events_folder / f"{event_id}_eod_table.csv"
            print(f"Loading EOD table from: {eod_file}")
            
            self.current_eod_table = None
            if eod_file.exists():
                self.current_eod_table = pd.read_csv(eod_file)
            self.load_tracking(event_id)
            self.correction_status.config(text=f"Corrections: {self.fish_corrections[event_id]}" if event_id in self.fish_corrections else "")
            
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
    
    def load_event_audio(self, entry):
        """Load audio data for the current event with memory optimization.
        entry: dict with 'audio_parts' list of Path objects (may be a single-item list)
        """
        
        # Clean up any existing audio data first
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            del self.current_audio
            self.current_audio = None
            gc.collect()
        
        audio_parts = entry['audio_parts'] if isinstance(entry, dict) else [entry]
        
        if AUDIO_AVAILABLE:
            try:
                if len(audio_parts) == 1:
                    self.current_audio, self.sample_rate = sf.read(str(audio_parts[0]))
                else:
                    parts_data = []
                    for part_file in audio_parts:
                        part_audio, self.sample_rate = sf.read(str(part_file))
                        parts_data.append(part_audio)
                    self.current_audio = np.concatenate(parts_data, axis=0)
                    del parts_data
                    gc.collect()
                if self.current_audio.dtype != np.float32:
                    self.current_audio = self.current_audio.astype(np.float32)
                print(f"Loaded audio: {self.current_audio.shape}, {self.current_audio.dtype}, {self.current_audio.nbytes / 1024 / 1024:.1f} MB")
            except Exception as e:
                self.current_audio = None
                print(f"Failed to load audio with soundfile: {e}")
        else:
            # Load with scipy as fallback
            try:
                import scipy.io.wavfile as wav
                if len(audio_parts) == 1:
                    self.sample_rate, audio_data = wav.read(str(audio_parts[0]))
                    if audio_data.dtype == np.int16:
                        self.current_audio = audio_data.astype(np.float32) / 32768.0
                    else:
                        self.current_audio = audio_data.astype(np.float32)
                    del audio_data
                else:
                    parts_data = []
                    for part_file in audio_parts:
                        rate, part_data = wav.read(str(part_file))
                        self.sample_rate = rate
                        if part_data.dtype == np.int16:
                            parts_data.append(part_data.astype(np.float32) / 32768.0)
                        else:
                            parts_data.append(part_data.astype(np.float32))
                    self.current_audio = np.concatenate(parts_data, axis=0)
                    del parts_data
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
        summary_file = self.events_folder / "all_event_summaries.csv"
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            
            # Extract event number from current event entry
            event_num = self.event_files[self.current_event_idx]['event_num']
            
            print(f"Looking for event_id {event_num} in summary")
            
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
            print("No all_event_summaries.csv file found")
    
    def extract_event_features(self):
        """Extract comprehensive features for the current event"""
        event_id = self.event_files[self.current_event_idx]['base_stem']
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
        event_id = self.event_files[self.current_event_idx]['base_stem']
        
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
        """Update the event information text display (three lines)"""
        event_id = self.event_files[self.current_event_idx]['base_stem']
        self.info_text.delete(1.0, tk.END)
        current_annotation = self.annotations.get(event_id, 'unannotated')
        annotation_info = ANNOTATION_CATEGORIES[current_annotation]
        
        line1 = f"{event_id}   |   {annotation_info['label']}"
        if current_annotation == 'clear_fish' and event_id in self.fish_counts:
            line1 += f", {self.fish_counts[event_id]} fish"
        if getattr(self, 'sample_strata', None) and event_id in self.sample_strata:
            line1 += f"   |   stratum {self.sample_strata[event_id]}"
        flags = self.event_flags.get(event_id, {})
        line1 += f"   |   trail_gaps={flags.get('trail_gaps', False)}"
        if self.view_window is not None:
            line1 += f"   |   view {self.view_window[0] / self.sample_rate:.0f}-{self.view_window[1] / self.sample_rate:.0f} s"
        
        line2 = ""
        if event_id in self.event_features:
            features = self.event_features[event_id]
            line2 = (f"Duration {features.get('duration_seconds', float('nan')):.1f} s, EODs {features.get('n_eods', 'N/A')}, "
                     f"channels {features.get('n_channels_with_eods', 'N/A')}")
            if 'mean_ipi' in features:
                line2 += (f", mean IPI {features['mean_ipi']:.3f} s, IPI CV {features['ipi_cv']:.2f}, "
                          f"amplitude CV {features.get('amplitude_cv', float('nan')):.2f}")
        
        line3 = ""
        if event_id in self.tracker_results and self.current_fish_ids is not None:
            tr = self.tracker_results[event_id]
            per_fish = ', '.join(f"{f}: {n}" for f, n in tr['tracker_pulses_per_fish'].items())
            line3 = (f"Tracker {tr['tracker_n_fish']} fish (pulses {per_fish}; unassigned {tr['n_unassigned']})   ->   "
                     f"corrected {tr['corrected_n_fish']} fish, {tr.get('n_pulse_corrections', 0)} pulse corrections   ['a' accepts]")
        
        self.info_text.insert(1.0, line1 + "\n" + line2 + "\n" + line3)

    def show_help(self):
        """Keyboard shortcuts and the labelling convention in a popup."""
        text = (
            "Keys\n"
            "  1 / 2 / 3   clear fish / clear noise / ambiguous\n"
            "  while waiting for a fish count: 1-9 count, a = accept corrected tracker count, u = unclear\n"
            "  g   toggle trail_gaps (one fish that leaves and reappears)\n"
            "  [ ] step 30 s windows, w whole event; Left/Right previous/next event; Space play audio\n"
            "  Lasso (left drag) on the IPI / location / amplitude panel selects pulses; l = leave toolbar zoom/pan mode\n"
            "  (or click the active toolbar button again); then 0-9 assigns them to that fish, n = new fish, Esc clears\n"
            "  Correction entry: m A B = merge fish B into A, d A = drop fish A, r = reset all corrections of the event\n\n"
            "Convention\n"
            "  1 fish: one continuous trail (one IPI band, one location trace, one amplitude envelope);\n"
            "          pauses with the same width / location / IPI before and after = 1 fish + trail_gaps\n"
            "  2 fish: two trails separable by eye. Merge tracker over-splits (m A B), drop spurious fragments (d A),\n"
            "          separate mixed fish_ids with the lasso, then accept the count with 'a'\n"
            "  Not separable with reasonable effort: fish count 'u' (unclear). Noise: 2.\n"
        )
        popup = tk.Toplevel(self.root)
        popup.title("Help")
        popup.transient(self.root)
        label = tk.Label(popup, text=text, justify=tk.LEFT, font=('Consolas', 10), padx=12, pady=10)
        label.pack()
        popup.bind('<Escape>', lambda e: popup.destroy())
        popup.bind('<Key-h>', lambda e: popup.destroy())
        popup.focus_set()
    
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
        
        # Create matplotlib figure; with tracking data three extra panels (IPI, location, amplitude) share the x-axis
        show_tracks = self.current_fish_ids is not None and self.current_eod_table is not None
        if show_tracks:
            self.fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.06, figure=self.fig, left=0.05, right=0.99, top=0.96, bottom=0.05)
            self.ax = self.fig.add_subplot(gs[0])
            ax_ipi = self.fig.add_subplot(gs[1], sharex=self.ax)
            ax_loc = self.fig.add_subplot(gs[2], sharex=self.ax)
            ax_amp = self.fig.add_subplot(gs[3], sharex=self.ax)
            fish_present = sorted(set(self.current_fish_ids))
            fish_color = {f: (UNASSIGNED_COLOR if f < 0 else FISH_COLORS[i % len(FISH_COLORS)])
                          for i, f in enumerate([f for f in fish_present if f >= 0])}
            fish_color[-1] = UNASSIGNED_COLOR
            marker_colors = np.array([fish_color[f] for f in self.current_fish_ids])
            x_pulse = self.current_eod_table['p1_idx'].values
            t_pulse = pd.to_datetime(self.current_eod_table['timestamp']).values.astype('datetime64[ns]').astype(np.int64) / 1e9
            # per-pulse panel values (IPI to the previous pulse of the same fish; NaN for the first), kept for the lasso
            y_ipi = np.full(len(x_pulse), np.nan)
            for f in fish_present:
                idx = np.where(self.current_fish_ids == f)[0]
                idx = idx[np.argsort(x_pulse[idx])]
                y_ipi[idx[1:]] = np.diff(t_pulse[idx]) * 1e3
            y_loc = self.current_eod_table['pulse_location'].values if 'pulse_location' in self.current_eod_table.columns else np.full(len(x_pulse), np.nan)
            y_amp = self.current_eod_table['eod_amplitude'].values
            self.panel_points = {ax_ipi: (x_pulse, y_ipi), ax_loc: (x_pulse, y_loc), ax_amp: (x_pulse, y_amp)}
            for f in fish_present:
                m = self.current_fish_ids == f
                label = f'fish {f} (n={m.sum()})' if f >= 0 else f'unassigned (n={m.sum()})'
                ax_ipi.plot(x_pulse[m], y_ipi[m], '.', ms=3, color=fish_color[f], label=label)
                ax_loc.plot(x_pulse[m], y_loc[m], '.', ms=3, color=fish_color[f])
                ax_amp.plot(x_pulse[m], y_amp[m], '.', ms=3, color=fish_color[f])
            if len(self.lasso_selected) > 0:
                sel = np.array(sorted(self.lasso_selected))
                for ax_p, (xp, yp) in self.panel_points.items():
                    ax_p.plot(xp[sel], yp[sel], 'o', ms=7, mfc='none', mec='k', mew=1.2)
            ax_ipi.set_ylabel('IPI (ms)')
            ax_ipi.set_yscale('log')
            ax_ipi.legend(loc='upper right', fontsize=7, ncol=4)
            ax_ipi.grid(True, alpha=0.3)
            ax_loc.set_ylabel('location (el.)')
            ax_loc.grid(True, alpha=0.3)
            ax_amp.set_ylabel('amplitude (V)')
            ax_amp.set_yscale('log')
            ax_amp.set_xlabel('Sample')
            ax_amp.grid(True, alpha=0.3)
            plt.setp(self.ax.get_xticklabels(), visible=False)
            plt.setp(ax_ipi.get_xticklabels(), visible=False)
            plt.setp(ax_loc.get_xticklabels(), visible=False)
            # lasso selection of pulses on the three lower panels (references kept, else garbage-collected)
            self.lasso_selectors = [LassoSelector(ax_p, onselect=lambda verts, ax_p=ax_p: self.on_lasso(ax_p, verts),
                                                  useblit=False, button=1) for ax_p in self.panel_points]
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 6))
            self.fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.07)
        
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
        
        offset_multiplier = np.max(self.current_eod_table['eod_amplitude']) * 1.5
        
        if n_channels > 1:
            
            for ch in range(n_channels - 1):
                # Calculate differential signal
                diff_signal = np.diff(self.current_audio[:, ch:ch+2], axis=1).flatten()

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
                
                self.ax.plot(x_coords, diff_signal[::step] + (ch + 0.5) * offset_multiplier, linewidth=0.5, label=f'Ch{ch}-{ch+1}', alpha=0.8)
                
                # Clean up temporary arrays
                del x_coords
                
                # Plot EOD markers (limit to avoid memory issues)
                if self.current_eod_table is not None and len(self.current_eod_table) > 0:
                    ch_mask = (self.current_eod_table['eod_channel'] == ch).values
                    ch_eods = self.current_eod_table[ch_mask]
                    
                    if len(ch_eods) > 0:
                        p1_idc = ch_eods['p1_idx'].values
                        p2_idc = ch_eods['p2_idx'].values

                        if show_tracks:
                            # markers coloured by (corrected) fish_id; legend entries once per fish
                            self.ax.scatter(p1_idc, diff_signal[p1_idc] + (ch + 0.5) * offset_multiplier,
                                            s=12, c=marker_colors[ch_mask], alpha=0.9, zorder=3)
                            self.ax.scatter(p2_idc, diff_signal[p2_idc] + (ch + 0.5) * offset_multiplier,
                                            s=12, c=marker_colors[ch_mask], alpha=0.9, marker='s', zorder=3)
                            if 'pulse_location' in ch_eods.columns:
                                self.ax.vlines(p1_idc, (ch + 0.5) * offset_multiplier, ch_eods['pulse_location'].values * offset_multiplier,
                                               colors=marker_colors[ch_mask], linewidth=0.5, alpha=0.5)
                            continue
                        
                        # # Recompute diff_signal just for markers (more memory efficient)
                        # valid_peaks = (p1_idc >= 0) & (p1_idc < len(diff_signal)) & (~np.isnan(p1_idc))
                        
                        # if np.any(valid_peaks):
                            # valid_peak_samples = p1_idc[valid_peaks].astype(int)
                            
                        self.ax.plot(p1_idc, diff_signal[p1_idc] + (ch + 0.5) * offset_multiplier, 'o', markersize=1, alpha=0.7, color='red',
                                    label='P1' if ch == 0 else "")
                        self.ax.plot(p2_idc, diff_signal[p2_idc] + (ch + 0.5) * offset_multiplier, 'o', markersize=1, alpha=0.7, color='blue',
                                    label='P2' if ch == 0 else "")
                        # Plot pulse_location visualization for this channel
                        if 'pulse_location' in ch_eods.columns:
                            for i, p1_idx in enumerate(p1_idc):
                                peak_loc = ch_eods['pulse_location'].iloc[i]
                                # p1_idx = ch_eods['p1_idx'].iloc[i]
                                # Draw thin line from channel offset to pulse_location offset
                                plt.plot([p1_idx, p1_idx], [(ch + 0.5) * offset_multiplier, peak_loc * offset_multiplier], 
                                        'k-', linewidth=0.5, alpha=0.6)
                                # Mark pulse_location with small black marker
                                plt.plot(p1_idx, peak_loc * offset_multiplier, 'ko', markersize=1, alpha=0.8)
                        
                # Clean up
                del diff_signal
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
        if not show_tracks:
            self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Amplitude (stacked by channel)' if n_channels > 1 else 'Amplitude')
        self.ax.legend(loc='upper right', fontsize=7, ncol=4)
        self.ax.grid(True, alpha=0.3)
        
        # Color-code by annotation
        event_id = self.event_files[self.current_event_idx]['base_stem']
        current_annotation = self.annotations.get(event_id, 'unannotated')
        bg_color = ANNOTATION_CATEGORIES[current_annotation]['color']
        
        self.ax.set_title(f'{event_id} - {ANNOTATION_CATEGORIES[current_annotation]["label"]}', fontsize=10, pad=2,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.3))
        
        # Draw part boundaries for split events
        entry = self.event_files[self.current_event_idx]
        if entry['is_split'] and self.current_eod_table is not None and 'part_audio_start_sample' in self.current_eod_table.columns:
            boundary_samples = sorted(self.current_eod_table['part_audio_start_sample'].unique())
            for i, bsamp in enumerate(boundary_samples):
                if bsamp > 0:
                    self.ax.axvline(x=bsamp, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                part_num = self.current_eod_table.loc[
                    self.current_eod_table['part_audio_start_sample'] == bsamp, 'audio_part'
                ].iloc[0] if 'audio_part' in self.current_eod_table.columns else (i + 1)
                self.ax.text(bsamp + 100, self.ax.get_ylim()[1] * 0.95, f'_part_{part_num}',
                             fontsize=7, color='gray', va='top')
        
        # Create canvas first (but don't pack yet)
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        
        # Add navigation toolbar for zoom/pan (pack first so it appears at top)
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(fill=tk.X, side=tk.TOP)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        # Window stepper / zoom kept across a re-render (lasso assignment, correction): the full view is
        # pushed to the toolbar's history first, so that "home" returns to the whole event
        if self.keep_xlim is not None or self.view_window is not None:
            self.toolbar.push_current()
            if self.keep_xlim is not None:
                self.ax.set_xlim(self.keep_xlim)
                self.keep_xlim = None
            else:
                self.ax.set_xlim(self.view_window[0], self.view_window[1])
            self.toolbar.push_current()
        
        # Now pack the canvas (it will appear below the toolbar)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas.draw()
        
        # Force garbage collection after plotting
        gc.collect()
    
    def annotate_event(self, category):
        """Annotate the current event with the given category"""
        if not self.event_files:
            return
            
        event_id = self.event_files[self.current_event_idx]['base_stem']
        self.annotations[event_id] = category
        
        print(f"Annotated {event_id} as {ANNOTATION_CATEGORIES[category]['label']}")
        
        # Check if fish count input is enabled and this is a clear_fish annotation
        if category == 'clear_fish' and self.fish_count_enabled:
            # Set up for fish count input
            self.waiting_for_fish_count = True
            self.pending_event_id = event_id
            self.fish_count_status.config(text="Waiting for fish count: [1]-[9], [a]ccept tracker count, or [u]nclear")
            
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
        if not self.keyboard_target_is_entry():
            if self.lasso_selected:
                self.assign_selection_to_fish(1)
            elif self.waiting_for_fish_count:
                self.set_fish_count('1')
            else:
                self.annotate_event('clear_fish')
    
    def handle_key_2(self, event):
        """Handle key '2' - annotate as clear noise or set fish count to 2"""
        if not self.keyboard_target_is_entry():
            if self.lasso_selected:
                self.assign_selection_to_fish(2)
            elif self.waiting_for_fish_count:
                self.set_fish_count('2')
            else:
                self.annotate_event('clear_noise')
    
    def handle_key_3(self, event):
        """Handle key '3' - annotate as ambiguous, or set fish count to 3 while waiting for a count"""
        if not self.keyboard_target_is_entry():
            if self.lasso_selected:
                self.assign_selection_to_fish(3)
            elif self.waiting_for_fish_count:
                self.set_fish_count('3')
            else:
                self.annotate_event('ambiguous')

    def handle_key_digit(self, event):
        """Keys 0, 4-9 - assign a lasso selection to that fish, or fish count while waiting for a count"""
        if self.keyboard_target_is_entry():
            return
        if self.lasso_selected:
            self.assign_selection_to_fish(int(event.char))
        elif self.waiting_for_fish_count and event.char != '0':
            self.set_fish_count(event.char)

    def handle_key_n(self, event):
        """Key 'n' - assign a lasso selection to a new fish"""
        if not self.keyboard_target_is_entry() and self.lasso_selected:
            self.assign_selection_to_fish('n')

    def handle_key_a(self, event):
        """Key 'a' - accept the (corrected) tracker fish count"""
        if not self.keyboard_target_is_entry() and self.waiting_for_fish_count:
            count = self.current_fish_count_from_tracker()
            if count is None:
                self.fish_count_status.config(text="No tracking data: enter the count manually")
            else:
                self.set_fish_count(str(count))
    
    def handle_key_u(self, event):
        """Handle key 'u' - set fish count to unclear"""
        if not self.keyboard_target_is_entry():
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
                'fish_corrections': self.fish_corrections,
                'pulse_corrections': self.pulse_corrections,
                'event_flags': self.event_flags,
                'sample_strata': getattr(self, 'sample_strata', {}),
                'tracker_results': self.tracker_results,
                'features': self.event_features,
                'metadata': {
                    'events_folder': str(self.events_folder),
                    'tracking_folder': str(self.tracking_folder) if self.tracking_folder else None,
                    'n_events': len(self.event_files),
                    'annotation_counts': {cat: sum(1 for a in self.annotations.values() if a == cat) 
                                        for cat in ANNOTATION_CATEGORIES.keys()},
                    'fish_count_counts': {count: sum(1 for c in self.fish_counts.values() if c == count) 
                                        for count in sorted(set(self.fish_counts.values()))},
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
                self.fish_corrections.update(load_data.get('fish_corrections', {}))
                self.pulse_corrections.update(load_data.get('pulse_corrections', {}))
                self.event_flags.update(load_data.get('event_flags', {}))
                self.tracker_results.update(load_data.get('tracker_results', {}))
                self.event_features.update(load_data.get('features', {}))
                
                # Update display (re-attach tracking so stored corrections apply to the current event)
                if self.event_files:
                    self.load_tracking(self.event_files[self.current_event_idx]['base_stem'])
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
                
                for event in self.event_files:
                    event_id = event['base_stem']
                    annotation = self.annotations.get(event_id, 'unannotated')
                    features = self.event_features.get(event_id, {})
                    fish_count = self.fish_counts.get(event_id, None)
                    
                    if annotation != 'unannotated':  # Only include annotated events
                        row = {
                            'event_id': event_id,
                            'annotation': annotation,
                            'fish_count': fish_count if annotation == 'clear_fish' else None,
                            'file_path': str(event['audio_parts'][0])
                        }
                        tracker = self.tracker_results.get(event_id, {})
                        row['tracker_n_fish'] = tracker.get('tracker_n_fish')
                        row['corrected_n_fish'] = tracker.get('corrected_n_fish')
                        row['fish_corrections'] = json.dumps(self.fish_corrections[event_id]) if event_id in self.fish_corrections else ''
                        row['n_pulse_corrections'] = len(self.pulse_corrections.get(event_id, {}))
                        row['trail_gaps'] = self.event_flags.get(event_id, {}).get('trail_gaps', False)
                        row['sample_stratum'] = getattr(self, 'sample_strata', {}).get(event_id, '')
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
                        for count in fish_count_stats.index:
                            f.write(f"  {count} fish: {fish_count_stats[count]}\n")
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
        print("  while waiting for a fish count: 1-9, a = accept tracker count, u = unclear")
        print("  g - toggle trail_gaps flag; [ ] - step 30 s windows, w - whole event")
        print("  Lasso on the IPI/location/amplitude panel selects pulses; 0-9 assigns them to that fish, n = new fish, Esc = clear")
        print("  Fish correction entry: m A B (merge B into A), d A (drop A), r (reset)")
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

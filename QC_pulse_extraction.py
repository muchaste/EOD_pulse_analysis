"""
Quality Control Script for EOD Pulse Extraction
Compares accepted vs filtered-out pulses to evaluate filtering effectiveness
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns
from scipy import stats
import tkinter as tk
from tkinter import filedialog

# Helper to convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

# Import shared functions from eod_functions module
from eod_functions import load_variable_length_waveforms, calculate_waveform_stats, plot_waveform_comparison, compare_table_features

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def main():
    """Main QC analysis function."""
    
    # Select files
    root = tk.Tk()
    root.withdraw()
    
    print("Select the OUTPUT directory containing the extracted pulse files...")
    output_dir = filedialog.askdirectory(title="Select Output Directory with Pulse Files")
    
    if not output_dir:
        print("No directory selected. Exiting.")
        return
    
    # Find all pulse files
    output_path = Path(output_dir)
    eod_tables = list(output_path.glob("*_eod_table.csv"))
    
    if not eod_tables:
        print("No EOD table files found. Make sure you've run the pulse extraction first.")
        return
    
    print(f"Found {len(eod_tables)} EOD table files:")
    for i, table_file in enumerate(eod_tables):
        print(f"  {i+1}. {table_file.stem}")
    
    # Select which file to analyze
    while True:
        try:
            choice = int(input(f"Select file to analyze (1-{len(eod_tables)}): ")) - 1
            if 0 <= choice < len(eod_tables):
                break
            else:
                print(f"Please enter a number between 1 and {len(eod_tables)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_table = eod_tables[choice]
    file_prefix = selected_table.stem.replace('_eod_table', '')
    
    print(f"\nAnalyzing: {selected_table.name}")
    print(f"File prefix: {file_prefix}")
    
    # Load data
    print("Loading accepted pulses...")
    accepted_table = pd.read_csv(selected_table)
    accepted_waveforms_path = output_path / f"{file_prefix}_eod_waveforms"
    accepted_waveforms = load_variable_length_waveforms(str(accepted_waveforms_path))
    
    print("Loading filtered-out pulses...")
    filtered_table_path = output_path / f"{file_prefix}_eod_table_filteredout.csv"
    filtered_waveforms_path = output_path / f"{file_prefix}_eod_waveforms_filteredout"
    
    if filtered_table_path.exists():
        filtered_table = pd.read_csv(filtered_table_path)
        filtered_waveforms = load_variable_length_waveforms(str(filtered_waveforms_path))
    else:
        print("Warning: No filtered-out files found. Make sure save_filtered_out=True in extraction script.")
        filtered_table = pd.DataFrame()
        filtered_waveforms = []
    
    # Calculate statistics
    print("\nCalculating statistics...")
    acc_stats = calculate_waveform_stats(accepted_waveforms, "Accepted")
    filt_stats = calculate_waveform_stats(filtered_waveforms, "Filtered")
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY CONTROL SUMMARY")
    print("="*60)
    
    total_pulses = len(accepted_waveforms) + len(filtered_waveforms)
    print(f"Total pulses detected: {total_pulses:,}")
    print(f"Accepted pulses: {len(accepted_waveforms):,} ({len(accepted_waveforms)/total_pulses*100:.1f}%)")
    print(f"Filtered pulses: {len(filtered_waveforms):,} ({len(filtered_waveforms)/total_pulses*100:.1f}%)")
    
    if acc_stats:
        print(f"\nACCEPTED PULSE STATISTICS:")
        print(f"  Count: {acc_stats['count']:,}")
        print(f"  Length: {acc_stats['length_stats']['mean']:.1f} ± {acc_stats['length_stats']['std']:.1f} samples")
        print(f"  Length range: {acc_stats['length_stats']['min']}-{acc_stats['length_stats']['max']} samples")
        print(f"  Peak-to-trough: {acc_stats['amplitude_stats']['peak_to_trough_mean']:.3f} ± {acc_stats['amplitude_stats']['peak_to_trough_std']:.3f}")
    
    if filt_stats:
        print(f"\nFILTERED PULSE STATISTICS:")
        print(f"  Count: {filt_stats['count']:,}")
        print(f"  Length: {filt_stats['length_stats']['mean']:.1f} ± {filt_stats['length_stats']['std']:.1f} samples")
        print(f"  Length range: {filt_stats['length_stats']['min']}-{filt_stats['length_stats']['max']} samples")
        print(f"  Peak-to-trough: {filt_stats['amplitude_stats']['peak_to_trough_mean']:.3f} ± {filt_stats['amplitude_stats']['peak_to_trough_std']:.3f}")
    
    # Width factor assessment
    if acc_stats:
        avg_length = acc_stats['length_stats']['mean']
        print(f"\nWIDTH FACTOR ASSESSMENT:")
        print(f"  Current width factor: 50")
        print(f"  Average waveform length: {avg_length:.1f} samples")
        
        if avg_length < 50:
            print("  → Width factor might be too low (waveforms truncated)")
        elif avg_length > 200:
            print("  → Width factor might be too high (too much padding)")
        else:
            print("  → Width factor appears appropriate")
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    plot_waveform_comparison(accepted_waveforms, filtered_waveforms, output_dir, file_prefix)
    
    if not filtered_table.empty:
        print("Generating table feature comparison...")
        compare_table_features(accepted_table, filtered_table, output_dir, file_prefix)
    
    # Save detailed statistics
    stats_output = {
        'file_analyzed': str(selected_table),
        'total_pulses': total_pulses,
        'accepted_count': len(accepted_waveforms),
        'filtered_count': len(filtered_waveforms),
        'acceptance_rate': len(accepted_waveforms) / total_pulses if total_pulses > 0 else 0,
        'accepted_stats': acc_stats,
        'filtered_stats': filt_stats
    }
    
    with open(output_path / f"{file_prefix}_QC_stats.json", 'w') as f:
        json.dump(stats_output, f, indent=2, default=convert_numpy)
    
    print(f"\nQuality control analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"  - {file_prefix}_QC_comparison.png")
    print(f"  - {file_prefix}_QC_table_features.png")
    print(f"  - {file_prefix}_QC_stats.json")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if filtered_waveforms and acc_stats and filt_stats:
        # Check if good pulses are being filtered out
        acc_mean_amp = acc_stats['amplitude_stats']['peak_to_trough_mean']
        filt_mean_amp = filt_stats['amplitude_stats']['peak_to_trough_mean']
        
        if filt_mean_amp > acc_mean_amp * 0.8:
            print("⚠️  WARNING: Filtered pulses have similar amplitudes to accepted ones.")
            print("   Consider relaxing filtering criteria.")
        
        if filt_stats['count'] > acc_stats['count']:
            print("⚠️  WARNING: More pulses are filtered than accepted.")
            print("   Check if filtering is too aggressive.")
        
        # Check width factor
        if acc_stats['length_stats']['mean'] > 150:
            print("💡 Consider reducing width factor to save disk space.")
        elif acc_stats['length_stats']['mean'] < 50:
            print("💡 Consider increasing width factor to capture full waveforms.")
    
    print("\nAnalysis complete! Review the plots to assess filtering effectiveness.")

if __name__ == "__main__":
    main()

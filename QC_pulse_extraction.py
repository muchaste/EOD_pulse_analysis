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
from eod_functions import load_variable_length_waveforms, calculate_waveform_stats

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def plot_waveform_comparison(accepted_waveforms, filtered_waveforms, output_path, file_prefix):
    """Create comprehensive comparison plots."""
    
    # Filter out empty waveforms
    acc_valid = [wf for wf in accepted_waveforms if len(wf) > 0]
    filt_valid = [wf for wf in filtered_waveforms if len(wf) > 0]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overlay plot of example waveforms
    plt.subplot(3, 4, 1)
    n_examples = min(100, len(acc_valid))  # Show up to 100 examples
    for i in range(n_examples):
        plt.plot(acc_valid[i], 'b-', alpha=0.1, linewidth=0.5)
    plt.title(f'Accepted Waveforms (n={len(acc_valid)})\nShowing {n_examples} examples')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    n_examples = min(100, len(filt_valid))  # Show up to 100 examples
    for i in range(n_examples):
        plt.plot(filt_valid[i], 'r-', alpha=0.1, linewidth=0.5)
    plt.title(f'Filtered-Out Waveforms (n={len(filt_valid)})\nShowing {n_examples} examples')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 2. Length distribution comparison
    plt.subplot(3, 4, 3)
    acc_lengths = [len(wf) for wf in acc_valid]
    filt_lengths = [len(wf) for wf in filt_valid]
    
    bins = np.linspace(0, max(max(acc_lengths) if acc_lengths else 0, 
                             max(filt_lengths) if filt_lengths else 0), 50)
    
    plt.hist(acc_lengths, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_lengths)})', 
             color='blue', density=True)
    plt.hist(filt_lengths, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_lengths)})', 
             color='red', density=True)
    plt.title('Waveform Length Distribution')
    plt.xlabel('Length (samples)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Amplitude distribution comparison
    plt.subplot(3, 4, 4)
    acc_amps = [np.max(wf) - np.min(wf) for wf in acc_valid]
    filt_amps = [np.max(wf) - np.min(wf) for wf in filt_valid]
    
    bins = np.linspace(0, max(max(acc_amps) if acc_amps else 0, 
                             max(filt_amps) if filt_amps else 0), 50)
    
    plt.hist(acc_amps, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_amps)})', 
             color='blue', density=True)
    plt.hist(filt_amps, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_amps)})', 
             color='red', density=True)
    plt.title('Peak-to-Trough Amplitude Distribution')
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Average waveform shapes
    plt.subplot(3, 4, 5)
    if acc_valid:
        # Normalize all waveforms to same length for averaging
        target_length = int(np.median([len(wf) for wf in acc_valid]))
        normalized_acc = []
        for wf in acc_valid[:200]:  # Use first 200 for averaging
            if len(wf) > 5:
                x_old = np.linspace(0, 1, len(wf))
                x_new = np.linspace(0, 1, target_length)
                wf_interp = np.interp(x_new, x_old, wf)
                normalized_acc.append(wf_interp)
        
        if normalized_acc:
            mean_acc = np.mean(normalized_acc, axis=0)
            std_acc = np.std(normalized_acc, axis=0)
            x = np.arange(target_length)
            plt.plot(x, mean_acc, 'b-', linewidth=2, label='Mean')
            plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, 
                           alpha=0.3, color='blue', label='±1 STD')
    
    plt.title('Average Accepted Waveform Shape')
    plt.xlabel('Normalized Sample')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 6)
    if filt_valid:
        # Normalize all waveforms to same length for averaging
        target_length = int(np.median([len(wf) for wf in filt_valid]))
        normalized_filt = []
        for wf in filt_valid[:200]:  # Use first 200 for averaging
            if len(wf) > 5:
                x_old = np.linspace(0, 1, len(wf))
                x_new = np.linspace(0, 1, target_length)
                wf_interp = np.interp(x_new, x_old, wf)
                normalized_filt.append(wf_interp)
        
        if normalized_filt:
            mean_filt = np.mean(normalized_filt, axis=0)
            std_filt = np.std(normalized_filt, axis=0)
            x = np.arange(target_length)
            plt.plot(x, mean_filt, 'r-', linewidth=2, label='Mean')
            plt.fill_between(x, mean_filt - std_filt, mean_filt + std_filt, 
                           alpha=0.3, color='red', label='±1 STD')
    
    plt.title('Average Filtered-Out Waveform Shape')
    plt.xlabel('Normalized Sample')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Skewness comparison
    plt.subplot(3, 4, 7)
    acc_skew = [stats.skew(wf) for wf in acc_valid if len(wf) > 3]
    filt_skew = [stats.skew(wf) for wf in filt_valid if len(wf) > 3]
    
    bins = np.linspace(-3, 3, 50)
    plt.hist(acc_skew, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_skew)})', 
             color='blue', density=True)
    plt.hist(filt_skew, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_skew)})', 
             color='red', density=True)
    plt.title('Skewness Distribution')
    plt.xlabel('Skewness')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Kurtosis comparison
    plt.subplot(3, 4, 8)
    acc_kurt = [stats.kurtosis(wf) for wf in acc_valid if len(wf) > 3]
    filt_kurt = [stats.kurtosis(wf) for wf in filt_valid if len(wf) > 3]
    
    bins = np.linspace(-2, 10, 50)
    plt.hist(acc_kurt, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_kurt)})', 
             color='blue', density=True)
    plt.hist(filt_kurt, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_kurt)})', 
             color='red', density=True)
    plt.title('Kurtosis Distribution')
    plt.xlabel('Kurtosis')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7-8. Peak and trough position distributions
    plt.subplot(3, 4, 9)
    acc_peak_pos = [np.argmax(wf) / len(wf) for wf in acc_valid if len(wf) > 0]
    filt_peak_pos = [np.argmax(wf) / len(wf) for wf in filt_valid if len(wf) > 0]
    
    bins = np.linspace(0, 1, 50)
    plt.hist(acc_peak_pos, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_peak_pos)})', 
             color='blue', density=True)
    plt.hist(filt_peak_pos, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_peak_pos)})', 
             color='red', density=True)
    plt.title('Peak Position Distribution')
    plt.xlabel('Relative Peak Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 10)
    acc_trough_pos = [np.argmin(wf) / len(wf) for wf in acc_valid if len(wf) > 0]
    filt_trough_pos = [np.argmin(wf) / len(wf) for wf in filt_valid if len(wf) > 0]
    
    bins = np.linspace(0, 1, 50)
    plt.hist(acc_trough_pos, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_trough_pos)})', 
             color='blue', density=True)
    plt.hist(filt_trough_pos, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_trough_pos)})', 
             color='red', density=True)
    plt.title('Trough Position Distribution')
    plt.xlabel('Relative Trough Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Length vs Amplitude scatter plot
    plt.subplot(3, 4, 11)
    acc_lengths = [len(wf) for wf in acc_valid]
    acc_amps = [np.max(wf) - np.min(wf) for wf in acc_valid]
    
    # Subsample for plotting if too many points
    if len(acc_lengths) > 5000:
        indices = np.random.choice(len(acc_lengths), 5000, replace=False)
        acc_lengths = [acc_lengths[i] for i in indices]
        acc_amps = [acc_amps[i] for i in indices]
    
    plt.scatter(acc_lengths, acc_amps, alpha=0.5, color='blue', s=1, label='Accepted')
    
    filt_lengths = [len(wf) for wf in filt_valid]
    filt_amps = [np.max(wf) - np.min(wf) for wf in filt_valid]
    
    # Subsample for plotting if too many points
    if len(filt_lengths) > 5000:
        indices = np.random.choice(len(filt_lengths), 5000, replace=False)
        filt_lengths = [filt_lengths[i] for i in indices]
        filt_amps = [filt_amps[i] for i in indices]
    
    plt.scatter(filt_lengths, filt_amps, alpha=0.5, color='red', s=1, label='Filtered')
    plt.title('Length vs Amplitude')
    plt.xlabel('Length (samples)')
    plt.ylabel('Peak-to-Trough Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Summary statistics text
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    summary_text = f"""
    FILTERING SUMMARY:
    
    Total Pulses: {len(accepted_waveforms) + len(filtered_waveforms)}
    Accepted: {len(accepted_waveforms)} ({len(accepted_waveforms)/(len(accepted_waveforms)+len(filtered_waveforms))*100:.1f}%)
    Filtered: {len(filtered_waveforms)} ({len(filtered_waveforms)/(len(accepted_waveforms)+len(filtered_waveforms))*100:.1f}%)
    
    ACCEPTED WAVEFORMS:
    Length: {np.mean([len(wf) for wf in acc_valid]):.1f} ± {np.std([len(wf) for wf in acc_valid]):.1f}
    Amplitude: {np.mean([np.max(wf)-np.min(wf) for wf in acc_valid]):.3f} ± {np.std([np.max(wf)-np.min(wf) for wf in acc_valid]):.3f}
    
    FILTERED WAVEFORMS:
    Length: {np.mean([len(wf) for wf in filt_valid]):.1f} ± {np.std([len(wf) for wf in filt_valid]):.1f}
    Amplitude: {np.mean([np.max(wf)-np.min(wf) for wf in filt_valid]):.3f} ± {np.std([np.max(wf)-np.min(wf) for wf in filt_valid]):.3f}
    
    WIDTH FACTOR ASSESSMENT:
    Width factor of 50 appears {'appropriate' if np.mean([len(wf) for wf in acc_valid]) > 100 else 'potentially too high'}
    based on average waveform length.
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}\\{file_prefix}_QC_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_table_features(accepted_table, filtered_table, output_path, file_prefix):
    """Compare tabular features between accepted and filtered events."""
    
    if accepted_table.empty and filtered_table.empty:
        print("No data to compare")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Amplitude ratio comparison
    if 'pp_ratio' in accepted_table.columns and 'pp_ratio' in filtered_table.columns:
        axes[0, 0].hist(accepted_table['pp_ratio'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[0, 0].hist(filtered_table['pp_ratio'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[0, 0].set_title('Amplitude Ratio Distribution')
        axes[0, 0].set_xlabel('Peak-to-Peak Ratio')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Duration comparison
    if 'pp_dur_us' in accepted_table.columns and 'pp_dur_us' in filtered_table.columns:
        axes[0, 1].hist(accepted_table['pp_dur_us'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[0, 1].hist(filtered_table['pp_dur_us'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[0, 1].set_title('Pulse Duration Distribution')
        axes[0, 1].set_xlabel('Duration (μs)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Differential vs single-ended
    if 'is_differential' in accepted_table.columns and 'is_differential' in filtered_table.columns:
        acc_diff = accepted_table['is_differential'].value_counts()
        filt_diff = filtered_table['is_differential'].value_counts()
        
        x = np.arange(len(acc_diff))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, acc_diff.values, width, label='Accepted', color='blue', alpha=0.7)
        axes[0, 2].bar(x + width/2, filt_diff.values, width, label='Filtered', color='red', alpha=0.7)
        axes[0, 2].set_title('Differential vs Single-ended')
        axes[0, 2].set_xlabel('Type (0=Single, 1=Differential)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Amplitude by channel
    if 'eod_amplitude' in accepted_table.columns and 'eod_amplitude' in filtered_table.columns:
        axes[1, 0].hist(accepted_table['eod_amplitude'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[1, 0].hist(filtered_table['eod_amplitude'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[1, 0].set_title('EOD Amplitude Distribution')
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Pulse width distribution
    if 'pulse_width' in accepted_table.columns and 'pulse_width' in filtered_table.columns:
        axes[1, 1].hist(accepted_table['pulse_width'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[1, 1].hist(filtered_table['pulse_width'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[1, 1].set_title('Pulse Width Distribution')
        axes[1, 1].set_xlabel('Width (s)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Waveform length distribution
    if 'waveform_length' in accepted_table.columns and 'waveform_length' in filtered_table.columns:
        axes[1, 2].hist(accepted_table['waveform_length'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[1, 2].hist(filtered_table['waveform_length'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[1, 2].set_title('Waveform Length Distribution')
        axes[1, 2].set_xlabel('Length (samples)')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}\\{file_prefix}_QC_table_features.png', dpi=300, bbox_inches='tight')
    plt.show()

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

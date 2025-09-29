"""
# 02_1_Concatenate_control_parameters.py
# Script to concatenate control parameters from extracted EODs from control/catalogue recordings and compute analysis parameters.
# Saves a parameter file for use in field recording extraction.
"""


import pandas as pd
import os

if __name__ == "__main__":
    # Select folder interactively
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    
    root = tk.Tk()
    root.withdraw()
    
    control_output_folder = filedialog.askdirectory(
        title="Select Control Extraction Output Folder"
    )
    
    if control_output_folder:
        # Find and load master_eod_table.csv if it exists
        master_file = os.path.join(control_output_folder, "master_eod_table.csv")
        if os.path.exists(master_file):
            combined_eod_table = pd.read_csv(master_file)
            print(f"Loaded master EOD table: {master_file} ({len(combined_eod_table)} events)")
        
            
            # Ask for safety margin
            safety_margin = simpledialog.askfloat(
                "Safety Margin",
                "Enter safety margin for parameter ranges (0.25 = 25% wider):",
                initialvalue=0.25,
                minvalue=0.0,
                maxvalue=1.0
            )
            
            if safety_margin is not None:
                # EOD width in microseconds (convert to seconds for parameters)
                if 'eod_width_uS' in combined_eod_table.columns:
                    width_us_range = (combined_eod_table['eod_width_uS'].min(), combined_eod_table['eod_width_uS'].max())
                    width_s_range = (width_us_range[0] / 1e6, width_us_range[1] / 1e6)
                else:
                    width_s_range = (3e-05, 0.001)  # Default values
                    print("  Warning: 'eod_width_uS' column not found, using defaults")
                
                # Amplitude ratio
                if 'eod_amplitude_ratio' in combined_eod_table.columns:
                    amp_ratio_range = (combined_eod_table['eod_amplitude_ratio'].min(), combined_eod_table['eod_amplitude_ratio'].max())
                elif 'amplitude_ratio' in combined_eod_table.columns:
                    amp_ratio_range = (combined_eod_table['amplitude_ratio'].min(), combined_eod_table['amplitude_ratio'].max())
                else:
                    amp_ratio_range = (0.1, 3.0)  # Default values
                    print("  Warning: amplitude ratio column not found, using defaults")
                
                # FFT peak frequency
                if 'fft_peak_freq' in combined_eod_table.columns:
                    fft_range = (combined_eod_table['fft_peak_freq'].min(), combined_eod_table['fft_peak_freq'].max())
                elif 'fft_freq_max' in combined_eod_table.columns:
                    fft_range = (combined_eod_table['fft_freq_max'].min(), combined_eod_table['fft_freq_max'].max())
                else:
                    fft_range = (500, 20000)  # Default values
                    print("  Warning: FFT frequency column not found, using defaults")
                
                print(f"  Data ranges found:")
                print(f"    EOD width: {width_s_range[0]*1e6:.1f} - {width_s_range[1]*1e6:.1f} μs")
                print(f"    Amplitude ratio: {amp_ratio_range[0]:.2f} - {amp_ratio_range[1]:.2f}")
                print(f"    FFT frequency: {fft_range[0]:.0f} - {fft_range[1]:.0f} Hz")
                
                # Apply safety margins
                width_margin = (width_s_range[1] - width_s_range[0]) * safety_margin
                amp_margin = (amp_ratio_range[1] - amp_ratio_range[0]) * safety_margin
                fft_margin = (fft_range[1] - fft_range[0]) * safety_margin
                
                # Create parameters with safety margins
                parameters = {
                    'min_width_s': max(1e-06, width_s_range[0] - width_margin),
                    'max_width_s': min(0.01, width_s_range[1] + width_margin),
                    'amplitude_ratio_min': max(0.01, amp_ratio_range[0] - amp_margin),
                    'amplitude_ratio_max': min(10.0, amp_ratio_range[1] + amp_margin),
                    'peak_fft_freq_min': max(50, fft_range[0] - fft_margin),
                    'peak_fft_freq_max': min(50000, fft_range[1] + fft_margin),
                    'safety_margin_used': safety_margin,
                    'n_events_analyzed': len(combined_eod_table)
                }
                
                print(f"  Parameters with {safety_margin*100:.0f}% safety margins:")
                print(f"    min_width_s: {parameters['min_width_s']:.2e}")
                print(f"    max_width_s: {parameters['max_width_s']:.2e}")
                print(f"    amplitude_ratio_min: {parameters['amplitude_ratio_min']:.2f}")
                print(f"    amplitude_ratio_max: {parameters['amplitude_ratio_max']:.2f}")
                print(f"    peak_fft_freq_min: {parameters['peak_fft_freq_min']:.0f}")
                print(f"    peak_fft_freq_max: {parameters['peak_fft_freq_max']:.0f}")
                
                # Save parameters to CSV
                param_df = pd.DataFrame({k: [v] for k, v in parameters.items()})
                param_file = os.path.join(control_output_folder, "computed_analysis_parameters_for_field_extraction.csv")
                param_df.to_csv(param_file, index=False)
                
                print(f"  Parameter file saved: {param_file}")
                
                if parameters:
                    print(f"\nParameter file creation completed!")
                    print(f"Files created in: {control_output_folder}")
                    print(f"  - combined_control_eod_table.csv")
                    print(f"  - analysis_parameters.csv")
            else:
                print("Parameter file creation cancelled.")
        else:
            print("No master EOD table found")
    else:
        print("No folder selected.")
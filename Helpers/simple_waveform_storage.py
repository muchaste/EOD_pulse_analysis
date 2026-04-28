"""
Simple Fixed-Length Waveform Storage Functions
=============================================

Simple storage functions for fixed-length EOD waveforms.
Much simpler than variable-length storage since all waveforms have identical length.
"""

import numpy as np
import json
from pathlib import Path


def save_fixed_length_waveforms(waveforms_list, output_path):
    """
    Save fixed-length waveforms as compressed .npz file.
    
    Parameters
    ----------
    waveforms_list : list of 1D arrays
        List of waveforms (all same length)
    output_path : str
        Base path for output files (without extension)
    
    Returns
    -------
    metadata : dict
        Basic metadata about the saved waveforms
    """
    if not waveforms_list:
        # Save empty file
        np.savez_compressed(f"{output_path}.npz", waveforms=np.array([]))
        metadata = {
            'n_waveforms': 0,
            'waveform_length': 0,
            'total_samples': 0
        }
    else:
        # Stack waveforms into 2D array (n_waveforms, waveform_length)
        waveforms_array = np.stack(waveforms_list, axis=0)
        
        # Save as compressed npz
        np.savez_compressed(f"{output_path}.npz", waveforms=waveforms_array)
        
        metadata = {
            'n_waveforms': len(waveforms_list),
            'waveform_length': len(waveforms_list[0]),
            'total_samples': waveforms_array.size,
            'shape': waveforms_array.shape,
            'dtype': str(waveforms_array.dtype)
        }
    
    # Save minimal metadata (optional - only if you need it)
    with open(f"{output_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, separators=(',', ':'))
    
    return metadata


def load_fixed_length_waveforms(base_path):
    """
    Load fixed-length waveforms from .npz file.
    
    Parameters
    ----------
    base_path : str
        Base path (without .npz extension)
    
    Returns
    -------
    waveforms_list : list of 1D arrays
        List of waveforms
    """
    try:
        # Load from npz file
        data = np.load(f"{base_path}.npz")
        waveforms_array = data['waveforms']
        
        if waveforms_array.size == 0:
            return []
        
        # Convert back to list of 1D arrays
        waveforms_list = [waveforms_array[i] for i in range(waveforms_array.shape[0])]
        
        return waveforms_list
        
    except FileNotFoundError:
        print(f"Warning: File {base_path}.npz not found")
        return []
    except Exception as e:
        print(f"Error loading waveforms: {e}")
        return []


# Compatibility wrapper to replace save_variable_length_waveforms
def save_variable_length_waveforms(waveforms_list, output_path):
    """
    Compatibility wrapper - just calls save_fixed_length_waveforms.
    Can replace the old function calls without changing the rest of the code.
    """
    return save_fixed_length_waveforms(waveforms_list, output_path)
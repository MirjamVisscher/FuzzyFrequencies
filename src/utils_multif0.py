#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:47:03 2025

@author: mirjam
"""
import numpy as np
from scipy.signal import butter, lfilter

from scipy.ndimage import binary_dilation

def fill_gaps(midis, max_gap=3):
    """Fill gaps of `max_gap` or fewer zeroes in a binary DataFrame."""
    filled = midis.copy()  # Copy DataFrame to avoid modifying original
    
    for col in filled.columns:
        if col == "timestamp":  # Skip timestamp column
            continue
        filled[col] = binary_dilation(filled[col].astype(bool), structure=np.ones(max_gap+1)).astype(int)
    
    return filled

def lowpass_filter(signal, cutoff=5000, fs=44100, order=4):
    """Applies a Butterworth low-pass filter to the signal.
    
    - cutoff: The cutoff frequency in Hz (e.g., 5000 Hz).
    - fs: The sample rate (default: 44100 Hz).
    - order: The order of the filter (higher = steeper cutoff).
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)  # Apply filter

def apply_fade_in_out(signal, fade_duration, fs):
    """Applies a fade-in and fade-out envelope to the signal.
    
    - fade_duration: Duration of fade-in and fade-out in seconds.
    - fs: Sample rate (Hz).
    """
    fade_samples = int(fade_duration * fs)  # Number of samples for fade

    if len(signal) < 2 * fade_samples:
        return signal  # If the note is too short, no fade needed

    # Create fade-in and fade-out envelopes
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    # Apply fade-in
    signal[:fade_samples] *= fade_in

    # Apply fade-out
    signal[-fade_samples:] *= fade_out

    return signal
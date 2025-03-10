#!/usr/bin/env python3 py310
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:35:13 2023

@author: mirjam
"""

import math
import pandas as pd
import csv
from scipy.special import kl_div
from pydub import AudioSegment
from scipy.stats import wasserstein_distance


class Constants:
    """A class containing the constants used
    Attributes:
        None
    """
    CHAMBER_PITCH = 440
    ALL_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    WHITE_KEYS = ['E', 'F', 'G', 'A', 'B', 'C', 'D']
    MODES = ['1', '2', '3', '3.4', '4', '5', '6', '7', '8', '9', '10', '11', '12', '?']

def get_miditone(pitch, chamber_pitch = Constants.CHAMBER_PITCH):
    """Calculate the miditone based on the frequency
    Returns:
        miditone(float): miditone corresponding to the frequency
    """
    if not math.isnan(pitch):
        miditone = 12 * math.log2(pitch / chamber_pitch) + 69
        return (miditone)
    else:
        return math.nan
    
def get_frequency(miditone):
    """Calculate the frequency based on the miditone
    Returns:
        frequency(float): frequency corresponding to the miditone
    """
    if not math.isnan(miditone):
        frequency = 2**(float(miditone-69)/12)*Constants.CHAMBER_PITCH
        return (frequency)
    else:
        return math.nan
    
class Tones:
    """A class containing the constant TONES: a dataframe containing pitch, 
    pitch class in numbers and in names
    Attributes:
        None
    """
    tones = pd.DataFrame()
    tones['miditone'] = range(21,109)
    tones['frequency'] = tones['miditone'].apply(get_frequency)
    tones['pitch_class_name'] = [Constants.ALL_KEYS[(i-12) % len(Constants.ALL_KEYS)] for i in tones['miditone']]
    tones['pitch_class'] = tones['pitch_class_name'].map(lambda x: Constants.ALL_KEYS.index(x))
    tones['pitch_name'] = tones['pitch_class_name'] + (tones['miditone'] // 12-1).astype(str)
    tones['white_key'] = tones['pitch_class_name'].isin(Constants.WHITE_KEYS).astype(int)
    TONES = tones
    
def get_pitch_name(midi):
    pitch_row = Tones.TONES[Tones.TONES['miditone'] == midi]
    if not pitch_row.empty:
        return pitch_row['pitch_name'].values[0]
    
def get_pitch_class(midi):
    pitch_row = Tones.TONES[Tones.TONES['miditone'] == midi]
    if not pitch_row.empty:
        return pitch_row['pitch_class'].values[0]

def get_pitch_class_name(midi):
    pitch_row = Tones.TONES[Tones.TONES['miditone'] == midi]
    if not pitch_row.empty:
        return pitch_row['pitch_class_name'].values[0]

def repair_frequency_file(path, delimiter='\t'):
    """Turn the output of multif0 into a csv that is readable by pandas by 
    padding the lines with commas.
    Returns:
        nothing, it replaces your malformed csv file with a readable file
    """
    # TODO the first row is now converted into column headers. Add new headers instead.
    # Read the input CSV file
    with open(path, 'r') as input_file:
        reader = csv.reader(input_file, delimiter=delimiter)
        rows = list(reader)
    
    # Find the maximum number of columns in any row
    max_columns = max(len(row) for row in rows)
    
    # Create a new list with properly formed columns
    new_rows = []
    for row in rows:
        new_row = row + [''] * (max_columns - len(row))
        new_rows.append(new_row)
    
    # Write the new data to the output CSV file
    with open(path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(new_rows)

# Function to get the length of a WAV file in seconds
def get_wav_length(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Length in seconds

# Function to get the last value in the first column of a CSV file
def get_csv_length(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[-1, 0]

def symmetric_kl_divergence(p, q, epsilon=1e-10):
    """Calculates the symmetric Kullback-Leibler distance between two 
    distributions.

    Args:
        p (list): A probability distribution.
        q (list): Another probability distribution.

    Returns:
        float: The Kullback_Leibler distance between p and q.
    """
    
    p = [pi + epsilon for pi in p]
    q = [qi + epsilon for qi in q]

    kl_divergence_pq = kl_div(p, q)
    kl_divergence_qp = kl_div(q, p)
    symmetric_kl = sum(0.5 * (kl_divergence_pq + kl_divergence_qp))
    return symmetric_kl

def squared_distance(p, q):
    """Calculates the squared distance between two distributions.

    Args:
        p (list): A probability distribution.
        q (list): Another probability distribution.

    Returns:
        float: The squared distance between p and q.
    """
    assert len(p) == len(q), f"Lists must have the same length: {len(p)} != {len(q)}"
    difference = [pi - qi for pi, qi in zip(p, q)]
    return sum(i**2 for i in difference)

def euclidean_distance(p,q):

    """Calculates the Euclidian distance between two distributions.

    Args:
        p (list): A probability distribution.
        q (list): Another probability distribution.

    Returns:
        float: The euclidian distance between p and q.
    """
    assert len(p) == len(q), f"Lists must have the same length: {len(p)} != {len(q)}"
    difference = [pi - qi for pi, qi in zip(p, q)]
    return math.sqrt(sum(i**2 for i in difference))
    
def manhattan_distance(p,q):
    """Calculates the Manhattan distance between two distributions.

    Args:
        p (list): A probability distribution.
        q (list): Another probability distribution.

    Returns:
        float: The Manhattan distance between p and q.
    """
    assert len(p) == len(q), f"Lists must have the same length: {len(p)} != {len(q)}"
    return sum(abs(pi - qi) for pi, qi in zip(p, q))


def earth_movers_distance(p, q):
    """Calculates the Earth Mover's Distance between two distributions.

    Args:
        p (list): A probability distribution.
        q (list): Another probability distribution.

    Returns:
        float: The Earth Mover's Distance between p and q.
    """
    # Calculate the Earth Mover's Distance directly
    emd = wasserstein_distance(p, q)
    
    return emd


def phase(profile, x):
    """
    Shifts the values in the third column of the given profile DataFrame by 12 times 'x' positions either up or down.

    Parameters:
        profile (DataFrame): The DataFrame containing the data.
        x (int): The number of positions to shift the values. Positive values shift downwards, negative values shift upwards.

    Returns:
        DataFrame: The profile DataFrame with the third column shifted accordingly.
    """
    
    shifted_series = pd.Series([0] * len(profile))
    
    for i in range(len(profile)):
        old_index = i + (12 * -x)
        if 0 <= old_index < len(profile):
            shifted_series[i] = profile[old_index]
    profile = shifted_series
    return profile

def transpose(pcp, transposition):
    """transposes a pitch class profile
    Returns:
        list transposed_pcp
        """
    transposed_pcp = [0] * len(pcp)
    for i in range(len(pcp)):
        transposed_pcp[(i + transposition) % len(pcp)] = pcp[i]
    return transposed_pcp


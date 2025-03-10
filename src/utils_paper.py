#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:00:28 2024

@author: mirjam
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import os

from Multif0Extraction import Multif0Extraction
from utils import get_pitch_class_name, get_pitch_name

def pitch_class_profile_fuzzypaper(recording, experiment, save=False, normalisation = False):
    plt.style.use('uu')
    pcp = recording.create_pitch_class_profiles(save=save, normalisation = normalisation)[['pitch_class_name', 'symbolic', 'multif0']]
    pcp.rename(columns={'symbolic': 'MusicXML', 'multif0': 'Multif0'}, inplace = True)
    ax = pcp.plot.bar(x='pitch_class_name', figsize=(8, 5))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)#move legend out of plot area
    ax.set_xlabel('pitch class')
    ax.set_ylabel('relative presence')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    try:
        final = get_pitch_class_name(recording.symbolic_file.final_midi())
    except:
        multif0_file = Multif0Extraction(recording.frequency_filename,recording.experiment)
        final = get_pitch_class_name(multif0_file.final_midi())
        
    finalwidth = 2.20
    for patch, label in zip(ax.patches, pcp['pitch_class_name'].tolist() * 2): 
        if label == final:
            patch.set_edgecolor('black')  # Set desired edge color
            patch.set_linewidth(finalwidth)  # Set desired line width
    
    handles, labels = ax.get_legend_handles_labels()

    # Add custom "Final" legend entry
    final_patch = mpatches.Patch(edgecolor='black', facecolor='none', linewidth=finalwidth, label='Final')
    handles.append(final_patch)
    labels.append('Final')
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = os.path.join('../results/figures', experiment, 'paper', 'profiles')
    os.makedirs(output_path, exist_ok=True)
    if save:
        fig = ax.figure
        fig.savefig(os.path.join(output_path, '1b pcp_'+ recording.title + '.png'), bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close(fig)       
    return ax

def pitch_profile_fuzzypaper(recording, experiment, save=False, normalisation=False):
    plt.style.use('uu')

    # Compute pitch profile once
    pitch_profile = recording.pitch_profile(save=save, normalisation=normalisation)

    # Select relevant columns
    multif0_col = next(iter([col for col in pitch_profile.columns if col.startswith("multif0")]), None)
    if not multif0_col:
        raise ValueError("No 'multif0' column found in pitch profile.")

    selected_columns = ['pitch_name', 'symbolic', multif0_col]
    pcp = pitch_profile[selected_columns]

    # Identify first and last relevant rows
    valid_rows = (pcp["symbolic"] > 0) | (pcp[multif0_col] > 0)
    first_idx, last_idx = valid_rows.idxmax(), valid_rows[::-1].idxmax()
    trimmed_pcp = pcp.loc[first_idx:last_idx]

    # Plot results
    ax = trimmed_pcp.plot.bar(x='pitch_name', figsize=(12, 4))
    ax.set_xlabel('pitch')
    ax.set_ylabel('relative presence')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    # Get final pitch name and highlight it
    final = _get_final_pitch(recording)
    final_width = 2.2
    for patch, label in zip(ax.patches, trimmed_pcp['pitch_name'].tolist() * 2):  
     if label == final:
         patch.set_edgecolor('black')
         patch.set_linewidth(final_width)

    plt.legend().remove()

    # Save figure if needed
    if save:
        output_path = os.path.join('../results/figures', experiment, 'paper', 'profiles')
        os.makedirs(output_path, exist_ok=True)
        fig_path = os.path.join(output_path, f'1a pp_{recording.title}.png')

        ax.figure.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(ax.figure)  # Close figure to free memory

    return ax

def _get_final_pitch(composition):
    """Retrieve the final pitch name safely."""
    try:
        return get_pitch_name(composition.symbolic_file.final_midi())
    except FileNotFoundError:  # Handle specific errors
        multif0_file = Multif0Extraction(composition.frequency_filename, composition.experiment)
        return get_pitch_name(multif0_file.final_midi())

def piano_roll_bp(bp, experiment, start_time=0, end_time=None, save=False):
    """Plot the frequencies of basic pitch extraction on a piano roll representation
       within a specified time window without correction for chamber pitch or transposition.
       Args:
           start_time (float): Start time in seconds for the plot.
           end_time (float): End time in seconds for the plot.
           save (bool): Whether to save the plot as an image file.
       Returns:
           matplotlib plot: Scatterplot of all voices.
    """
    
    odir = os.path.join('../results/figures', experiment, 'paper', 'piano_rolls')
    os.makedirs(odir, exist_ok=True)
    
    F_coef = range(24, 96)  # range of miditones to show in the plot
    fig, ax1 = plt.subplots(figsize=(20, 15))
    cmap = plt.get_cmap("Greys")
    norm = plt.Normalize(vmin=0, vmax=127)
    
    # Adjust end_time if it's not provided
    if end_time is None:
        end_time = bp.freqs['end_time_s'].max()  # Use the max end time in the data

    # Filter data for the specified time window
    time_filtered_data = bp.freqs[(bp.freqs['start_time_s'] <= end_time) & (bp.freqs['end_time_s'] >= start_time)]
    
    # Plot each note as a horizontal bar within the time window
    for _, row in time_filtered_data.iterrows():
        color = cmap(norm(row['velocity']))
        ax1.plot([max(row['start_time_s'], start_time), min(row['end_time_s'], end_time)], 
                 [row['miditone'], row['miditone']],
                 color=color, linewidth=10)
    
    # Draw horizontal lines based on pitch values
    for pitch in F_coef:
        mod12 = pitch % 12
        if mod12 in {2, 4, 5, 7, 11, 0}:
            color = 'black'
            linewidth = 0.7
        elif mod12 in {1, 3, 6, 8, 10}:
            color = 'lightgray'
            linewidth = 0.5
        elif mod12 == 9:
            color = 'red'
            linewidth = 1
        else:
            continue
        ax1.axhline(y=pitch, color=color, linewidth=linewidth, linestyle='-')
    
    # Set x-axis limits based on start and end time
    ax1.set_xlim(start_time, end_time)

    x_tick_interval = 5 #if time_window > 60 else max(time_window / 10, 1)  # Adjust intervals
    x_ticks = np.arange(start_time, end_time + x_tick_interval, x_tick_interval)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis='x', labelsize=25)
    
    # Set y-axis ticks and labels
    ax1.set_yticks(F_coef)
    ax1.set_yticklabels([pitch if (pitch + 3) % 12 == 0 else "" for pitch in F_coef], fontsize=25)

    # Create a secondary y-axis on the right
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(F_coef)
    ax2.set_yticklabels([pitch if (pitch + 3) % 12 == 0 else "" for pitch in F_coef], fontsize=25)

    # Set labels, title, and y-axis limits
    ax1.set_xlabel('Time (seconds)', size=25)
    ax1.set_ylabel('MIDI pitch', size=25)
    # ax2.set_ylabel('MIDI pitch', size=25)
    # plt.title('Basic pitch of ' + os.path.splitext(os.path.basename(bp.file_name))[0], fontsize=50)
    
    # Set tick label sizes
    ax1.tick_params(axis='x', labelsize=25)
    ax1.tick_params(axis='y', labelsize=25)
    ax2.tick_params(axis='y', labelsize=25)
    
    # Save plot if required
    output_path = os.path.join(odir, os.path.splitext(os.path.basename(bp.file_name))[0] + '.png')
    plt.tight_layout()
    if save:
        plt.savefig(output_path)
    
    # Show the plot
    plt.show()
    
    # Close the plot to free memory
    plt.close()

def piano_roll_mp(mp, experiment, save=False, start_time=0, end_time=None):
    """Plot the frequencies of Multif0Extraction on a piano roll representation
    within a specified time window, without correction for chamber pitch or transposition.

    Args:
        save (bool): Whether to save the plot as an image file.
        start_time (float): Start time of the time window in seconds.
        end_time (float): End time of the time window in seconds.
    Returns:
        matplotlib plot: Scatterplot of all voices within the specified time window.
    """

    fig, ax0 = plt.subplots(figsize=(20, 15))

    # Transpose freqs for plotting
    X = mp.freqs.T.to_numpy()  # Convert DataFrame to NumPy array
    Fs = mp.fs_hcqt
    T_coef = np.arange(X.shape[1]) / Fs  # time converted to seconds

    # Define end_time if not provided
    if end_time is None:
        end_time = T_coef.max()

    # Mask the data within the start_time and end_time window
    time_mask = (T_coef >= start_time) & (T_coef <= end_time)
    X = X[:, time_mask]  # Select only the time window of interest
    T_coef = T_coef[time_mask]  # Adjust T_coef to match the selected time window

    # Prepare data for scatter plot
    y_coords, x_coords = np.where(X > 0)  # Get coordinates of non-zero values
    x_coords = T_coef[x_coords]
    F_coef = np.arange(X.shape[0])  # Define y-axis indices
    y_coords = F_coef[y_coords]
    colors = X[X > 0]  # Use the non-zero values directly

    # Plot the matrix as scatter plot with small dots
    ax0.scatter(x_coords, y_coords, c=colors, cmap='gray_r', marker='o')

    # Set x-axis ticks and labels according to start_time and end_time
    x_ticks = np.arange(start_time, end_time + 1, 5)
    ax0.set_xticks(x_ticks)
    ax0.tick_params(axis='x', labelsize=25)

    # Set y-axis ticks and labels
    ax0.set_yticks(F_coef)
    ax0.set_yticklabels([label if (label + 3) % 12 == 0 else "" for label in mp.freqs.T.index], fontsize=25)
    
    # Create a secondary y-axis on the right
    ax1 = ax0.twinx()
    ax1.set_yticks(F_coef)
    ax1.set_yticklabels([label if (label + 3) % 12 == 0 else "" for label in mp.freqs.T.index], fontsize=25)

    # Synchronize the limits of both y-axes
    ax1.set_ylim(ax0.get_ylim())
    
    # Draw horizontal lines based on pitch values
    for pitch in F_coef:
        mod12 = (pitch + 24) % 12  # Adjusted modulo for pitch values
        if mod12 in {2, 4, 5, 7, 11, 0}:  # 0 is the same as 12 in modulo 12
            color = 'black'
            linewidth = 0.7
        elif mod12 in {1, 3, 6, 8, 10}:
            color = 'lightgray'
            linewidth = 0.5
        elif mod12 == 9:
            color = 'red'
            linewidth = 1
        else:
            continue

        # Draw horizontal line at the given pitch
        ax0.axhline(y=pitch, color=color, linestyle='-', linewidth=linewidth, alpha=0.7)

    # Axes labels
    ax0.set_xlabel('Time (seconds)', fontsize=25)
    ax0.set_ylabel('MIDI pitch', fontsize=25)

    plt.tight_layout()

    # Save the figure
    if save:
        odir = os.path.join('../results/figures', experiment, 'paper', 'piano_rolls')
        os.makedirs(odir, exist_ok=True)
        plt.savefig(os.path.join(odir, f"{mp.title}.png"))
    plt.show()
    plt.close()
    return plt



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:18:20 2023

@author: mirjam
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from utils import get_pitch_class, Tones

class MultipitchExtraction:
    """A class representing a set of frequencies, extracted with the multipitch algorithm by Weiss et al (2024).
    
    Attributes:
        file_name(string): name of the file
        title(string): name of the file without file extention
        experiment(string): name of the experiment
        freqs(dataframe): with columns:
            timestamp (float): The timestamp of the frequency.
            {24 - 96}: Energy of the miditone
        tail(int): the number of timeframes used to determine the lowest final
    """
    def __init__(self, file_name, experiment, model_name):
        self.file_name = file_name
        self.title = file_name.replace('.csv', '')
        self.experiment = experiment
        path = os.path.join('..', 'data',  'raw', experiment, model_name)
        freqs = pd.read_csv(os.path.join(path, file_name), header=0)
        freqs.columns = ['timeslice'] + list(range(24, 96))
        freqs.set_index('timeslice', inplace=True)
        self.freqs = freqs.apply(pd.to_numeric, errors='coerce')
        #fs_hcqt derived from the multipitch algorithm. I don't know whether it is constant
        self.fs_hcqt = 43.06640625
        self.model_name = model_name
        
        """settings that affect the final detector """
        # threshold to take energy values into accounts
        self.profilethreshold = 0.28 # threshold both for profiles
        self.finalthreshold = 0.4 # threshold for final 
        #last non-empty thresholded rows to build candidates for the final.
        self.tailrows = 200
        # number of observations that the final should have in the last 200 non-empty thresholded timeslices
        self.tailthreshold = 35
        #number of top candidates in the pitch class profile that can be a final
        self.pcptop = 7 
    
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"\nfile_name:\n{self.file_name}\n\nexperiment:\n{self.experiment}\n\ntitle:\n{self.title}\n\ntail:\n{self.tail}"
        return information
        
    def piano_roll(self, save=False):
        """Plot the frequencies of FrequencyFile on a piano roll representation
        without correction for chamber pitch or transposition.
        Args:
            save (bool): Whether to save the plot as an image file.
        Returns:
            matplotlib plot: Scatterplot of all voices.
        """
    
        fig, ax0 = plt.subplots(figsize=(60, 15))
    
        # Transpose freqs for plotting
        X = self.freqs.T.to_numpy()  # Convert DataFrame to NumPy array
        Fs = self.fs_hcqt
        T_coef = np.arange(X.shape[1]) / Fs
        
        # Use indices for y-axis if X.index are not directly MIDI pitches
        F_coef = np.arange(X.shape[0])
        
        # Prepare data for scatter plot
        y_coords, x_coords = np.where(X > 0)  # Get coordinates of non-zero values
        x_coords = T_coef[x_coords]
        y_coords = F_coef[y_coords]
        colors = X[X > 0]  # Use the non-zero values directly without flatten()
    
        # Plot the matrix as scatter plot with small dots
        ax0.scatter(x_coords, y_coords, c=colors, cmap='gray_r', marker='o')
        
        x_ticks = np.arange(T_coef.min(), T_coef.max() + 1, 15)
        ax0.set_xticks(x_ticks)  # Set x-axis ticks to every 15 seconds
        ax0.tick_params(axis='x', labelsize=25)
        
        # Set y-axis ticks and labels
        ax0.set_yticks(F_coef)  # Set y-axis ticks
        ax0.set_yticklabels([label if (label+3) % 12 == 0 else "" for label in self.freqs.T.index], fontsize=25)
        ax0.tick_params(axis='x', labelsize=25)
        
        # Create a secondary y-axis on the right
        ax1 = ax0.twinx()
        ax1.set_yticks(F_coef)
        ax1.set_yticklabels([label if (label+3) % 12 == 0 else "" for label in self.freqs.T.index], fontsize=25)

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
    
        # Axes labels and title
        ax0.set_xlabel('Time (seconds)', fontsize=25)
        ax0.set_ylabel('MIDI pitch', fontsize=25)
        ax0.set_title(f'Multi-pitch extraction with model {self.model_name} for {self.title}', fontsize=50)
    
        plt.tight_layout()
    
        # Save the figure
        if save:
            output_path = os.path.join('..', 'results', 'figures', self.experiment, 'piano_roll', self.model_name)
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, self.title + '.png'))
    
        # plt.show()
        plt.close()
        return plt


            
    def lowest_final(self):
        """Calculate the lowest frequency that has more than 29 instances in the tail of voice1
        but only if the final is part of the top n in the pitch class profile.
        This is the Wiering solution
        Returns:
            float: frequency of the last note of the lowest voice.
        """
      
        pcp = self.pitch_class_profile(transposition = 0, normalisation = False)
        binary_freqs = (self.freqs > self.finalthreshold).astype(int)
        
        # Select the last self.tailrows non-empty rows from the thresholded multipitch extraction
        row_sums = binary_freqs.sum(axis=1)
        filtered_rows = binary_freqs[row_sums > 0]
        last_tail_rows = filtered_rows.tail(self.tailrows)
        
        
        # select the last self.tail rows where the sum of the row >0
        # tail_counts = make a count of each miditone (column) the count is called tail_count
        tail_counts= last_tail_rows.sum(axis=0)
        # tail_counts = pd.DataFrame({})
        tail_counts_df = pd.DataFrame({'midi_tone': tail_counts.index, 'tail_count': tail_counts.values})
        tail_counts_df['pitch_class'] = tail_counts_df['midi_tone'].apply(lambda x: get_pitch_class(x))
        
        pcp_top = pcp.nlargest(self.pcptop, 'pcp_audio_proportion')['pitch_class']
        candidates = tail_counts_df[tail_counts_df['pitch_class'].isin(pcp_top)]
        
        try:
            #first try to get the final from the most frequent pitch classes
            candidates = candidates.loc[candidates['tail_count']>self.tailthreshold]
            lowest_final = min(candidates['midi_tone'])
        except:
            #else select the final from all miditones in the last self.tailrows non-empty thresholded rows
            candidates = tail_counts_df.loc[tail_counts_df['tail_count']>self.tailthreshold]
            lowest_final = min(candidates['midi_tone'])
        
        
        return lowest_final
           
    
    def final_midi(self):
        """Get the midi number closest to the frquency of the last note of the 
        lowest voice
        Returns:
            int: midi_number of the last note of the lowest voice
        """
   
        final_midi = self.lowest_final()
        return final_midi
    
    def final_name(self):
        """Get the note name of the last note of the lowest voice
        Returns:
            str: note name of the last note of the lowest voice
        """
        final_midi = self.final_midi()
        #TODO implement librosa's pitch algorithm instead of Tones.TONES
        final_name = Tones.TONES.loc[Tones.TONES['miditone'] == final_midi, 'pitch_name'].values[0]
        return final_name

    
    def pitch_profile(self, transposition=0, normalisation=False):
        """Calculate the pitch profile of the FrequencyFile
        Arguments:
            transposition: the transposition is defined by the difference 
                between the lowest finals of the frequency file and of the 
                symbolic file
            normalisation: if True, the pitch will be normalised to a final on C
                    
        Returns:
            dataframe: 'pitch_name', 'pp_audio_proportion', 'miditone'
        """
        # Convert all values in self.freqs above self.threshold to 1, all others to 0
        binary_freqs = (self.freqs > self.profilethreshold).astype(int)
        
        # Calculate proportionsper miditone
        pitch_class_counts= binary_freqs.sum()
        total_count = pitch_class_counts.sum()
        proportions = pitch_class_counts / total_count
        proportions_df = pd.DataFrame({'miditone': proportions.index, 'pp_audio_proportion': proportions.values})
        
        
        #transpose if desired for comparison to symbolic files
        proportions_df['miditone'] = proportions_df['miditone']+transposition
        
        # normalise if desired for comparision to other recordings
        if normalisation == True:
            difference = self.final_midi()%12
            if difference > 5:
                difference = -(12-difference)
            proportions_df['miditone'] = proportions_df['miditone']-difference
        
        
        # Merge with pitch class names from Tones.TONES
        pitch_profile = proportions_df.merge(Tones.TONES[['miditone', 'pitch_name']], 
                                              on='miditone', how='outer')
        pitch_profile = pitch_profile.dropna(subset=['pitch_name'])
        pitch_profile = pitch_profile.reset_index(drop=True)
        
        pitch_profile['pp_audio_proportion'] = pitch_profile['pp_audio_proportion'].fillna(0)
        
       
        
        # Return the DataFrame with proportions and pitch class names
        return pitch_profile
    
    def pitch_class_profile(self, transposition =0, normalisation=False):
        """Calculate the pitch class profile of the multipitchextraction
        Arguments:
            transposition: the transposition is defined by the difference 
                between the lowest finals of the frequency file and of the 
                symbolic file
            normalisation: if True, the pitch will be normalised to a final on C
                    
        Returns:
            dataframe: 'pcp_audio_proportion', 'pitch_class_name'
        """
        
        pcp = self.pitch_profile(transposition = transposition, normalisation = normalisation)
        pcp = pcp.merge(Tones.TONES[['miditone', 'pitch_class_name', 'pitch_class']], on='miditone', how='outer')
        pcp = pcp.drop(columns=['miditone', 'pitch_name'])
        pcp = pcp.groupby(['pitch_class', 'pitch_class_name']).sum().reset_index()
        pcp= pcp.rename(columns={'pp_audio_proportion': 'pcp_audio_proportion'})
        
        return pcp


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

from utils import get_miditone, get_pitch_class, Tones

class BasicpitchExtraction:
    """A class representing a set of frequencies, extracted with the basicpitch 
    algorithm by Bittner et al (2022).
    
    Attributes:
        file_name(string): name of the file
        title(string): name of the file without file extention
        experiment(string): name of the experiment
        freqs(dataframe): with columns:
            timestamp (float): The timestamp of the frequency.
            {24 - 96}: Energy of the miditone
        tail(int): the number of timeframes used to determine the lowest final
    """
    def __init__(self, file_name, experiment):
        self.file_name = file_name
        self.title = file_name.replace('.csv', '')
        self.experiment = experiment
        path = os.path.join('..', 'data',  'raw', experiment, 'basicpitch')
                
        self.freqs = pd.read_csv(os.path.join(path, file_name), usecols=[0, 1, 2, 3])
        self.freqs.rename(columns={'pitch_midi': 'miditone'}, inplace=True)
        
        """ We have chosen not to threshold the velocities in the basicpitch extractions
        because there is not too much noise from low dynamics in the data."""
        # tail in seconds
        self.tail = 4.5
        # threshold for counting the final in seconds
        self.tailthreshold = 0.4
        # number of candidate pitch classes in the pcp
        self.top = 7
        # self.loudnessthreshold lowers the dynamic threshold to an effective level
        self.loudnessthreshold = 8
        # threshold for selecting pitch candidates above a min level in the pitch profile
        self.pp_threshold = 0.015
        
        
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"\nfile_name:\n{self.file_name}\n\nexperiment:\n{self.experiment}\n\ntitle:\n{self.title}\n"
        return information
    
    
        
    def piano_roll(self, save = False):
        """Plot the frequencies of basic pitch extraction on a piano roll representation
            without correction for chamber pitch or transposition.
            Args:
                save (bool): Whether to save the plot as an image file.
            Returns:
                matplotlib plot: Scatterplot of all voices.
        """

        odir = os.path.join('..', 'results', 'figures', self.experiment, 'piano_roll', 'basicpitch')
        os.makedirs(odir, exist_ok=True)
        
        F_coef = range(24, 96)  # range of miditones to show in the plot
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(60, 25))
        cmap = plt.get_cmap("Greys")
        # Normalize the velocity to [0, 1] range
        norm = plt.Normalize(vmin=0, vmax=127)
        
        # Plot each note as a horizontal bar
        for _, row in self.freqs.iterrows():
            color = cmap(norm(row['velocity']))  # Map velocity to grayscale
            ax1.plot([row['start_time_s'], row['end_time_s']], [row['miditone'], row['miditone']],
                     color=color, linewidth=10)
        
        # Draw horizontal lines based on pitch values
        for pitch in F_coef:
            mod12 = pitch % 12
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

            ax1.axhline(y=pitch, color=color, linewidth=linewidth, linestyle='-')
        
        # Set x-axis ticks to every 15 seconds
        x_ticks = np.arange(0, ax1.get_xlim()[1] + 1, 15)
        ax1.set_xticks(x_ticks)
        ax1.tick_params(axis='x', labelsize=25)
        
        # Set y-axis ticks and labels
        ax1.set_yticks(F_coef)  # Set y-axis ticks
        ax1.set_yticklabels([pitch if (pitch + 3) % 12 == 0 else "" for pitch in F_coef], fontsize=25)
    
        # Create a secondary y-axis on the right
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Sync y-limits with the primary y-axis
        ax2.set_yticks(F_coef)
        ax2.set_yticklabels([pitch if (pitch + 3) % 12 == 0 else "" for pitch in F_coef], fontsize=25)
        
        
    
        # Set labels, title, and y-axis limits
        ax1.set_xlabel('Time (seconds)', size=25)
        ax1.set_ylabel('MIDI pitch', size = 25)
        # ax1.set_ylim(24, 95)  # Set the y-axis limits for the primary y-axis
        ax2.set_ylabel('MIDI pitch', size = 25)
        plt.title('Basic pitch of ' + os.path.splitext(os.path.basename(self.file_name))[0], fontsize=50)
        
        # Set the tick labels font size
        ax1.tick_params(axis='x', labelsize=25)  # x-axis (bottom and top)
        ax1.tick_params(axis='y', labelsize=25)  # y-axis (left)
        ax2.tick_params(axis='y', labelsize=25)  # Secondary y-axis (right)

        # Construct the output path
        output_path = os.path.join(odir, os.path.splitext(os.path.basename(self.file_name))[0] + '.png')
        
        plt.tight_layout()
        # Save the plot if required
        if save:
            plt.savefig(output_path)
        
        # Show the plot
        # plt.show()
        
        # Close the plot to free memory
        plt.close()
             
    def lowest_final(self):
        """Calculate the lowest of the last miditones
        Returns:
            int: miditone
        """
        
        pcp = self.pitch_class_profile(transposition = 0, normalisation = False)
        pp = self.pitch_profile(transposition = 0, normalisation = False)
        pitch_candidates = pp.loc[pp['pp_audio_proportion']>self.pp_threshold]['miditone']
        
        # create a dynamic loudness threshold above which the final canditates 
        # are consdidered. This is weighted average velocity minus a threshold
        weighted_velocity_sum = (self.freqs['velocity'] * self.freqs['duration']).sum()
        total_duration = self.freqs['duration'].sum()
        dynamic_loudnessthreshold = round(weighted_velocity_sum / total_duration)-self.loudnessthreshold
        
        loudfreqs = self.freqs.loc[self.freqs['velocity'] > dynamic_loudnessthreshold]
        loudfreqs = loudfreqs.loc[loudfreqs['duration'] > self.tailthreshold]
        # selecting the pitch_canditates
        loudfreqs = loudfreqs[loudfreqs['miditone'].isin(pitch_candidates)]
        # get last timestamp above the dynamic loudness threshold and the tailthreshold
        end_time = loudfreqs['end_time_s'].max()
        
        # select the last seconds before the end time
        tailrows = self.freqs.loc[self.freqs['end_time_s'] > end_time - self.tail]
        # compute the presence of the pitches
        presence = tailrows.groupby('miditone').agg({
            'duration': 'sum',
            'velocity': 'max', 
            'end_time_s': 'max'
            }).reset_index()
        
       
        present_candidates = presence.loc[(presence['duration'] > self.tailthreshold)&(presence['velocity'] > (dynamic_loudnessthreshold-8))]
        ppp_candidates = present_candidates[present_candidates['miditone'].isin(pitch_candidates)].copy()
        pcp_top = pcp.nlargest(self.top, 'pcp_audio_proportion')['pitch_class']
        ppp_candidates.loc[:,'pitch_class'] = ppp_candidates['miditone'].apply(lambda x: get_pitch_class(x))



        try:
            #first try to get the final from the most frequent pitch classes
            candidates = ppp_candidates[ppp_candidates['pitch_class'].isin(pcp_top)]
            candidates = candidates.loc[candidates['duration']>self.tailthreshold]
            lowest_final = min(candidates['miditone'])
            # lowest_velocity = candidates.loc[candidates['miditone'] == lowest_final, 'velocity'].values[0]

        except:
            #else select the final from all miditones in the last self.tailrows non-empty thresholded rows
            candidates = ppp_candidates.loc[ppp_candidates['duration']>self.tailthreshold]
            lowest_final = min(candidates['miditone'])
        
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
    
    def pitch_profile(self, transposition=0, normalisation=False, weighted = True): 
        """Calculate the pitch profile of the basic pitch extraction
        Arguments:
            transposition: the transposition is defined by the difference 
                between the lowest finals of the frequency file and of the 
                symbolic file
            normalisation: if True, the pitch will be normalised to a final on C
                    
        Returns:
            dataframe: 'pitch_name', 'pp_audio_proportion', 'miditone'
        """
        
        frequencies = self.freqs
        
        frequencies['dur'] = frequencies['end_time_s'] - frequencies['start_time_s']
        if weighted == True: 
            frequencies['duration'] = (frequencies['dur'] * frequencies['velocity'])
        else:
            frequencies['duration'] = frequencies['dur']
        
        grouped_durations = frequencies.groupby('miditone')['duration'].sum().reset_index()
        total_duration = grouped_durations['duration'].sum()
        grouped_durations['pp_audio_proportion'] = grouped_durations['duration'] / total_duration

        #transpose if desired for comparison to symbolic files
        grouped_durations['miditone'] = grouped_durations['miditone']+transposition
        
        # normalise if desired for comparision to other recordings
        if normalisation == True:
            difference = self.final_midi()%12
            if difference > 5:
                difference = -(12-difference)
            grouped_durations['miditone'] = grouped_durations['miditone']-difference
        
        
        # Merge with pitch class names from Tones.TONES
        pitch_profile = grouped_durations.merge(Tones.TONES[['miditone', 'pitch_name']], 
                                              on='miditone', how='outer')
        pitch_profile = (pitch_profile
                         .dropna(subset=['pitch_name'])
                         .reset_index(drop=True)
                          .drop(columns=['duration'])
                         )
        
        pitch_profile['pp_audio_proportion'] = pitch_profile['pp_audio_proportion'].fillna(0)
        
        return pitch_profile
    
    def pitch_class_profile(self, transposition =0, normalisation=False):
        """Calculate the pitch class profile of the FrequencyFile
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


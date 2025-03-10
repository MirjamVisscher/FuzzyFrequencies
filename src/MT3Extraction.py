#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:39:22 2025

@author: mirjam
"""



import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from utils import Tones

class MT3Extraction:
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
        self.title = file_name.replace('.mid', '')
        self.experiment = experiment
        self.path = os.path.join('..', 'data',  'raw', experiment, 'MT3', self.file_name)
        self.midi_data = pretty_midi.PrettyMIDI(self.path)
        metadata = pd.read_csv('../data/raw/'+experiment+'/experiment_metadata.csv')
        self.metadata = metadata.loc[metadata['composition']==self.title]
        try: self.metadata = self.metadata.loc[self.metadata['exclude']!=1]
        except: pass
        try:
            self.audio_final= int(self.metadata.audio_final.iloc[0])# changeID 1
        except:
            pass
    
    def lowest_final(self):
        """Calculate the lowest of the last miditones
        Returns:
            int: miditone
        """
        final = self.audio_final
        return final
    
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
        final_name = Tones.TONES.loc[Tones.TONES['miditone'] == final_midi, 'pitch_name'].values[0]
        return final_name
    

        
    def pitch_profile(self, transposition=0, normalisation=False):
        """Calculate the pitch profile of the basic pitch extraction.
        Returns:
            dataframe: 'pitch_name', 'pp_audio_proportion', 'miditone'
        """
        # Extract all note pitches
        pitches = []
        for instrument in self.midi_data.instruments:
            for note in instrument.notes:
                duration = note.end - note.start
                pitches.extend([note.pitch] * int(duration * 100))
                

        if not pitches:
            print(f"Warning: No pitched notes found in {self.path}")
            return pd.DataFrame(columns=['pitch_name', 'pp_audio_proportion', 'miditone'])

        # Count occurrences of each pitch
        pitch_counts = np.bincount(pitches)
        total_pitches = sum(pitch_counts)

        
        pitch_proportions = pitch_counts / total_pitches

        # Create DataFrame
        pitch_profile_df = pd.DataFrame({
            'pp_audio_proportion': pitch_proportions,
            'miditone': np.arange(len(pitch_counts))
        })
        
        
        #transpose if desired for comparison to symbolic files
        pitch_profile_df['miditone'] = pitch_profile_df['miditone']+transposition
        
        # normalise if desired for comparision to other recordings
        if normalisation == True:
            difference = self.final_midi()%12
            if difference > 5:
                difference = -(12-difference)
            pitch_profile_df['miditone'] = pitch_profile_df['miditone']-difference
        

        pitch_profile = pitch_profile_df.merge(Tones.TONES[['miditone', 'pitch_name']], 
                                              on='miditone', how='outer')
        pitch_profile = (pitch_profile
                         .dropna(subset=['pitch_name'])
                         .reset_index(drop=True)
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
    
    def piano_roll(self, save = False):
        """Generate a piano roll visualization with enhanced formatting."""
        
        # Ensure there are notes in the file
        if not any(instrument.notes for instrument in self.midi_data.instruments):
            print(f"Warning: No notes found in {self.path}")
            return
        
        # Create directory for saving plots
        output_dir = os.path.join('..', 'results', 'figures', self.experiment, 'piano_roll', 'MT3')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(60, 20))

        # Define pitch range for grid lines
        F_coef = range(24, 96)  # Define MIDI pitch range

        # Draw horizontal guide lines
        for pitch in F_coef:
            mod12 = pitch % 12
            if mod12 in {2, 4, 5, 7, 11, 0}:  # Stronger lines for these pitches
                color = 'black'
                linewidth = 0.7
            elif mod12 in {1, 3, 6, 8, 10}:  # Lighter grid lines
                color = 'lightgray'
                linewidth = 0.5
            elif mod12 == 9:  # Red line for a reference pitch
                color = 'red'
                linewidth = 1
            else:
                continue
            ax1.axhline(y=pitch, color=color, linewidth=linewidth, linestyle='-', alpha = 0.7)

        # Plot each instrument's notes
        for i, instrument in enumerate(self.midi_data.instruments):
            for note in instrument.notes:
                ax1.plot([note.start, note.end], [note.pitch, note.pitch], 
                         linewidth=6, label=instrument.name if i == 0 else "", alpha=0.7, color='black')

        # Set labels and formatting
        ax1.set_xlabel("Time (s)", fontsize=25)
        ax1.set_ylabel("MIDI Pitch", fontsize=25)
        ax1.set_yticks(F_coef)  # Set y-axis ticks
        ax1.set_yticklabels([pitch if (pitch + 3) % 12 == 0 else "" for pitch in F_coef], fontsize=25)

        # Create a secondary y-axis on the right
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Sync y-limits with the primary y-axis
        ax2.set_yticks(F_coef)
        ax2.set_yticklabels([pitch if (pitch + 3) % 12 == 0 else "" for pitch in F_coef], fontsize=25)
        
        ax1.tick_params(axis='x', labelsize=25)  # Sets x-axis tick label size

        ax1.set_title(f"Piano Roll for {os.path.basename(self.path)}")
        ax1.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save == True:
        # Save before showing
            save_path = os.path.join(output_dir, f'{self.title}.png')
            plt.savefig(save_path)
            print(f"Saved piano roll: {save_path}")

        


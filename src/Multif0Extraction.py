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
import matplotlib.gridspec as gridspec
from scipy.io.wavfile import write

from utils import get_miditone, get_pitch_class, Tones, Constants, get_pitch_name, get_frequency
from utils_multif0 import apply_fade_in_out, fill_gaps

class Multif0Extraction:
    """A class representing a set of frequencies. the number of voices may 
    vary depending on the audio file where the frequencies are extracted from.
    Attributes:
        file_name(string): name of the file
        title(string): name of the file without file extention
        experiment(string): name of the experiment
        freqs(dataframe): with columns:
            timestamp (float): The timestamp of the frequency.
            voice1 to voicen (float): frequency of that voice on timestamp t.
        tail(int): the number of timeframes used to determine the lowest final
    """
    def __init__(self, file_name, experiment):
        self.file_name = file_name
        self.title = file_name.replace('.csv', '')
        self.experiment = experiment
        
        # Read the frequency data and column mapping
        path = '../data/raw/'+experiment+'/multif0/'
        freqs = pd.read_csv(os.path.join(path, file_name), header=None)
        self.voices = freqs.shape[1]-1
        freqs.columns = ['timestamp'] + [f'voice{i}' for i in range(1, len(freqs.columns))]
        
        self.freqs = freqs.apply(pd.to_numeric, errors='coerce')
        self.tail = 200
    
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"file_name:\n{self.file_name}\n\nexperiment:\n{self.experiment}\ntitle:\n{self.title}"
        return information
        
    def pitch_deviation(self):
        """Calculate the deviation from the concert pitch of the audio file.
        Returns:
            float: the deviation in midi distance from most frequent pitch
                """
        # get the most frequent multif0
        longfreqs = self.freqs.drop(columns=['timestamp']).melt().dropna().drop(columns=['variable'])
        longfreqs = longfreqs.value_counts().sort_index().reset_index()
        # most_frequent_index = longfreqs[0].idxmax()
        most_frequent_index = longfreqs[longfreqs.columns[1]].idxmax()
        # get the neighboring frequencies, 2 below, 2 above
        peak = longfreqs.loc[most_frequent_index - 2 : most_frequent_index + 2].copy()
        # make sure the values are numeric
        peak['value'] = pd.to_numeric(peak['value'], errors='coerce')
        # get the miditone of both freqencies and calculate the weighted mean
        peak['miditone'] = peak['value'].apply(get_miditone)
        # weighted_avg = (peak['miditone'] * peak[0]).sum() / peak[0].sum()
        weighted_avg = (peak['miditone'] * peak[peak.columns[0]]).sum() / peak[peak.columns[0]].sum()
        # get the difference in cents between that miditone and the miditone, rounded to the closest integer
        deviation = weighted_avg-round(weighted_avg)
        
        return deviation
    
    def concert_pitch(self):
        """Calculate the concert pitch of the audio file.
        Returns:
            float: The frequency of the pitch most close to a 440Hz.
        """
        concert_pitch = get_frequency(69 + self.pitch_deviation())
        
        return concert_pitch
    
            
    def lowest_final(self, correction=True):
        """Calculate the lowest frequency that has more than 29 instances in the tail of voice1
        but only if the final is part of the top n in the pitch class profile.
        This is the Wiering solution
        Returns:
            float: frequency of the last note of the lowest voice.
        """

        if correction == True: 
            chamber_pitch = self.concert_pitch()
            
        else:
            chamber_pitch = Constants.CHAMBER_PITCH
        
        pcp = self.pitch_class_profile(transposition = 0, normalisation = False)
        
        tail_values = self.freqs.voice1.dropna().tail(self.tail).to_frame()
        tail_values['midi_tone'] = tail_values['voice1'].apply(lambda x: round(get_miditone(x, chamber_pitch)))
        tail_values['pitch_class'] = tail_values['midi_tone'].apply(lambda x: get_pitch_class(x))
        
        top = 7 #number of top candidates in the pitch class profile that can be a final
        pcp_top7 = pcp.nlargest(top, 'pcp_audio_proportion')['pitch_class']
        candidates = tail_values[tail_values['pitch_class'].isin(pcp_top7)]        
        
        #in case there are no candidates in the last 200 frequencies of voice1, take all frequencies
        if len(candidates)>0:
            midi_tone_counts = candidates['midi_tone'].value_counts()
        else:
            midi_tone_counts = tail_values['midi_tone'].value_counts()
        
        
        # Filter values where count is greater than 29
        midi_tone_counts = midi_tone_counts[midi_tone_counts > 29].index
        # Select the lowest value from the filtered values
        lowest_value = min(midi_tone_counts)
    
        # Select the first frequency corresponding to the lowest midi_tone
        lowest_final = tail_values[tail_values['midi_tone'] == lowest_value]['voice1'].iloc[0]
                
        return lowest_final

    
    def final_midi(self, correction = True):
        """Get the midi number closest to the frquency of the last note of the 
        lowest voice
        Returns:
            int: midi_number of the last note of the lowest voice
        """
        if correction == True: 
            chamber_pitch = self.concert_pitch()
            
        else:
            chamber_pitch = Constants.CHAMBER_PITCH
        # print(chamber_pitch)
        final_midi = round(get_miditone(self.lowest_final(),chamber_pitch))
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
              
    def piano_roll(self):
        """plot the frequencies of Multif0Extraction on a piano roll representation
        without correction for chamber pitch neither for transposition.
        Returns:
            matplotlib plot: scatterplot of all voices.
        """

        selected_columns = self.freqs.drop(['timestamp'], axis=1)

        # Getting the minimum and maximum values
        lowest = selected_columns.min().min()
        highest = selected_columns.max().max()
        tones = Tones.TONES[(Tones.TONES['frequency'] >= lowest) & (Tones.TONES['frequency'] <= highest)]
        
        #create output path
        output_path = os.path.join('../results/figures', self.experiment, 'piano_roll','multif0')
        os.makedirs(output_path, exist_ok=True)       
        
        whitekeys = tones[tones['white_key'] == 1].frequency.to_list()
        aas = tones[tones['pitch_class_name']=='A'].frequency
        semitones = tones.frequency
        
        # Create the figure
        plt.figure(figsize=(60, 10))
    
        for i, column in enumerate(self.freqs.columns[1:]):
            color = plt.cm.brg(i / len(self.freqs.columns[1:]))
            plt.scatter(self.freqs['timestamp'], self.freqs[column], label=column, color=color, s=10)
                
        # Set labels, scales, and title
        plt.xlabel('Timestamp in seconds')
        plt.ylabel('Pitch')
        plt.yscale('log')
    
        for freq in semitones:
            plt.axhline(y=freq, color='lightgray', linewidth=0.5)
    
        for freq in whitekeys:
            plt.axhline(y=freq, color='black', linewidth=0.5)
    
        for freq in aas:
            plt.axhline(y=freq, color='red', linewidth=0.5)
        
        x_tick_interval = 10
        x_ticks = range(0, int(self.freqs['timestamp'].max()) + 1, x_tick_interval)
        plt.xticks(x_ticks)
        whitekey_tones = tones[tones['white_key'] == 1]

        # Customize y-axis ticks
        plt.yticks(whitekey_tones['frequency'], whitekey_tones['pitch_name'])
        plt.minorticks_off()
    
        # Save the figure
        plt.title(self.title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, self.title + '.png'))
        plt.show()
        plt.close()
        return plt
    
    def piano_roll_hist(self, histogram=False):
        """Plot the frequencies of Multif0Extraction on a piano roll representation
        without correction for chamber pitch or transposition and includes a histogram on the right of the plot
        Returns:
            matplotlib plot: scatterplot of all voices.
        """
        # Drop the 'timestamp' column for processing frequencies
        selected_columns = self.freqs.drop(['timestamp'], axis=1)
        
        # Get the minimum and maximum frequency values
        lowest = selected_columns.min().min()
        highest = selected_columns.max().max()
    
        # TODO: Implement librosa's pitch algorithm instead of Tones.TONES if needed
        tones = Tones.TONES[(Tones.TONES['frequency'] >= lowest) & (Tones.TONES['frequency'] <= highest)]
        
        # Create output path
        output_path = os.path.join('../results/figures', self.experiment, 'piano_roll_histogram')
        os.makedirs(output_path, exist_ok=True)
        
        # Filter tones for plotting
        whitekeys = tones[tones['white_key'] == 1].frequency.to_list()
        aas = tones[tones['pitch_class_name'] == 'A'].frequency
        semitones = tones.frequency
        
        # Create the figure with gridspec
        fig = plt.figure(figsize=(60, 10))
        # gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 0.5], wspace=0)
        
        # Piano roll plot
        ax0 = plt.subplot(gs[0])
        for i, column in enumerate(self.freqs.columns[1:]):
            # color = plt.cm.brg(i / len(self.freqs.columns[1:]))
            # ax0.scatter(self.freqs['timestamp'], self.freqs[column], label=column, color='blue', s=2, marker='h')
            plt.scatter(self.freqs['timestamp'], self.freqs[column], label=column, color='#C00A35', s=5)
        
        # Set labels, scales, and title for piano roll
        ax0.set_xlabel('Timestamp in seconds')
        ax0.set_ylabel('Pitch')
        ax0.set_yscale('log')
        
        # Add horizontal lines for semitones, white keys, and 'A' notes
        for freq in semitones:
            ax0.axhline(y=freq, color='gray', linestyle='--', linewidth=0.5)
        for freq in whitekeys:
            ax0.axhline(y=freq, color='black', linewidth=0.5)
        for freq in aas:
            ax0.axhline(y=freq, color='red', linewidth=0.5)
        
        # Set x-axis ticks
        x_tick_interval = 10
        x_ticks = range(0, int(self.freqs['timestamp'].max()) + 1, x_tick_interval)
        ax0.set_xticks(x_ticks)
        
        # Customize y-axis ticks to show white key pitch names
        whitekey_tones = tones[tones['white_key'] == 1]
        ax0.set_yticks(whitekey_tones['frequency'])
        ax0.set_yticklabels(whitekey_tones['pitch_name'])
        ax0.minorticks_off()
        
        # Set x-axis limits to remove unnecessary white space
        ax0.set_xlim(left=0, right=self.freqs['timestamp'].max())
        
        # Save the figure
        # ax0.set_title(self.title)
        ax0.set_title('Piano roll of Takarazuka, Hilliard Ensemble, Coro dell\'Accademia corale di Lecco, Venere Lute Quartet')
        
        if histogram == True:
            # Histogram plot
            pitch_profile = self.pitch_profile(transposition=0, normalisation=False, correction=True)
            pitch_profile['frequency'] = pitch_profile['miditone'].apply(get_frequency)
            
            # Calculate bar heights based on frequency differences
            frequencies = pitch_profile['frequency']
            frequency_diffs = np.diff(frequencies) / 2
            heights = np.hstack([frequency_diffs, frequency_diffs[-1]])  # Repeat the last difference for the final bar
            
            
            ax1 = plt.subplot(gs[1], sharey=ax0)
            ax1.barh(pitch_profile['frequency'], pitch_profile['pp_audio_proportion'], color='#C00A35', height=heights)
            ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Remove y-axis labels and ticks
            ax1.set_xlabel('Pitch class profile')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, self.title + '.png'))
        plt.show()
        plt.close()
        
        return plt
    
    def pitch_profile(self, transposition, normalisation, correction=True):
        """Calculate the pitch profile of the Multif0Extraction
        Arguments:
            transposition: the transposition is defined by the difference 
                between the lowest finals of the frequency file and of the 
                symbolic file
            normalisation: if True, the pitch will be normalised to a final on C
            correction: if True, the chamber pitch will be detected and the 
                extracted pitches will be brought close to 440 Hz
        Returns:
            dataframe: 'miditone', 'pp_audio_proportion', 'pitch_name'
        """
        # print('transposition in class Multif0Extraction: '+str(transposition))
        if correction == True: 
            chamber_pitch = self.concert_pitch()
        else:
            chamber_pitch = Constants.CHAMBER_PITCH
        
        longfreqs = (self
                     .freqs.drop(columns=['timestamp'])
                     .melt()
                     .dropna()
                     .drop(columns=['variable'])
                     .rename(columns={'value': 'frequency'})
                     )
        longfreqs['miditone'] = (longfreqs['frequency']
                                     .apply(lambda frequency: get_miditone(frequency, chamber_pitch))
                                     .round()
                                     )
        longfreqs['miditone'] = longfreqs['miditone']+transposition
        if normalisation == True:
            difference = self.final_midi()%12
            if difference > 5:
                difference = -(12-difference)
            longfreqs['miditone'] = longfreqs['miditone']-difference

        pitch_profile = (longfreqs['miditone']
                         .value_counts(normalize=True)
                         .rename('pp_audio_proportion')
                         .to_frame()
                         .reset_index()
                         .rename(columns={'index': 'miditone'})
                         )
        #TODO implement librosa's pitch algorithm instead of Tones.TONES
        pitch_profile = pitch_profile.merge(Tones.TONES[['miditone', 'pitch_name']], on='miditone', how='outer')
        pitch_profile = pitch_profile.dropna(subset=['pitch_name'])
        pitch_profile = pitch_profile.reset_index(drop=True).fillna(0)
        
        return pitch_profile
    
    def pitch_class_profile(self, transposition, normalisation, correction=True):
        """Calculate the pitch class profile of the Multif0Extraction
        Arguments:
            transposition: the transposition is defined by the difference 
                between the lowest finals of the frequency file and of the 
                symbolic file
            normalisation: if True, the pitch will be normalised to a final on C
            correction: if True, the chamber pitch will be detected and the 
                extracted pitches will be brought close to 440 Hz
        
        Returns:
            dataframe: 'pitch_class_name', 'pitch_class', 'pcp_audio_proportion'
        """

        
        if correction == True: 
            chamber_pitch = self.concert_pitch()
            
        else:
            chamber_pitch = Constants.CHAMBER_PITCH

        
        longfreqs = (self
                     .freqs.drop(columns=['timestamp'])
                     .melt()
                     .dropna()
                     .drop(columns=['variable'])
                     .rename(columns={'value': 'frequency'})
                     )
        longfreqs['miditone'] = (longfreqs['frequency']
                                      .apply(lambda frequency: get_miditone(frequency, chamber_pitch))
                                      .round()
                                      )
        
        longfreqs['miditone'] = longfreqs['miditone']+transposition
        if normalisation == True:
            difference = self.final_midi()%12
            if difference > 5:
                difference = -(12-difference)
            longfreqs['miditone'] = longfreqs['miditone']-difference
        
        longfreqs= longfreqs.merge(Tones.TONES[['miditone', 'pitch_class']], on='miditone', how='left')

        pitch_class_profile = (longfreqs['pitch_class']
                          .value_counts(normalize=True)
                          .rename('pcp_audio_proportion')
                          .to_frame()
                          .sort_values(by='pitch_class')
                          .reset_index()
                          )
        
        pitch_class_profile = (pitch_class_profile
                               .merge(Tones.TONES[['pitch_class', 'pitch_class_name']], on='pitch_class', how='outer')
                               .drop_duplicates()
                               .sort_values(by='pitch_class')
                               .reset_index()
                               .drop(columns='index')
                               .fillna({'pcp_audio_proportion': 0})
                               )
        
        return pitch_class_profile

    
    def make_midi_mf0(self, correction=True, replace=False):
        """Calculates the cadence pattern.
        
        Parameters:
            correction (bool): If True, adjusts pitches to 440 Hz.
            replace (bool): If False, stops execution if the output file exists.
            
        Returns:
            DataFrame or None: The MIDI data if created, otherwise None.
        """
        
        # Define output directory and file path
        odir = os.path.join('..', 'data', 'processed', self.experiment, 'midi_mf0')
        os.makedirs(odir, exist_ok=True)
        output_path = os.path.join(odir, self.file_name)
    
        # Check if file exists and return early if replace=False
        if os.path.exists(output_path) and not replace:
            print(f"File {output_path} already exists. Skipping execution.")
            return None  # Or raise an exception if you prefer
        
        # Determine chamber pitch
        chamber_pitch = self.concert_pitch() if correction else Constants.CHAMBER_PITCH
    
        # Convert frequencies to MIDI tones
        midis = pd.DataFrame(index=self.freqs.index, columns=self.freqs.columns)
        midis['timestamp'] = self.freqs['timestamp']
    
        for index, row in self.freqs.iterrows():
            for col in self.freqs.columns[1:]:
                if pd.notnull(row[col]):
                    midis.at[index, col] = round(get_miditone(row[col], chamber_pitch))
    
        # Save to file
        midis.to_csv(output_path)
        return midis

    """
    FROM HERE, WE EXPERIMENT WITH CADENCE PATTERNS
    Probably, I should do that in a different file
    """
    



    def cadence_pattern(self, correction = False):
        """Calculates the cadence pattern
        correction: if True, the chamber pitch will be detected and the 
            extracted pitches will be brought close to 440 Hz
            Returns:
                a count of the cadence tones
                """
        #turn the frequencies into miditones
        if correction == True: 
            chamber_pitch = self.concert_pitch()
            
        else:
            chamber_pitch = Constants.CHAMBER_PITCH
        
        midis = pd.DataFrame(index=self.freqs.index, columns=self.freqs.columns)
        midis['timestamp'] = self.freqs['timestamp']

        for index, row in self.freqs.iterrows():
            for col in self.freqs.columns[1:]:
                if pd.notnull(row[col]):
                    midis.at[index, col] = round(get_miditone(row[col], chamber_pitch))
        midis['chord'] = midis.iloc[:, 1:].apply(lambda x: ''.join(str(value) for value in x if not pd.isnull(value)), axis=1)
        midis['chord'] = midis['chord'].replace('', np.nan)
        
        #create a rolling chord
        def most_frequent_chord(row, window_size=30):
            window = row.name
            window_chords = midis.loc[window:window+window_size, 'chord']
            most_frequent = window_chords.mode()
            if not most_frequent.empty:
                return most_frequent.iloc[0]
            else:
                return None
            
        midis['rolling_chord'] = midis.apply(most_frequent_chord, axis=1)
        filtered_midis = midis[midis['chord'] == midis['rolling_chord']]
        
        def consecutive_values(series, threshold=15):
            return (series.groupby(series.ne(series.shift()).cumsum())
                          .transform('count')
                          .ge(threshold))
        
        # Filter the first row of each occurence of 15 or more consecutive identic chords
        filtered_midis_consecutive = filtered_midis[consecutive_values(filtered_midis['chord'])]
        filtered_midis_consecutive = filtered_midis_consecutive.dropna(subset=['voice2'])
        filtered_midis_unique = filtered_midis_consecutive[(filtered_midis_consecutive['chord'] != filtered_midis_consecutive['chord'].shift()) | (filtered_midis_consecutive.index == filtered_midis_consecutive.index[0])]
        
        #create the set of intervals that are present in this chord relative to the bass
        voice_columns = self.freqs.columns[2:7]
        intervals = {}
        for voice_col in voice_columns:
            interval_col = (filtered_midis_unique[voice_col] - filtered_midis_unique['voice1'])%12
            filtered_midis_unique.loc[:, f'interval{voice_col[-1]}'] = interval_col
        #indicate whether there is a dissonance
        dissonant_values = [1, 2, 5, 6, 10, 11]
        interval_columns = [col for col in filtered_midis_unique.columns if col.startswith('interval')]
        dissonance_mask = filtered_midis_unique[interval_columns].isin(dissonant_values).any(axis=1)
        filtered_midis_unique.loc[:, 'dissonance'] = dissonance_mask
        
        for column in filtered_midis_unique.columns:
            if column.startswith('voice'):
                # Apply the get_pitch_name function to the column and create a new column with the pitch names
                filtered_midis_unique.loc[:,column + '_pitch_name'] = filtered_midis_unique[column].apply(get_pitch_name)
        # odir = os.path.join('..', 'data', 'processed', experiment, 'midis')
        # os.makedirs(odir, exist_ok=True)
        # midis.to_csv(os.path.join(odir, self.file_name))
        return midis  
    
    def sonify(self, cents, max_gap=5, smooth=False):
        """ This script sonifies the midi or frequencies of the Multif0 extraction
        and save the wav file to 
        '..', 'results', 'output', self.experiment, 'sonification', 'Multif0_MIDI' for cents = 100
        and '..', 'results', 'output', self.experiment, 'sonification', 'Multif0_freq' for cents = 20
        Args:
           cents: size of the frequency bins before sonifying
           max_gap: maximum size of notegaps between timestamps to be filled
        """
        
        if cents == 100:
            midis = self.make_midi_mf0(replace=True)
            
            frequencies = midis.copy()
            frequencies.iloc[:, 1:] = 440 * (2 ** ((frequencies.iloc[:, 1:] - 69) / 12))
            
        elif cents == 20:
            frequencies = self.freqs
        else:
            print('please choose between 20 and 100 cents')
            pass

        # midi to columns
        freqs_wide= (
            frequencies.reset_index(drop=True)  # Keep the original index as a column
            .melt(id_vars=["timestamp"], value_name="frequency")
            .drop(columns=["variable"])
            .dropna()
            .astype({"frequency": int})
            .drop_duplicates(subset=["timestamp", "frequency"])
            .assign(present=1)
            .pivot_table(index="timestamp", columns="frequency", values="present", aggfunc="first")
            .fillna(0)
            # .set_index("index")  # Restore the original index
        )

        if cents == 100:
            freqs_wide = fill_gaps(freqs_wide, max_gap=max_gap)
            odir = os.path.join('..', 'data', 'processed', self.experiment, 'midiswide')
            os.makedirs(odir, exist_ok=True)
            freqs_wide.to_csv(os.path.join(odir, self.file_name))

        timestamps = freqs_wide.index.to_numpy()
        fs = 44100  # Sample rate
        duration = timestamps[-1]  # Total duration in seconds
        time = np.linspace(0, duration, int(fs * duration))  # Full time array
        waveform = np.zeros_like(time)


        def find_onsets_offsets(values, timestamps): # Function to find onsets and offsets
            """ Returns (onsets, offsets) for a single MIDI tone """
            onsets = timestamps[(values.shift(1, fill_value=0) == 0) & (values == 1)]
            offsets = timestamps[(values.shift(1, fill_value=0) == 1) & (values == 0)]
            
            if len(onsets) > len(offsets):  # Handle last note still sounding
                offsets = np.append(offsets, timestamps[-1])  # Assume it ends at last timestamp
            
            return list(zip(onsets, offsets))  # Pairs of (start, end)

        for frequency in freqs_wide.columns:
            onsets_offsets = find_onsets_offsets(freqs_wide[frequency], timestamps)
            
            for onset, offset in onsets_offsets:
                start_idx = int(onset * fs)
                end_idx = int(offset * fs)
                
                # Compute duration and adjust to nearest full cycle
                note_duration = (end_idx - start_idx) / fs  # in seconds
                period = 1 / frequency
                n_cycles = round(note_duration / period)  # Find closest integer cycles
                adjusted_duration = n_cycles * period  # Snap duration to full cycles
                
                # Recalculate end index based on adjusted duration
                adjusted_end_idx = start_idx + int(adjusted_duration * fs)
                if adjusted_end_idx >= len(time):  # Ensure we don't exceed array bounds
                    adjusted_end_idx = len(time) - 1
        
                # Generate sine wave starting at 0 and ending at 0
                t = time[start_idx:adjusted_end_idx]
                note_waveform = np.sin(2 * np.pi * frequency * t)
                
                # Apply fade-in and fade-out (optional, but helps smooth transitions further)
                note_waveform = apply_fade_in_out(note_waveform, fade_duration=0.01, fs=fs)
        
                # Add to the overall waveform
                waveform[start_idx:adjusted_end_idx] += note_waveform      
        waveform /= np.max(np.abs(waveform))  # Normalize waveform
        
        if cents ==100:
            sonidir = os.path.join('..', 'results', 'output', self.experiment, 'sonification', 'Multif0_MIDI')
        if cents ==20:
            sonidir = os.path.join('..', 'results', 'output', self.experiment, 'sonification', 'Multif0_freq')
        os.makedirs(sonidir, exist_ok=True)
        opath = os.path.join(sonidir,self.title+'.wav')
        write(opath, fs, (waveform * 32767).astype(np.int16))


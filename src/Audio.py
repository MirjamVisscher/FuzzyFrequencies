#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:06:51 2023

@author: mirjam
"""

import os
import pandas as pd
import numpy as np
import essentia.standard as es
import matplotlib.pyplot as plt
import librosa

from Constants import ToneConstants
from Multif0Extraction import Multif0Extraction

class Audio:
    """A class representing a recording of a composition in a .wav format.
    Attributes:
        filename(string): filename of the composition
        experiment: experiment of which the recording is part of
     """
    def __init__(self, composition, experiment):
        self.composition = composition
        self.experiment = experiment
        self.metadata = pd.read_csv('../data/raw/'+self.experiment+'/experiment_metadata.csv')
        self.hop_size = 256 # for Librosa and Essentia
        self.frame_size = 2048 #for Essentia
        try: 
            self.metadata = self.metadata.loc[self.metadata['exclude']!=1]
        except:
            pass
        self.file_name = self.metadata.loc[(self.metadata['composition']==self.composition)].audio.iloc[0]# changeID 1
        self.multif0_filename = self.metadata.loc[(self.metadata['composition']==self.composition)].mf0.iloc[0] # changeID 1
        
        self.path = '../data/raw/'+experiment+'/audio/'
        self.file_path = os.path.join(self.path, self.file_name)
        try:
            self.y, self.sr = librosa.load(self.file_path)
        except:
            print('loading Librosa audio failed')
        
        plt.style.use('uu')
  
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"title:\n{self.file_name}\n\nexperiment:\n{self.experiment}"
        return information
    def hpcp(self):
        try:
            #get essentia object
            esAudio = es.MonoLoader(filename=self.file_path)()
            print('Essentia audio object loaded')
        except:
            print('loading Essentia audio object failed')
        # Essentia algorithms
        window = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        spectral_peaks = es.SpectralPeaks()
        hpcp = es.HPCP()
        
        # Store all HPCP frames
        hpcp_frames = []
        
        # Process the entire track frame by frame
        for frame in es.FrameGenerator(esAudio, frameSize=self.frame_size, hopSize=self.hop_size):
            spec = spectrum(window(frame))  # Compute spectrum
            peaks_freq, peaks_mag = spectral_peaks(spec)  # Compute spectral peaks
            hpcp_output = hpcp(peaks_freq, peaks_mag)  # Compute HPCP
            hpcp_frames.append(hpcp_output)  # Store HPCP for this frame
        
        # Convert to numpy array
        hpcp_matrix = np.array(hpcp_frames)
        
        # Aggregate the HPCP across all frames (e.g., by averaging)
        hpcp_mean = np.mean(hpcp_matrix, axis=0)
        hpcp_shifted = np.roll(hpcp_mean, -3) # HPCP starts on A; convert to C-B vector
        return hpcp_shifted
        
        
    def chroma(self, chromatype):
        
        if chromatype == 'multif0':
            global_chroma = Multif0Extraction(self.multif0_filename, self.experiment).pitch_class_profile(transposition=0, normalisation=False ).pcp_audio_proportion
        elif chromatype == 'hpcp':
            global_chroma = self.hpcp()
            global_chroma = global_chroma / np.sum(global_chroma)
        else:
            if chromatype == 'cens':
                pcp_chroma = librosa.feature.chroma_cens(y=self.y, sr=self.sr, hop_length = self.hop_size)
            elif chromatype == 'cqt':
                pcp_chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, hop_length = self.hop_size)
            elif chromatype == 'vqt':
                pcp_chroma = librosa.feature.chroma_vqt(y=self.y, sr=self.sr,
                                                   intervals='ji5',
                                                   bins_per_octave=12,
                                                   hop_length=256)
            
            else:
                raise ValueError(f"Unsupported chromatype: {chromatype}")
            
            global_chroma = pcp_chroma.mean(axis=1)
            global_chroma = global_chroma / np.sum(global_chroma)
        return global_chroma
        
        
    def show_chroma(self):
        plt.figure(figsize=(10, 5))
        n_chroma = 12
        bar_width = 0.2
        index = np.arange(n_chroma)
        chromatic = ToneConstants.ALL_KEYS
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.bar(index - 1.5*bar_width, self.chroma('multif0'), bar_width, label='multif0')
        ax.bar(index - 0.5*bar_width, self.chroma('cens'), bar_width, label='CENS')
        ax.bar(index + 0.5*bar_width, self.chroma('cqt'), bar_width, label='CQT')
        ax.bar(index + 1.5*bar_width, self.chroma('vqt'), bar_width, label='VQT')
        ax.set_xlabel('Chroma')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Global chroma comparison for {self.file_name}')
        ax.set_xticks(index)  # Set numerical indices as ticks
        ax.set_xticklabels(chromatic)  # Set chroma names as tick labels
        ax.legend()
        plt.show()
        plt.close()
        return plt


    def cqt_pp(self, transposition):
        """
        This function is a dericative of librosa.chroma_cqt(), but without the 
        mapping to chroma.
    
        """
        n_octaves= 7
        bins_per_octave = 36

        # Build the CQT
        C = np.abs(
            librosa.cqt(
                self.y,
                sr=self.sr,
                hop_length=self.hop_size,
                n_bins=n_octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
            ))
        pitch_profile = np.sum(C, axis=1)

        # Normalize so that the sum is 1
        pitch_profile /= np.sum(pitch_profile)
        
        # Define MIDI range from A0 (MIDI 21) to C8 (MIDI 108)
        midi_bins = np.arange(21, 109)
        midi_profile = np.zeros_like(midi_bins, dtype=np.float32)
        
        # CQT bins start at C1, corresponding to MIDI 24
        start_midi = 24
        n_semitones = len(pitch_profile) // (bins_per_octave // 12)  # Convert to semitone bins
        
        # Aggregate 1/3-semitone bins into full semitones
        for i in range(n_semitones):
            midi_note = start_midi + i
            if midi_note in midi_bins:
                start_bin = i * (bins_per_octave // 12)
                end_bin = (i + 1) * (bins_per_octave // 12)
                midi_profile[midi_note - 21] = np.sum(pitch_profile[start_bin:end_bin])
        if transposition != 0:
            midi_profile = np.roll(midi_profile, shift=transposition)
    
            # Zero out values that wrapped around
            if transposition > 0:  # Shift up, cut off high-end values
                midi_profile[:transposition] = 0
            elif transposition < 0:  # Shift down, cut off low-end values
                midi_profile[transposition:] = 0
        return midi_profile
    
            
    
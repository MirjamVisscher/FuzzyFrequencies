#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:00:52 2024

@author: mirjam
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import Constants, get_pitch_class_name
from Multif0Extraction import Multif0Extraction

# plt.style.use('uu')


class Cycle:
    """A class representing a modal cycle in the Pitch Profiles study
    Attributes:
        experimentname(string): name of the experiment
        cycle_name (string): name of the cycle
        composer (string): full name of the composer of the cycle
        multif0_files(list): list of frequency files in this experiment
    """
    
    def __init__(self, experiment, composer, cycle_name):
        self.experiment = experiment
        self.cycle_name = cycle_name
        self.composer = composer
        try:
            
            self.metadata = pd.read_csv('../data/raw/'+experiment+'/experiment_metadata.csv', dtype={'mode': str})
        except:
            self.metadata = pd.read_csv('../data/raw/'+experiment+'/experiment_metadata.csv')
        try:
            self.metadata = self.metadata.loc[self.metadata['cycle'] == self.cycle_name]
        except:
            print(f'Either the composer or the cycle are not in the metadata. Please check the file experiment_metadata.csv in your {self.experiment} folder.')
        try:
            self.metadata = self.metadata.sort_values(by=['mode', 'nr. playlist'], ascending=True)
        except:
            pass
        try: self.metadata = self.metadata.loc[self.metadata['exclude']!=1]
        except:
            pass
        try:
            self.metadata = self.metadata[(self.metadata['symbolic_is_audio'] != 'n') & (self.metadata['final_safe'] != 'n')]
        except:
            pass
        
        self.compositions = self.metadata['composition']
        self.multif0_files = self.metadata['file_name']
        
        
    def __str__(self):
        """ Print instance information of this class """
        information = (f"experiment:\n {self.experiment}\n"
                       f"cycle:\n {self.cycle_name}\n"
                       f"composer:\n {self.composer}\n"
                       f"compositions:\n {self.compositions}\n"
                       f"frequency files:{self.multif0_files}")
        return information
    
    def get_profile(self, file_name, normalisation):
        print(file_name+' '+str(normalisation))
        mf0 = Multif0Extraction(file_name, self.experiment)
        pcp = pd.DataFrame(mf0.pitch_class_profile(transposition=0, normalisation=normalisation, correction=True))
        final = get_pitch_class_name(mf0.final_midi())
        return pcp, final

    def plot_combined_pitch_class_profiles(self, mode, normalisation):
        """ Plot combined pitch class profiles for one specified mode in a cycle        
        """
        all_pcps = []
        finals = []
        compositions = []
        # tonal_types = []
        
        modelist = self.metadata.loc[self.metadata['mode'] == mode]
        
        for index, row in modelist.iterrows():
            
            file_name = row['file_name']
            print(file_name)#DEBUG
            pcp, final = self.get_profile(file_name, normalisation)
            all_pcps.append(pcp)
            finals.append(final)
            compositions.append(row['composition'])
            # tonal_types.append(row['tonal_type'])
        
        # Number of rows to plot
        num_rows = len(modelist)
        
        pitch_classes = all_pcps[0]['pitch_class_name']
        
        # Set the width of each bar and the positions of the bars
        bar_width = 0.8 / num_rows
        positions = range(len(pitch_classes))
        
        plt.figure(figsize=(12, 3))
        
        # for i, (pcp, final, composition, tonal_type) in enumerate(zip(all_pcps, finals, compositions, tonal_types)):
        for i, (pcp, final, composition) in enumerate(zip(all_pcps, finals, compositions)):
            proportions = pcp['pcp_audio_proportion']
            bar_positions = [pos + (i * bar_width) for pos in positions]
            # print(tonal_type)#DEBUG
            # label = tonal_type+ ', '+final+ ', '+composition
            label = final+ ', '+composition
            
            bars = plt.bar(bar_positions, proportions, bar_width, label=label, alpha=0.6)
            
            if normalisation == False:
                # Find the index of the final pitch class
                final_index = pcp[pcp['pitch_class_name'] == final].index[0]
                bars[final_index].set_edgecolor('black')
                bars[final_index].set_linewidth(2)
                bars[final_index].set_alpha(1)
            
        plt.title(f'Mode {mode}')
        plt.xticks([pos + (bar_width * (num_rows / 2)) for pos in positions], pitch_classes)
        # Place the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
  
    
    def plot_combined_pitch_class_profiles_multiple(self, mode, normalisation, ax=None):
        """Plot combined pitch class profiles for all files in metadata for a given mode."""
        
        all_pcps, finals, compositions = [], [], []
        
        # Filter metadata for the given mode
        modelist = self.metadata.loc[self.metadata['mode'] == mode]
    
        for _, row in modelist.iterrows():
            file_name = row['file_name']
            pcp, final = self.get_profile(file_name, normalisation)
            all_pcps.append(pcp)
            finals.append(final)
            compositions.append(row['composition'])
    
        num_rows = len(modelist)
        pitch_classes = all_pcps[0]['pitch_class_name'].values  # Extract pitch class names
    
        # Bar settings
        bar_width = 0.8 / num_rows
        positions = range(len(pitch_classes))
    
        # Create figure if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
    
        # Plot bars for each composition
        for i, (pcp, final, composition) in enumerate(zip(all_pcps, finals, compositions)):
            proportions = pcp['pcp_audio_proportion']
            bar_positions = [pos + (i * bar_width) for pos in positions]
    
            bars = ax.bar(bar_positions, proportions, bar_width, label=f'{composition}', alpha=0.6)
    
            # Highlight final pitch class if normalisation is False
            if not normalisation:
                final_index = pcp[pcp['pitch_class_name'] == final].index[0]
                bars[final_index].set_edgecolor('black')
                bars[final_index].set_linewidth(2)
                bars[final_index].set_alpha(1)
    
        # Formatting
        ax.set_xlabel('Pitch Class')
        ax.set_ylabel('Proportion')
        ax.set_title(f'Mode {mode}')
        ax.set_xticks([pos + (bar_width * (num_rows / 2)) for pos in positions])
        ax.set_xticklabels(pitch_classes)
    
        # Modify legend
        handles, labels = ax.get_legend_handles_labels()
        final_patch = mpatches.Patch(edgecolor='black', facecolor='none', linewidth=2, label='Final')
        handles.append(final_patch)
        labels.append('Final')
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_combined_pitch_class_profiles_single(self, mode, normalisation, ax=None):
        """Plot combined pitch class profiles for all files in metadata for a given mode."""
        
        all_pcps, finals, compositions = [], [], []
        
        # Filter metadata for the given mode
        modelist = self.metadata.loc[self.metadata['mode'] == mode]
    
        for _, row in modelist.iterrows():
            file_name = row['file_name']
            pcp, final = self.get_profile(file_name, normalisation)
            all_pcps.append(pcp)
            finals.append(final)
            compositions.append(row['composition'])
    
        num_rows = len(modelist)
        pitch_classes = all_pcps[0]['pitch_class_name'].values  # Extract pitch class names
    
        # Bar settings
        bar_width = 0.8 / num_rows
        positions = range(len(pitch_classes))
    
        # Create figure if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
    
        # Plot bars for each composition
        for i, (pcp, final, composition) in enumerate(zip(all_pcps, finals, compositions)):
            proportions = pcp['pcp_audio_proportion']
            bar_positions = [pos + (i * bar_width) for pos in positions]
    
            bars = ax.bar(bar_positions, proportions, bar_width, label=f'{composition}', alpha=0.6)
    
            # Highlight final pitch class if normalisation is False
            if not normalisation:
                final_index = pcp[pcp['pitch_class_name'] == final].index[0]
                bars[final_index].set_edgecolor('black')
                bars[final_index].set_linewidth(2)
                bars[final_index].set_alpha(1)

        ax.set_title(f'Mode {mode}: {composition}')
        ax.set_xticks([pos + (bar_width * (num_rows / 2)) for pos in positions])
        ax.set_xticklabels(pitch_classes)
        plt.legend('',frameon=False)

    
    def plot_combined_pitch_class_profiles_all_modes(self, normalisation):
        """ Plot combined pitch class profiles for all unique modes in a single combined plot """
        unique_modes = self.metadata['mode'].unique()
        num_modes = len(unique_modes)
        
        # Define the number of rows and columns for the subplot grid
        num_cols = 1
        num_rows = num_modes
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(11, 2.5*num_modes), sharex=False)
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical space between subplots
        
        for i, mode in enumerate(unique_modes):
            ax = axs[i]
            ax.set_title(f'Mode {mode}')
            self.plot_combined_pitch_class_profiles_multiple(mode, normalisation, ax=ax)
            ax.legend(loc='upper left')
            if normalisation== False:
                # Set the x-ticks and labels individually for each subplot
                ax.set_xticks(range(len(Constants.ALL_KEYS)))
                ax.set_xticklabels(Constants.ALL_KEYS)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymin + 1.5 * (ymax - ymin))
    
        
        plt.xlabel('Pitch Class')  # Common x-axis label for all subplots
        plt.ylabel('Proportion')   # Common y-axis label for all subplots
        plt.tight_layout()
        # plt.title(f'{self.composer}, {self.cycle_name}')
        
        output_path = os.path.join('../results/figures', self.experiment,'cycles')
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, self.composer+ ', '+self.cycle_name + '11.png'))
        plt.show()
        plt.close()
        
    def plot_combined_pitch_class_profiles_all_modes_2col(self, normalisation):
        """ Plot combined pitch class profiles for all unique modes in a single combined plot """
        unique_modes = self.metadata['mode'].unique()
        num_modes = len(unique_modes)
    
        # Define the number of rows and columns for the subplot grid
        num_cols = 2
        num_rows = (num_modes + 1) // num_cols  # Calculate rows needed for 2 columns
    
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(11, 2.5*num_rows), sharex=False)
        fig.subplots_adjust(hspace=0.5)  # Adjust vertical space between subplots
    
        for i, mode in enumerate(unique_modes):
            row, col = i // num_cols, i % num_cols  # Determine row and column
            ax = axs[row, col]
            # ax = axs[i // num_cols, i % num_cols]  # Adjust indexing for 2D array of axes
            ax.set_title(f'Mode {mode}')
            self.plot_combined_pitch_class_profiles_single(mode, normalisation, ax=ax)
            # ax.legend(loc='upper left')
            if not normalisation:
                ax.set_xticks(range(len(Constants.ALL_KEYS)))
                ax.set_xticklabels(Constants.ALL_KEYS)
            ymin, ymax = ax.get_ylim()
            # ax.set_ylim(ymin, ymin + 1.5 * (ymax - ymin))
            if col == 0:
                ax.set_ylabel('Proportion')

            # Set labels only for the bottom row
            if row == num_rows - 1:
                ax.set_xlabel('Pitch Class')
        # Hide any unused subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axs[j // num_cols, j % num_cols])
    

    
        output_path = os.path.join('../results/figures', self.experiment, 'cycles')
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, self.composer + ', ' + self.cycle_name + '11.png'))
        plt.show()
        plt.close()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:06:51 2023

@author: mirjam
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from Multif0Extraction import Multif0Extraction
from MultipitchExtraction import MultipitchExtraction
from BasicpitchExtraction import BasicpitchExtraction
from SymbolicFile import SymbolicFile
from Audio import Audio
from utils import transpose, get_pitch_class_name

class Composition:
    """A class representing a composition of a composition in a .wav format.
    Attributes:
        title(string): title of the composition
        symbolic_file(string): name of the symbolic file that represents this 
        composition
        audio_files(string): names of the audio files that represent this 
        composition
        composer(string): composer of the composition
        composition_year(int): estimated year of composition
        modal_cyle(string): title of the modal cycle the composition is part of
        mode(int): mode of the composition as indicated by the composer
     """
    def __init__(self, title, experiment):
        self.title = title
        self.experiment = experiment
        metadata = pd.read_csv('../data/raw/'+experiment+'/experiment_metadata.csv')
        self.metadata = metadata.loc[metadata['composition']==title]
        try: self.metadata = self.metadata.loc[self.metadata['exclude']!=1]
        except:
            pass
        try:
            self.metadata = self.metadata[(self.metadata['symbolic_is_audio'] != 'n') & (self.metadata['final_safe'] != 'n')]
        except:
            pass
        
        try:
            self.symbolic_filename = self.metadata.symbolic.iloc[0]
            self.symbolic_file = SymbolicFile(self.symbolic_filename, experiment)
            self.normalisation = False #if there is a symbolic file, then do not normalise the multif0 files to final C
        except IndexError:
            # Handle the case where the symbolic file is not found
            print(f"Symbolic file not found for {self.title} Skipping...")
            self.symbolic_file = None  # Set self.symbolic_file to some default value or None
            self.symbolic_filename = None
            self.normalisation = True #if there is no symbolic file, then normalise all files to C as final
            
        try:
            self.mp214c_filenames = self.metadata.multipitch_214c
        except:
            pass
        try:
            self.mp195f_filenames = self.metadata.multipitch_195f 
        except:
            pass
        
        try:
            self.basicpitch_filenames = self.metadata.basicpitch 
        except:
            pass
        self.multif0_filenames = self.metadata.mf0
        self.performers = self.metadata.performer
        # Check if there is more than one performer in 'multif0_file'

        try:
            # self.audio_file = self.metadata.loc[self.metadata['type']=='audio_file'].file_name # changeID 1
            self.audio_file = self.metadata.audio # changeID 1
        except:
            pass
        
        try:
            self.composer = self.metadata.composer[0]
        except:
            self.composer = ' '
        # self.composition_year = 
        # self.modal_cycle = 
        # self.mode = 
  
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"title:\n{self.title}\n\nexperiment:\n{self.experiment}\n\nsymbolic_file:\n{self.symbolic_filename}\n\nmultif0_files:\n{self.multif0_filenames}\n\nmultipitch 214c files:\n{self.mp214c_filenames}\n\nmultipitch 195f files:\n{self.mp195f_filenames}"
        return information
  
    def pitch_profile(self, normalisation=False, save=False):
        """Create a picture of the pitch class profiles of the mxl (if available) 
        and the extracted frequencies
        Returns:
            dataframe with symbolic, miditone, pitch_name, f{method}_{performer}
            """
        output_dir = os.path.join('..', 'results', 'output', self.experiment, 'pp')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pp_all = pd.DataFrame(columns=['miditone', 'pitch_name'])
        # add symbolic
        try:
            # combine the pitch profiles of symbolic and multif0 files
            pp_sym = self.symbolic_file.pitch_profile(normalisation=normalisation).rename(columns={'pp_sym_proportion': 'symbolic'})
            # get the final of the symbolic file as a reference for transposition
            symbolic_final = self.symbolic_file.final_midi()
            
        except:
            pp_sym = None
            symbolic_final = None
            print('Symbolic file not found')
        pp_all = pd.merge(pp_all, pp_sym, on=['miditone', 'pitch_name'], how='outer')
        
        
        try:
        
        # add multif0
            for multif0_filename, performer in zip(self.multif0_filenames, self.performers):
                multif0_file = Multif0Extraction(multif0_filename,self.experiment)
                audio_final = multif0_file.final_midi()
                try:
                    if normalisation == False:
                        transposition = symbolic_final - audio_final
                    else:
                        transposition = 0
                except:
                    transposition = 0
                pp_f0 = multif0_file.pitch_profile(normalisation=normalisation,transposition=transposition)
                pp_f0.rename(columns={'pp_audio_proportion': f'multif0_{performer}'}, inplace=True)
                # this line below seems superfluous
                # pp_all = pd.merge(pp_all, pp_freq, on=['miditone', 'pitch_name'], how='outer')
                
            
            try:
                pp_sym_f0 = pd.merge(pp_sym, pp_f0, on=['miditone', 'pitch_name'], how='outer')
            except:
                pp_sym_f0 = pp_f0
            pp_all = pp_sym_f0
            
        except:
            pass
        
        try:
        #add multipitch
            for model_name in ['195f', '214c']:
                for multipitch_filename, performer in zip(self.multipitch_filenames, self.performers):
                    mp = MultipitchExtraction(multipitch_filename, self.experiment, model_name = model_name)
                    audio_final = mp.final_midi()
                    
                    try:
                        if normalisation == False:
                            transposition = symbolic_final - audio_final
                        else:
                            transposition = 0
                    except:
                        transposition = 0
                    
                
                    pp_mp = mp.pitch_profile(normalisation=normalisation,transposition=transposition)
                    # print(pp_mp)
                    pp_mp.rename(columns={'pp_audio_proportion': f'multipitch_{performer}'}, inplace=True)
                    # pp_all = pd.merge(pp_all, pp_mp, on=['miditone', 'pitch_name'], how='outer')
                    try:
                        pp_sym_f0_mp = pd.merge(pp_all, pp_mp, on=['miditone', 'pitch_name'], how='outer')
                    except:
                        pp_sym_f0_mp = pp_all
                        
                    pp_all = pp_sym_f0_mp
                
        except:
            pass
        
        """ basicpitch """
        try:
         #add multipitch
             for basicpitch_filename, performer in zip(self.basicpitch_filenames, self.performers):
                 bp = BasicpitchExtraction(basicpitch_filename, self.experiment)
                 audio_final = bp.final_midi()
                 
                 try:
                     if normalisation == False:
                         transposition = symbolic_final - audio_final
                     else:
                         transposition = 0
                 except:
                     transposition = 0
                 
             
                 pp_bp = bp.pitch_profile(normalisation=normalisation,transposition=transposition)
                 # print(pp_bp)
                 pp_bp.rename(columns={'pp_audio_proportion': f'basicpitch_{performer}'}, inplace=True)
                 try:
                     pp_sym_f0_mp_bp = pd.merge(pp_all, pp_bp, on=['miditone', 'pitch_name'], how='outer')
                 except:
                     pp_sym_f0_mp_bp = pp_all
                     
                 pp_all = pp_sym_f0_mp_bp
        except:
            pass
        
        """ basicpitch """
        pp= pp_all

        # Set the pandas option to opt-in to the future behavior
        pd.set_option('future.no_silent_downcasting', True)
        
        # Fill NA values, sort by 'miditone', and infer object types without downcasting
        pp = pp.fillna(0).sort_values(by='miditone').infer_objects(copy=False)
        pp.reset_index(drop=True, inplace=True)
        if save == True:
            pp.to_csv(os.path.join(output_dir,self.title+'.csv'))
        return pp
    
    def show_pitch_profile(self, save = False):
        plt.style.use('uu')
        pp = self.pitch_profile(save = save)
        #create a bar plot
        ax = pp.drop(['miditone'], axis=1).plot.bar(x='pitch_name', title = self.title, figsize=(16, 5))
        output_path = os.path.join('../results/figures', self.experiment, 'pp')
        os.makedirs(output_path, exist_ok=True) 
        if save:
            fig = ax.figure
            
            fig.savefig(os.path.join(output_path, self.title + '.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
        return ax
    
    def pitch_class_profile(self, normalisation=False, save=False):
        """Create a picture of the pitch profiles of the mxl (if available) 
        and the extracted frequencies
        
        Args:
            chromatype: multif0, cens, vqt, cqt, 195f, 214c, basicpitch
            
            borrow_final_mf0: if the final of the chromatype is insufficient, take the final of the mf0 extraction
        Returns:
            dataframe with 'pitch_class', 'symbolic', 'pitch_class_name', f{method}_{performer}
        """
        output_dir = os.path.join('..', 'results', 'output', self.experiment, 'pcp')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ''' add symbolic '''
        try:
            pcp_sym = self.symbolic_file.pitch_class_profile(normalisation=normalisation).rename(columns={'pcp_sym_proportion': 'symbolic'})
            symbolic_final = self.symbolic_file.final_midi()
        except:
            pcp_sym = None
            symbolic_final = None          
        
        pcp_all = pd.DataFrame(columns=['pitch_class', 'pitch_class_name'])
        pitch_class_profile = pd.DataFrame()
        
        try:
            # add the multif0 information
            for multif0_filename, performer in zip(self.multif0_filenames, self.performers):
                multif0_file = Multif0Extraction(multif0_filename, self.experiment)
                
                audio_final = multif0_file.final_midi()
                #get the transposition
                try:
                    if normalisation == False:
                        transposition = symbolic_final - audio_final
                    else:
                        transposition = 0
                except:
                    transposition = 0
                # extract the croma according to the chroma type    
                pcp_freq = multif0_file.pitch_class_profile(normalisation=normalisation,transposition=transposition)
                pcp_freq.rename(columns={'pcp_audio_proportion': f'multif0_{performer}'}, inplace=True)
                pcp_all = pd.merge(pcp_all, pcp_freq, on=['pitch_class', 'pitch_class_name'], how='outer')
                try:
                    pitch_class_profile = pd.merge(pcp_sym, pcp_all, on=['pitch_class', 'pitch_class_name'], how='outer')
                except:
                    pitch_class_profile = pcp_all
                    print('multif0 extraction not available')
        except:
            pass
        
        try:
            for model_name in ['195f', '214c']:
                if model_name == '214c':
                    multipitch_filenames = self.mp214c_filenames
                if model_name == '195f':
                    multipitch_filenames = self.mp195f_filenames
                
                for multipitch_filename, performer, multif0_filename in zip(multipitch_filenames, self.performers, self.multif0_filenames):
                    mp = MultipitchExtraction(multipitch_filename, self.experiment, model_name = model_name)
                    
                    audio_final = mp.final_midi()
                                    
                    try:
                        if normalisation == False:
                            transposition = symbolic_final - audio_final
                        else:
                            transposition = 0
                    except:
                        transposition = 0
                    
                
                    pcp_mp = mp.pitch_class_profile(normalisation=normalisation,transposition=transposition)
                    pcp_mp.rename(columns={'pcp_audio_proportion': f'multipitch_{performer}'}, inplace=True)
                    pcp_all = pd.merge(pcp_all, pcp_mp, on=['pitch_class', 'pitch_class_name'], how='outer')
                    try:
                        pitch_class_profile = pd.merge(pcp_sym, pcp_all, on=['pitch_class', 'pitch_class_name'], how='outer')
                    except:
                        pitch_class_profile = pcp_all
                        print('multipitch extraction not available')
        except:
            pass            
        """basicpitch"""
        # add the multipitch information
        try:
            for basicpitch_filename, performer, multif0_filename in zip(self.basicpitch_filenames, self.performers, self.multif0_filenames):
                
                bp = BasicpitchExtraction(basicpitch_filename, self.experiment)
                
                
                audio_final = bp.final_midi()
                
                try:
                    if normalisation == False:
                        transposition = symbolic_final - audio_final
                    else:
                        transposition = 0
                except:
                    transposition = 0
                
            
                pcp_bp = bp.pitch_class_profile(normalisation=normalisation,transposition=transposition)
                pcp_bp.rename(columns={'pcp_audio_proportion': f'basicpitch_{performer}'}, inplace=True)
                pcp_all = pd.merge(pcp_all, pcp_bp, on=['pitch_class', 'pitch_class_name'], how='outer')
                try:
                    pitch_class_profile = pd.merge(pcp_sym, pcp_all, on=['pitch_class', 'pitch_class_name'], how='outer')
                except:
                    pitch_class_profile = pcp_all
                    print('multipitch extraction not available')
        except:
            pass
        """basicpitch"""
                    
        for chromatype in ['advanced', 'multif0', 'multipitch', 'basicpitch']:
            # print('chromatype is not in [advanced, multif0, multipitch], but it is '+chromatype)
            
            try:
                #get the final from the multif0 or the multipitch extraction
                try:
                    multif0_file = Multif0Extraction(multif0_filename, self.experiment)
                    audio_final = multif0_file.final_midi()
                except:
                    mp = MultipitchExtraction(self.multipitch_filenames.iloc[0], self.experiment, model_name = chromatype)
                    audio_final = mp.final_midi()
                
                try:
                    if normalisation == False:
                        transposition = symbolic_final - audio_final
                    else:
                        transposition = 0
                except:
                    transposition = 0
                
                # get the chromatype pitch class profile and transpose to symbolic key
                pcp_freq = Audio(self.title, self.experiment).chroma(chromatype)
                pcp_freq_transposed = transpose(pcp_freq, transposition)
                pcp_sym[chromatype] = pcp_freq_transposed
                pitch_class_profile = pcp_sym.reset_index()
                
            except:
                print('No audio file available')
                
        if save == True:
            pitch_class_profile.to_csv(os.path.join(output_dir,self.title+'.csv'))
        return pitch_class_profile
    
    def show_pitch_class_profile(self, save = False):
        plt.style.use('uu')
        pcp = self.pitch_class_profile(save=save)
        ax = pcp.drop(['pitch_class'], axis=1).plot.bar(x='pitch_class_name', title = self.title, figsize=(10, 5))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)#move legend out of plot area
        
        final = get_pitch_class_name(self.symbolic_file.final_midi())
        finalwidth = 2.20
        for patch, label in zip(ax.patches, pcp['pitch_class_name']):
            if label == final:
                patch.set_edgecolor('black')  # Set desired edge color
                patch.set_linewidth(finalwidth)  # Set desired line width
        
        
        output_path = os.path.join('../results/figures', self.experiment, 'pcp')
        os.makedirs(output_path, exist_ok=True)
        if save:
            fig = ax.figure
            fig.savefig(os.path.join(output_path, self.title + '.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            
        return ax
        
    def show_pitch_class_profile_paper(self, chromatype, save=False, substract=0):
        """
        Create a plot of the pitch class profiles of the mxl (if available) and the extracted frequencies.
        If there are multiple performances in the pitch class profile, the symbolic pitch class profile appears
        as wide bins in the background, and the multiple performances appear in the foreground.
        
        Arguments:
        chromatype (str): Type of chroma representation, options include 'multif0', 'cens', 'vqt', 'cqt'.
        save (bool): Whether to save the plot. Default is False.
        substract (int): Number of performers not to show in the plot. Default is 0.
        
        Returns:
        matplotlib.axes._subplots.AxesSubplot: The plot object.
        """
        plt.style.use('uu')
        if substract == 0:
            pcp = self.pitch_class_profile(chromatype)
        else:
            pcp = self.pitch_class_profile(chromatype).iloc[:, :-substract]
        plot_title = self.metadata.composer[0]+ ', '+ self.title.replace('_', ' ')
        final = get_pitch_class_name(self.symbolic_file.final_midi())
        finalwidth = 2.20 #linewidth of the box around the final
        
        # If there is a symbolic file and more than one performer
        if 'symbolic' in pcp.columns and len(pcp.columns)>5:
            # plot a grey wide bar for the symbolic pitch class profile
            ax = (pcp.drop(['pitch_class'], axis=1)
                  .plot.bar(x='pitch_class_name', y='symbolic',
                            color='lightslategrey', alpha=0.5, width=0.9,
                            edgecolor='lightslategrey', linewidth=1,
                            # title=plot_title,
                            figsize=(12, 5))
                  )
            
            # Add different edge color to the bar with the name in the variable 'final'
            for patch, label in zip(ax.patches, pcp['pitch_class_name']):
                if label == final:
                    patch.set_edgecolor('black')  # Set desired edge color
                    patch.set_linewidth(finalwidth)  # Set desired line width
            # Create a custom patch for the legend
            final_patch = mpatches.Patch(edgecolor='dimgrey', facecolor='white', linewidth=finalwidth, label='final')#TEMP
            
            # Plot other columns
            (pcp.drop([ 'symbolic','pitch_class'], axis=1)
             .plot.bar(x='pitch_class_name', ax=ax, width=0.7, legend=True, alpha=0.9)
             )
            
        else:
            # Plot the pitch class profiles of symbolic and audio next to each other
            ax = (pcp.drop(['pitch_class'], axis=1)
                  .plot.bar(x='pitch_class_name', 
                            # title=plot_title,
                            figsize=(10, 5))
                  )
        
        # add custom patch to legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(final_patch)
        labels.append('final')
        # Move legend out of plot area
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=20)
        
        # set label names
        ax.set_xlabel('Pitch class', fontsize = 20)
        ax.set_ylabel('Proportion', fontsize = 20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
   
        #save the plot
        output_path = os.path.join('../results/figures', self.experiment, 'pcp', chromatype)
        os.makedirs(output_path, exist_ok=True)
        
        if save:
            fig = ax.figure
            fig.savefig(os.path.join(output_path, self.title + '.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
        
        return ax
    


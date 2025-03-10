#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:06:51 2023

@author: mirjam
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

from Multif0Extraction import Multif0Extraction
from MultipitchExtraction import MultipitchExtraction
from BasicpitchExtraction import BasicpitchExtraction
from SymbolicFile import SymbolicFile
from Audio import Audio
from utils import transpose, get_pitch_class_name, get_pitch_name, get_pitch_class

class Recording:
    """A class representing a recording of a composition in a .wav format.
    Attributes:
        title(string): title of the composition
        symbolic_file(string): name of the symbolic file that represents this 
        composition
        audio_file(string): names of the audio file that represent this 
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
            self.mp214c_filename = self.metadata.multipitch_214c.iloc[0] 
        except:
            pass
        try:
            self.mp195f_filename = self.metadata.multipitch_195f.iloc[0]
            
        except:
            pass
        try:
            self.basicpitch_filename = self.metadata.basicpitch.iloc[0]
        except:
            pass
        
        self.multif0_filename = self.metadata.mf0.iloc[0]
        self.performer = self.metadata.performer.iloc[0]
        try:
            self.audio_file = self.metadata.audio.iloc[0]
        except:
            pass
        try:
            self.audio_final= int(self.metadata.audio_final.iloc[0])
        except:
            pass
        try:
            self.composer = self.metadata.composer.iloc[0]
        except:
            self.composer = ' '
        self.basename = os.path.splitext(self.multif0_filename)[0]
    
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"title:\n{self.title}\n\nexperiment:\n{self.experiment}\n\nsymbolic_file:\n{self.symbolic_filename}\n\nfrequency_files:\n{self.multif0_filename}\n\nmultipitch 214c files:\n{self.mp214c_filename}\n\nmultipitch 195f files:\n{self.mp195f_filename}"
        return information
  
    def pitch_profile(self, normalisation=False, save=False):
        """Create the pitch profiles for all chromatypes available
        ['symbolic', 'multif0', '195f', '214c', 'basicpitch', 'cqt']
        Arguments:
            bool normalisation: if True, the pitch will be normalised to a final on C
            bool save: save the pitch profile as .csv
            
        Returns:
            dataframe ['miditone', 'pitch_name' 'symbolic', 'multif0', '195f', '214c', 'basicpitch']
            """
        if normalisation == False:
            output_dir = os.path.join('..', 'results', 'output', self.experiment, 'pp')
        if normalisation == True:
            output_dir = os.path.join('..', 'results', 'output', self.experiment, 'pp_normalised')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        pp_all = pd.DataFrame(columns=['miditone', 'pitch_name'])
        
        ''' add symbolic '''
        try:
            # combine the pitch profiles of symbolic and multif0 files
            pp_sym = self.symbolic_file.pitch_profile(normalisation=normalisation).rename(columns={'pp_sym_proportion': 'symbolic'})
            # get the final of the symbolic file as a reference for transposition
            symbolic_final = self.symbolic_file.final_midi()
        except:
            pp_sym = None
            symbolic_final = None
            print('Symbolic file not found')
        try:
            pp_all = pd.merge(pp_all, pp_sym, on=['miditone', 'pitch_name'], how='outer')
        except:
            pp_all =pp_all
        ''' add multif0 '''
        try:
            frequency_file = Multif0Extraction(self.multif0_filename,self.experiment)
            audio_final = frequency_file.final_midi()
            try:
                if normalisation == False:
                    transposition = symbolic_final - audio_final
                else:
                    transposition = 0
            except:
                transposition = 0
            pp_f0 = frequency_file.pitch_profile(normalisation=normalisation,transposition=transposition)
            pp_f0.rename(columns={'pp_audio_proportion': 'multif0'}, inplace=True)
                 
            try:
                pp_sym_f0 = pd.merge(pp_sym, pp_f0, on=['miditone', 'pitch_name'], how='outer')
            except:
                pp_sym_f0 = pp_f0
            pp_all = pp_sym_f0
        except:
            pass
        ''' add 195f and 214c '''
        for chromatype in [ '195f', '214c']:
            try:
                if chromatype == '214c':
                    mp = MultipitchExtraction(self.mp214c_filename, self.experiment, model_name = chromatype)
                if chromatype == '195f':
                    mp = MultipitchExtraction(self.mp195f_filename, self.experiment, model_name = chromatype)
                audio_final = mp.final_midi()
                
                try:
                    if normalisation == False:
                        transposition = symbolic_final - audio_final
                    else:
                        transposition = 0
                except:
                    transposition = 0
                
            
                pp_mp = mp.pitch_profile(normalisation=normalisation,transposition=transposition)
                pp_mp.rename(columns={'pp_audio_proportion': f'{chromatype}'}, inplace=True)
                try:
                    pp_sym_f0_mp = pd.merge(pp_all, pp_mp, on=['miditone', 'pitch_name'], how='outer')
                except:
                    pp_sym_f0_mp = pp_all
                pp_all = pp_sym_f0_mp
            except:
                pass
                
        """ add basicpitch """
        try:
            bp = BasicpitchExtraction(self.basicpitch_filename, self.experiment)
            audio_final = bp.final_midi()
            
            try:
                if normalisation == False:
                    transposition = symbolic_final - audio_final
                else:
                    transposition = 0
            except:
                transposition = 0
            pp_bp = bp.pitch_profile(normalisation=normalisation,transposition=transposition)
            pp_bp.rename(columns={'pp_audio_proportion': 'basicpitch'}, inplace=True)
            try:
                pp_sym_f0_mp_bp = pd.merge(pp_all, pp_bp, on=['miditone', 'pitch_name'], how='outer')
            except:
                pp_sym_f0_mp_bp = pp_all
            pp_all = pp_sym_f0_mp_bp
        except:
            pass
       
        pp= pp_all
        pd.set_option('future.no_silent_downcasting', True)
        pp = pp.fillna(0).sort_values(by='miditone').infer_objects(copy=False)
        pp.reset_index(drop=True, inplace=True)
        if save == True:
            pp.to_csv(os.path.join(output_dir,self.basename+'.csv'))
        
        
        return pp
    
    def show_pitch_profile(self, save = False, normalisation=False):
        """Create the pitch profiles figures for all chromatypes available
        ['symbolic', 'multif0', '195f', '214c', 'basicpitch']
        Arguments:
            bool normalisation: if True, the pitch will be normalised to a final on C
            bool save: save the pitch profile as .csv
            
        Returns:
            ax object
            """
        plt.style.use('uu')
        pp = self.pitch_profile(save=save, normalisation=normalisation)
        
        #create a bar plot
        ax = pp.drop(['miditone'], axis=1).plot.bar(x='pitch_name', title = self.title, figsize=(16, 5))
        
        try:
            final = get_pitch_name(self.symbolic_file.final_midi())
        except:
            frequency_file = Multif0Extraction(self.multif0_filename,self.experiment)
            final = get_pitch_name(frequency_file.final_midi())
        if normalisation==False:    
            finalwidth = 2.20
            for patch, label in zip(ax.patches, pp['pitch_name']):
                if label == final:
                    patch.set_edgecolor('black')  # Set desired edge color
                    patch.set_linewidth(finalwidth)  # Set desired line width
        
        output_path = os.path.join('../results/figures', self.experiment, 'pp')
        os.makedirs(output_path, exist_ok=True) 
        if save:
            fig = ax.figure
            
            fig.savefig(os.path.join(output_path, self.basename + '.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
        return ax
    
    def create_pitch_class_profiles(self, normalisation=False, save=False): 
        #TODO To reduce code, create the pcp from the pp
        """Create the pitch profiles for all chromatypes available
        ['symbolic', 'multif0', '195f', '214c', 'basicpitch', 'cqt', 'hpcp']
        Args:
            bool save: if True, save the dataframe
            Bool bormalisation: if True, all input will be transposed to the final C
        Returns:
            dataframe ['pitch_class', 'pitch_class_name', 'symbolic', 'multif0', '195f', '214c', 'basicpitch', 'cqt', 'vqt','cens']
            """
        if normalisation == False:
            output_dir = os.path.join('..', 'results', 'output', self.experiment, 'pcp')
        if normalisation == True:
            output_dir = os.path.join('..', 'results', 'output', self.experiment, 'pcp_normalised')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pcp_all = pd.DataFrame(columns=['pitch_class', 'pitch_class_name'])
        pitch_class_profile = pd.DataFrame()
        ''' add symbolic '''
        try:
            pcp_sym = self.symbolic_file.pitch_class_profile(normalisation=normalisation).rename(columns={'pcp_sym_proportion': 'symbolic'})
            symbolic_final = self.symbolic_file.final_midi()
        except:
            pcp_sym = None
            symbolic_final = None          
        
        ''' add multif0 '''                
        try:
            
            frequency_file = Multif0Extraction(self.multif0_filename, self.experiment)
            mf0_final = frequency_file.final_midi()
            #get the transposition
            try:
                if normalisation == False:
                    transposition = symbolic_final - mf0_final
                else:
                    transposition = 0
            except:
                transposition = 0
            # extract the croma according to the chroma type    
            pcp_freq = frequency_file.pitch_class_profile(normalisation=normalisation,transposition=transposition)
            pcp_freq.rename(columns={'pcp_audio_proportion': 'multif0'}, inplace=True)
            pcp_all = pd.merge(pcp_all, pcp_freq, on=['pitch_class', 'pitch_class_name'], how='outer')
            try:
                pitch_class_profile = pd.merge(pcp_sym, pcp_all, on=['pitch_class', 'pitch_class_name'], how='outer')
            except:
                pitch_class_profile = pcp_all
                print('multif0 extraction not available')
        except:
            pass
        
        ''' add 195f and 214c '''
        for chromatype in ['214c','195f']:
            try:
                if chromatype == '214c':
                    mp = MultipitchExtraction(self.mp214c_filename, self.experiment, model_name = chromatype)
                    
                if chromatype == '195f':
                    mp = MultipitchExtraction(self.mp195f_filename, self.experiment, model_name = chromatype)
                
    
                mp_final = mp.final_midi()
                                
                try:
                    if normalisation == False:
                        transposition = symbolic_final - mp_final
                    else:
                        transposition = 0
                except:
                    transposition = 0
                
            
                pcp_mp = mp.pitch_class_profile(normalisation=normalisation,transposition=transposition)
                pcp_mp.rename(columns={'pcp_audio_proportion': f'{chromatype}'}, inplace=True)
                pcp_all = pd.merge(pcp_all, pcp_mp, on=['pitch_class', 'pitch_class_name'], how='outer')
                try:
                    pitch_class_profile = pd.merge(pcp_sym, pcp_all, on=['pitch_class', 'pitch_class_name'], how='outer')
                except:
                    pitch_class_profile = pcp_all
                    print('multipitch extraction not available')
            except:
                pass
                    
        """add basicpitch"""
        try:
                
            bp = BasicpitchExtraction(self.basicpitch_filename, self.experiment)
            
            
            bp_final = bp.final_midi()
            
            try:
                if normalisation == False:
                    transposition = symbolic_final - bp_final
                else:
                    transposition = 0
            except:
                transposition = 0
            
        
            pcp_bp = bp.pitch_class_profile(normalisation=normalisation,transposition=transposition)
            pcp_bp.rename(columns={'pcp_audio_proportion': 'basicpitch'}, inplace=True)
            pcp_all = pd.merge(pcp_all, pcp_bp, on=['pitch_class', 'pitch_class_name'], how='outer')
            try:
                pitch_class_profile = pd.merge(pcp_sym, pcp_all, on=['pitch_class', 'pitch_class_name'], how='outer')
            except:
                pitch_class_profile = pcp_all
                print('multipitch extraction not available')
        except:
            pass
        
        """add cqt, vqt, cens"""
        # for chromatype in ['hpcp', 'cqt', 'vqt', 'cens']: #change ID 10
        try:
            audio_object = Audio(self.title, self.experiment)
        except:
            print(f'no audio file {self.audio_file} found')
        for chromatype in ['hpcp', 'cqt']: #change ID 10
            try:
                #get the final from the multif0 or the multipitch extraction
                try:
                    audio_final = self.audio_final #if an audio_final is provided in the metadata
                except:
                    try:
                        frequency_file = Multif0Extraction(self.multif0_filename, self.experiment)
                        audio_final = frequency_file.final_midi() #else take the final from the mf0 method
                    except:
                        mp = MultipitchExtraction(self.mp214c_filename, self.experiment, model_name = '214c')
                        audio_final = mp.final_midi() #else take the final from the multipitch method
                try:
                    if normalisation == False:
                        transposition = symbolic_final - audio_final
                        
                    else:
                        transposition = 0
                except:
                    transposition = 0
                
                # get the chromatype pitch class profile and transpose to symbolic key
                
                pcp_freq = audio_object.chroma(chromatype)
                if normalisation == True: 
                    transposition = 0-(get_pitch_class(audio_final))
                    
                #insert normalisation here, which is basically the rolling of the pitch class number of the audio_final
                pcp_freq_transposed = transpose(pcp_freq, transposition)
                #put this pcp_freq_transposed as a new column in pcp_all, the column name should be chromatype
                pitch_class_profile[chromatype]=pcp_freq_transposed
            except:
                print(f'No chroma extraction of audio file of {self.title} possible')
                
        pitch_class_profile = pitch_class_profile.fillna(0).sort_values(by='pitch_class')
        if save == True:
            pitch_class_profile.to_csv(os.path.join(output_dir,self.basename+'.csv'))
        return pitch_class_profile
    
    
    def show_pitch_class_profile(self, save = False, normalisation = False):
        plt.style.use('uu')
        pcp = self.create_pitch_class_profiles(save=save, normalisation =normalisation)
        ax = pcp.drop(['pitch_class'], axis=1).plot.bar(x='pitch_class_name', title = self.title, figsize=(10, 5))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)#move legend out of plot area
        
        try:
            final = get_pitch_class_name(self.symbolic_file.final_midi())
        except:
            frequency_file = Multif0Extraction(self.multif0_filename,self.experiment)
            final = get_pitch_class_name(frequency_file.final_midi())
        if normalisation == False:
            finalwidth = 2.20
            for patch, label in zip(ax.patches, pcp['pitch_class_name']):
                if label == final:
                    patch.set_edgecolor('black')  # Set desired edge color
                    patch.set_linewidth(finalwidth)  # Set desired line width
        
        
        output_path = os.path.join('../results/figures', self.experiment, 'pcp')
        os.makedirs(output_path, exist_ok=True)
        if save:
            fig = ax.figure
            fig.savefig(os.path.join(output_path, self.basename + '.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            
        return ax
   

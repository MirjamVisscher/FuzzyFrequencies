#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:55:14 2023
use env py310

@author: mirjam
"""

import os
import pandas as pd
import csv
import gc
import matplotlib.pyplot as plt

from Multif0Extraction import Multif0Extraction
from Composition import Composition
from Recording import Recording
from MultipitchExtraction import MultipitchExtraction
from BasicpitchExtraction import BasicpitchExtraction
from Cycle import Cycle

from utils import repair_frequency_file, euclidean_distance, squared_distance, manhattan_distance

class Experiment:
    """A class representing a experiment in the Pitch Profiles study
    Attributes:
        experimentname(string): name of the experiment
        experiment_type(string): type of the experiment
        symbolic_files(list): list of the symbolic files in this experiment
        multif0_files(list): list of multif0 files in this experiment
    """
    # TODO in its current state, the Audio class takes the first audio filename 
    # of a composition in the metadata. It is impossible to compare multiple recordings
    # This does not hinder the types of experiment I want to run, but it is 
    # an architecural flaw that needs to be solved
    
    def __init__(self, experiment):
        self.experiment = experiment
        self.metadata = pd.read_csv('../data/raw/'+experiment+'/experiment_metadata.csv')
        try:
            self.metadata = self.metadata.sort_values(by= 'mode')
        except:
            pass
        try: 
            self.metadata = self.metadata.loc[self.metadata['exclude']!=1] 

        except:
            pass
        try:
            self.metadata = self.metadata[(self.metadata['symbolic_is_audio'] != 'n') & (self.metadata['final_safe'] != 'n')]
        except:
            pass

        try:
            self.symbolic_files = self.metadata['symbolic'] 
        except:
            self.symbolic_files = None
        self.multif0_files = self.metadata['mf0']
        try:
            self.mp195f_files = self.metadata['multipitch_195f']
        except:
            pass
        try:
            self.mp214c_files = self.metadata['multipitch_214c'] 
        except:
            pass
        
        try:
            self.basicpitch_files = self.metadata['bascpitch'] 
        except:
            pass
        try:
            self.metrics = self.metadata[['composition','voices','instrumentation_category','year_recording']]
        except:
            pass
        
        try:
            self.modes = self.metadata['mode']
        except:
            pass
        
        self.compositions = self.metadata['composition'].unique()
        
        n_compositions = len(self.compositions)
        # Check for multif0 files alignment
        if n_compositions > 1 and len(self.multif0_files) != n_compositions:
            print('Please create an experiment_datametadata.csv file that contains one composition with multiple recordings '
              'or that contains multiple compositions with one composition per recording.')
            return 
        
        # Set experiment_type based on number of compositions
        if n_compositions > 1:
            self.experiment_type = 'oneOnOne'
            aim = 'analysis of the alignment of symbolic encodings and pitch extractions and also for analysis of modal cycles.'
        elif n_compositions >= 1:
            self.experiment_type = 'performance_effect'
            aim = 'analysis of the effect of performance on the pitch extraction'
        print('the type of this experiment is '+self.experiment_type+'\nThis means that the the data is fit for '+aim)
        
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"experiment :\n {self.experiment}\n\nsymbolic_files :\n {self.symbolic_files}\n\ncompositions:\n {self.compositions}\n\nfrequency files:{self.multif0_files}"
        return information
    
    def repair_multif0_files(self, print = True):
        """Repairs all multif0 files of your project in one go.
        Returns:
            nothing, it replaces your malformed csv files with readable files
        """
        directory = '../data/raw/'+ self.experiment+'/multif0'
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    if print==True:
                        print(file_path)
                    repair_frequency_file(file_path)
        return 'done'
    
    def sonify_multif0(self, cents):
        """create wav files of the mf0 extractions for the complete experiment 
        and save them to 
        '..', 'results', 'output', self.experiment, 'sonification', 'Multif0_MIDI' for cents = 100
        and '..', 'results', 'output', self.experiment, 'sonification', 'Multif0_freq' for cents = 20
        """
        for multif0_file in self.multif0_files:
            mf0= Multif0Extraction(multif0_file, self.experiment)
            mf0.sonify(cents)
    
    def make_midi_mf0s(self):
        """create mf0 extractions with midi values instead of frequencies
        for the complete experiment and save them
        """
        for multif0_file in self.multif0_files:
            mf0= Multif0Extraction(multif0_file, self.experiment)
            mf0.make_midi_mf0()
    
    def repair_basicpitch_files(self):
        """Turn the output of basicpitch into a csv that is readable by pandas by 
        padding the lines with commas.
        Returns:
            nothing, it replaces your malformed csv file with a readable file
        """
        # TODO the first row is now converted into column headers. Add new headers instead.
        # Read the input CSV file
        idir = os.path.join('..', 'data', 'raw', self.experiment, 'basicpitch')
        files = os.listdir(idir)
        
        for file in files:
            path = os.path.join(idir, file)
            with open(path, 'r') as input_file:
                reader = csv.reader(input_file)
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
            print(f'{path} repaired.')
        return 'done'
                      
    def check_zero_voices(self):
        """Checks all multif0 files in the experiment whether these contain 
        zero voice columns
        Returns:
            list of multif0 files with 0 voice columns
        """
        voices_dict = {}
        for mf0 in self.multif0_files:
            voices_dict[mf0] = Multif0Extraction(mf0, self.experiment).voices
            zeroes = {key: value for key, value in voices_dict.items() if value == 0}
        return zeroes
    
    def piano_rolls(self, chromatype, batch_size=50, replace = False):
        """Plots piano rolls for each multif0 file in the experiment folders
        possible chromatypes are: 'multif0', 'basicpitch', '195f', '214c'
        Saves the plots to '../results/figures/'+self.experiment+'/piano_roll/'+chromatype
        Returns: nothing
        """
        plt.ioff()
        output_path = os.path.join('../results/figures', self.experiment, 'piano_roll',chromatype)
        os.makedirs(output_path, exist_ok=True)
        
        if chromatype == 'multif0':
            
            multif0_files = os.listdir('../data/raw/'+self.experiment+'/multif0/')
            
            if len(multif0_files)>0:
                batch_size = batch_size
                number_f0_files = len(multif0_files)
                num_batches = (number_f0_files + batch_size - 1) // batch_size  # Calculate total number of batches
                
                
                for batch_index in range(num_batches):
                    start_index = batch_index * batch_size
                    end_index = min((batch_index + 1) * batch_size, number_f0_files)
                    batch_files = multif0_files[start_index:end_index]
                
                    print(f"Processing batch {batch_index + 1}/{num_batches}:")
                    for file in batch_files:
                        title = file.replace('.csv', '')
                        png_file = os.path.join(output_path, title + '.png')
                        if replace == False:
                            if not os.path.exists(png_file):
                            
                                try:
                                    Multif0Extraction(file, self.experiment).piano_roll()
                                except Exception as e:
                                    print(f"Failed to create piano roll of {file}: {e}")
                        else:
                            try:
                                Multif0Extraction(file, self.experiment).piano_roll()
                            except Exception as e:
                                print(f"Failed to create piano roll of {file}: {e}")
                        
                    if batch_index < num_batches - 1:
                        choice = input("Continue to the next batch? (y/n): ").strip().lower()
                        if choice != 'y':
                            print("Stopping processing.")
                            break
                    else:
                        print("All batches processed for multiple f0.")
        
        if chromatype == '214c' or chromatype == '195f':
            if chromatype == '195f':
                output_pathmp = output_path
                mp195f_files = os.listdir('../data/raw/'+self.experiment+'/195f/')
                nfiles = len(mp195f_files)
            if chromatype == '214c':
                output_pathmp = output_path
                mp214c_files = os.listdir('../data/raw/'+self.experiment+'/214c/')
                nfiles = len(mp214c_files)
            if nfiles > 0:
                batch_size = batch_size
                
                num_batches = (nfiles + batch_size - 1) // batch_size  # Calculate total number of batches
                
                i=0
                for batch_index in range(num_batches):
                    print(batch_index)
                    start_index = batch_index * batch_size
                    end_index = min((batch_index + 1) * batch_size, nfiles)
                    if chromatype == '195f':
                        batch_files = mp195f_files[start_index:end_index]
                    if chromatype == '214c':
                        batch_files = mp214c_files[start_index:end_index]
                    
                    
                    
                    print(f"Processing batch {batch_index + 1}/{num_batches}:")
                    for file in batch_files:
                        print('processing multpitch file of type '+ chromatype+' number '+ str(i))
                        i=i+1
                        title = file.replace('.csv', '')
                        png_file = os.path.join(output_pathmp, title + '.png')
                        if replace == False:
                            if not os.path.exists(png_file):
                                print(png_file + ' does not exist yet')
                            
                                try:
                                    MultipitchExtraction(file, self.experiment, chromatype).piano_roll(save = True)
                                except Exception as e:
                                    print(f"Failed to create piano roll of {file}: {e}")
                        else:
                            try:
                                MultipitchExtraction(file, self.experiment, chromatype).piano_roll(save = True)
                            except Exception as e:
                                print(f"Failed to create piano roll of {file}: {e}")
                        
                    if batch_index < num_batches - 1:
                        choice = input("Continue to the next batch? (yes/no): ").strip().lower()
                        if choice != 'y':
                            print("Stopping processing.")
                            break
                    else:
                        print("All batches processed for multipitch.")
                    # Manually trigger garbage collection
                    gc.collect()
                    
                    # Close all open figures
                    plt.close('all')
                    
                    
            plt.ion()
        if chromatype == 'basicpitch':
            basicpitch_files = os.listdir('../data/raw/'+self.experiment+'/basicpitch/')[::-1]


            if len(basicpitch_files)>0:
                batch_size = batch_size
                number_bp_files = len(basicpitch_files)
                num_batches = (number_bp_files + batch_size - 1) // batch_size  # Calculate total number of batches
                
                
                for batch_index in range(num_batches):
                    start_index = batch_index * batch_size
                    end_index = min((batch_index + 1) * batch_size, number_bp_files)
                    batch_files = basicpitch_files[start_index:end_index]
                
                    print(f"Processing batch {batch_index + 1}/{num_batches}:")
                    for file in batch_files:
                        title = file.replace('.csv', '')
                        png_file = os.path.join(output_path, title + '.png')
                        print(png_file)
                        if replace == False:
                            if not os.path.exists(png_file):
                                print(png_file + ' does not exist yet')
                                try:
                                    BasicpitchExtraction(file, self.experiment).piano_roll(save=True)
                                except Exception as e:
                                    print(f"Failed to create piano roll of {file}: {e}")
                        else:
                            try:
                                BasicpitchExtraction(file, self.experiment).piano_roll(save=True)
                            except Exception as e:
                                print(f"Failed to create piano roll of {file}: {e}")
                        
                    if batch_index < num_batches - 1:
                        choice = input("Continue to the next batch? (yes/no): ").strip().lower()
                        if choice != 'y':
                            print("Stopping processing.")
                            break
                    else:
                        print("All batches processed for basicpitch.")
                    gc.collect()
    
    def show_pcp(self):
        """plot all pitch class profiles in the experiment and save them as .png
        saves all pitch class profiles as .csv
        Returns:
            None
        """
        
        for composition in self.compositions:
            print(composition)
            if self.experiment_type == 'oneOnOne':
                composition = Recording(composition,self.experiment)
            if self.experiment_type == 'performance_effect':    
                print('this has to be solved yet...')
                composition = Composition(composition,self.experiment)
            try:
                composition.show_pitch_class_profile(save=True, normalisation=False)
            except:
                pass
              
    def show_pp(self):
        """plot all pitch profiles in the experiment and save them as .png
        saves all pitch profiles as .csv
        Returns:
            None
        """
        for composition in self.compositions:
            print(composition)
            if self.experiment_type == 'oneOnOne':
                composition = Recording(composition,self.experiment)
            if self.experiment_type == 'performance_effect':
                print('this has to be solved yet...')
                composition = Composition(composition,self.experiment)
                
            try:
                composition.show_pitch_profile(save=True, normalisation=False)
            except:
                pass
      
        
    def plot_cycles(self):
        """ For each cycle:
            Plot combined pitch class profiles for all unique modes in a single 
            combined plot"""
            
        try:
            cycles = self.metadata[['composer', 'cycle']].drop_duplicates()
            for index, row in cycles.iterrows():
                print(row['composer'])
                print(row['cycle'])
                Cycle(self.experiment, row.composer, row.cycle).plot_combined_pitch_class_profiles_all_modes()
                
        except:
            print('There are no cycles specified in this experiment.')


    def save_profiles(self, profile_type, normalisation):
        """
        This function creates and saves the pp's and pcp's for all chromatypes available
        Args:
            str profile_type: choose between 'pcp' and 'pp'
            bool normalisation: 
        Returns:
            None
            """
        
        if profile_type== 'pcp':    
            if self.experiment_type == 'oneOnOne':
                for composition in self.compositions:
                    print(composition)
                    Recording(composition, self.experiment).create_pitch_class_profiles(save=True, normalisation=normalisation)
            elif self.experiment_type == 'performance_effect':
                print('please implement the Composition.pitch_class_profiles() method similar to the Recording.pitch_class_profiles method.')
                for composition in self.compositions:
                    print(composition)
                    Composition(composition, self.experiment)._pitch_class_profile(save=True, normalisation=normalisation)
        
        if profile_type== 'pp':    
            if self.experiment_type == 'oneOnOne':
                for composition in self.compositions:
                    print(composition)
                    Recording(composition, self.experiment).pitch_profile(save=True, normalisation=normalisation)
            elif self.experiment_type == 'performance_effect':
                print('please implement the Composition.pitch_class_profiles() method similar to the Recording.pitch_class_profiles method.')
                for composition in self.compositions:
                    print(composition)
                    Composition(composition, self.experiment).pitch_profile(save=True, normalisation=normalisation)
        
    
    def distances(self, profile_type, distance_type):
        """
        Create a dataframe with distances of an extraction method to symbolic profiles
        for each composition (experiment_type = 'oneOnOne') or recording (experiment_type = 'performance_effect')
        
        Args:
            profile_type: choose between 'pcp' and 'pp'
            distance_type: choose between 'euclidean', ´squared', 'manhattan'
        Returns:
            dataframe ['composition','composer','distance']
        """
        
        if self.experiment_type == 'oneOnOne':
            idir = os.path.join('..', 'results', 'output', self.experiment, profile_type)
            odir = os.path.join('..', 'results', 'output', self.experiment, 'distances')
            if not os.path.exists(odir):
                os.makedirs(odir)
            # chromatypes = ['multif0', '195f', '214c', 'basicpitch', 'vqt', 'cqt', 'cens'] #changeID 10
            chromatypes = ['multif0', '195f', '214c', 'basicpitch', 'cqt', 'hpcp'] #changeID 10
            rows = []
            for composition in self.compositions:
                profile_path = os.path.join(idir, composition + '.csv') #make filename findable
                
                if os.path.exists(profile_path):
                    profile = pd.read_csv(profile_path)
                    distancerow = {'composition': composition}
                    
                    for chromatype in chromatypes:
                        if chromatype in profile.columns:
                            if distance_type == 'euclidean':
                                distance = euclidean_distance(profile['symbolic'], profile[chromatype])
                            if distance_type == 'squared':
                                distance = squared_distance(profile['symbolic'], profile[chromatype])
                            if distance_type == 'manhattan':
                                distance = manhattan_distance(profile['symbolic'], profile[chromatype])
                        else:
                            distance = None
                        distancerow[chromatype] = distance
                    
                    rows.append(distancerow)
                else:
                    print(f'no {profile_type} file available for {composition}')
            
            distances = pd.DataFrame(rows)
            # Remove columns where all values are None
            distances = distances.dropna(axis=1, how='all')
            distances.to_csv(os.path.join(odir, distance_type+' distances_' + profile_type + '.csv'))
        
        if self.experiment_type == 'performance_effect':
            # TODO
            distances = None
            print('this needs to be developed. the code is similar to the oneOnOne approach, but loops through the recordings')
        
        return distances
  

    
    def create_finals(self):
        '''create a list of finals and compare them to the ground truth
    
        Returns:
            dataframe with finals for each extraction method
            dataframe with analysis of the finals
            '''
            
        if self.experiment_type == 'performance_effect':
            return 'this method is fit only for experiments of the type oneOnOne'
        
        data = []
    
        for composition in self.compositions:
            print(composition)

            row = {'composition': composition}
            rec = Recording(composition, self.experiment)
            
            row['final_truth'] = None
            row['final_symbolic'] = None
            row['final_mf0'] = None
            row['final_mp_195f'] = None
            row['final_mp_214c'] = None
            row['final_bp'] = None
            
            try:
                row['final_truth'] = rec.audio_final
            except:
                pass
                      
            try:
                row['final_symbolic'] = rec.symbolic_file.final_midi()
            except:
                pass
        
            try:
                row['final_mf0'] = Multif0Extraction(rec.frequency_filename, self.experiment).final_midi()
            except:
                print('mf0 failed')
        
            try:
                row['final_bp'] = BasicpitchExtraction(rec.basicpitch_filename, self.experiment).final_midi()
            except:
                print(composition + ': bp failed')
            
            try:
                row['final_mp_195f'] = MultipitchExtraction(rec.mp195f_filename, self.experiment, model_name='195f').final_midi()

            except:
                print(composition+ ': mp failed')
                
            try:
                row['final_mp_214c'] = MultipitchExtraction(rec.mp214c_filename, self.experiment, model_name='214c').final_midi()
            except:
                print(composition+ ': mp failed')
        
            data.append(row)
            print(row)

        def categorize_value(value):
            if value in [0, 1]:
                return 'pp'
            elif value % 12 in [0, 1]:
                return 'pcp'
            else:
                return 'false'
        def calculate_percentages(column):
            total = len(column)
            pp_pct = (column == 'pp').sum() / total
            pcp_pct = (column == 'pcp').sum() / total
            pp_pcp_pct = ((column == 'pp') | (column == 'pcp')).sum() / total
            false_pct = (column == 'false').sum() / total
            return [pp_pct, pcp_pct, pp_pcp_pct, false_pct]
    
        finals = pd.DataFrame(data, columns=['composition', 'final_truth', 'final_symbolic', 'final_mf0', 'final_mp_195f', 'final_mp_214c', 'final_bp'])
        
        finals['delta_mf0'] = abs(finals['final_mf0']-finals['final_truth'])
        finals['delta_mp_195f'] = abs(finals['final_mp_195f']-finals['final_truth'])
        finals['delta_mp_214c'] = abs(finals['final_mp_214c']-finals['final_truth'])
        finals['delta_bp'] = abs(finals['final_bp']-finals['final_truth'])
        finals['category_mf0'] = finals['delta_mf0'].apply(categorize_value)
        finals['category_mp195f'] = finals['delta_mp_195f'].apply(categorize_value)
        finals['category_mp_214c'] = finals['delta_mp_214c'].apply(categorize_value)
        finals['category_bp'] = finals['delta_bp'].apply(categorize_value) 
        
        odir = os.path.join('..', 'results', 'output', self.experiment, 'finals')
        if not os.path.exists(odir):
            os.makedirs(odir)
        finals.to_csv(os.path.join(odir, 'finals.csv'))

        category_percentages = pd.DataFrame(
            index=['pp', 'pcp', 'pp+pcp', 'false'],
            columns=['mf0', 'mp_195f', 'mp_214c', 'bp']
        )

        category_percentages['mf0'] = calculate_percentages(finals['category_mf0'])
        category_percentages['mp_195f'] = calculate_percentages(finals['category_mp195f'])
        category_percentages['mp_214c'] = calculate_percentages(finals['category_mp_214c'])
        category_percentages['bp'] = calculate_percentages(finals['category_bp'])
        
        category_percentages.to_csv(os.path.join(odir, 'finals_tested.csv'))
            
        return finals, category_percentages
        


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:36:39 2024

@author: mirjam
"""
import argparse
from Experiment import Experiment
from MultipitchExtraction import MultipitchExtraction
from BasicpitchExtraction import BasicpitchExtraction
from Recording import Recording
from Cycle import Cycle
from utils_stats import perform_dunn_test, visualise_distances, get_statistics, kruskal_wallis_test
from utils_cluster import create_clusters_with_subplots, create_clusters_with_evaluation
from utils_performance_effect import create_metrics, plot_exploration, plot_metrics, regression_analysis
from utils_paper import pitch_profile_fuzzypaper, pitch_class_profile_fuzzypaper, piano_roll_bp, piano_roll_mp

experiment = 'CANTO_JRP'
experiment = 'Palestrina'
canto_jrp =Experiment(experiment)

def initial_repair_and_quality_assurance(canto_jrp):
    """this function adjusts the basicpitch and multif0 extraction files
    and checks whether there are any multif0 extractions with 0 voices (these
    files are erroneous and should be extracted again or excluded from the 
    experiment)
    """
    canto_jrp.repair_basicpitch_files()
    canto_jrp.repair_multif0_files()
    zeroes = canto_jrp.check_zero_voices() # make sure that the mf0 extractions have values
    lengths, length_check = canto_jrp.compare_length_multif0_audio # make sure that the length of the mf0 files approximates the length of the recordings
    return zeroes, length_check

def visualisation(canto_jrp):
    """
    These steps are optional: useful for visual inspection, but not necessary for 
    the analysis.
    """
    canto_jrp.piano_rolls(chromatype='multif0', batch_size=700, replace = False) 
    canto_jrp.piano_rolls(chromatype='basicpitch', batch_size=700, replace = False) 
    canto_jrp.piano_rolls(chromatype='195f', batch_size=700, replace = False) 
    canto_jrp.piano_rolls(chromatype='214c', batch_size=700, replace = False) 
    canto_jrp.show_pcp() 
    canto_jrp.show_pp()

def mf0_to_midi(canto_jrp):
    """ 
    This function turns the frequencies in the multif0 extractions info midi tones
    and stores them in ../data/processed/{experiment}//midi_mf0/
    """
    canto_jrp.make_midi_mf0s()

def finals_analysis(canto_jrp):
    """perform the analysis of the finals against the ground truth given in the 
    experiment_metadata´
    """
    canto_jrp.create_finals()

def profiles(canto_jrp):
    "create and save the pitch (class) profiles"""
    canto_jrp.save_profiles('pp', normalisation=False)
    canto_jrp.save_profiles('pcp', normalisation=False)
    canto_jrp.save_profiles('pp', normalisation=True)
    canto_jrp.save_profiles('pcp', normalisation=True)#there is a faster way, by transposing the non-normalised profiles
    
def visualise_profiles(canto_jrp):
    """optional step to visualise the pitch (class) profiles"""
    canto_jrp.show_pcp()
    canto_jrp.show_pp()

def distances(canto_jrp): 
    """calculate and save the distances between profiles extracted with various 
    methods and profiles created with the symbolic encocodings
    """
    for distance_type in ['euclidean', 'squared', 'manhattan']:
        canto_jrp.distances('pp', distance_type )
        canto_jrp.distances('pcp', distance_type )

def distance_analysis(canto_jrp):
    """this is the statistical analysis of the distances between profiles of the 
    various extraction methods and the symbolic ground truth"""
    distance_types = ['euclidean', 'squared', 'manhattan']
    for distance_type in distance_types:
        for profile_type in ['pcp', 'pp']:
            perform_dunn_test(experiment, profile_type, distance_type)
            for plot_type in ['line', 'histogram']:            
                visualise_distances(experiment, profile_type, plot_type, distance_type, save = True)
            get_statistics(experiment, profile_type,distance_type)
            kruskal_wallis_test(experiment, profile_type, distance_type)

def cluster_analysis(canto_jrp):
    """This is the visualisation and statistical analysis of the clustering of 
    the profiles derived by the various methods"""
    
    method = 'tSNE'
    profile_types = ['pp', 'pcp']
    # if mirroring is not needed, use mirror=None
    create_clusters_with_subplots(experiment, 'pp', method, mirror=[(1,1), (-1,-1), (-1,1), (-1,1), (-1,1)])
    create_clusters_with_subplots(experiment, 'pcp', method, mirror=[(1,1), (1,1), (1,1), (1,1), (1,1), (-1,1), (1,1)])
    for profile_type in profile_types:   
        combined_clusters_pp, evaluation_df_pp = create_clusters_with_evaluation(experiment, profile_type, method)

def performance_analysis(canto_jrp, experiment, chromatype='multif0'):
    """
    create analysis plots and statistics for the analysis of the variety in 
    the performance metrics number of voices, instrumentation category and decade of recording
    """
    metrics = create_metrics(canto_jrp, experiment, chromatype)
    plot_exploration(metrics, experiment, chromatype)
    plot_metrics(metrics, experiment, 'violin', chromatype)
    regression_analysis(metrics, experiment, chromatype)

def paper_pitch_c_profiles(canto_jrp):
    """create pitch class profile Figure 1"""
    title = '1010-1012 Jos2513 Josquin, Virgo salutiferi | Ave Maria'
    recording = Recording(title, experiment)
    pitch_profile_fuzzypaper(recording, experiment, save=True, normalisation = False)
    pitch_class_profile_fuzzypaper(recording, experiment, save=True, normalisation = False)

def paper_extraction_problems(canto_jrp):
    """create piano rolls Figure 4 and 5"""
    bp_file_name = '0503 Jos2109 O bone et dulcissime Jesu.csv'
    bp = BasicpitchExtraction(bp_file_name, experiment)
    piano_roll_bp(bp, experiment, start_time=300, end_time = 345, save=True)
    
    """ create piano roll for Multipitch"""
    mp_file_name =  '0629 Rue1002b Missa Almana - Gloria.csv'
    model_name = '214c'
    mp = MultipitchExtraction(mp_file_name, experiment, model_name)
    piano_roll_mp(mp, experiment, start_time=355, end_time = 370, save=True)
    
def plot_cycles_paper():
    cycle = Cycle('chromaticism_modes', 'Gesualdo, Carlo', 'Libro Sesto')
    cycle.plot_combined_pitch_class_profiles_all_modes(normalisation=False)
    cycle = Cycle('modal_cycles_1131', 'Palestrina, Giovanni Pierluigi da', 'Vergine')
    cycle.plot_combined_pitch_class_profiles_all_modes_2col(normalisation=False)


def main(experiment_name, new_experiment, compute_results, visualise, midi_creation):
    """ Main function to replicate the results from the paper. """
    canto_jrp = Experiment(experiment_name)

    if new_experiment == True:
        initial_repair_and_quality_assurance(canto_jrp)
    
    if compute_results == True:
        """ Recreate the results in the article Fuzzy Frequencies """
        finals_analysis(canto_jrp)
        profiles(canto_jrp)
        distances(canto_jrp)
        distance_analysis(canto_jrp)
        cluster_analysis(canto_jrp)
        performance_analysis(canto_jrp, experiment, chromatype='multif0')
        paper_pitch_c_profiles(canto_jrp)
        paper_extraction_problems(canto_jrp)
        plot_cycles_paper()
    
    if visualise == True:
        """ Create visualisations of the data """
        visualisation(canto_jrp)
    if midi_creation == True:
        """ Create a MIDI representation of the Multif0 extractions for easier analysis """
        mf0_to_midi(canto_jrp)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate the results of the paper Fuzzy Frequencies.")
    
    # Required argument: experiment name
    parser.add_argument("experiment_name", type=str, help="The name of the experiment to use.")

    # Optional flags (default: False)
    parser.add_argument("--new_experiment", action="store_true", help="Perform initial repair and quality assurance.")
    parser.add_argument("--compute_results", action="store_true", help="Compute the results similiar to the Fuzzy Frequncies article.")
    parser.add_argument("--visualise", action="store_true", help="Generate visualisations of the data.")
    parser.add_argument("--midi_creation", action="store_true", help="Create a MIDI representation of the extracted data.")

    args = parser.parse_args()
    
    main(args.experiment_name, args.new_experiment, args.compute_results ,args.visualise, args.midi_creation)
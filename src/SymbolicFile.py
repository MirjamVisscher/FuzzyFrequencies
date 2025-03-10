#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:18:20 2023

@author: mirjam
"""

import pandas as pd
from music21 import converter, stream, note, chord

from utils import Constants, Tones, get_pitch_name


class SymbolicFile:
    """A class representing the musicXML representation of a composition.
    Attributes:
         file_name(string): name of the file
         title(string): name of the file without file extention
         experiment(string): name of the experiment
    """
    
    def __init__(self, file_name, experiment):
        self.file_name = file_name
        self.experiment_name = experiment
        self.title = file_name.replace('.mxl', '').replace('.musicxml', '')
        
        #load and parse mxlfile
        input_folder = '../data/raw/'+experiment+'/symbolic/'
        input_path = input_folder+file_name
        self.score = converter.parse(input_path)
    
    def __str__(self):
        """ Print instance information of this class
        """
        information = f"file_name:\n{self.file_name}\n\nexperiment:\n{self.experiment_name}\ntitle:\n{self.title}"
        return information


    def partinfo(self):
        """Extract information about the last chord from SymbolicFile
        Returns:
            series: final_chord
            float: lowest_midi
            string: lowest_name
        """
        # get the partnames
        
        def get_note_midi(element):
            if isinstance(element, note.Note):
                midi = element.pitch.midi
                note_name = element.pitch.nameWithOctave
                
            elif isinstance(element, chord.Chord):
                chord_notes = element.notes  # Get all notes in the chord
                chord_midi = [n.pitch.midi for n in chord_notes]  # Get MIDI values of each note
                midi = min(chord_midi)  # Select the lowest MIDI value
                note_name = get_pitch_name(midi)  # Concatenate pitch names with octaves
            return note_name, midi

        def last_element(score, part, measure):
            elements = score[part][measure]

            for element in reversed(elements):
                if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                    return element
            return None  # Return None if no Note or Chord found in the measure


        def last_measure_element(score, part):
            
            lastmeasure = len(score[part])
            for measure in range(-1, -lastmeasure, -1):  # Iterate backward through measures
                
                last_element_in_measure = last_element(score, part, measure)
                if last_element_in_measure is not None:
                    note_name, midi = get_note_midi(last_element_in_measure)
                    return note_name, midi, measure
            return None, None, None  # Return None if no Note or Chord found in any measure
        
        parts_list = []
        for i in range(len(self.score)):
            if isinstance(self.score[i], stream.Part):
                part_df = pd.DataFrame({'partnr': [i], 'partname': [self.score[i].partName],
                                        'lastmeasure': [None], 'midinr': [None], 'final_note': [None]})
                parts_list.append(part_df)
        parts = pd.concat(parts_list, ignore_index=True)


        for part in parts.partnr:
            for measure in reversed(range(len(self.score[part]))):
                
                if isinstance(self.score[part][measure], stream.Measure):
                    
                    lastmeasure = self.score[part][measure].number
                    parts.loc[parts['partnr'] == part, 'lastmeasure'] = lastmeasure
                    break  # Exit the inner loop once the condition is met

        for index, row in parts.iterrows():#for each voice
            partnr = row['partnr']
            note_name, midi, measure = last_measure_element(self.score, partnr)
            parts.loc[parts['partnr'] == partnr, 'midinr'] = int(midi)
            parts.loc[parts['partnr'] == partnr, 'final_note'] = note_name
            parts.loc[parts['partnr'] == partnr, 'last_measured'] = measure
            
        max_last_measured = parts['last_measured'].max()
        parts = parts[parts['last_measured'] == max_last_measured]

        lowest_midi = min(parts.midinr)
        lowest_name = parts.loc[parts['midinr'] == lowest_midi, 'final_note'].unique()
        lowest_name = str(lowest_name[0])
        final_chord = parts.midinr
        return final_chord, lowest_midi, lowest_name



    def count_notes(self):
        """Calculate the number of notes in a mxl or musicxml file.
        Args:
            input_folder (string): Path to the folder containing the file.
            file_name (string): Name of the file.
        Returns:
            int: The number of notes.
        """
               # Initialize a counter for notes
        note_count = 0
        
        # Iterate through all notes in the score
        for element in self.score.recurse():
            if isinstance(element, note.Note):
                note_count += 1
        
        return note_count

    
    def final_midi(self):
        """
        Extract the midi number of the last note of voice 1 from SymbolicFile
        """
        return self.partinfo()[1]
        
    def final_name(self):
        """
        Extract the name of the last note of voice 1 from SymbolicFile
        """
        return self.partinfo()[2]
        
    def final_chord(self):
        """
        Extract the midi numbers of the last chord from SymbolicFile
        """
        return self.partinfo()[0]
    
    # def pitch_class_profile(self, normalisation=False):
    #     """Calculate the pitch class profile of the SymbolicFile
    #     Arguments:
    #         normalisation: if True, the pitch will be normalised to a final on C
    #     Returns:
    #         dataframe: 'pcp_audio_proportion', 'pitch_class_name'
    #     """
    #     pitch_class_profile = [0] * 12
    #     total_duration = 0
    #     for note in self.score.flatten().notes:
    #         if note.isChord:
    #             for sub_note in note:
    #                 pitch_class = sub_note.pitch.pitchClass
    #                 pitch_class_profile[pitch_class] += sub_note.duration.quarterLength
    #                 total_duration += sub_note.duration.quarterLength
    #         else:
    #             pitch_class = note.pitch.pitchClass
    #             pitch_class_profile[pitch_class] += note.duration.quarterLength
    #             total_duration += note.duration.quarterLength
                
    #     pitch_class_profile = [count / total_duration for count in pitch_class_profile]
    #     pitch_class_profile = (pd.DataFrame(pitch_class_profile,columns=['pcp_sym_proportion'])
    #                               .rename_axis('pitch_class')
    #                               .reset_index()
    #                               )
        
    #     if normalisation == True:
    #         difference = self.final_midi()%12
    #         if difference > 5:
    #             difference = -(12-difference)
    #         pitch_class_profile['pitch_class'] = (pitch_class_profile['pitch_class']-difference+12)%12
    #         pitch_class_profile = pitch_class_profile.sort_values(by='pitch_class')

    #     pitch_class_profile= pitch_class_profile.set_index('pitch_class')
    #     pitch_class_profile['pitch_class_name'] = Constants.ALL_KEYS
        
    #     return pitch_class_profile
    
    def pitch_class_profile(self, normalisation=False):
        """Calculate the pitch class profile of the SymbolicFile
        Arguments:
            normalisation: if True, the pitch will be normalised to a final on C
        Returns:
            dataframe: 'pcp_audio_proportion', 'pitch_class_name'
        """
        pitch_class_profile = [0] * 12
        total_duration = 0
        for mnote in self.score.flatten().notes:  # Change 'note' to 'mnote'
            if mnote.isChord:
                for sub_note in mnote:  # Change 'note' to 'mnote'
                    pitch_class = sub_note.pitch.pitchClass
                    pitch_class_profile[pitch_class] += sub_note.duration.quarterLength
                    total_duration += sub_note.duration.quarterLength
            else:
                pitch_class = mnote.pitch.pitchClass  # Change 'note' to 'mnote'
                pitch_class_profile[pitch_class] += mnote.duration.quarterLength
                total_duration += mnote.duration.quarterLength
                    
        pitch_class_profile = [count / total_duration for count in pitch_class_profile]
        pitch_class_profile = (pd.DataFrame(pitch_class_profile, columns=['pcp_sym_proportion'])
                                .rename_axis('pitch_class')
                                .reset_index()
                               )
        
        if normalisation:
            difference = self.final_midi() % 12
            if difference > 5:
                difference = -(12 - difference)
            pitch_class_profile['pitch_class'] = (pitch_class_profile['pitch_class'] - difference + 12) % 12
            pitch_class_profile = pitch_class_profile.sort_values(by='pitch_class')
    
        pitch_class_profile = pitch_class_profile.set_index('pitch_class')
        pitch_class_profile['pitch_class_name'] = Constants.ALL_KEYS
        
        return pitch_class_profile

    
    # def pitch_profile(self, normalisation=False):
    #     """Calculate the pitch profile of the FrequencyFile
    #     Arguments:
    #         normalisation: if True, the pitch will be normalised to a final on C
    #     Returns:
    #         dataframe: 'miditone', 'pp_audio_proportion'
    #     """
    #     # Initialize an empty dictionary to store pitch profile information
    #     pitch_profile = {}
    
    #     # Initialize a variable to track the total duration of all notes
    #     total_duration = 0
    
    #     # Loop through each note in the music score
    #     for note in self.score.flatten().notes:
    #         # Skip None values (rests)
    #         if note is None:
    #             continue
    
    #         # Check if the note is a chord (multiple simultaneous notes)
    #         if note.isChord:
    #             # Loop through each sub-note in the chord
    #             for sub_note in note:
    #                 # Extract MIDI information
    #                 midi_number = sub_note.pitch.midi
    
    #                 # Update pitch profile dictionary
    #                 pitch_profile[midi_number] = {
    #                     'pp_sym_proportion': pitch_profile.get(midi_number, {}).get('pp_sym_proportion', 0) + sub_note.duration.quarterLength,
    #                 }
    
    #                 # Update total duration
    #                 total_duration += sub_note.duration.quarterLength
    #         else:
    #             # Extract MIDI information for a single note
    #             midi_number = note.pitch.midi
    
    #             # Update pitch profile dictionary
    #             pitch_profile[midi_number] = {
    #                 'pp_sym_proportion': pitch_profile.get(midi_number, {}).get('pp_sym_proportion', 0) + note.duration.quarterLength,
    #             }
    
    #             # Update total duration
    #             total_duration += note.duration.quarterLength
    
    #     # Normalize the 'pp_sym_proportion' values based on the total duration
    #     for midi_number, data in pitch_profile.items():
    #         data['pp_sym_proportion'] /= total_duration
    
    #     # Create a DataFrame from the pitch_profile dictionary with 'miditone' as the index
    #     pitch_profile_df = pd.DataFrame(list(pitch_profile.items()), columns=['miditone', 'Data'])
        
    #     if normalisation == True:
    #         difference = self.final_midi()%12
    #         if difference > 5:
    #             difference = -(12-difference)
    #         pitch_profile_df['miditone'] = pitch_profile_df['miditone']-difference
        
    #     pitch_profile_df  = pitch_profile_df.set_index('miditone')
    #     pitch_profile_df = pd.concat([pitch_profile_df, pitch_profile_df['Data'].apply(pd.Series)], axis=1).drop(['Data'], axis=1)
        
        

    #     # Find the minimum and maximum MIDI values in the pitch profile
    #     min_midi = pitch_profile_df.index.min()
    #     max_midi = pitch_profile_df.index.max()
    
    #     # Create a range of MIDI values
    #     midi_range = list(range(int(min_midi), int(max_midi) + 1))
    
    #     # Create a DataFrame with missing MIDI values and corresponding pitch names
    #     missing_midi_df = pd.DataFrame({'miditone': midi_range})
    
    #     # Merge the original pitch profile with the DataFrame of missing MIDI values, filling NaN values with 0.0
    #     full_pitch_profile = pitch_profile_df.merge(missing_midi_df, left_index=True, right_on='miditone', how='outer').fillna(0.0)
    
    #     # Sort the final pitch profile DataFrame by 'miditone' and reset the index
    #     full_pitch_profile = full_pitch_profile.sort_values(by='miditone').reset_index(drop=True)
    
    #     # Add 'pitch_name' to the full_pitch_profile as a last step
    #     #TODO implement librosa's or music21's pitch algorithm instead of Tones.TONES  
    #     full_pitch_profile = full_pitch_profile.merge(Tones.TONES[['miditone', 'pitch_name']], on='miditone', how='outer').fillna(0)
    
    #     # Return the complete pitch profile DataFrame
    #     return full_pitch_profile
    def pitch_profile(self, normalisation=False):
        """Calculate the pitch profile of the FrequencyFile
        Arguments:
            normalisation: if True, the pitch will be normalised to a final on C
        Returns:
            dataframe: 'miditone', 'pp_audio_proportion'
        """
        # Initialize an empty dictionary to store pitch profile information
        pitch_profile = {}
    
        # Initialize a variable to track the total duration of all notes
        total_duration = 0
    
        # Loop through each note in the music score
        for mnote in self.score.flatten().notes:  # Change 'note' to 'mnote'
            # Skip None values (rests)
            if mnote is None:
                continue
    
            # Check if the note is a chord (multiple simultaneous notes)
            if mnote.isChord:
                # Loop through each sub-note in the chord
                for sub_note in mnote:
                    # Extract MIDI information
                    midi_number = sub_note.pitch.midi
    
                    # Update pitch profile dictionary
                    pitch_profile[midi_number] = {
                        'pp_sym_proportion': pitch_profile.get(midi_number, {}).get('pp_sym_proportion', 0) + sub_note.duration.quarterLength,
                    }
    
                    # Update total duration
                    total_duration += sub_note.duration.quarterLength
            else:
                # Extract MIDI information for a single note
                midi_number = mnote.pitch.midi  # Change 'note' to 'mnote'
    
                # Update pitch profile dictionary
                pitch_profile[midi_number] = {
                    'pp_sym_proportion': pitch_profile.get(midi_number, {}).get('pp_sym_proportion', 0) + mnote.duration.quarterLength,
                }
    
                # Update total duration
                total_duration += mnote.duration.quarterLength
    
        # Normalize the 'pp_sym_proportion' values based on the total duration
        for midi_number, data in pitch_profile.items():
            data['pp_sym_proportion'] /= total_duration
    
        # Create a DataFrame from the pitch_profile dictionary with 'miditone' as the index
        pitch_profile_df = pd.DataFrame(list(pitch_profile.items()), columns=['miditone', 'Data'])
    
        if normalisation == True:
            difference = self.final_midi() % 12
            if difference > 5:
                difference = -(12 - difference)
            pitch_profile_df['miditone'] = pitch_profile_df['miditone'] - difference
    
        pitch_profile_df = pitch_profile_df.set_index('miditone')
        pitch_profile_df = pd.concat([pitch_profile_df, pitch_profile_df['Data'].apply(pd.Series)], axis=1).drop(['Data'], axis=1)
    
        # Find the minimum and maximum MIDI values in the pitch profile
        min_midi = pitch_profile_df.index.min()
        max_midi = pitch_profile_df.index.max()
    
        # Create a range of MIDI values
        midi_range = list(range(int(min_midi), int(max_midi) + 1))
    
        # Create a DataFrame with missing MIDI values and corresponding pitch names
        missing_midi_df = pd.DataFrame({'miditone': midi_range})
    
        # Merge the original pitch profile with the DataFrame of missing MIDI values, filling NaN values with 0.0
        full_pitch_profile = pitch_profile_df.merge(missing_midi_df, left_index=True, right_on='miditone', how='outer').fillna(0.0)
    
        # Sort the final pitch profile DataFrame by 'miditone' and reset the index
        full_pitch_profile = full_pitch_profile.sort_values(by='miditone').reset_index(drop=True)
    
        # Add 'pitch_name' to the full_pitch_profile as a last step
        #TODO implement librosa's or music21's pitch algorithm instead of Tones.TONES  
        full_pitch_profile = full_pitch_profile.merge(Tones.TONES[['miditone', 'pitch_name']], on='miditone', how='outer').fillna(0)
    
        # Return the complete pitch profile DataFrame
        return full_pitch_profile

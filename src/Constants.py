#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:51:28 2023

@author: mirjam
"""

import pandas as pd
from utils import get_frequency


class ToneConstants:
    """A class containing the constants used
    Attributes:
        None
    """
    CHAMBER_PITCH = 440
    ALL_KEYS = ['c', 'c#','d', 'd#', 'e', 'f', 'f#', 'g', 'g#','a', 'a#', 'b']
    WHITE_KEYS = ['e', 'f', 'g', 'a', 'b', 'c', 'd']   
    
class Tones:
    """A class containing the constant TONES: a dataframe containing pitch, 
    pitch class in numbers and in names
    Attributes:
        None
    """
    tones = pd.DataFrame()
    tones['miditone'] = range(21,109)
    tones['frequency'] = tones['miditone'].apply(get_frequency)
    tones['pitch_class_name'] = [ToneConstants.ALL_KEYS[(i-12) % len(ToneConstants.ALL_KEYS)] for i in tones['miditone']]
    tones['pitch_name'] = tones['pitch_class_name'] + (tones['miditone'] // 12-1).astype(str)
    tones['white_key'] = tones['pitch_class_name'].isin(ToneConstants.WHITE_KEYS).astype(int)
    TONES = tones
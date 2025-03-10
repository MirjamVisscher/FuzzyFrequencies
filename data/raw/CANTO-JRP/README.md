# CANTO-JRP Dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14991371.svg)](https://doi.org/10.5281/zenodo.14991371) [![GitHub](https://img.shields.io/badge/GitHub-Fuzzy%20Frequencies-181717?logo=github&logoColor=white&color=800080)](https://github.com/MirjamVisscher/FuzzyFrequencies)  [![Spotify Playlist](https://img.shields.io/badge/Spotify-CANTO--JRP-1DB954?logo=spotify&logoColor=1DB954)](https://open.spotify.com/playlist/2QyBpYbo1W5fZhjrIx1uew?si=ef55e1ae74294179) [![Review status](https://img.shields.io/badge/Peer%20Review-Under%20review-yellow.svg)]() 
<!-- [![MIT license](https://img.shields.io/badge/License-MIT-green.svg)](https://lbesson.mit-license.org/) -->
[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

**Authors** 
Mirjam Visscher <a href="https://orcid.org/0000-0003-2152-0278"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="14px" height="14px"></a>  Frans Wiering <a href="https://orcid.org/0000-0002-2984-8932"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="14px" height="14px"></a>


**Project** [CANTOSTREAM](https://www.projects.science.uu.nl/ics-cantostream/)

**Department** Information and Computing Sciences, Utrecht University, The Netherlands 

**Date** 8 March 2025

---

## General  
This `README.md` file provides an overview of the CANTO-JRP Dataset, which is based on compositions from the Josquin Research Project, limited to those available on Spotify at the time the dataset was created.
Due to copyright restrictions, the recordings themselves are not publicly available. Instead, the dataset includes multiple f0 estimations (from various models), symbolic encodings, and metadata.

The dataset is described in the paper (currently submitted, not yet accepted):

Visscher, M.; Wiering, F. Fuzzy Frequencies: Finding tonal structures in audio recordings of Renaissance polyphony. *Journal name* 2025, 1-22.

## Folder Structure  
Each set of multiple f0 extractions is stored in its own folder. 
Due to their large size, these folders are compressed into separate `tar.gz` files.
To use the data with the accompanying code from [GitHub](https://github.com/MirjamVisscher/FuzzyFrequencies), download and extract the relevant folders into `FuzzyFrequencies/data/raw/CANTO-JRP/`.
If you're only interested in the Multif0 extractions, you can download just the `experiment_metadata.csv` and the `multif0.tar.gzip`.

### Folders and files in this dataset
- **195f**  
- **214c**  
- **basicpitch**  
- **MT3**  
- **multif0**  
- **symbolic**  
- experiment_metadata.csv
- README.md
  
## Description of dataset items
This section provides an overview of the data in each folder, how the data should be used, and its purpose.

| Folder   | File types                          | Source                        |number of files|format|
|-----------  |----------                          |----------                     |----------     |-----------|
| 195f        | Multipitch extractions, model 195f  | Fuzzy Frequencies             |637|CSV|
| 214c        | Multipitch extractions, model 214c  | Fuzzy Frequencies             |637|CSV|
| basicpitch  | Basicpitch extractions              | Fuzzy Frequencies             |637|CSV|
| MT3         | MT3 extractions                     | Fuzzy Frequencies             |172|MIDI|
| multif0     | Multif0 extractions                 | Fuzzy Frequencies             |637|CSV|
| symbolic    | Symbolic encodings                  | [Josquin Research Project (JRP)](https://josquin.stanford.edu/) |637|MusicXML|
|

__Note__ The number of MT3 extractions is lower than those from the other models. Due to the MT3 model’s lower performance on our dataset and the high computational cost, we only processed audio files smaller than approximately 10 MB.

## File Types
### Metadata

The file `experiment_metadata.csv` file contains information about each composition from the JRP that was available on Spotify at the time this dataset was created. 
This file serves both as a reference for users of the dataset and as a specification file for the [GitHub code](https://github.com/MirjamVisscher/FuzzyFrequencies).

| Field             | Format  | Description       |
|-------            |-------  |-------            |
| id                | integer | row identifier    |
| nr_playlist       | string  | position(s) in the playlist |
| composer          | string  | composer's surname|
| composition       | string  | name of the composition |
| voices            | integer | number of voices |
| experiment        | string  | experiment name, needed for the code |
| performer         | string  | performer(s) of the recording |
| Album             | string  | album name of the recording |
| year_recording    | integer | year of recording |
| audio_final       | integer | MIDI tone of the lowest note of the final chord |
| symbolic_is_audio | string  | extent to which recording and encoding are the same (yes, almost, no)|
| instrumentation   | string  | instrumentation v(ocal) and (i)nstrumental |
| instrumentation_category | string | category of instrumentation (vocal, instrumental, mixed) |
| final_safe        | string  | extent to which the audio final is the same as the (transposed) encoded final (yes, no, pitch class profile)|
| not repeated      | string  | whether there is repetition of the encoding in the recording (yes, no) |
| repetitions       | string  | rough specification of the repetitions |
| comments          | string  | extra comments, mainly instruments used |
| symbolic          | string  | file name of the symbolic encoding |
| audio             | string  | file name of the recording |
| mf0               | string  | file name of the Multif0 extraction |
| basicpitch        | string  | file name of the Basicpitch extraction |
| multipitch_214c   | string  | file name of the Multipitch extraction, model 214c |
| multipitch_195f   | string  | file name of the Multipitch extraction, model 195f |
| MT3               | string  | file name of the MT3 extraction |
|

### Multipitch extraction
The Multipitch extractions include a column for each MIDI tone, with cell values representing the loudness of the pitch at a given timestamp.

| Field   | Format  | Description                                 |
|-------  |-------  |-------                                      |
| [empty] | integer | timestamp index, sample rate = 43.06640625  |
| 1       | float   | loudness of MIDI tone 1 + 24 = 25           |
| ..      | ..      | ..                                          |
| 71      | float   | loudness of MIDI tone 71 + 24 = 85          |
| 

### Basicpitch extraction
The Basicpitch extractions include a row for each detected note and its corresponding loudness.

| Field         | Format  | Description                                     |
|-------        |-------  |-------                                          |
| start_time_s  |float    |	start time of the pitch in seconds              |
| end_time_s	  |float    |	end time of the pitch in seconds                |
| pitch_midi    |integer  | MIDI tone of the pitch                          |
| velocity      |integer  | MIDI equivalent of loudness                     |	
| pitch_bend    |inteher  | multiple columns of microtonal pitch deviations |
|


### MT3 extraction
The MT3 extractions are provided in [MIDI](https://www.midi.org/) format (Musical Instrument Digital Interface). MIDI is an industry standard music technology protocol used to represent musical data and allow communication between musical devices. For more details, see the [MIDI specifications](https://web.archive.org/web/20191023033822/https://www.midi.org/specifications-old/item/standard-midi-files-smf).

### Multif0 extraction
The Multif0 extractions do not have meaningful headers; the first column contains the timestamps, the subsequent columns contain 'voice' columns, without voice leading. By default, the leftmost voice column contains the lowest detected frequency. 

| Field  | Format  | Description                                            |
|------- |-------  |-------                                                 |
|0.0     |float    |timestamp in seconds, time sample rate = 86.1328125     |
|[empty] |float    |frequency of the lowest voice at that time stamp, frequency sample rate = 20 cents                                            |
| ..     |..       |..                                                      |
|[empty]        |float    |frequency of the highest voice at that time stamp, frequency sample rate = 20 cents                                            |
|
      


### Symbolic encoding
The symbolic encodings are provided in MusicXML format. For an introduction to this format, please see the [MusicXML tutorial](https://www.w3.org/2021/06/musicxml40/tutorial/introduction/)

## Codebook  
In this section, we specify for each file type how the data was collected or created.

For 611 out of the 902 works on the [JRP website](https://josquin.stanford.edu/), usable recordings have been found on Spotify; these are collected in the [__Spotify playlist__](https://open.spotify.com/playlist/2QyBpYbo1W5fZhjrIx1uew?si=ef55e1ae74294179).

The __Basicpitch__ extractions are created by applying the model by Bittner et al. (2022) [1] to the set of audio recordings.

The __Multipitch__ extractions are created by applying the model by Weiß and Müller (2024) [4] to the set of audio recordings, with model 214c and 195f.

The __MT3__ extractions are created by extracting the audio files smaller than ~110 MB using the [Colab notebook](https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb.) provided by Gardner et al (2022) [3].

The __Multif0__ extractions are created by applying the model by Cuesta et al. (2020) [2] to the audio files.

The __symbolic encodings__ are downloaded from [The Josquin Research Project](https://josquin.stanford.edu/)

The files `experiment_metadata.csv` and README.md have been handcrafted by the first author.

## License  
[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

This dataset is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
This means you are free to:
- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for **commercial purposes**.

For full license details, see the [Creative Commons License Page](https://creativecommons.org/licenses/by-nc/4.0/).

<!-- This work is licensed under the [MIT License](https://opensource.org/licenses/MIT). -->

## Contribute

- To report errors, e.g. in composition dates, please create an [issue](https://github.com/MirjamVisscher/FuzzyFrequencies) describing the error.
- Contact me if you want to contribute data
- Also do not hesitate to contact me for further questions: [m.e.visscher@uu.nl](mailto:m.e.visscher@uu.nl)

### Cite

Finally, if you use the code in a research project, please reference it as:

Visscher, M.; Wiering, F. Fuzzy Frequencies: Finding tonal structures in audio recordings of Renaissance polyphony. *Journal name* 2025, 1-22.

```bibtex
@article{visscher2025fuzzy,
  author       = {Mirjam Visscher and
                  Frans Wiering},
  title        = {Fuzzy Frequencies: Finding tonal structures in audio recordings of Renaissance polyphony},
  journal      = {*upon publication*},  % Add actual journal name if available
  year         = {2025},
  volume       = {*upon publication*},  % Add volume if available
  number       = {*upon publication*},  % Add issue number if available
  pages        = {*upon publication*},  % Add page range if available
  doi          = {*upon publication*},
}
```
## References 

[1] Bittner, R.M.; Bosch, J.J.; Rubinstein, D.; Meseguer-Brocal, G.; Ewert, S. A lightweight instrument-agnostic model for polyphonic note transcription and multipitch estimation. In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Singapore, 2022.

[2] Cuesta, H.; McFee, B.; Gómez, E. Multiple f0 estimation in vocal ensembles using convolutional neural networks. In Proceedings of the International Society for Music Information Retrieval (ISMIR), Montréal, Canada, 2020.

[3] Gardner, J.P.; Simon, I.; Manilow, E.; Hawthorne, C.; Engel, J. MT3: Multi-task multitrack music transcription. In Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[4] Weiß, C.; Müller, M. From music scores to audio recordings: Deep pitch-class representations for measuring tonal structures. *ACM Journal on Computing and Cultural Heritage* 2024.
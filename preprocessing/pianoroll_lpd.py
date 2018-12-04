from __future__ import print_function
import pandas as pd

import numpy as np
import pretty_midi
from music21 import *
from midi2audio import FluidSynth
from IPython.display import display, Image, Audio

import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

import os
import glob

import pypianoroll as pr
from pypianoroll import Multitrack, Track
import copy

from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

def traverse_dir(root_dir, extension='.npz'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))
    return file_list


root_dir = './lpd_5_cleansed/'
file_list = traverse_dir(root_dir)

tensor_file = np.zeros((0,768,128,5))
i = 0
for file in file_list:
    i += 1
    print(i)
    multitrack = pr.load(file)
    track_list = [3,0,2,1,4]
    file_len = max(multitrack.tracks[0].pianoroll.shape[0],
                   multitrack.tracks[1].pianoroll.shape[0],
                   multitrack.tracks[2].pianoroll.shape[0],
                   multitrack.tracks[3].pianoroll.shape[0],
                   multitrack.tracks[4].pianoroll.shape[0])
    dbar_len = int(np.floor(file_len/768.0))
    if dbar_len > 0:
        tensor_new = []
        
        for track_idx in track_list:
            track = multitrack.tracks[track_idx].pianoroll
            if (track.shape[0]==0):
                track_re = np.zeros((dbar_len,768,128))
                tensor_new.append(track_re)
            else:
                track_re = np.reshape(track[:dbar_len*768],(dbar_len,768,128))
                tensor_new.append(track_re)
        temp = np.stack(tensor_new,axis=3)
        tensor_file = np.concatenate((tensor_file,temp))
        #print(np.shape(tensor_file))
    else:
        continue

###################################
a = tensor_file.astype(bool)
np.save('./lpd_4dbar_24/tra/tra_phr.npy',a)
###################################
data = np.load('./lpd_4dbar_24/tra/tra_phr.npy')
data_reshape = np.reshape(data,(-1,128,5))
data_reshape_84 = data_reshape[:,24:108,:]
new_len = int(len(data_reshape_84[:,0,0])/2)
data_new = np.full((new_len, 84, 5), False, dtype=bool)

for track in range(0,5):
    print('data_load')
    for note in range(84):
        print(note)
        for time in range(new_len):
            [time1, time2] = data_reshape_84[2*time:(2*time+2),note,track]
            data_new[time,note,track] = (time1 or time2)
    np.save('./lpd_4dbar_12/tra/tra_phr.npy',data_new)
    print('data_save')


##################################
#         pianoroll.mid          #
##################################
data_new = np.load('./lpd_4dbar_12/tra/tra_phr.npy')
data_new_reshape = np.reshape(data_new,(-1,384,84,5))
gen_pr = np.zeros((len(data_new_reshape[:]),384, 128, 5), dtype=bool)
gen_pr[:,:,24:108,:] = data_new_reshape
# Create a `pypianoroll.Track` instance
# program can be choosed to apply various instruments
for idx in range(0,len(gen_pr)):
    track_bass = Track(pianoroll=gen_pr[idx,:,:,0], program=33, is_drum=False,
                  name= 'bass')
    track_drum = Track(pianoroll=gen_pr[idx,:,:,1], program=0, is_drum=True,
                  name= 'drum')
    track_guitar = Track(pianoroll=gen_pr[idx,:,:,2], program=25, is_drum=False,
                  name= 'guitar')
    track_piano = Track(pianoroll=gen_pr[idx,:,:,3], program=0, is_drum=False,
                  name= 'piano')
    track_string = Track(pianoroll=gen_pr[idx,:,:,4], program=41, is_drum=False,
                  name= 'string')
    
    # Create a `pypianoroll.Multitrack` instance
    multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string], tempo=120.0, beat_resolution=12)
    
    # Write the `pypianoroll.Multitrack` instance to a MIDI file
    directory = './lpd_mid/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_o = directory + 'lpd_%s'%idx + '.mid'
    multitrack.write(path_o)

#################################
#         midi_transpose        #
#################################
# import numpy as np
# import csv
# import os
# from music21 import *
# import pretty_midi

data_new = np.load('./lpd_4dbar_12/tra/tra_phr.npy')
data_new_reshape = np.reshape(data_new,(-1,384,84,5))

# measuring scale
for i in range(24912):
    file_i = './lpd_mid/lpd_%s'%i + '.mid'
    print(i)
    ##i_acc = i.split('.')[0]+'_acc.mid'
    ##file_acc = '/Users/haominliu/MidiNet/hymn/accompany_44/new_tune/%s'% i_acc
    try:
        test = converter.parse(file_i)
    except:
        print('converter.parse continue')
        continue
    ##test_acc = converter.parse(file_acc)
    
    a = test.analyze('key')
    
    scale = a.tonicPitchNameWithCase
    print('scale',scale)
    # create a dictionary to match values to notes
    trans = {'C':0,'D-':-1,'D':-2,'E-':-3,'E':-4,'F':-5,'G-':+6,'G':+5,'A-':+4,'A':+3,'B-':+2,'B':+1,
                   'C#':-1,       'D#':-3,              'F#':+6,       'G#':+4,       'A#':+2,
             'a':0,'b-':-2,'b':-2,'c':-3,'d-':-4,'d':-5,'e-':+6,'e':+5,'f':+4,'g-':+3,'g':+2,'a-':+1,
                   'a#':-2,              'c#':-4,       'd#':+6,              'f#':+3,       'g#':+1
            }
      
    ####################### npy transpose ########################
    
    track_list = [0,2,3,4]
    for track in track_list:
        data_new_reshape[i,:,:,track] = np.roll(data_new_reshape[i,:,:,track], trans[scale], axis=1)
        
np.save('./lpd_4dbar_12_C/tra/tra_phr.npy',data_new_reshape)
print('data_save')

#################################
#            midi2wav           #
#################################
# from music21 import *
# from midi2audio import FluidSynth
# from IPython.display import display, Image, Audio
fs = FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2') # arch
for idx in range(24912):
    print(idx)
    fs.midi_to_audio('./lpd_mid_C/lpd_%s'%idx + '.mid', './lpd_wav_C/lpd_%s'%idx + '.wav')

#################################
#  chord extraction by Madmom   #
#################################
names = ["C:min",
         "D:min",
         "E:min",
         "F:min",
         "G:min",
         "A:min",
         "B:min",
         "C:maj",
         "D:maj",
         "E:maj",
         "F:maj",
         "G:maj",
         "A:maj",
         "B:maj",
         "N",
         "C#:min",
         "D#:min",
         "E#:min",
         "F#:min",
         "G#:min",
         "A#:min",
         "B#:min",
         "C#:maj",
         "D#:maj",
         "E#:maj",
         "F#:maj",
         "G#:maj",
         "A#:maj",
         "B#:maj"]
         
notes = [[24,27,31], #Cmin
         [26,29,33], #Dmin
         [28,31,35], #Emin
         [29,32,36], #Fmin
         [31,34,38], #Gmin
         [21,24,28], #Amin
         [23,26,30], #Bmin
         [24,28,31], #Cmaj
         [26,30,33], #Dmaj
         [28,32,35], #Emaj
         [29,33,36], #Fmaj
         [31,35,38], #Gmaj
         [21,25,28], #Amaj
         [23,27,30], #Bmaj
         [], #N
         [25,28,32], #C#min
         [27,30,34], #D#min
         [29,32,36], #E#min
         [30,33,37], #F#min
         [32,35,39], #G#min
         [22,25,29], #A#min
         [24,27,31], #B#min
         [25,29,32], #C#maj
         [27,31,34], #D#maj
         [29,33,36], #E#maj
         [30,34,37], #F#maj
         [32,36,39], #G#maj
         [22,26,29], #A#maj
         [24,28,31]] #B#maj
         
notes_dict = {}
for i in range(len(names)):
    notes_dict[names[i]] = notes[i]

dcp = DeepChromaProcessor()
decode = DeepChromaChordRecognitionProcessor()
data_npy_C = np.load('./lpd_4dbar_12_C/tra/tra_phr.npy')
gen = np.zeros((24912, 384, 84, 6), dtype=bool)
gen[:,:,:,0:5] = data_npy_C
gen_list = []
num = 0 # valid num
for i in range(24912):
    print(i)
    try:    
        chroma = dcp('./lpd_wav_C/lpd_%s'%i + '.wav')
        cho = decode(chroma)
    except Exception as e:
        print(e)
        continue
    
    cho_list = []
    c = 0
    duration = cho[-1][1] - cho[0][0]
    duration_beat = duration/32.0
    last_center = cho[-1][1] - duration_beat/2.0
    first_center = cho[0][0] + duration_beat/2.0
    beat_list = np.linspace(first_center, last_center, num=32)
    ##print(a)
    
    for beat in beat_list:
        if (cho[c][0]<= beat) & (beat <= cho[c][1]):
            cho_list.append(cho[c][2])
        else:
            cho_list.append(cho[c+1][2])
            c += 1
    n = 0
    ####### "N" chord cal #######
    for t in range(32):
        if cho_list[t] == 'N':
            n += 1
        else:
            pass
        
    if n >= 5:
        print('n:%s'%n)
        continue
    else:
        pass
    ###############################
    for t in range(32):
        if cho_list[t] == 'N':
            ##print('N')
            continue
        else:
            ##print(cho_list[t])
            for chord in notes_dict[cho_list[t]]:
                gen[i,12*t:12*(t+1),chord,5] = [True]*12
    
    gen_list.append(gen[i])
    print(np.shape(gen_list))
    num += 1
    
np.save('./lpd_4dbar_12_C/tra/phr_chord_clean.npy',gen_list[:22080])
np.save('./lpd_4dbar_12_C/val/phr_chord_clean.npy',gen_list[22080:])
print('data_save')


#################################
#  test lpd + chord mid .       #
#################################
# data = np.load('./lpd_4dbar_12_C/tra/phr_chord_clean.npy')
# gen_pr = np.zeros((len(data[:]),384, 128, 6), dtype=bool)
# gen_pr[:,:,24:108,:] = data
# idx = 1000
# track_bass = Track(pianoroll=gen_pr[idx,:,:,0], program=33, is_drum=False,
#               name= 'bass')
# track_drum = Track(pianoroll=gen_pr[idx,:,:,1], program=0, is_drum=True,
#               name= 'drum')
# track_guitar = Track(pianoroll=gen_pr[idx,:,:,2], program=25, is_drum=False,
#               name= 'guitar')
# track_piano = Track(pianoroll=gen_pr[idx,:,:,3], program=0, is_drum=False,
#               name= 'piano')
# track_string = Track(pianoroll=gen_pr[idx,:,:,4], program=41, is_drum=False,
#               name= 'string')
# track_chord = Track(pianoroll=gen_pr[idx,:,:,5], program=0, is_drum=False,
#               name= 'chord')

# # Create a `pypianoroll.Multitrack` instance
# multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string, track_chord], tempo=120.0, beat_resolution=12)

# # Write the `pypianoroll.Multitrack` instance to a MIDI file
# directory = './'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# path_o = directory + 'lpd_test.mid'
# multitrack.write(path_o)
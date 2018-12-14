from __future__ import print_function
import pandas as pd

import numpy as np
import pretty_midi
from music21 import *
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
%matplotlib inline
# For putting audio in the notebook
import IPython.display


import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six

import os
import glob
import pypianoroll as pr
import copy

def traverse_dir(root_dir, extension='.npz'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))
    return file_list

###########################################
# From lpd_5_cleansed to lpd_4dbar_12_npy #
###########################################
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
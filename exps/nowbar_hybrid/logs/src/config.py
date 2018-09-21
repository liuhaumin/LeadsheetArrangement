'''
Model Configuration
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from shutil import copyfile
import os
import SharedArray as sa
import tensorflow as tf
import glob

print('[*] config...')

# class Dataset:
##TRACK_NAMES = ['mel', 'acc']
TRACK_NAMES = ['bass', 'drums', 'guitar', 'piano', 'strings']


def get_colormap():
    colormap = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., .5, 0.],
                         [0., .5, 1.]])
    
    ##colormap = np.array([[1., 0., 0.],
    ##                     [0., 1., 0.]])
    return tf.constant(colormap, dtype=tf.float32, name='colormap')

###########################################################################
# Training
###########################################################################

class TrainingConfig:
    is_eval = True
    batch_size = 64
    #batch_size = 32
    epoch = 3
    iter_to_save = 100
    sample_size = 64
    print_batch = True
    ##drum_filter = np.tile([1,0.3,0,0,0,0.3], 16) ## for 96 timesteps per bar
    drum_filter = np.tile([1,0.3,0,0,0,0.3], 8)  ## for 48 timesteps per bar
    
    scale_mask = [1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.]
    inter_pair = [(0,2), (0,3), (0,4), (2,3), (2,4), (3,4)]
    ##inter_pair = [(0,1)]
    track_names = TRACK_NAMES
    track_dim = len(track_names)
    eval_map = np.array([
                    [1, 1, 1, 1, 1],  # metric_is_empty_bar
                    [1, 1, 1, 1, 1],  # metric_num_pitch_used
                    [1, 0, 1, 1, 1],  # metric_too_short_note_ratio
                    [1, 0, 1, 1, 1],  # metric_polyphonic_ratio
                    [1, 0, 1, 1, 1],  # metric_in_scale
                    [0, 1, 0, 0, 0],  # metric_drum_pattern
                    [1, 0, 1, 1, 1]   # metric_num_chroma_used
                ])
    

    ##eval_map = np.array([
    ##                [1,1],  # metric_is_empty_bar
    ##                [1,1],  # metric_num_pitch_used
    ##                [1,1],  # metric_too_short_note_ratio
    ##                [1,1],  # metric_polyphonic_ratio
    ##                [1,1],  # metric_in_scale
    ##                [0,0],  # metric_drum_pattern
    ##                [1,1]   # metric_num_chroma_used
    ##            ])

    exp_name = 'exp'
    gpu_num = '1'


###########################################################################
# Model Config
###########################################################################

class ModelConfig:
    output_w = 48
    output_h = 84
    lamda = 10
    batch_size = 64
    #batch_size = 32
    beta1 = 0.5
    beta2 = 0.9
    lr = 2e-4
    is_bn = True
    colormap = get_colormap()

# image
class MNISTConfig(ModelConfig):
    output_w = 28
    output_h = 28
    z_dim = 74
    output_dim = 1

# RNN
class RNNConfig(ModelConfig):
    track_names = ['All']
    track_dim = 1
    output_bar = 4
    #output_bar = 8
    z_inter_dim = 128
    output_dim = 1
    #output_dim = 5
    acc_idx = None
    state_size = 128

class RNNHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 2
    output_bar = 4
    z_inter_dim = 32
    z_intra_dim = 32
    output_dim = 1
    acc_idx = None
    state_size = 32

# condi 2 track TT
class NowBarRNNHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    acc_idx = 0
    track_dim = 2
    output_bar = 4
    z_inter_dim = 32
    z_intra_dim = 32
    output_dim = 1
    state_size = 32
    
# condi 6 track LPD
##class NowBarRNNHybridConfig(ModelConfig):
##    track_names = TRACK_NAMES
##    acc_idx = 5
##    track_dim = 6
##    output_bar = 4
##    z_inter_dim = 32
##    z_intra_dim = 32
##    output_dim = 1
##    state_size = 32

    
# onebar
class OneBarHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 5
    acc_idx = None
    z_inter_dim = 64
    z_intra_dim = 64
    output_dim = 1

class OneBarJammingConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 5
    acc_idx = None
    z_intra_dim = 128
    output_dim = 1

class OneBarComposerConfig(ModelConfig):
    track_names = ['All']
    track_dim = 1
    acc_idx = None
    z_inter_dim = 128
    output_dim = 5

# nowbar
class NowBarHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 5
    ##acc_idx = 4
    z_inter_dim = 64
    z_intra_dim = 64
    output_dim = 1
    type_ = 0 # 0. chord sequence 1. chroma sequence 2. chroma vector  3. chord label
    acc_output_w = 48 # chord sequence: 48, chroma sequence: 48, chroma vector: 4, chord vector: 4
    acc_output_h = 84 # chord sequence: 84, chroma sequence: 12, chroma vector:12, chord vector: 84

class NowBarJammingConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 5
    acc_idx = 4
    z_intra_dim = 128
    output_dim = 1

class NowBarComposerConfig(ModelConfig):
    track_names = ['All']
    track_dim = 1
    acc_idx = 4
    z_inter_dim = 128
    output_dim = 5

# Temporal
class TemporalHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    # track_dim = 5
    track_dim = 1
    output_bar = 4
    z_inter_dim = 32
    z_intra_dim = 32
    acc_idx = None
    output_dim = 1

class TemporalJammingConfig(ModelConfig):
    track_names = TRACK_NAMES
    track_dim = 5
    output_bar = 4
    z_intra_dim = 64
    output_dim = 1

class TemporalComposerConfig(ModelConfig):
    track_names = ['All']
    track_dim = 1
    output_bar = 4
    z_inter_dim = 64
    acc_idx = None
    output_dim = 5

class NowBarTemporalHybridConfig(ModelConfig):
    track_names = TRACK_NAMES
    acc_idx = 4
    track_dim = 5
    output_bar = 4
    z_inter_dim = 32
    z_intra_dim = 32
    acc_idx = 4
    output_dim = 1

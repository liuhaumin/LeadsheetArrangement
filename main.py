from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pprint import pprint
import SharedArray as sa

from model.core import *
from model.components import *
from input_data import *
from config import *

#assign GPU


if __name__ == '__main__':

    """ Create TensorFlow Session """
    with tf.Session() as sess:
        
    # === Prerequisites ===
    # Step 1 - Initialize the training configuration        
        t_config = TrainingConfig
        t_config.exp_name = 'exps/nowbar_hybrid'        
    
    # Step 2 - Select the desired model
        model = NowbarHybrid(NowBarHybridConfig)
        
    # Step 3 - Initialize the input data object
        input_data = InputDataNowBarHybrid(model)
        
    # Step 4 - Load training data
        path_x_train_bar = 'tra_X_bars'
        path_y_train_bar = 'tra_y_bars'
        input_data.add_data_sa(path_x_train_bar, path_y_train_bar, 'train') # x: input, y: conditional feature
        
    # Step 5 - Initialize a museGAN object
        musegan = MuseGAN(sess, t_config, model)
        
    # === Training ===
        musegan.train(input_data)
    
    # === Load a Pretrained Model ===
        musegan.load(musegan.dir_ckpt)
    
    # === Generate Samples ===
        path_x_test_bar = 'val_X_bars'
        path_y_test_bar = 'val_y_bars'
        input_data.add_data_sa(path_x_test_bar, path_y_test_bar, key='test')
        musegan.gen_test(input_data, is_eval=True)



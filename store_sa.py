from __future__ import print_function

import SharedArray as sa
import numpy as np
import os
from sklearn.utils import shuffle


def save_on_sa(data_dir, use_only_84_keys = True, rescale = True, postfix=''):
    print('Reading...')
    print('[*]',data_dir)

    ##data_prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano', 'Chord']
    ##data_prefix = ['mel_phr','acc_phr']
    
    subdirs = ['tra', 'val']
#    subdirs = ['val']

    for sd in subdirs:
        data = []
        
        # midi setting for training
#         data_X = np.load(os.path.join(data_dir, sd, 'x_bar_chroma.npy'))
#         if sd is 'tra':
#             data_y = np.load(os.path.join(data_dir, sd ,'y_bar_chroma.npy'))
#         else:
#             data_y = np.load(os.path.join(data_dir, sd ,'y_bar_chroma_humanlifeleadsheet.npy'))
#             #data_y = np.load(os.path.join(data_dir, sd ,'y_bar_chroma.npy'))
            
            
        # midi setting for testing
        data_X = np.load(os.path.join(data_dir, 'val', 'x_bar_chroma.npy'))
        if sd is 'tra':
            data_y = np.load(os.path.join(data_dir, 'val' ,'y_bar_chroma.npy'))
        else:
            data_y = np.load(os.path.join(data_dir, 'val' ,'y_bar_chroma_test.npy'))
            #data_y = np.load(os.path.join(data_dir, sd ,'y_bar_chroma.npy'))

        print(data_X.dtype)
        print(data_y.dtype)
        
        if sd is 'tra':
            print(sd)
            print('Shuffling...')
            data_X, data_y = shuffle(data_X, data_y, random_state=0)
            ##data_X = shuffle(data_X, random_state=0)
        else:
            print(sd)
            pass
        name = sd + '_X_' + postfix
        print(name, data_X.shape)
        tmp_arr_x = sa.create(name, data_X.shape, dtype=bool)
        np.copyto(tmp_arr_x, data_X)

        name = sd + '_y_' + postfix
        print(name, data_y.shape)
        ##tmp_arr_y = sa.create(name, data_y.shape, dtype=bool)
        tmp_arr_y = sa.create(name, data_y.shape, dtype=float)
        np.copyto(tmp_arr_y, data_y)

# if __name__ == '__main__':
    
print(' loading data !!')

try:
    sa.delete('tra_X_bars')
except:
    # print('error tra_X_bars')
    pass
try:
    sa.delete('tra_y_bars')
except:
    # print('error tra_y_bars')
    pass
try:
    sa.delete('val_X_bars')
except:
    # print('error val_X_bars')
    pass
try:
    sa.delete('val_y_bars')
except:
    # print('error val_y_bars')
    pass

save_on_sa('./data/chord_roll', postfix='bars')
print('----------loading data completed !!--------------')
##save_on_sa('../../wayne/v3.0/dataset/data_bar', postfix='bars')
##save_on_sa('./data_tab_4dbar_12', postfix='phrs')
##save_on_sa('./data/chroma_roll', postfix='bars')
##save_on_sa('./data/chroma_beat', postfix='bars')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pprint import pprint
import SharedArray as sa
import pickle

from musegan.core import *
from musegan.components import *
from input_data import *
from config import *


from musegan.libs.utils import *

if __name__ == '__main__':

    """ Create TensorFlow Session """

    t_config = TrainingConfig


    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    gen_dir = 'interpolation/try'

    with tf.Session(config=config) as sess:

        # Onebar
        #t_config.exp_name = 'exps/onebar_hybrid'
        #model = NowbarHybrid(OneBarHybridConfig)
        #input_data = InputDataNowBarHybrid(model)

        # RNN_Hybrid
        t_config.exp_name = 'exps/RNN_hybrid'
        model = RNNHybrid(RNNHybridConfig)
        input_data = InputDataRNNHybrid(model)
        

        musegan = MuseGAN(sess, t_config, model)

        musegan.load(musegan.dir_ckpt)

        z_interpolation = dict()

    #################### SINGLE TRACK ###################
        # gen_dir = 'interpolation/gen/intra_4'
        # st_z_inter = np.random.normal(0, 0.1, [64]).astype(np.float32)

        # st_z_intra_inv = np.random.normal(0, 0.1, [64, 64, 4]).astype(np.float32)
        # st_z_intra_v = np.random.normal(0, 0.1, [64, 1]).astype(np.float32)
        # ed_z_intra_v = np.random.normal(0, 0.1, [64, 1]).astype(np.float32)

        # intra_v_list = np.array([st_z_intra_v] + slerp(st_z_intra_v, ed_z_intra_v, steps=62) + [ed_z_intra_v])
        # z_interpolation['inter'] = np.tile(st_z_inter, (64, 1))
        # z_interpolation['intra'] = np.concatenate([st_z_intra_inv, intra_v_list], axis=2)

        # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)

        # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)


    # print(z_interpolation['intra'].shape)
    # np.concatenate([ st_z_intra_v], axis=2)
    # z_list_inter = np.array([st_z_inter] +  + [ed_z_inter])
    # z_list_intra = np.array([st_z_intra] + slerp(st_z_intra, ed_z_intra, steps=62) + [ed_z_intra])

        # ################### GRID interpoaltion ###################
        
        for i in range(200):
            #gen_dir = 'interpolation/gen/bilerp'
            gen_dir = 'interpolation/gen_4dbar/interp_%s'%i
    
            # inter_a0 =  np.random.normal(0, 0.1, [64]).astype(np.float32)
            # intra_b0 =  np.random.normal(0, 0.1, [64, 5]).astype(np.float32)
    
            # inter_a1 = inter_a0 + 0.0005
            # intra_b1 = intra_b0 + 0.0005
            steps = 8 # 8x8 square
            
            inter_i00 = np.random.normal(0, 0.01,[32]).astype(np.float32)
            intra_i00 = np.random.normal(0, 0.01,[32, 2]).astype(np.float32)
            inter_v00 = np.random.normal(0, 0.01,[4, 32]).astype(np.float32)
            intra_v00 = np.random.normal(0, 0.01,[4, 32, 2]).astype(np.float32)
            
            inter_i01 = np.random.normal(0, 0.01,[32]).astype(np.float32)
            intra_i01 = np.random.normal(0, 0.01,[32, 2]).astype(np.float32)
            inter_v01 = np.random.normal(0, 0.01,[4, 32]).astype(np.float32)
            intra_v01 = np.random.normal(0, 0.01,[4, 32, 2]).astype(np.float32)
            
            inter_i10 = np.random.normal(0, 0.01,[32]).astype(np.float32)
            intra_i10 = np.random.normal(0, 0.01,[32, 2]).astype(np.float32)
            inter_v10 = np.random.normal(0, 0.01,[4, 32]).astype(np.float32)
            intra_v10 = np.random.normal(0, 0.01,[4, 32, 2]).astype(np.float32)
            
            inter_i11 = np.random.normal(0, 0.01,[32]).astype(np.float32) 
            intra_i11 = np.random.normal(0, 0.01,[32, 2]).astype(np.float32)
            inter_v11 = np.random.normal(0, 0.01,[4, 32]).astype(np.float32) 
            intra_v11 = np.random.normal(0, 0.01,[4, 32, 2]).astype(np.float32)
            
            grid_inter_i = bislerp(inter_i01, inter_i11, inter_i00, inter_i10, steps)
            grid_intra_i = bislerp(intra_i01, intra_i11, intra_i00, intra_i10, steps)
            grid_inter_v = bislerp(inter_v01, inter_v11, inter_v00, inter_v10, steps)
            grid_intra_v = bislerp(intra_v01, intra_v11, intra_v00, intra_v10, steps)
            
            # self set corner
            ##inter_i0 = np.ones([32]).astype(np.float32) * -0.001
            ##intra_i0 = np.ones([32, 2]).astype(np.float32) * -0.001
            ##inter_i1 = np.ones([32]).astype(np.float32) * 0.001
            ##intra_i1 = np.ones([32, 2]).astype(np.float32) * 0.001
            ##inter_v0 = np.ones([4, 32]).astype(np.float32) * -0.001
            ##intra_v0 =  np.ones([4, 32, 2]).astype(np.float32) * -0.001
            ##inter_v1 = np.ones([4, 32]).astype(np.float32) * 0.001
            ##intra_v1 = np.ones([4, 32, 2]).astype(np.float32) * 0.001
    
            ##grid_list_v = bilerp(inter_v0, inter_v1, intra_v0, intra_v1, 8)
            #grid_list = interp(grid, inter_i, inter_v, intra_i, intra_v, num)
    
            #z_interpolation['z_inter_i'] = np.array([t[0] for t in grid_list])
            z_interpolation['z_inter_i'] = np.array([t for t in grid_inter_i])
            z_interpolation['z_inter_v'] = np.array([t for t in grid_inter_v])
            z_interpolation['z_intra_i'] = np.array([t for t in grid_intra_i])
            z_interpolation['z_intra_v'] = np.array([t for t in grid_intra_v])
            
            # print(inter.shape, intra.shape)
    
            result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)
            make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)
    

            # ################### OLD ###################
            # init
    
    
            # z_interpolation['inter'] = np.reshape(st_z_inter, (1,64))
            # z_interpolation['intra'] = np.reshape(st_z_intra, (1,64,5))
    
            # gen_dir = 'interpolation/gen/init'
    
            # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)
    
            # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)
    
            # # inter
            # gen_dir = 'interpolation/gen/inter'
            # print(gen_dir)
    
            # z_interpolation['inter'] = np.tile(st_z_inter, (64, 1))
            # z_interpolation['intra'] = z_list_intra
    
            # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)
    
            # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)
    
            # # intra
            # gen_dir = 'interpolation/gen/intra'
            # print(gen_dir)
    
            # z_interpolation['inter'] = z_list_inter
            # z_interpolation['intra'] = np.tile(st_z_intra, (64, 1, 1))
    
            # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)
    
            # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)
    
    
            # # both
            # gen_dir = 'interpolation/gen/both'
            # print(gen_dir)
    
            # z_interpolation['inter'] = z_list_inter
            # z_interpolation['intra'] = z_list_intra
    
            # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)
    
            # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)
    


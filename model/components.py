from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from model.libs.ops import *
from model.libs.utils import *
from model.modules import *

class Model:
    def get_model_info(self, quiet=True):
        num_parameter_g = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.g_vars])
        num_parameter_d = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.d_vars])
        num_parameter_all = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.vars])

        if not quiet:
            print('# of parameters in G (generator)                 |', num_parameter_g)
            print('# of parameters in D (discriminator)             |', num_parameter_d)
            print('# of parameters in total                         |', num_parameter_all)

        return num_parameter_g, num_parameter_d, num_parameter_all

    def _build_optimizer(self, config):
        # self.print_vars(self.g_vars)
        with tf.variable_scope('Opt'):

            self.d_optim = tf.train.AdamOptimizer(config.lr, beta1=config.beta1, beta2=config.beta2) \
                                   .minimize(self.d_loss, var_list=self.d_vars)

            self.g_optim = tf.train.AdamOptimizer(config.lr, beta1=config.beta1, beta2=config.beta2) \
                                   .minimize(self.g_loss, var_list=self.g_vars)

    def print_vars(self, var_list):
        print('================================================')
        for v in var_list:
            print(v)




#######################################################################################################################
# NowBar
#######################################################################################################################

class Nowbar(Model):
    def _build_graph(self, config):
        self._build_encoder(config)
        self._build_generator(config)
        self._build_discriminator(config)
        self.print_vars(self.e_vars)
        self.g_vars = self.g_vars + self.e_vars

        self._build_optimizer(config)
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def _build_encoder(self, config):
        with tf.variable_scope('E') as scope:
            if config.type_ is not None:
                ##self.acc_track = tf.slice(self.x, [0, 0, 0, config.acc_idx], [-1, -1, -1, 1]) # take piano as condition
                print('not NONE')
                self.acc_track = self.y
                BE = BarEncoder()
                self.nowbar = BE(in_tensor=self.acc_track, type_=config.type_)
            else:
                print('NONE')
                self.nowbar = None

            self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def _build_generator(self, config):
        with tf.variable_scope('G') as scope:
            self.all_tracks = []

            for tidx in range(config.track_dim):
                with tf.variable_scope(config.track_names[tidx]):
                    BG = BarGenerator(output_dim=self.output_dim)
                    tmp_track = BG(in_tensor=self.z_final_list[tidx], nowbar=self.nowbar, type_=config.type_)

                self.all_tracks.append(tmp_track)

            self.prediction = tf.concat([t for t in self.all_tracks], 3)
            # print(self.prediction.get_shape())
            self.prediction_binary = to_binary_tf(self.prediction)
            self.prediction_chroma = to_chroma_tf(self.prediction_binary)

            self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            ## summary
            prediction_image = to_image_tf(self.prediction, config.colormap)
            self.summary_prediction_image = tf.summary.image('prediction/G', prediction_image,
                                                             max_outputs=10)

    def _build_discriminator(self, config):
        with tf.variable_scope('D') as scope:

            BD = BarDiscriminator()

            self.input_real = self.x
            self.input_fake = self.prediction

            _, self.D_real = BD(self.input_real, nowbar=self.nowbar, type_=config.type_, reuse=False)
            _, self.D_fake = BD(self.input_fake, nowbar=self.nowbar, type_=config.type_, reuse=True)

            ## compute gradient panelty
            # reshape data
            re_real = tf.reshape(self.input_real, [-1, config.output_w * config.output_h * config.track_dim])
            re_fake = tf.reshape(self.input_fake, [-1, config.output_w * config.output_h * config.track_dim])

            # sample alpha from uniform
            alpha = tf.random_uniform(
                                shape=[config.batch_size,1],
                                minval=0.,
                                maxval=1.)
            differences = re_fake - re_real
            interpolates = re_real + (alpha*differences)

            # feed interpolate into D
            X_hat = tf.reshape(interpolates, [-1, config.output_w, config.output_h, config.track_dim])
            _, self.D_hat = BD(X_hat, nowbar=self.nowbar, type_=config.type_, reuse=True)

            # compute gradients panelty
            gradients = tf.gradients(self.D_hat, [interpolates])[0]
            slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2) * config.lamda

            #loss
            self.d_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real)
            self.g_loss = -tf.reduce_mean(self.D_fake)
            self.d_loss += gradient_penalty

            self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

class NowbarHybrid(Nowbar):
    def __init__(self, config):
        with tf.variable_scope('NowbarHybrid'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_intra_dim = config.z_intra_dim
            self.z_inter_dim = config.z_inter_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_intra = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim], name='z_intra')
            self.z_inter = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim], name='z_inter')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w, config.output_h, config.track_dim], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, config.acc_output_w, config.acc_output_h,1], name='y')
            print("self.y",self.y.get_shape())
            # to list
            self.z_final_list =  []

            for tidx in range(config.track_dim):
                z_intra =  tf.squeeze(tf.slice(self.z_intra, [0, 0, tidx], [-1, -1, 1]), squeeze_dims=2)
                z_track = tf.concat([z_intra, self.z_inter], 1)
                self.z_final_list.append(z_track)

            self._build_graph(config)

class NowbarJamming(Nowbar):
    def __init__(self, config):
        with tf.variable_scope('NowbarJamming'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_intra_dim = config.z_intra_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_intra = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim], name='z_intra')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w, config.output_h, config.track_dim], name='x')

            # to list
            self.z_final_list =  []

            for tidx in range(config.track_dim):
                z_track =  tf.squeeze(tf.slice(self.z_intra, [0, 0, tidx], [-1, -1, 1]), squeeze_dims=2)
                self.z_final_list.append(z_track)

            self._build_graph(config)

class NowbarComposer(Nowbar):
    def __init__(self, config):
        with tf.variable_scope('NowbarComposer'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_inter_dim = config.z_inter_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_inter = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim], name='z_inter')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w, config.output_h, self.output_dim], name='x')

            # to list
            self.z_final_list =  [self.z_inter]

            self._build_graph(config)

#######################################################################################################################
# Temporal
#######################################################################################################################

class Temporal(Model):
    def _build_graph(self, config):
        self._build_encoder(config)
        self._build_bar_generator(config)
        self._build_discriminator(config)

        self.g_vars = self.g_vars + self.e_vars
        self.print_vars(self.g_vars)

        self._build_optimizer(config)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def _build_encoder(self, config):

        with tf.variable_scope('E') as scope:
            self.nowbar_list = []
            self.acc_track_list = []

            x_tmp = tf.reshape(self.x,  [-1, config.output_bar, config.output_w, config.output_h, config.track_dim]) #(64, 4, 96, 84, 2)
            BE = BarEncoder()

            for bidx in range(config.output_bar):
                if config.acc_idx is not None:
                    acc_track = tf.slice(x_tmp, [0, bidx, 0, 0, config.acc_idx], [-1, 1, -1, -1, 1]) # take melody as condition
                    acc_track  = tf.squeeze(acc_track, [1]) #(?, 96, 84, 1) 
                    ##nowbar = BE(acc_track, reuse=False) 
                    nowbar = BE(acc_track, reuse=(bidx > 0)) 
                    
                    ##print('nowbar0:',nowbar[0].get_shape()) #(?, 96, 7, 16)
                    ##print('nowbar1:',nowbar[1].get_shape()) #(?, 96, 1, 16)
                    ##print('nowbar6:',nowbar[6].get_shape()) #(?, 2, 1, 16)
                    self.acc_track_list.append(acc_track) #(2, ?, 96, 84, 1)
                else:
                    nowbar = None

                self.nowbar_list.append(nowbar) #(4, ?, 96, 84, 1)
            self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def _build_bar_generator(self, config):
        with tf.variable_scope('G') as scope:

            # gen phrase
            self.phrase = [[None]*config.track_dim for _ in range(config.output_bar)]

            for bidx in range(config.output_bar):
                ##with tf.variable_scope('bar%s'%bidx, reuse=False):
                for tidx in range(config.track_dim):
                    if tidx is config.acc_idx:
                        tmp_track = self.acc_track_list[bidx]
                    else:
                        ##with tf.variable_scope(config.track_names[tidx], reuse=False):
                        with tf.variable_scope(config.track_names[tidx], reuse=(bidx > 0)):
                            BG = BarGenerator(output_dim=self.output_dim)
                            tmp_track = BG(in_tensor=self.z_final[bidx][tidx],
                                           nowbar= self.nowbar_list[bidx],
                                           type_=0,
                                           reuse=(bidx > 0))    ##reuse = False)
                    self.phrase[bidx][tidx] = tmp_track
            
            for track in self.phrase:#
                print(track)
                for bar in track:
                    print(bar)  #
                    
            self.prediction = tf.concat([tf.concat([bar for bar in track], 3) for track in self.phrase], 1)
            # print(self.prediction.get_shape())
            self.prediction_binary = to_binary_tf(self.prediction)
            self.prediction_chroma = to_chroma_tf(self.prediction_binary)

            self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            ## summary
            prediction_image = to_image_tf(self.prediction, config.colormap)
            self.summary_prediction_image = tf.summary.image('prediction/G', prediction_image,
                                                             max_outputs=10)
    def _build_discriminator(self, config):
        with tf.variable_scope('D') as scope:

            BD = BarDiscriminator()
            PD = PhraseDiscriminator()

            # real & fake
            self.input_real = tf.reshape(self.x, [-1, config.output_w, config.output_h, config.track_dim])
            self.input_fake = tf.reshape(self.prediction, [-1, config.output_w, config.output_h, config.track_dim])

            self.D_real_h5, _ = BD(self.input_real, reuse=False)
            self.D_fake_h5, _ = BD(self.input_fake, reuse=True)

            self.D_real_h5_r = tf.reshape(self.D_real_h5, [-1, config.output_bar, 128])
            self.D_fake_h5_r = tf.reshape(self.D_fake_h5, [-1, config.output_bar, 128])

            self.D_real = PD(self.D_real_h5_r, reuse=False)
            self.D_fake = PD(self.D_fake_h5_r, reuse=True)

            ## compute gradient panelty
            # reshape data
            re_real = tf.reshape(self.input_real, [-1, config.output_bar * config.output_w * config.output_h * config.track_dim])
            re_fake = tf.reshape(self.input_fake, [-1, config.output_bar * config.output_w * config.output_h * config.track_dim])

            # sample alpha from uniform
            print(re_real.get_shape()[0])
            alpha = tf.random_uniform(
                                shape=[config.batch_size,1],
                                minval=0.,
                                maxval=1.)

            differences = re_fake - re_real
            interpolates = re_real + (alpha*differences)

            # feed interpolate into D
            X_hat = tf.reshape(interpolates, [-1, config.output_w, config.output_h, config.track_dim])
            self.D_hat_h5, _ = BD(X_hat, reuse=True)
            self.D_hat_h5_r  = tf.reshape(self.D_hat_h5, [-1, config.output_bar, 128])
            self.D_hat = PD(self.D_hat_h5_r, reuse=True)

            # compute gradients panelty
            gradients = tf.gradients(self.D_hat, [interpolates])[0]
            slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2) * config.lamda

            #loss
            self.d_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real)
            self.g_loss = -tf.reduce_mean(self.D_fake)
            self.d_loss += gradient_penalty

            self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

class TemporalHybrid(Temporal):
    def __init__(self, config):
        with tf.variable_scope('TemporalHybrid'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_intra_dim = config.z_intra_dim
            self.z_inter_dim = config.z_inter_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_intra_v = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim],
                 name='z_intra_v') # input_latent_i_t
            self.z_intra_i = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim],
                name='z_intra_i') # input_latent_i
            self.z_inter_v = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim]
                , name='z_inter_v')  # input_latent_t
            self.z_inter_i = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim]
                , name='z_inter_i')  # input_latent

            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w*config.output_bar, config.output_h, config.track_dim], name='x')

            self._build_phrase_generator(config)

            # to list
            self.z_final = [[None]*config.track_dim for _ in range(config.output_bar)]

            for bidx in range(config.output_bar):
                for tidx in range(config.track_dim):

                    tz_inter_v = tf.squeeze(tf.slice(self.z_inter_v_hat, [0, 0, bidx], [-1, -1, 1]), axis=2)
                    tz_intra_v = tf.squeeze(tf.slice(self.z_intra_v_hat[tidx], [0, 0, bidx], [-1, -1, 1]), axis=2)
                    tz_inter_i = self.z_inter_i
                    tz_intra_i = tf.squeeze(tf.slice(self.z_intra_i, [0, 0, tidx], [-1, -1, 1]), axis=2)

                    self.z_final[bidx][tidx] = tf.concat([tz_inter_v, tz_intra_v, tz_inter_i, tz_intra_i], 1)

            self._build_graph(config)

    def _build_phrase_generator(self, config):
        with tf.variable_scope('G') as scope:
            PG = PhraseGenerator(output_dim=config.z_inter_dim)

            # arrange time variant latents
            self.z_inter_v_hat = PG(self.z_inter_v, reuse=False)
            self.z_intra_v_hat = []
            for tidx in range(config.track_dim):
                tz_intra_v = tf.squeeze(tf.slice(self.z_intra_v, [0, 0, tidx], [-1, -1, 1]), squeeze_dims=2)
                tz_intra_v_hat = PG(tz_intra_v, reuse=True)
                self.z_intra_v_hat.append(tz_intra_v_hat)

class TemporalJamming(Temporal):
    def __init__(self, config):
        with tf.variable_scope('TemporalJamming'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_intra_dim = config.z_intra_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_intra_v = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim],
                 name='z_intra_v') # input_latent_i_t
            self.z_intra_i = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim],
                name='z_intra_i') # input_latent_i

            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w*config.output_bar, config.output_h, config.track_dim], name='x')

            self._build_phrase_generator(config)

            # to list
            self.z_final = [[None]*config.track_dim for _ in range(config.output_bar)]

            for bidx in range(config.output_bar):
                for tidx in range(config.track_dim):
                    tz_intra_v = tf.squeeze(tf.slice(self.z_intra_v_hat[tidx], [0, 0, bidx], [-1, -1, 1]), axis=2)
                    tz_intra_i = tf.squeeze(tf.slice(self.z_intra_i, [0, 0, tidx], [-1, -1, 1]), axis=2)

                    self.z_final[bidx][tidx] = tf.concat([ tz_intra_v, tz_intra_i], 1)

            self._build_graph(config)

    def _build_phrase_generator(self, config):
        with tf.variable_scope('G') as scope:
            PG = PhraseGenerator(output_dim=config.z_intra_dim)

            # arrange time variant latents
            self.z_intra_v_hat = []
            for tidx in range(config.track_dim):
                tz_intra_v = tf.squeeze(tf.slice(self.z_intra_v, [0, 0, tidx], [-1, -1, 1]), squeeze_dims=2)
                tz_intra_v_hat = PG(tz_intra_v, reuse=tidx>0)
                self.z_intra_v_hat.append(tz_intra_v_hat)

class TemporalComposer(Temporal):
    def __init__(self, config):
        with tf.variable_scope('TemporalComposer'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_inter_dim = config.z_inter_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z_inter_v = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim]
                , name='z_inter_v')  # input_latent_t
            self.z_inter_i = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim]
                , name='z_inter_i')  # input_latent

            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w*config.output_bar, config.output_h, self.output_dim], name='x')

            self._build_phrase_generator(config)

            # to list
            self.z_final = [[None]*config.track_dim for _ in range(config.output_bar)]

            for bidx in range(config.output_bar):
                for tidx in range(config.track_dim):
                    tz_inter_v = tf.squeeze(tf.slice(self.z_inter_v_hat, [0, 0, bidx], [-1, -1, 1]), axis=2)
                    tz_inter_i = self.z_inter_i

                    self.z_final[bidx][tidx] = tf.concat([tz_inter_v, tz_inter_i], 1)

            self._build_graph(config)

    def _build_phrase_generator(self, config):
        with tf.variable_scope('G') as scope:
            PG = PhraseGenerator(output_dim=config.z_inter_dim)

            # arrange time variant latents
            self.z_inter_v_hat = PG(self.z_inter_v, reuse=False)

#######################################################################################################################
# RNN
#######################################################################################################################

class RNNComposer(Temporal):
    def __init__(self, config):
        with tf.variable_scope('RNNComposer'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_inter_dim = config.z_inter_dim
            self.output_dim = config.output_dim
            self.output_bar = config.output_bar

            # placeholder
            self.z_inter = tf.placeholder(tf.float32, shape=[config.batch_size, config.output_bar, config.z_inter_dim], name='z_inter')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w*config.output_bar, config.output_h, self.output_dim], name='x')

            self.z_final = [[None]*config.track_dim for _ in range(config.output_bar)]

            self._build_phrase_generator_rnn(config)

            for bidx in range(config.output_bar):
                for tidx in range(config.track_dim):
                    tz_inter = tf.squeeze(tf.slice(self.z_inter_v_hat, [0, bidx, 0], [-1, 1, -1]), axis=1)
                    print(tz_inter.get_shape()) # (64, 128)
                    self.z_final[bidx][tidx] = tz_inter

            self._build_graph(config)


    def _build_phrase_generator_rnn(self, config):
        with tf.variable_scope('G') as scope:
            self.cell = tf.contrib.rnn.LSTMBlockCell(config.state_size)
            self.initial_state = self.cell.zero_state(config.batch_size, tf.float32)
            self.z_inter_v_hat, last_state = tf.nn.dynamic_rnn(self.cell, self.z_inter, initial_state=self.initial_state)
            print(self.z_inter_v_hat.get_shape()) # (64, 8, 128)

            self.r_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            print(self.r_vars)
            
            
class RNNHybrid(Temporal):
    def __init__(self, config):
        with tf.variable_scope('RNNHybrid'):

            # set key vars
            self.track_dim = config.track_dim
            self.z_inter_dim = config.z_inter_dim
            self.z_intra_dim = config.z_intra_dim
            self.output_dim = config.output_dim
            self.output_bar = config.output_bar

            # placeholder
            self.z_inter_v = tf.placeholder(tf.float32, shape=[config.batch_size, config.output_bar, config.z_inter_dim],
                                            name='z_inter_v') #(64, 4, 32)
            self.z_inter_i = tf.placeholder(tf.float32, shape=[config.batch_size, config.z_inter_dim], name='z_inter_i') #(64, 32)
            self.z_intra_v = tf.placeholder(tf.float32, shape=[config.batch_size, config.output_bar, config.z_intra_dim,
                                                               config.track_dim], name='z_intra_v') #(64, 4, 32, 2)
            self.z_intra_i = tf.placeholder(tf.float32, shape=[config.batch_size, config.z_intra_dim, config.track_dim],
                                            name='z_intra_i') #(64, 32, 2)
            
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w*config.output_bar, config.output_h, self.track_dim],
                                    name='x') #(?, 384, 84, 2)

            # to list
            self.z_final = [[None]*config.track_dim for _ in range(config.output_bar)]

            self._build_phrase_generator_rnn(config)

            for bidx in range(config.output_bar):
                for tidx in range(config.track_dim):
                    tz_inter_v = tf.squeeze(tf.slice(self.z_inter_v_hat, [0, bidx, 0], [-1, 1, -1]), axis=1) #(64, 32)
                    tz_intra_v = tf.squeeze(tf.slice(self.z_intra_v_hat[tidx], [0,bidx,0], [-1, 1, -1]), axis=1) #(64, 32)
                    ##print('tz_inter_v:',tz_inter_v.get_shape()) # (64, 128)
                    ##print('tz_intra_v%s:'%tidx,tz_intra_v.get_shape())
                    tz_inter_i = self.z_inter_i #(64, 32)
                    tz_intra_i = tf.squeeze(tf.slice(self.z_intra_i, [0,0,tidx], [-1, -1, 1]), axis=2) #(64, 32)
                    self.z_final[bidx][tidx] = tf.concat([tz_inter_v, tz_intra_v, tz_inter_i, tz_intra_i], 1) #(64,128)
            self._build_graph(config)


    def _build_phrase_generator_rnn(self, config):
        with tf.variable_scope('G') as scope:
            with tf.variable_scope('inter'):
                self.cell = tf.contrib.rnn.LSTMBlockCell(config.state_size)
                self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 2)##
                self.initial_state = self.cell.zero_state(config.batch_size, tf.float32)
                self.z_inter_v_hat, last_state = tf.nn.dynamic_rnn(self.cell, self.z_inter_v, initial_state=self.initial_state)
                print(self.z_inter_v_hat.get_shape()) # (64, 4, 32)
        
        ##with tf.variable_scope('G', reuse=True) as scope:
            
            self.z_intra_v_hat = []
            for tidx in range(config.track_dim):
                with tf.variable_scope(config.track_names[tidx]):
                    print('z_intra_v',self.z_intra_v.get_shape()) # (64, 4, 32, 2)
                    tz_intra_v = tf.squeeze(tf.slice(self.z_intra_v, [0, 0, 0, tidx], [-1, -1, -1, 1]), axis=3) # (64, 4, 32)
                    tz_intra_v_hat, last_state = tf.nn.dynamic_rnn(self.cell, tz_intra_v, initial_state=self.initial_state) 
                    self.z_intra_v_hat.append(tz_intra_v_hat) # (2, 64, 4, 32)
                
            self.r_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            print(self.r_vars)   
            


#######################################################################################################################
# GAN
#######################################################################################################################

class ImageMNIST(Model):
    def __init__(self, config):
        with tf.variable_scope('NowbarComposer'):

            self.z_dim = config.z_dim
            self.output_dim = config.output_dim

            # placeholder
            self.z = tf.placeholder(tf.float32, shape=[None, config.z_dim], name='z_inter')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w, config.output_h, self.output_dim], name='x')

            self._build_graph(config)

    def _build_graph(self, config):
        self._build_generator(config)
        self._build_discriminator(config)
        self._build_optimizer(config)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    def _build_generator(self, config):
        with tf.variable_scope('G') as scope:
            G = ImageGenerator(output_dim=self.output_dim)
            self.prediction = G(in_tensor=self.z)
            # print(self.prediction.get_shape())

            self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            ## summary
            prediction_image = to_image_tf(self.prediction, config.colormap)
            self.summary_prediction_image = tf.summary.image('prediction/G', prediction_image,
                                                             max_outputs=10)

    def _build_discriminator(self, config):
        with tf.variable_scope('D') as scope:

            D = ImageDiscriminator()

            self.input_real = self.x
            self.input_fake = self.prediction

            self.D_real = D(self.input_real, reuse=False)
            self.D_fake = D(self.input_fake, reuse=True)

            epsilon = tf.random_uniform([], 0.0, 1.0)

            X_hat = epsilon * self.input_real + (1 - epsilon) * self.input_fake
            D_hat = D(X_hat, reuse=True)

            self.d_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real)
            self.g_loss = tf.reduce_mean(self.D_fake)

            gp = tf.gradients(D_hat, X_hat)[0]
            gp = tf.sqrt(tf.reduce_sum(tf.square(gp), axis=1))
            gp = tf.reduce_mean(tf.square(gp - 1.0) * config.lamda)

            self.d_loss += gp
            self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

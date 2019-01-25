from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.libs.ops import *
from model.libs.utils import *

class PhraseGenerator(object):
    def __init__(self, name='PG', output_dim=1, is_bn=True):
        self.output_dim = output_dim
        self.name = name
        self.is_bn = is_bn

    def __call__(self, in_tensor, reuse=False):
        
        ############## . paul115236 ############
        with tf.variable_scope(self.name, reuse=reuse):
            h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
            h0 = relu(batch_norm(transconv2d(h0, [2, 1], 1024, kernels=[2, 1],
                                            strides=[2, 1], name='h1'), self.is_bn))
            h1 = relu(batch_norm(transconv2d(h0, [4, 1], self.output_dim,
                                            kernels=[3, 1], strides=[1, 1], name='h2'), self.is_bn))
            h1 = tf.transpose(tf.squeeze(h1, axis=2), [0, 2, 1])

        return h1
        #########################################
        

        #with tf.variable_scope(self.name, reuse=reuse):
        #    h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
        #    h0 = relu(batch_norm(transconv2d(h0, [2, 1], 1024, kernels=[2, 1],
        #                                    strides=[2, 1], name='h1'), self.is_bn))
        #    h1 = relu(batch_norm(transconv2d(h0, [4, 1], self.output_dim,
        #                                    kernels=[3, 1], strides=[1, 1], name='h2'), self.is_bn))
        #    h1 = tf.transpose(tf.squeeze(h1, axis=2), [0, 2, 1])
        #
        #return h1

class BarGenerator(object):
    def __init__(self, name='BG', output_dim=1, is_bn=True):
        self.output_dim = output_dim
        self.name = name
        self.is_bn = is_bn

    def __call__(self, in_tensor, nowbar=None, type_=None, reuse=False):
        print('BarGenerator:', type_ )
        if type_ is 0:  # chord sequence
            with tf.variable_scope(self.name, reuse=reuse):
                # shape(in_tensor): (B, dim)=(64,128)
                h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]])) #(64, 1, 1, 128) 
                h0 = relu(batch_norm(transconv2d(h0, [1, 1], 1024, kernels=[1, 1], strides=[1, 1], name='h0'), self.is_bn))

                h1 = tf.reshape(h0, [-1, 2, 1, 512])
                h1 = concat_prev(h1, nowbar[5] if nowbar else None)
                h1 = relu(batch_norm(transconv2d(h1, [4, 1], 512, kernels=[2, 1], strides=[2, 1], name='h1'), self.is_bn))

                h2 = concat_prev(h1, nowbar[4] if nowbar else None)
                h2 = relu(batch_norm(transconv2d(h2, [8, 1], 256, kernels=[2, 1], strides=[2, 1], name='h2'), self.is_bn))

                h3 = concat_prev(h2, nowbar[3] if nowbar else None)
                h3 = relu(batch_norm(transconv2d(h3, [16, 1], 256, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn))

                h4 = concat_prev(h3, nowbar[2] if nowbar else None)
                h4 = relu(batch_norm(transconv2d(h4, [48, 1], 128, kernels=[3, 1], strides=[3, 1], name='h4'), self.is_bn))

                h5 = concat_prev(h4, nowbar[1] if nowbar else None)
                h5 = relu(batch_norm(transconv2d(h5, [48, 7], 64, kernels=[1, 7], strides=[1, 1], name='h5'), self.is_bn))

                h6 = concat_prev(h5, nowbar[0] if nowbar else None)
                h6 = transconv2d(h6, [48, 84], self.output_dim, kernels=[1, 12], strides=[1, 12], name='h6')

            return tf.nn.tanh(h6)
        
        if type_ is 1: # chroma sequence
            with tf.variable_scope(self.name, reuse=reuse):

                h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
                h0 = relu(batch_norm(transconv2d(h0, [1, 1], 1024, kernels=[1, 1], strides=[1, 1], name='h0'), self.is_bn))

                h1 = relu(batch_norm(transconv2d(h0, [1, 12], 512, kernels=[1, 12], strides=[1, 12], name='h1'), self.is_bn))     
                
                h2 = concat_prev(h1, nowbar[5] if nowbar else None)
                h2 = relu(batch_norm(transconv2d(h2, [2, 12], 256, kernels=[2, 1], strides=[2, 1], name='h2'), self.is_bn))
                
                h3 = concat_prev(h2, nowbar[4] if nowbar else None)
                h3 = relu(batch_norm(transconv2d(h3, [4, 12], 256, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn))

                h4 = concat_prev(h3, nowbar[3] if nowbar else None)
                h4 = relu(batch_norm(transconv2d(h4, [8, 12], 128, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn))

                h5 = concat_prev(h4, nowbar[2] if nowbar else None)
                h5 = relu(batch_norm(transconv2d(h5, [16, 12], 128, kernels=[2, 1], strides=[2, 1], name='h5'), self.is_bn))

                h6 = concat_prev(h5, nowbar[1] if nowbar else None)
                h6 = relu(batch_norm(transconv2d(h6, [48, 12], 64, kernels=[3, 1], strides=[3, 1], name='h6'), self.is_bn))

                h7 = concat_prev(h6, nowbar[0] if nowbar else None)
                h7 = transconv2d(h7, [48, 84], self.output_dim, kernels=[1, 7], strides=[1, 7], name='h7')

            return tf.nn.tanh(h7)

        elif type_ is 2: # chroma vector 4
            with tf.variable_scope(self.name, reuse=reuse):

                h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]]))
                h0 = relu(batch_norm(transconv2d(h0, [1, 1], 1024, kernels=[1, 1], strides=[1, 1], name='h0'), self.is_bn))

                h1 = relu(batch_norm(transconv2d(h0, [1, 12], 512, kernels=[1, 12], strides=[1, 12], name='h1'), self.is_bn))     
                
                h2 = relu(batch_norm(transconv2d(h1, [2, 12], 256, kernels=[2, 1], strides=[2, 1], name='h2'), self.is_bn))
                
                h3 = relu(batch_norm(transconv2d(h2, [4, 12], 256, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn))

                h4 = concat_prev(h3, nowbar[0] if nowbar else None)
                h4 = relu(batch_norm(transconv2d(h4, [8, 12], 128, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn))

                h5 = relu(batch_norm(transconv2d(h4, [16, 12], 128, kernels=[2, 1], strides=[2, 1], name='h5'), self.is_bn))

                h6 = relu(batch_norm(transconv2d(h5, [48, 12], 64, kernels=[3, 1], strides=[3, 1], name='h6'), self.is_bn))

                h7 = transconv2d(h6, [48, 84], self.output_dim, kernels=[1, 7], strides=[1, 7], name='h7')

            return tf.nn.tanh(h7)

        elif type_ is 3: # chord label
            with tf.variable_scope(self.name, reuse=reuse):

                h0 = tf.reshape(in_tensor, tf.stack([-1, 1, 1, in_tensor.get_shape()[1]])) #(64, 1, 1, 128)
                h0 = relu(batch_norm(transconv2d(h0, [1, 1], 1024, kernels=[1, 1], strides=[1, 1], name='h0'), self.is_bn))

                h1 = tf.reshape(h0, [-1, 2, 1, 512])
                h1 = concat_prev(h1, nowbar[5] if nowbar else None)
                h1 = relu(batch_norm(transconv2d(h1, [4, 1], 512, kernels=[2, 1], strides=[2, 1], name='h1'), self.is_bn))

                h2 = concat_prev(h1, nowbar[4] if nowbar else None)
                h2 = relu(batch_norm(transconv2d(h2, [8, 1], 256, kernels=[2, 1], strides=[2, 1], name='h2'), self.is_bn))

                h3 = concat_prev(h2, nowbar[3] if nowbar else None)
                h3 = relu(batch_norm(transconv2d(h3, [16, 1], 256, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn))

                h4 = concat_prev(h3, nowbar[2] if nowbar else None)
                h4 = relu(batch_norm(transconv2d(h4, [48, 1], 128, kernels=[3, 1], strides=[3, 1], name='h4'), self.is_bn))

                h5 = concat_prev(h4, nowbar[1] if nowbar else None)
                h5 = relu(batch_norm(transconv2d(h5, [48, 7], 64, kernels=[1, 7], strides=[1, 1], name='h5'), self.is_bn))

                h6 = concat_prev(h5, nowbar[0] if nowbar else None)
                h6 = transconv2d(h6, [48, 84], self.output_dim, kernels=[1, 12], strides=[1, 12], name='h6')

            return tf.nn.tanh(h7)

class BarEncoder(object):
    def __init__(self, name='BE', is_bn=True):
        self.name = name
        self.is_bn = is_bn

    def __call__(self, in_tensor, type_=None, reuse=False):
        print('BarEncoder:', type_)
        if type_ is 0: # chord sequence
            with tf.variable_scope(self.name, reuse=reuse):
                h0 = lrelu(batch_norm(conv2d(in_tensor, 16, kernels=[1, 12], strides=[1, 12], name='h0'), self.is_bn)) #(?, 48, 7, 16)
                h1 = lrelu(batch_norm(conv2d(h0, 16, kernels=[1, 7], strides=[1, 7], name='h1'), self.is_bn)) #(?, 48, 1, 16)
                h2 = lrelu(batch_norm(conv2d(h1, 16, kernels=[3, 1], strides=[3, 1], name='h2'), self.is_bn)) #(?, 16, 1, 16)
                h3 = lrelu(batch_norm(conv2d(h2, 16, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn)) #(?, 8, 1, 16)
                h4 = lrelu(batch_norm(conv2d(h3, 16, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn)) #(?, 4, 1, 16)
                h5 = lrelu(batch_norm(conv2d(h4, 16, kernels=[2, 1], strides=[2, 1], name='h5'), self.is_bn)) #(?, 2, 1, 16)
    
            return [h0, h1, h2, h3, h4, h5, in_tensor]

        if type_ is 1: # chroma sequence
            with tf.variable_scope(self.name, reuse=reuse):
                h0 = in_tensor # (?,48,12,1)
                h1 = lrelu(batch_norm(conv2d(h0, 16, kernels=[3, 1], strides=[3, 1], name='h1'), self.is_bn)) #(?, 16, 12, 16)
                h2 = lrelu(batch_norm(conv2d(h1, 16, kernels=[2, 1], strides=[2, 1], name='h2'), self.is_bn)) #(?, 8, 12, 16)
                h3 = lrelu(batch_norm(conv2d(h2, 16, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn)) #(?, 4, 12, 16)
                h4 = lrelu(batch_norm(conv2d(h3, 16, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn)) #(?, 2, 12, 16)
                h5 = lrelu(batch_norm(conv2d(h4, 16, kernels=[2, 1], strides=[2, 1], name='h5'), self.is_bn)) #(?, 1, 12, 16)
                
                ##h0 = in_tensor # (?,48,12)
            return [h0, h1, h2, h3, h4, h5]
        
        if type_ is 2: # chroma vector
            with tf.variable_scope(self.name, reuse=reuse):
                h0 = in_tensor
            return [h0]
    
        if type_ is 3: # chord label
            with tf.variable_scope(self.name, reuse=reuse):
                h0 = lrelu(batch_norm(conv2d(in_tensor, 16, kernels=[1, 12], strides=[1, 12], name='h0'), self.is_bn)) #(?, 48, 7, 16)
                h1 = lrelu(batch_norm(conv2d(h0, 16, kernels=[1, 7], strides=[1, 7], name='h1'), self.is_bn)) #(?, 48, 1, 16)
                h2 = lrelu(batch_norm(conv2d(h1, 16, kernels=[3, 1], strides=[3, 1], name='h2'), self.is_bn)) #(?, 16, 1, 16)
                h3 = lrelu(batch_norm(conv2d(h2, 16, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn)) #(?, 8, 1, 16)
                h4 = lrelu(batch_norm(conv2d(h3, 16, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn)) #(?, 4, 1, 16)
                h5 = lrelu(batch_norm(conv2d(h4, 16, kernels=[2, 1], strides=[2, 1], name='h5'), self.is_bn)) #(?, 2, 1, 16)
    
            return [h0, h1, h2, h3, h4, h5]
class BarDiscriminator(object):

    def __init__(self, name='BD'):
        self.name = name

    def __call__(self, in_tensor, nowbar=None, type_=None, reuse=False):
        print('BarDiscriminator:', type_)
        if type_ is 0: # chord sequence
            with tf.variable_scope(self.name, reuse=reuse):
                ## conv
                print('nowbar',tf.shape(nowbar[6]))
                print('in_tensor',tf.shape(in_tensor))
                h0 = tf.concat([in_tensor,nowbar[6]],3)
                h0 = lrelu(conv2d(h0, 128, kernels=[1, 12], strides=[1, 12], name='h0')) # (64, 48, 7, 128)
                h1 = lrelu(conv2d(h0, 128, kernels=[1, 7], strides=[1, 7], name='h1')) # (64, 48, 1, 128)
                h2 = lrelu(conv2d(h1, 128, kernels=[2, 1], strides=[2, 1], name='h2')) # (64, 24, 1, 128)
                h3 = lrelu(conv2d(h2, 128, kernels=[2, 1], strides=[2, 1], name='h3')) # (64, 12, 1, 128)
                h4 = lrelu(conv2d(h3, 256, kernels=[4, 1], strides=[2, 1], name='h4')) # (64, 5, 1, 256)
                h5 = lrelu(conv2d(h4, 512, kernels=[3, 1], strides=[2, 1], name='h5')) # (64, 2, 1, 512)
    
                ## linear
                h6 = tf.reshape(h5, [-1, np.product([s.value for s in h5.get_shape()[1:]])])
                h6 = lrelu(linear(h6, 1024, name='h6'))
                h7 = linear(h6, 1, name='h7')
            return h5, h7

        if type_ is 1: # chroma sequence
            with tf.variable_scope(self.name, reuse=reuse):
                ## conv
                h0 = lrelu(conv2d(in_tensor, 128, kernels=[1, 7], strides=[1, 7], name='h0')) # (64, 48, 12, 128)
                
                h1 = concat_prev(h0, nowbar[0] if nowbar else None)
                h1 = lrelu(conv2d(h1, 128, kernels=[3, 1], strides=[3, 1], name='h1')) # (64, 16, 12, 128)
                
                h2 = concat_prev(h1, nowbar[1] if nowbar else None)
                h2 = lrelu(conv2d(h2, 128, kernels=[2, 1], strides=[2, 1], name='h2')) # (64, 8, 12, 128)
                
                h3 = concat_prev(h2, nowbar[2] if nowbar else None)
                h3 = lrelu(conv2d(h3, 128, kernels=[2, 1], strides=[2, 1], name='h3')) # (64, 4, 12, 128)
                
                h4 = concat_prev(h3, nowbar[3] if nowbar else None)
                h4 = lrelu(conv2d(h4, 256, kernels=[2, 1], strides=[2, 1], name='h4')) # (64, 2, 12, 256)
                
                h5 = concat_prev(h4, nowbar[4] if nowbar else None)
                h5 = lrelu(conv2d(h5, 512, kernels=[2, 1], strides=[2, 1], name='h5')) # (64, 1, 12, 512)
    
    
                ## linear
                h6 = concat_prev(h5, nowbar[5] if nowbar else None) # (64, 1, 12, 528)
                h6 = tf.reshape(h6, [-1, np.product([s.value for s in h6.get_shape()[1:]])]) # (64, 12*528)
                h6 = lrelu(linear(h6, 1024, name='h6')) #(64, 1024)
                h7 = linear(h6, 1, name='h7') # (64, 1)
            return h5, h7

        if type_ is 2: # chroma vector 4
            with tf.variable_scope(self.name, reuse=reuse):
                ## conv
                h0 = lrelu(conv2d(in_tensor, 128, kernels=[1, 7], strides=[1, 7], name='h0')) # (64, 48, 12, 128)
                
                h1 = lrelu(conv2d(h0, 128, kernels=[3, 1], strides=[3, 1], name='h1')) # (64, 16, 12, 128)
                
                h2 = lrelu(conv2d(h1, 128, kernels=[2, 1], strides=[2, 1], name='h2')) # (64, 8, 12, 128)
                
                h3 = lrelu(conv2d(h2, 128, kernels=[2, 1], strides=[2, 1], name='h3')) # (64, 4, 12, 128)
                
                h4 = concat_prev(h3, nowbar[0] if nowbar else None)
                h4 = lrelu(conv2d(h4, 256, kernels=[2, 1], strides=[2, 1], name='h4')) # (64, 2, 12, 256)
                
                h5 = lrelu(conv2d(h4, 512, kernels=[2, 1], strides=[2, 1], name='h5')) # (64, 1, 12, 512)
    
    
                ## linear
                h6 = tf.reshape(h5, [-1, np.product([s.value for s in h5.get_shape()[1:]])])
                h6 = lrelu(linear(h6, 1024, name='h6'))
                h7 = linear(h6, 1, name='h7')
            return h5, h7
        
        if type_ is 3: # chord label
            with tf.variable_scope(self.name, reuse=reuse):
                ## conv
                h0 = lrelu(conv2d(in_tensor, 128, kernels=[1, 7], strides=[1, 7], name='h0')) # (64, 48, 12, 128)
                h1 = lrelu(conv2d(h0, 128, kernels=[1, 12], strides=[1, 12], name='h1')) # (64, 48, 1, 128)
                h2 = lrelu(conv2d(h1, 128, kernels=[3, 1], strides=[3, 1], name='h2')) # (64, 16, 1, 128)
                h3 = lrelu(conv2d(h2, 128, kernels=[2, 1], strides=[2, 1], name='h3')) # (64, 8, 1, 128)
                h4 = lrelu(conv2d(h3, 256, kernels=[2, 1], strides=[2, 1], name='h4')) # (64, 4, 1, 256)
                h5 = lrelu(conv2d(h4, 512, kernels=[2, 1], strides=[2, 1], name='h5')) # (64, 2, 1, 512)
    
                ## linear
                h6 = tf.reshape(h5, [-1, np.product([s.value for s in h5.get_shape()[1:]])])
                h6 = lrelu(linear(h6, 1024, name='h6'))
                h7 = linear(h6, 1, name='h7')
            return h5, h7

class PhraseDiscriminator(object):
    def __init__(self, name='PD'):
        self.name = name

    def __call__(self, in_tensor, reuse):
        with tf.variable_scope(self.name, reuse=reuse):

            ############# paul115236 ##############
            ## conv
            #h0 = lrelu(conv2d(tf.expand_dims(in_tensor, axis=2), 512,
            #                  kernels=[2, 1], strides=[2, 1], name='h0'))
            #h1 = lrelu(conv2d(h0, 256, kernels=[2, 1], strides=[1, 1],name='h1'))
            #h2 = lrelu(conv2d(h1, 128, kernels=[3, 1], strides=[3, 1],name='h2'))
            ## linear
            #h3 = tf.reshape(h2, [-1, np.product([s.value for s in h2.get_shape()[1:]])])
            #h3 = lrelu(linear(h3, 1024, name='h3'))
            #h4 = linear(h3, 1, name='h4')
        #return h4
            #########################################
            
            ## conv
            h0 = lrelu(conv2d(tf.expand_dims(in_tensor, axis=2), 512,
                              kernels=[2, 1], strides=[1, 1], name='h0'))
            h1 = lrelu(conv2d(h0, 128, kernels=[3, 1], strides=[3, 1],name='h1'))

            ## linear
            h2 = tf.reshape(h1, [-1, np.product([s.value for s in h1.get_shape()[1:]])])
            h2 = lrelu(linear(h2, 1024, name='h2'))
            h3 = linear(h2, 1, name='h3')
        return h3


class ImageGenerator(object):
    def __init__(self, name='image_G', output_dim=3, is_bn=True):
        self.output_dim = output_dim
        self.name = name
        self.is_bn = is_bn


    def __call__(self, in_tensor, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):

            # linear
            h0 = relu(batch_norm(linear(in_tensor, 128*7*7, name='h0'), self.is_bn))
            h0 = tf.reshape(h0, [-1, 7, 7, 128])

            #convnet
            h1 = relu(batch_norm(transconv2d(h0, [14, 14], 256, kernels=[4, 4], strides=[2, 2], name='h1', padding = 'SAME'), self.is_bn))
            h2 = relu(batch_norm(transconv2d(h1, [28, 28], self.output_dim, kernels=[4, 4], strides=[2, 2], name='h2', padding = 'SAME'), self.is_bn))

        return tf.nn.tanh(h2)


class  ImageDiscriminator(object):
    def __init__(self, name='image_D'):
        self.name = name

    def __call__(self, in_tensor, reuse):
        with tf.variable_scope(self.name, reuse=reuse):

            ## conv
            h0 = lrelu(batch_norm(conv2d(in_tensor, 256, kernels=[4, 4], strides=[2, 2], name='h0'), True))
            h1 = lrelu(batch_norm(conv2d(h0, 256, kernels=[4, 4], strides=[2, 2], name='h1'), True))

            ## linear
            h1 = tf.reshape(h1, [-1, np.product([s.value for s in h1.get_shape()[1:]])])
            h2 = lrelu(linear(h1, 1024, name='h2'))
            h3 = linear(h2, 1, name='h3')

        return h3



from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import glob
import os
from lib import *

class MusicHighlighter(object):
    def __init__(self):
        self.dim_feature = 64

        # During training or testing, we both use batch normalization
        self.bn_params = {'is_training': True, 
                          'center': True, 'scale': True,
                          'updates_collections': None}
        
        # place holder
        self.x = tf.placeholder(tf.float32, shape=[None, None, 128])
        self.pos_enc = tf.placeholder(tf.float32, shape=[None, None, self.dim_feature*4])
        self.num_chunk = tf.placeholder(tf.int32)

    def conv(self, inputs, filters, kernel, stride):
        dim = inputs.get_shape().as_list()[-2]
        return tf.contrib.layers.conv2d(inputs, filters, 
                                        [kernel, dim], [stride, dim],
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params=self.bn_params)

    def fc(self, inputs, num_units, act=tf.nn.relu):
        return tf.contrib.layers.fully_connected(inputs, num_units,
                                                 activation_fn=act,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,
                                                 normalizer_params=self.bn_params)

    def attention(self, inputs, dim):
        outputs = self.fc(inputs, dim, act=tf.nn.tanh)
        outputs = self.fc(outputs, 1, act=None)
        attn_score = tf.nn.softmax(outputs, dim=1)
        return attn_score

    def build_model(self):
        # 2D Conv. feature extraction
        net = tf.expand_dims(self.x, axis=3)
        net = self.conv(net, self.dim_feature, 3, 2)
        net = self.conv(net, self.dim_feature*2, 4, 2)
        net = self.conv(net, self.dim_feature*4, 4, 2)

        # Global max-pooling
        net = tf.squeeze(tf.reduce_max(net, axis=1), axis=1)

        # Restore shape [batch_size, num_chunk, dim_feature]
        net = tf.reshape(net, [1, self.num_chunk, self.dim_feature*4])

        # Attention mechanism
        attn_net = net + self.pos_enc
        attn_net = self.fc(attn_net, self.dim_feature*4)
        attn_net = self.fc(attn_net, self.dim_feature*4)
        attn_score = self.attention(attn_net, self.dim_feature*4)

        '''
        # This part is only used in training.
        # Outputs prediction
        net = self.fc(net, 1024)
        chunk_predictions = self.fc(net, self.n_class, act=tf.nn.softmax)
        overall_predictions = tf.squeeze(tf.matmul(attn_score, chunk_predictions, transpose_a=True), axis=1)
        loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(overall_predictions), axis=1))
        '''
        return attn_score

    def extract(self, length=30, save_score=True, save_thumbnail=True, save_wav=True):
        fs = sorted(glob.glob('./input/*.mp3'))

        # check output folder
        if not os.path.isdir('./output'):
            os.mkdir('./output')
            os.mkdir('./output/score')
            os.mkdir('./output/highlight')
            os.mkdir('./output/audio')
        else:
            if not os.path.isdir('./output/score'):
                os.mkdir('./output/score')
            if not os.path.isdir('./output/highlight'):
                os.mkdir('./output/highlight')
            if not os.path.isdir('./output/audio'):
                os.mkdir('./output/audio')
        
        attn_score = self.build_model()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, './model/model')

            for f in fs:
                # read and split to chunk
                name = f.split('/')[-1][:-4]
                print(name, 'processing...')
                audio, spectrogram, duration = audio_read(f)
                n_chunk, remainder = np.divmod(duration, 3)
                chunk_spec = chunk(spectrogram, n_chunk)

                # model
                pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=self.dim_feature*4)
                feed = {self.num_chunk: n_chunk, self.x: chunk_spec, self.pos_enc: pos}
                attn = sess.run([attn_score], feed_dict=feed)[0]
                attn = np.repeat(attn, 3)
                attn = np.append(attn, np.zeros(remainder))
                
                # score
                attn = attn / attn.max()
                if save_score:
                    np.save('./output/score/{}.npy'.format(name), attn)
                
                # thumbnail
                attn = attn.cumsum()
                attn = np.append(attn[length], attn[length:] - attn[:-length])
                index = np.argmax(attn)
                highlight = [index, index + length]

                if save_thumbnail:
                    np.save('./output/highlight/{}.npy'.format(name), highlight)

                if save_wav:
                    librosa.output.write_wav('./output/audio/{}.wav'.format(name), audio[highlight[0]*22050:highlight[1]*22050], 22050)
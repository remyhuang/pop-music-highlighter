import tensorflow as tf

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
        self.build_model()

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
        self.attn_score = self.attention(attn_net, self.dim_feature*4)

        ## This part is only used in training ##
        # net = self.fc(net, 1024)
        # chunk_predictions = self.fc(net, 190, act=tf.nn.softmax)
        # overall_predictions = tf.squeeze(tf.matmul(attn_score, chunk_predictions, transpose_a=True), axis=1)
        # loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(overall_predictions), axis=1))
        self.saver = tf.train.Saver(tf.global_variables())

    def calculate(self, sess, x, pos_enc, num_chunk):
        feed_dict = {self.x: x, self.pos_enc: pos_enc, self.num_chunk: num_chunk}
        attn_score = sess.run(self.attn_score, feed_dict=feed_dict)
        return attn_score
import tensorflow as tf
import numpy as np

class ARKernel():
    def __init__(self, layer_name):
        self.layer_name = layer_name

    def __call__(self, x, is_training):
        # x: N x 2 x h x w x 1
        # [batch, in_depth, in_height, in_width, in_channels]
        with tf.variable_scope(self.layer_name + "_significance"):
            # conv1_1 = np.zeros((self.order, 1, 1)).astype(dtype=np.float32)
            # assert self.order == 2
            # conv1_1[0, 0, 0] = 1.37
            # conv1_1[1, 0, 0] = -0.37
            # # tf.nn.conv3d()
            # out = tf.layers.conv3d(
            #     x, 1, (self.order, 1, 1),
            #     strides=(1, 1, 1), padding='same',
            #     name='ar2',
            #     kernel_initializer=lambda x: conv1_1,
            # )
            x0 = tf.pad(x, paddings=[[0,0], [0,0], [1,1], [1,1],[0,0]], mode="CONSTANT")
            s_out0 = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv3d(
                x0, 64, (1, 3, 3),
                padding="valid",
                name='sig0',
            ), training=is_training))

            s_out0 = tf.pad(s_out0, paddings=[[0,0], [0,0], [1,1], [1,1],[0,0]])
            s_out1 = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.layers.conv3d(
                    s_out0, 128, (1, 3, 3),
                    padding="valid",
                    name='sig1',
                ), training=is_training))

            s_out1 = tf.pad(s_out1, paddings=[[0,0], [0,0], [1,1], [1,1],[0,0]])
            s_out2 = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.layers.conv3d(
                    s_out1, 128, (1, 3, 3),
                    padding="valid",
                    name='sig2',
                ), training=is_training))


            s_out2 = tf.pad(s_out2, paddings=[[0,0], [0,0], [1,1], [1,1],[0,0]])
            s_out3 = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.layers.conv3d(
                    s_out2, 64, (1, 3, 3),
                    padding="valid",
                    name='sig3',
                ), training=is_training))

            s_out3 = tf.pad(s_out3, paddings=[[0,0], [0,0], [1,1], [1,1],[0,0]])
            s_out4 = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.layers.conv3d(
                    s_out3, 1, (1, 3, 3),
                    padding="valid",
                    name='sig4',
                ), training=is_training))

            s_out4 = tf.pad(s_out4, paddings=[[0,0], [0,0], [1,1], [1,1],[0,0]])
            s_out5 = tf.nn.leaky_relu(
                tf.layers.batch_normalization(tf.layers.conv3d(
                    s_out4, 1, (1, 3, 3),
                    padding="valid",
                    name='sig5',
                ), training=is_training))
            s_out6 = tf.nn.sigmoid(s_out5)

        with tf.variable_scope(self.layer_name + "_offset"):

            o_out0 = tf.layers.conv3d(
                x, 1, (1, 1, 1),
                strides=(1, 1, 1),
                name='offset',
            )
            o_out1 = tf.layers.batch_normalization(o_out0, training=is_training)
            o_out2 = tf.nn.leaky_relu(o_out1)

        out = s_out6 * o_out2
        out = tf.layers.conv3d(out, 1, (5, 1, 1))
        return out



if __name__ == '__main__':
    model = ARKernel("test")
    data = tf.random_uniform((10, 5, 20, 20, 1))
    out = model(data, is_training=False)
    print("out.shape", out.shape)

import tensorflow as tf


def leaky_relu(x, alpha=0.2):
    # return tf.maximum(tf.minimum(0.0, alpha * x), x)
    return tf.maximum(0.0, x)

#拼接条件向量
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(y)[3]])], 3)


class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc = tf.layers.dense(
                x, self.nb_units,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf.layers.dense(
                    fc, self.nb_units,
                    # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation=tf.identity
                )

                fc = tf.layers.batch_normalization(fc,training=True)
                #fc = tf.keras.layers.BatchNormalization(fc)

                # fc = leaky_relu(fc)
                fc = tf.nn.tanh(fc)

            output = tf.layers.dense(
                fc, 1,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
            )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# Discriminator for images, takes (bs,dim) as input
class Discriminator_img(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256, dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]

            x = z[:, :self.input_dim]
            #条件信息
            y = z[:, self.input_dim:]
            #one-hot，标签信息
            y.set_shape([None, 10])
            yb = tf.reshape(y, shape=[bs, 1, 1, 10])
            #调整x形状将y作为条件嵌入
            if self.dataset == "mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset == "cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            x = conv_cond_concat(x, yb)

            #多层 CNN 进行特征提取
            conv = tf.layers.conv2d(
                x, 32, [4, 4], strides=[2, 2],
                activation=tf.identity
            )
            # (bs, 14, 14, 32)
            conv = leaky_relu(conv)
            conv = conv_cond_concat(conv, yb)
            for _ in range(self.nb_layers - 1):
                # conv = tcl.convolution2d(conv, 64, [4, 4], [2, 2],
                #                          activation_fn=tf.identity
                #                          )
                # conv = tc.layers.batch_norm(conv, decay=0.9, scale=True, updates_collections=None)
                conv = tf.layers.conv2d(
                    conv, 64, [4, 4], strides=[2, 2],
                    activation=tf.identity
                )

                conv = tf.layers.batch_normalization(
                    conv, momentum=0.9, scale=True, training=True,
                )

                conv = leaky_relu(conv)
            # (bs, 7, 7, 32)
            # fc = tf.reshape(conv, [bs, -1])
            #展平 + y 作为额外输入
            fc = tf.layers.flatten(conv)
            # (bs, 1568)
            fc = tf.concat([fc, y], axis=1)
            #全连接层
            fc = tf.layers.dense(
                fc, 1024,
                activation=tf.identity
            )

            fc = tf.layers.batch_normalization(
                fc, momentum=0.9, scale=True, training=True
            )
            fc = leaky_relu(fc)

            fc = tf.concat([fc, y], axis=1)
            #标量
            output = tf.layers.dense(
                fc, 1,
                activation=tf.identity
            )

            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator_img_ucond(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256, dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            if self.dataset == "mnist":
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset == "cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            conv = tf.layers.conv2d(
                x, 32, [4, 4], strides=[2, 2],
                activation=tf.identity
            )

            # (bs, 14, 14, 32)
            conv = leaky_relu(conv)
            for _ in range(self.nb_layers - 1):
                conv = tf.layers.conv2d(
                    conv, 64, [4, 4], strides=[2, 2],
                    activation=tf.identity
                )

                conv = tf.layers.batch_normalization(
                    conv, momentum=0.9, scale=True, training=True
                )

                conv = leaky_relu(conv)
            # (bs, 7, 7, 32)
            # fc = tf.reshape(conv, [bs, -1])
            fc = tf.layers.flatten(conv)
            # (bs, 1568)

            fc = tf.layers.dense(
                fc, 1024,
                activation=tf.identity
            )

            fc = tf.layers.batch_normalization(
                fc, momentum=0.9, scale=True, training=True
            )

            fc = leaky_relu(fc)

            output = tf.layers.dense(
                fc, 1,
                activation=tf.identity
            )

            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tf.layers.dense(
                z, self.nb_units,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                # weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation=tf.identity
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf.layers.dense(
                    fc, self.nb_units,
                    # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    # weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation=tf.identity
                )
                fc = leaky_relu(fc)

            output = tf.layers.dense(
                fc, self.output_dim,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                # weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation=tf.identity
            )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# generator for images, G()
class Generator_img(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, dataset='mnist', is_training=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            #print(f"Batch size: {bs}")
            # print_op = tf.print('Batch size:', bs)
            # with tf.control_dependencies([print_op]):
            #     # 确保在打印之后才进行 reshape 操作
            #     bs = tf.identity(bs)
            #输入 z 的最后 10 维 代表类别信息
            y = z[:, -10:]

            yb = tf.reshape(y, shape=[bs, 1, 1, 10])

            #全连接层
            fc = tf.layers.dense(
                z, 1024,
                activation=tf.identity
            )

            fc = tf.layers.batch_normalization(
                            fc, momentum=0.9, scale=True, training=self.is_training
                        )
            # fc = tf.keras.layers.BatchNormalization(
            #     momentum=0.9,
            #     scale=True
            # )(fc, training=self.is_training)

            fc = leaky_relu(fc)
            fc = tf.concat([fc, y], 1)

            if self.dataset == 'mnist':
                fc = tf.layers.dense(
                    fc, 7 * 7 * 128,
                    activation=tf.identity
                )

                fc = tf.reshape(fc, tf.stack([bs, 7, 7, 128]))

            elif self.dataset == 'cifar10':
                fc = tf.layers.dense(
                    fc, 8 * 8 * 128,
                    activation=tf.identity
                )
                fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tf.layers.batch_normalization(
                fc, momentum=0.9, scale=True, training=self.is_training
            )
            # fc = tf.keras.layers.BatchNormalization(
            #     momentum=0.9,
            #     scale=True  # 注意：Keras 中参数名为 'scale'，但实际对应的是 'gamma' 是否可训练
            # )(fc, training=self.is_training)

            fc = leaky_relu(fc)
            fc = conv_cond_concat(fc, yb)
            #after conv_cond_concat: (?, 7, 7, 138)
            #反卷积上采样
            conv = tf.layers.conv2d_transpose(
                fc, 64, [4, 4], strides=[2, 2],padding='SAME',
                activation=tf.identity
            )
            # print(f"Shape before batch_normalization layer: {conv.shape}")# (?, 14, 14, 64)
            conv = tf.layers.batch_normalization(
                conv, momentum=0.9, scale=True, training=self.is_training
            )
            # conv = tf.keras.layers.BatchNormalization(
            #     momentum=0.9,
            #     scale=True  # 注意：Keras 中参数名为 'scale'，但实际对应的是 'gamma' 是否可训练
            # )(conv, training=self.is_training)
            # print(f"Shape after batch_normalization layer: {conv.shape}")# (?, 14, 14, 64)
            #14 14 64
            conv = leaky_relu(conv)
            if self.dataset == 'mnist':
                output = tf.layers.conv2d_transpose(
                    conv, 1, [4, 4], strides=[2, 2],padding='SAME',
                    activation=tf.nn.sigmoid
                )
                # print(f"Shape after final deconvolution layer for MNIST: {output.shape}")
                output = tf.reshape(output, [bs, -1])
                # print(f"Final output shape after reshaping for MNIST: {output.shape}")
            elif self.dataset == 'cifar10':
                output = tf.layers.conv2d_transpose(
                    conv, 3, [4, 4], strides=[2, 2],padding='SAME',
                    activation=tf.nn.sigmoid
                )

                output = tf.reshape(output, [bs, -1])

            # (0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator_img_ucond(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, dataset='mnist', is_training=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.is_training = is_training

    def __call__(self, z, reuse=True):
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            fc = tf.layers.dense(
                z, 1024,
                activation=tf.identity
            )
            fc = tf.layers.batch_normalization(
                fc, momentum=0.9, scale=True, training=self.is_training
            )

            fc = leaky_relu(fc)

            if self.dataset == 'mnist':
                fc = tf.layers.dense(
                    fc, 7 * 7 * 128,
                    activation=tf.identity
                )
                fc = tf.reshape(fc, tf.stack([bs, 7, 7, 128]))
            elif self.dataset == 'cifar10':
                fc = tf.layers.dense(
                    fc, 8 * 8 * 128,
                    activation=tf.identity
                )
                fc = tf.reshape(fc, tf.stack([bs, 8, 8, 128]))
            fc = tf.layers.batch_normalization(
                fc, momentum=0.9, scale=True, training=self.is_training
            )

            fc = leaky_relu(fc)
            conv = tf.layers.conv2d_transpose(
                fc, 64, [4, 4], strides=[2, 2],
                activation=tf.identity
            )

            # (bs, 14, 14, 64)
            conv = tf.layers.batch_normalization(
                conv, momentum=0.9, scale=True, training=self.is_training
            )

            conv = leaky_relu(conv)
            if self.dataset == 'mnist':
                output = tf.layers.conv2d_transpose(
                    conv, 1, [4, 4], strides=[2, 2],
                    activation=tf.nn.sigmoid
                )

                output = tf.reshape(output, [bs, -1])
            elif self.dataset == 'cifar10':
                output = tf.layers.conv2d_transpose(
                    conv, 3, [4, 4], strides=[2, 2],
                    activation=tf.nn.sigmoid
                )

                output = tf.reshape(output, [bs, -1])
            # (0,1) by tanh
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, input_dim, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            # with tf.variable_scope(self.name) as vs:
            #     if reuse:
            #         vs.reuse_variables()
            fc = tf.layers.dense(
                x, self.nb_units,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf.layers.dense(
                    fc, self.nb_units,
                    # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation=tf.identity
                )
                fc = leaky_relu(fc)

            output = tf.layers.dense(
                fc, self.output_dim,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
            )
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
            return output[:, 0:self.feat_dim], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# encoder for images, H()
class Encoder_img(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, dataset='mnist', cond=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset
        self.cond = cond

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            if self.dataset == "mnist":
                # print(x.get_shape())
                x = tf.reshape(x, [bs, 28, 28, 1])
            elif self.dataset == "cifar10":
                x = tf.reshape(x, [bs, 32, 32, 3])
            conv = tf.layers.conv2d(
                x, 64, [4, 4], strides=[2, 2],
                activation=tf.identity
            )

            conv = leaky_relu(conv)
            for _ in range(self.nb_layers - 1):
                conv = tf.layers.conv2d(
                    conv, self.nb_units, [4, 4], strides=[2, 2],
                    activation=tf.identity
                )

                conv = leaky_relu(conv)
            conv = tf.layers.flatten(conv)
            fc = tf.layers.dense(
                conv, 1024, activation=tf.identity
            )

            # if self.cond:
            #     output = tcl.fully_connected(
            #         fc, self.output_dim+10,
            #         activation_fn=tf.identity
            #         )
            #     logits = output[:, self.output_dim:]
            #     y = tf.nn.softmax(logits)
            #     return output[:,:self.output_dim], y, logits
            # else:
            output = tf.layers.dense(
                fc, self.output_dim,
                activation=tf.identity
            )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


if __name__ == '__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian

    g_net = Generator_img(input_dim=10, output_dim=784, name='g_net', nb_layers=2, nb_units=256, dataset='mnist')
    d_net = Discriminator_img(input_dim=784 + 10, name='d_net', nb_layers=2, nb_units=256, dataset='mnist')
    x = tf.placeholder(tf.float32, [32, 10], name='x')
    y = g_net(x, reuse=False)
    z = tf.placeholder(tf.float32, [32, 784 + 10], name='z')
    dz = d_net(z, reuse=False)

    print(z, dz)


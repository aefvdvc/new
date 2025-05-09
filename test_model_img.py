import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()  # ½ûÓÃ TensorFlow 2.x ÐÐÎª

def leaky_relu(x, alpha=0.2):
    return tf1.maximum(alpha * x, x)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf1.concat([x, y * tf1.ones([tf1.shape(x)[0], tf1.shape(x)[1], tf1.shape(x)[2], tf1.shape(y)[3]])], 3)

class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf1.variable_scope(self.name, reuse=reuse):
            fc = tf1.layers.dense(
                x, self.nb_units,
                activation=None,
                name="fc_initial"
            )
            fc = leaky_relu(fc)
            for i in range(self.nb_layers - 1):
                fc = tf1.layers.dense(
                    fc, self.nb_units,
                    activation=None,
                    name=f"fc_{i}"
                )
                fc = tf1.layers.batch_normalization(fc, training=True)
                fc = tf1.nn.tanh(fc)

            output = tf1.layers.dense(
                fc, 1,
                activation=None,
                name="output"
            )
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]

class Discriminator_img(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256, dataset='mnist'):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.dataset = dataset

    def __call__(self, z, reuse=True):
        with tf1.variable_scope(self.name, reuse=reuse):
            bs = tf1.shape(z)[0]
            x = z[:, :self.input_dim]
            y = z[:, self.input_dim:]
            y.set_shape([None, 10])
            yb = tf1.reshape(y, shape=[bs, 1, 1, 10])
            if self.dataset == "mnist":
                x = tf1.reshape(x, [bs, 28, 28, 1])
            elif self.dataset == "cifar10":
                x = tf1.reshape(x, [bs, 32, 32, 3])
            x = conv_cond_concat(x, yb)
            conv = tf1.layers.conv2d(
                x, 32, [4, 4], [2, 2],
                activation=None,
                name="conv_initial"
            )
            conv = leaky_relu(conv)
            conv = conv_cond_concat(conv, yb)
            for i in range(self.nb_layers - 1):
                conv = tf1.layers.conv2d(
                    conv, 64, [4, 4], [2, 2],
                    activation=None,
                    name=f"conv_{i}"
                )
                conv = tf1.layers.batch_normalization(conv, training=True)
                conv = leaky_relu(conv)

            fc = tf1.layers.flatten(conv)
            fc = tf1.concat([fc, y], axis=1)
            fc = tf1.layers.dense(
                fc, 1024,
                activation=None,
                name="fc_1"
            )
            fc = tf1.layers.batch_normalization(fc, training=True)
            fc = leaky_relu(fc)

            fc = tf1.concat([fc, y], axis=1)
            output = tf1.layers.dense(
                fc, 1,
                activation=None,
                name="output"
            )
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]

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
        with tf1.variable_scope(self.name, reuse=reuse):
            bs = tf1.shape(z)[0]
            y = z[:, -10:]
            yb = tf1.reshape(y, shape=[bs, 1, 1, 10])
            fc = tf1.layers.dense(
                z, 1024,
                activation=None,
                name="fc_initial"
            )
            fc = tf1.layers.batch_normalization(fc, training=self.is_training)
            fc = leaky_relu(fc)
            fc = tf1.concat([fc, y], 1)

            if self.dataset == 'mnist':
                fc = tf1.layers.dense(
                    fc, 7 * 7 * 128,
                    activation=None,
                    name="fc_mnist"
                )
                fc = tf1.reshape(fc, [bs, 7, 7, 128])
            elif self.dataset == 'cifar10':
                fc = tf1.layers.dense(
                    fc, 8 * 8 * 128,
                    activation=None,
                    name="fc_cifar10"
                )
                fc = tf1.reshape(fc, [bs, 8, 8, 128])
            fc = tf1.layers.batch_normalization(fc, training=self.is_training)
            fc = leaky_relu(fc)
            fc = conv_cond_concat(fc, yb)

            conv = tf1.layers.conv2d_transpose(
                fc, 64, [4, 4], [2, 2],
                activation=None,
                name="deconv_1"
            )
            conv = tf1.layers.batch_normalization(conv, training=self.is_training)
            conv = leaky_relu(conv)

            if self.dataset == 'mnist':
                output = tf1.layers.conv2d_transpose(
                    conv, 1, [4, 4], [2, 2],
                    activation=tf1.nn.sigmoid,
                    name="output_mnist"
                )
                output = tf1.reshape(output, [bs, -1])
            elif self.dataset == 'cifar10':
                output = tf1.layers.conv2d_transpose(
                    conv, 3, [4, 4], [2, 2],
                    activation=tf1.nn.sigmoid,
                    name="output_cifar10"
                )
                output = tf1.reshape(output, [bs, -1])
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]

if __name__ == '__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian

    g_net = Generator_img(input_dim=10, output_dim=784, name='g_net', nb_layers=2, nb_units=256, dataset='mnist')
    d_net = Discriminator_img(input_dim=784 + 10, name='d_net', nb_layers=2, nb_units=256, dataset='mnist')
    x = tf1.placeholder(tf1.float32, [32, 10], name='x')
    y = g_net(x, reuse=False)
    z = tf1.placeholder(tf1.float32, [32, 784 + 10], name='z')
    dz = d_net(z, reuse=False)

    print(z, dz)
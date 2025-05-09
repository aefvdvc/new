import tensorflow as tf
import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior()


def leaky_relu(x, alpha=0.2):
    return tf1.maximum(tf1.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, input_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc = tf1.layers.dense(
                x, self.nb_units,
                activation=None
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf1.layers.dense(
                    fc, self.nb_units,
                    activation=None
                )
                fc = tf1.layers.batch_normalization(fc)
                fc = tf1.nn.tanh(fc)

            output = tf1.layers.dense(
                fc, 1,
                activation=None
            )
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        with tf1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tf1.layers.dense(
                z, self.nb_units,
                activation=None
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf1.layers.dense(
                    fc, self.nb_units,
                    activation=None
                )
                fc = leaky_relu(fc)

            output = tf1.layers.dense(
                fc, self.output_dim,
                activation=None
            )
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]


class Generator_resnet(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def residual_block(self, x, dim):
        e = tf1.layers.dense(x, self.nb_units, activation=None)
        e = leaky_relu(e)
        e = tf1.layers.dense(x, dim, activation=None)
        e = leaky_relu(e)
        return x + e

    def __call__(self, z, reuse=True):
        with tf1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tf1.layers.dense(
                z, self.nb_units // 2,
                activation=None
            )
            fc = leaky_relu(fc)
            fc = tf1.layers.dense(
                z, self.nb_units,
                activation=None
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = self.residual_block(fc, self.nb_units)

            fc = tf1.layers.dense(
                z, self.nb_units,
                activation=None
            )
            fc = leaky_relu(fc)
            fc = tf1.layers.dense(
                z, self.nb_units // 2,
                activation=None
            )
            fc = leaky_relu(fc)

            output = tf1.layers.dense(
                fc, self.output_dim,
                activation=None
            )
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]


class Generator_res(object):
    def __init__(self, input_dim, label_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        z_latent = z[:, :self.input_dim]
        z_label = z[:, self.input_dim:]
        with tf1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tf1.layers.dense(
                z, self.nb_units,
                activation=None
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf1.concat([fc, z_label], axis=1)
                fc = tf1.layers.dense(
                    fc, self.nb_units,
                    activation=None
                )
                fc = leaky_relu(fc)
            output = tf1.layers.dense(
                fc, self.output_dim,
                activation=None
            )
            return output

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]


class Generator_Bayes(object):
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2, nb_units=256,
                 constrain=False):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.constrain = constrain

    def __call__(self, z, reuse=True):
        with tf1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            z1 = z[:, :self.input_dim1]
            z2 = z[:, self.input_dim1:]

            fc1 = tf1.layers.dense(
                z1, self.nb_units,
                activation=None,
                name='z1_0'
            )
            fc1 = leaky_relu(fc1)

            fc2 = tf1.layers.dense(
                z, self.nb_units,
                activation=None,
                name='z2_0'
            )
            fc2 = leaky_relu(fc2)

            for i in range(self.nb_layers - 1):
                z = fc1
                fc1 = tf1.layers.dense(
                    fc1, self.nb_units,
                    activation=None,
                    name='z1_%d' % (i + 1)
                )
                fc1 = leaky_relu(fc1)

                fc2 = tf1.concat([z, fc2], axis=1)
                fc2 = tf1.layers.dense(
                    fc2, self.nb_units,
                    activation=None,
                    name='z2_%d' % (i + 1)
                )
                fc2 = leaky_relu(fc2)

            output1 = tf1.layers.dense(
                fc1, self.output_dim1,
                activation=None,
                name='z1_last'
            )
            fc2 = tf1.concat([fc1, fc2], axis=1)
            output2 = tf1.layers.dense(
                fc2, self.output_dim2,
                activation=None,
                name='z2_last'
            )
            if self.constrain:
                output2_phi = output2[:, 1:2]
                output2_sigma2 = output2[:, 2:3]
                output2_nu = output2[:, 3:4]
                output2_phi = tf1.tanh(output2_phi)
                output2_sigma2 = tf1.abs(output2_sigma2)
                output2 = tf1.concat([output2[:, 0:1], output2_phi, output2_sigma2, output2_nu, output2[:, -2:]],
                                     axis=1)
            return [output1, output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf1.global_variables() if self.name + '/z1' in var.name]
        vars_z2 = [var for var in tf1.global_variables() if self.name + '/z2' in var.name]
        all_vars = [var for var in tf1.global_variables() if self.name in var.name]
        return [vars_z1, vars_z2, all_vars]


class Generator_PCN(object):
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2, nb_units=256):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        with tf1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            z1 = z[:, :self.input_dim1]
            z2 = z[:, self.input_dim1:]

            fc1 = tf1.layers.dense(
                z1, self.nb_units,
                activation=None,
                name='z1_0'
            )
            fc1 = leaky_relu(fc1)
            fc_cross = tf1.layers.dense(
                z1, self.nb_units,
                kernel_initializer=tf1.zeros_initializer(),
                bias_initializer=None,
                activation=None,
                name='zc_0'
            )
            fc_cross = leaky_relu(fc_cross)

            fc2 = tf1.layers.dense(
                z2, self.nb_units,
                activation=None,
                name='z2_0'
            )
            fc2 = leaky_relu(fc2)
            fc2 = tf1.add(fc2, fc_cross)

            for i in range(self.nb_layers - 1):
                z = fc1
                fc1 = tf1.layers.dense(
                    fc1, self.nb_units,
                    activation=None,
                    name='z1_%d' % (i + 1)
                )
                fc1 = leaky_relu(fc1)

                fc_cross = tf1.layers.dense(
                    z, self.nb_units,
                    activation=None,
                    kernel_initializer=tf1.zeros_initializer(),
                    bias_initializer=None,
                    name='zc_%d' % (i + 1)
                )
                fc_cross = leaky_relu(fc_cross)

                fc2 = tf1.layers.dense(
                    fc2, self.nb_units,
                    activation=None,
                    name='z2_%d' % (i + 1)
                )
                fc2 = leaky_relu(fc2)
                fc2 = tf1.add(fc2, fc_cross)

            output1 = tf1.layers.dense(
                fc1, self.output_dim1,
                activation=None,
                name='z1_last'
            )
            output_cross = tf1.layers.dense(
                fc1, self.output_dim2,
                activation=None,
                kernel_initializer=tf1.zeros_initializer(),
                name='zc_last'
            )

            output2 = tf1.layers.dense(
                fc2, self.output_dim2,
                activation=None,
                name='z2_last'
            )
            output2 = tf1.add(output2, output_cross)
            return [output1, output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf1.global_variables() if self.name + '/z1' in var.name]
        vars_z2 = [var for var in tf1.global_variables() if self.name + '/z2' in var.name]
        vars_zc = [var for var in tf1.global_variables() if self.name + '/zc' in var.name]
        return [vars_z1, vars_z2, vars_zc]


class Encoder(object):
    def __init__(self, input_dim, output_dim, feat_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, x, reuse=True):
        with tf1.variable_scope(self.name, reuse=tf1.AUTO_REUSE) as vs:
            fc = tf1.layers.dense(
                x, self.nb_units,
                activation=None
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf1.layers.dense(
                    fc, self.nb_units,
                    activation=None
                )
                fc = leaky_relu(fc)

            output = tf1.layers.dense(
                fc, self.output_dim,
                activation=None
            )
            logits = output[:, self.feat_dim:]
            y = tf1.nn.softmax(logits)
            return output[:, 0:self.feat_dim], y, logits

    @property
    def vars(self):
        return [var for var in tf1.global_variables() if self.name in var.name]


if __name__ == '__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
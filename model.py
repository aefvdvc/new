import tensorflow as tf
#�����ֻ����x
# G ������� 10 ��ȫ���Ӳ㣬ÿ���� 512 �����ؽڵ㣬
# �� H ������� 10 ��ȫ���Ӳ㣬ÿ���� 256 �����ؽڵ㡣
# Dx ��������ĸ�ȫ���Ӳ㣬ÿ���� 256 �����ؽڵ㣬
# �� Dz �����������ȫ���Ӳ㣬ÿ���� 128 �����ص�Ԫ��
# leaky_relu ���������Ϊÿ�����ز��еķ����Ա任

#x>0 x | x<=0 ��x
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)
    # return tf.maximum(0.0, x)
    # return tf.nn.tanh(x)
    # return tf.nn.elu(x)

#�����->�� Leaky ReLU->���ز�*1-> Batch Normalization �� Tanh->�����
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

            #0.0 fc = tcl.fully_connected(
            #     x, self.nb_units,
            #     # weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #     activation_fn=tf.identity
            # )
            fc = tf.layers.dense(
                x, self.nb_units,
                # kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = tf.layers.dense(
                    fc, self.nb_units,
                    # kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation=tf.identity
                )

                fc = tf.layers.batch_normalization(fc)

                # fc = leaky_relu(fc)  # �����Ҫ Leaky ReLU������ʹ�� tf.nn.leaky_relu

                fc = tf.nn.tanh(fc)

            output = tf.layers.dense(
                fc, 1,
                # kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=tf.identity
            )
            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# fcnͬ��
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

#���ڲв������������z��������
class Generator_resnet(object):
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
    #�в��
    def residual_block(self, x, dim):
        e = tf.layers.dense(x, self.nb_units, activation=tf.identity)
        e = leaky_relu(e)
        e = tf.layers.dense(x, dim, activation=tf.identity)
        e = leaky_relu(e)
        return x + e

    def __call__(self, z, reuse=True):
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = tf.layers.dense(
                z, self.nb_units / 2,
                activation=tf.identity
            )
            fc = leaky_relu(fc)
            fc = tf.layers.dense(
                z, self.nb_units,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                # weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation=tf.identity
            )
            fc = leaky_relu(fc)
            for _ in range(self.nb_layers - 1):
                fc = self.residual_block(fc, self.nb_units)

            fc = tf.layers.dense(
                z, self.nb_units,
                # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                # weights_regularizer=tcl.l2_regularizer(2.5e-5),
                activation=tf.identity
            )
            fc = leaky_relu(fc)
            fc = tf.layers.dense(
                z, self.nb_units / 2,
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

#skip connection
class Generator_res(object):  # skip connection
    def __init__(self, input_dim, label_dim, output_dim, name, nb_layers=2, nb_units=256):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        z_latent = z[:, :self.input_dim]
        z_label = z[:, self.input_dim:]
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
            #��Ծ����ÿһ���У���ǰ����� fc ���� z_label ƴ�ӳ��µ����룬ͨ�� tf.concat([fc, z_label], axis=1) ����ƴ��
            for _ in range(self.nb_layers - 1):
                fc = tf.concat([fc, z_label], axis=1)
                fc = tf.layers.dense(
                    fc, self.nb_units,
                    # weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    # weights_regularizer=tcl.l2_regularizer(2.5e-5),
                    activation=tf.identity
                )
                fc = leaky_relu(fc)
            # fc = tf.concat([fc,z_label],axis=1)
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

#��Ҷ˹�ƶ�
class Generator_Bayes(object):  # y1,y2 = f(x1,x2) where p(y1|x1,x2) = p(y1|x1)
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
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            z1 = z[:, :self.input_dim1]
            z2 = z[:, self.input_dim1:]

            fc1 = tf.layers.dense(
                z1, self.nb_units,
                activation=tf.identity,
                name='z1_0'
            )

            fc1 = leaky_relu(fc1)

            fc2 = tf.layers.dense(
                z, self.nb_units,
                activation=tf.identity,
                name='z2_0'
            )

            fc2 = leaky_relu(fc2)

            for i in range(self.nb_layers - 1):
                z = fc1
                fc1 = tf.layers.dense(
                    fc1, self.nb_units,
                    activation=tf.identity,
                    name='z1_%d' % (i + 1)
                )
                fc1 = leaky_relu(fc1)

                fc2 = tf.concat([z, fc2], axis=1)
                fc2 = tf.layers.dense(
                    fc2, self.nb_units,
                    activation=tf.identity,
                    name='z2_%d' % (i + 1)
                )
                fc2 = leaky_relu(fc2)

            output1 = tf.layers.dense(
                fc1, self.output_dim1,
                activation=tf.identity,
                name='z1_last'
            )

            fc2 = tf.concat([fc1, fc2], axis=1)

            output2 = tf.layers.dense(
                fc2, self.output_dim2,
                activation=tf.identity,
                name='z2_last'
            )

            if self.constrain:
                #�ֱ��ǵڶ���-������
                output2_phi = output2[:, 1:2]
                output2_sigma2 = output2[:, 2:3]
                output2_nu = output2[:, 3:4]
                # -1 �� 1
                output2_phi = tf.tanh(output2_phi)
                output2_sigma2 = tf.abs(output2_sigma2)
                # output2_nu = tf.abs(output2_nu)
                #������еĲ��֣������޸ĺ�� output2_phi��output2_sigma2��output2_nu ��δ�޸ĵĲ��֣�������ƴ�������������µ� output2��
                #ƴ�Ӳ���ͨ�� axis=1 ����ɣ������еķ�����Щ���ֺϲ���
                output2 = tf.concat([output2[:, 0:1], output2_phi, output2_sigma2, output2_nu, output2[:, -2:]], axis=1)
            return [output1, output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf.global_variables() if self.name + '/z1' in var.name]
        vars_z2 = [var for var in tf.global_variables() if self.name + '/z2' in var.name]
        all_vars = [var for var in tf.global_variables() if self.name in var.name]
        return [vars_z1, vars_z2, all_vars]


class Generator_PCN(object):  # partially connected network, z1<--f1(z1), z2<--f2(z1,z2)
    def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2, name, nb_layers=2, nb_units=256):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units

    def __call__(self, z, reuse=True):
        # with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs:
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            z1 = z[:, :self.input_dim1]
            z2 = z[:, self.input_dim1:]

            fc1 = tf.layers.dense(
                z1, self.nb_units,
                activation=tf.identity,
                name='z1_0'
            )

            fc1 = leaky_relu(fc1)
            # cross connections
            fc_cross = tf.layers.dense(
                z1, self.nb_units,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=None,
                activation=tf.identity,
                name='zc_0'
            )

            fc_cross = leaky_relu(fc_cross)

            fc2 = tf.layers.dense(
                z2, self.nb_units,
                activation=tf.identity,
                name='z2_0'
            )

            fc2 = leaky_relu(fc2)
            fc2 = tf.add(fc2, fc_cross)
            # fc2 = tf.concat([fc2,fc_cross],axis=1)

            for i in range(self.nb_layers - 1):
                z = fc1
                fc1 = tf.layers.dense(
                    fc1, self.nb_units,
                    activation=tf.identity,
                    name='z1_%d' % (i + 1)
                )
                fc1 = leaky_relu(fc1)

                # cross connection
                fc_cross = tf.layers.dense(
                    z, self.nb_units,
                    activation=tf.identity,
                    kernel_initializer=tf.zeros_initializer(),
                    bias_initializer=None,
                    name='zc_%d' % (i + 1)
                )
                fc_cross = leaky_relu(fc_cross)

                fc2 = tf.layers.dense(
                    fc2, self.nb_units,
                    activation=tf.identity,
                    name='z2_%d' % (i + 1)
                )
                fc2 = leaky_relu(fc2)
                fc2 = tf.add(fc2, fc_cross)
                # fc2 = tf.concat([fc2,fc_cross],axis=1)

            output1 = tf.layers.dense(
                fc1, self.output_dim1,
                activation=tf.identity,
                name='z1_last'
            )

            # cross connection
            output_cross = tf.layers.dense(
                fc1, self.output_dim2,
                activation=tf.identity,
                kernel_initializer=tf.zeros_initializer(),
                name='zc_last'
            )

            output2 = tf.layers.dense(
                fc2, self.output_dim2,
                activation=tf.identity,
                name='z2_last'
            )

            output2 = tf.add(output2, output_cross)
            return [output1, output2]

    @property
    def vars(self):
        vars_z1 = [var for var in tf.global_variables() if self.name + '/z1' in var.name]
        vars_z2 = [var for var in tf.global_variables() if self.name + '/z2' in var.name]
        vars_zc = [var for var in tf.global_variables() if self.name + '/zc' in var.name]
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
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            # with tf.variable_scope(self.name) as vs:
            #     if reuse:
            #         vs.reuse_variables()
            fc = tf.layers.dense(
                x, self.nb_units,
                activation=tf.identity,
                # kernel_initializer=tf.random_normal_initializer(stddev=0.02)
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


if __name__ == '__main__':
    import numpy as np
    import time
    from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian

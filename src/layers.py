import tensorflow as tf

### modified from DeepFRI

class GraphConv1(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(GraphConv1, self).__init__(**kwargs) 

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        D_hat = tf.linalg.diag(1./(eps + tf.math.sqrt(tf.reduce_sum(A_hat, axis=2))))
        F=tf.matmul(D_hat, A_hat)
        rrr=tf.matmul(tf.matmul(D_hat, A_hat), D_hat)
        return tf.matmul(tf.matmul(D_hat, A_hat), D_hat)

    def call(self, inputs):
        output = tf.keras.backend.batch_dot(self._normalize(inputs[1]),inputs[0])
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config
        
class GraphConv2(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(GraphConv2, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        D_hat = tf.linalg.diag(1./(eps + tf.math.sqrt(tf.reduce_sum(A_hat, axis=2))))
        F=tf.matmul(D_hat, A_hat)
        rrr=tf.matmul(tf.matmul(D_hat, A_hat), D_hat)
        return tf.matmul(tf.matmul(D_hat, A_hat), D_hat)

    def call(self, inputs2):
        output = tf.keras.backend.batch_dot(self._normalize(inputs2[1]),inputs2[0])
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class Reshape_out(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(Reshape_out, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.output_layer = tf.keras.layers.Dense(2*output_dim)
        self.reshape = tf.keras.layers.Reshape(target_shape=(output_dim, 2))
        self.softmax = tf.keras.layers.Softmax(axis=-1, name='labels')

    def call(self, x):
        x = self.output_layer(x)
        x = self.reshape(x)
        out = self.softmax(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config


class get_sum(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(get_sum, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        x_pool = tf.reduce_sum(x, axis=self.axis)
        return x_pool

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis,
        })
        return config

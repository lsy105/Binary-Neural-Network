from keras import backend as K
import keras.layers as kl
import keras
import tensorflow as tf
from tensorflow.python.framework import ops

def Binarization(x):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
        return tf.sign(x, name="Sign")

class Binary_Conv2D(kl.Layer):
    def __init__(self, 
                 kernel_num,
                 kernel_size,
                 binarize_input=True,
                 stride=1,
                 pad=0,
                 use_bias=True,
                 kernel_initializer=keras.initializers.glorot_uniform(),
                 bias_initializer=keras.initializers.Zeros(),
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Binary_Conv2D, self).__init__(**kwargs)
        self.kernel_num_ = kernel_num
        self.kernel_size_ = kernel_size
        self.kernel_initializer_ = kernel_initializer
        self.pad_ = pad
        self.stride_ = stride
        self.binarize_input_ = binarize_input

    def build(self, input_shape):
        print(input_shape)
        # Create float point weights.
        self.float_kernel = self.add_weight(name='float_kernel', 
                                            shape=(self.kernel_size_, self.kernel_size_, 
                                                   input_shape[-1], self.kernel_num_),
                                            initializer=self.kernel_initializer_,
                                            trainable=True)
        # Create binary weights 
        self.binary_kernel = self.add_weight(name='binary_kernel', 
                                             shape=(self.kernel_size_, self.kernel_size_, 
                                                    input_shape[-1], self.kernel_num_),
                                             initializer= self.kernel_initializer_,
                                             trainable=False)
        super(Binary_Conv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.binarize_input_ == True:
            x = Binarization(x)
        self.binary_kernel = Binarization(K.clip(self.float_kernel, -1.0, 1.0))
        x = K.spatial_2d_padding(x, padding=((self.pad_, self.pad_), (self.pad_, self.pad_)))
        return K.conv2d(x, self.binary_kernel, strides=(self.stride_, self.stride_), padding='valid', data_format='channels_last')

    def compute_output_shape(self, input_shape):
        W = int((input_shape[1] - self.kernel_size_ + 2 * self.pad_) / self.stride_ + 1)
        return (input_shape[0], W, W, self.kernel_num_)


class Binary_Dense(kl.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim_ = output_dim
        super(Binary_Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.float_kernel = self.add_weight(name='float_Dense',
                                            shape=(input_shape[1], self.output_dim_),
                                            initializer=keras.initializers.glorot_uniform(seed=None),
                                            trainable=True)
        self.binary_kernel = self.add_weight(name='binary_Dense',
                                             shape=(input_shape[1], self.output_dim_),
                                             initializer=keras.initializers.glorot_uniform(seed=None),
                                             trainable=False)
        super(Binary_Dense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = Binarization(x)
        self.binary_kernel = Binarization(K.clip(self.float_kernel, -1.0, 1.0))
        return K.dot(x, self.binary_kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim_)

class HardTanh(kl.Layer):
    def __init__(self, **kwargs):
        super(HardTanh, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HardTanh, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.clip(x, -1.0, 1.0)

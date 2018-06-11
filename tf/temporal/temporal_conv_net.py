import tensorflow as tf
from temporal_block import TemporalBlock
class TemporalConvNet(tf.layers.Layer):
    """Temporal Convolutional Network
    """

    def __init__(self, num_channels,
                       kernel_size = 2,
                       dropout = 0.2,
                       trainable = True,
                       name = None,
                       dtype = None,
                       activity_regularizer = None,
                       **kwargs):
            super(TemporalConvNet, self).__init__(trainable = trainable,
                                                  dtype = dtype,
                                                  activity_regularizer = activity_regularizer,
                                                  name = name,
                                                  **kwargs)
            self.layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                out_channels = num_channels[i]
                self.layers.append(
                    TemporalBlock(out_channels,
                                  kernel_size,
                                  strides = 1,
                                  dilation_rate = dilation_size,
                                  dropout = dropout,
                                  name = "tblock_{}".format(i))
                )

    def call(self, inputs, training = True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training = training)
        return outputs

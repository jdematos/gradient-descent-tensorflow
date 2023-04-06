import tensorflow as tf


class DigitClassifierModel(tf.keras.Model):
    def __init__(self, input_shape, num_nds, num_cls):
        super(DigitClassifierModel, self).__init__()

        # Define layers
        self.flatten_layer = tf.keras.layers.Flatten(input_shape=input_shape)
        self.dense_layer_1 = tf.keras.layers.Dense(units=num_nds, activation=tf.nn.relu)
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        self.dense_layer_2 = tf.keras.layers.Dense(
            units=num_cls, activation=tf.nn.softmax
        )

    def call(self, inputs, training=False):
        # Define the computation performed in the forward pass
        x = self.flatten_layer(inputs)
        x = self.dense_layer_1(x)
        if training:
            x = self.dropout_layer(x, training=True)
        x = self.dense_layer_2(x)
        return x

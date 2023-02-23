"""
This file is meant as a rough test of implementing custom optimizers in tensorflow.
A cleaner and more organized repo will follow this first version.

"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.metrics import Accuracy, Mean, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Optimizer
from tensorflow.nn import relu, softmax
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops, state_ops

# from keras.optimizers.optimizer_experimental import optimizer

# Compiles into a callable tensorflow graph
@tf.function
def train_step(images,labels):
    # Record operations for auto differentiation
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Accumulate values
    train_loss(loss)
    train_accuracy(labels, predictions)

# Compiles into a callable tensorflow graph
@tf.function
def test_step(images,labels):
    predictions = model(images)
    loss = loss_function(labels, predictions)
    # Accumulate values
    test_loss(loss)
    test_accuracy(labels, predictions)
        

class DigitClassifierModel(tf.keras.Model):
    
    def __init__(self, input_shape=None, num_nds= None, num_cls=None):
        super(DigitClassifierModel, self).__init__()
        self.input_layer = Flatten(input_shape=input_shape)
        self.hidden_layer = Dense(units=num_nds, activation=relu)
        self.output_layer = Dense(units=num_cls, activation=softmax)
    
    def call(self, value):
        value = self.input_layer(value)
        value = self.hidden_layer(value)
        value = self.output_layer(value)
        return value 


class BatchGradientDecent(Optimizer):

    # def __init__(self, lr=1e-2, **kwargs):
    #     super(BatchGradientDecent, self).__init__(**kwargs)
    #     with backend.name_scope(self.__class__.__name__):
    #         self._learning_rate = backend.variable(lr, name="lr")
    #         self.iterations = backend.variable(0, dtype="int64", name="iterations")

    # def _create_all_weights(self, params):
    #     # Set parent class variable
    #     self.weights = [
    #         backend.zeros(
    #             backend.int_shape(p), 
    #             dtype=backend.dtype(p)
    #         ) 
    #         for p in params
    #     ]
    #     return self.weights
    
    # def get_updates(self, loss, params):
    #     grads = self.get_gradients(loss, params)
    #     accumulators = self._create_all_weights(params)
    #     self.updates = [state_ops.assign_add(self.iterations, 1)]
    #     # Set parent class variable
    #     lr = self.lr
    #     for p, g, acc in zip(params, grads, accumulators):
    #         new_p = p + (lr * g)
    #         self.updates.append(state_ops.assign(p, new_p))
    #     return self.updates
    
    # def get_config(self):
    #     config = {
    #         "lr":float(backend.get_value(self.lr))
    #     }
    #     base_config = super(BatchGradientDecent, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items())) 

    """ - `build`: Create your optimizer-related variables, such as `momentums` in
        SGD optimizer.
      - `update_step`: Implement your optimizer's updating logic.
      - `get_config`: serialization of the optimizer, include all hyper
        parameters.
    """
    def __init__(self, lr=1e-2, **kwargs):
         super(BatchGradientDecent, self).__init__(name="BDG", **kwargs)
         self._learning_rate = self._build_learning_rate(lr)

    def build(self, var_list):
        super(BatchGradientDecent, self).build(var_list)
        if hasattr(self, "_built") and self._built: return
        self.accumulators=[]
        for var in var_list:
            self.accumulators.append(
                self.add_variable_from_reference(
                    model_variable=var,
                    variable_name="acc"
                )
            )
        self._built = True
        
    def update_step(self, gradient, variable):
        lr = tf.cast(x=self._learning_rate, dtype=variable.dtype, name="cast_lr")
        var_key = self._var_key(variable)
        w = self.accumulators[self._index_dict[var_key]]
        variable.assign_add((gradient*-lr)+w)

    def get_config(self):
        base_config = super(BatchGradientDecent, self).get_config()
        base_config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate),
            }
        )


if __name__ == "__main__":

    EPOCHS = 5
    BATCH_SIZE = 32
    SHUFFLE_NUM = 10000

    # Get data
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    print(type(x_train))
    # Normalize data
    x_train = np.divide(x_train, float(255))
    x_test = np.divide(x_test, float(255))
    # Batch and shuffle data
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(SHUFFLE_NUM).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    # Instantiate the model
    model = DigitClassifierModel(input_shape=(28,28), num_nds=512, num_cls=10)

    # Optimizer and loss object
    # `from_logits` set to false because the output is produced by a softmax activation function
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = BatchGradientDecent() #tf.keras.optimizers.Adam()

    # Accumulates values over epochs
    train_loss = Mean(name="train_loss")
    train_accuracy = SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = Mean(name="test_loss")
    test_accuracy = SparseCategoricalAccuracy(name="test_accuracy")


    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for train_images, train_labels in train_ds:
            train_step(train_images, train_labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )


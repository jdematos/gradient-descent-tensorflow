import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy


class Trainer:
    def __init__(self, train_ds, test_ds, epochs, model, optimizer):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.epochs = epochs
        self.optimizer = optimizer
        self.model = model

        # Define the loss and metrics
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=False)
        self.train_loss = Mean(name="train_loss")
        self.train_acc = SparseCategoricalAccuracy(name="train_accuracy")
        self.test_loss = Mean(name="test_loss")
        self.test_acc = SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # Make predictions on the current batch of data
            predictions = self.model(images)
            # Compute the loss value for this batch
            loss = self.loss_fn(labels, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update the training loss and accuracy
        self.train_loss(loss)
        self.train_acc(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # Make predictions on the test set
        predictions = self.model(images)
        # Compute the loss value for this batch
        loss = self.loss_fn(labels, predictions)

        # Update the test loss and accuracy
        self.test_loss(loss)
        self.test_acc(labels, predictions)

    def train(self):
        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.test_loss.reset_states()
            self.test_acc.reset_states()

            # Train the model on the training data
            for images, labels in self.train_ds:
                self.train_step(images, labels)

            # Evaluate the model on the test data
            for images, labels in self.test_ds:
                self.test_step(images, labels)

            # Print the metrics for this epoch
            template = (
                "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
            )
            print(
                template.format(
                    epoch + 1,
                    self.train_loss.result(),
                    self.train_acc.result() * 100,
                    self.test_loss.result(),
                    self.test_acc.result() * 100,
                )
            )

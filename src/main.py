import tensorflow as tf

from models import DigitClassifierModel
from models import Trainer  # Specifically for digit classifier
from optimizers import BGDOptimizer


if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 32
    SHUFFLE_NUM = 10000

    # Get data
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Batch and shuffle data
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(SHUFFLE_NUM)
        .batch(BATCH_SIZE)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
        BATCH_SIZE
    )

    # Instantiate the model
    model = DigitClassifierModel(input_shape=(28, 28), num_nds=512, num_cls=10)
    # Instantiate the optimizer
    optimizer = BGDOptimizer(lr=1e-2)

    # Run trainer
    Trainer(train_ds, test_ds, EPOCHS, model, optimizer).train()

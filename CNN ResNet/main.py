import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import shutil
import h5py
import pickle
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Conv2D, Lambda, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers.schedules import ExponentialDecay
from pathlib import Path

global_dataset = None

dataset_file = Path(r"C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/dane10_compressed.h5") # set the path to training dataset file
save_path = Path(r"C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska_results")

if save_path.exists() and save_path.is_dir():
    shutil.rmtree(save_path)  # remove directory
save_path.mkdir(parents=True, exist_ok=True)
print(f"Directory '{save_path}' is ready")


def get_dataset_size(file_path):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f['inputs']
        return dataset.shape[0]


def model_configuration():
    global global_dataset  # use the global dataset variable

    # Ensure the dataset is loaded
    if global_dataset is None:
        load_dataset()

    # Generic configuration
    width, height, channels = None, None, 1
    batch_size = 1
    validation_split = 0.2
    verbose = 1
    n = 3  # number of residual blocks in a single group
    init_fm_dim = 16  # initial number of feature maps; doubles as the feature map size halves
    shortcut_type = "identity"  # shortcut type: "identity" or "projection"

    num_samples = get_dataset_size(dataset_file)
    train_size = (1 - validation_split) * num_samples
    val_size = validation_split * num_samples

    # Calculate parameters
    maximum_number_iterations = 80
    print(f"Maximum number of iterations: {maximum_number_iterations}")
    steps_per_epoch = np.ceil(train_size / batch_size).astype(int)
    val_steps_per_epoch = np.ceil(val_size / batch_size).astype(int)
    epochs = tensorflow.cast(
        tensorflow.math.ceil(maximum_number_iterations / steps_per_epoch),
        dtype=tensorflow.int64
    )

    # Define the loss function
    loss = tensorflow.keras.losses.MeanSquaredError()

    # Set layer initializer
    initializer = tensorflow.keras.initializers.HeNormal()

    # Define the optimizer
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # TensorBoard callback for monitoring
    tensorboard = TensorBoard(
        save_path / "logs",
        histogram_freq=1,
        write_steps_per_second=True,
        write_images=True,
        update_freq='epoch'
    )

    # Model checkpoint callback for saving weights
    checkpoint = ModelCheckpoint(
        save_path / "epoch_{epoch:02d}_model_checkpoint.keras",
        save_freq="epoch"
    )

    def add_data_point(line, new_x, new_y):
        """
        Adds a new data point (new_x, new_y) to an existing Matplotlib line plot.

        Parameters:
        - line: The Line2D object from plt.plot() (used to update the plot).
        - new_x: New x-coordinate.
        - new_y: New y-coordinate.
        """
        # Get current data
        x_data, y_data = line.get_xdata(), line.get_ydata()

        # Append new data point
        x_data = list(x_data) + [new_x]
        y_data = list(y_data) + [new_y]

        # Update the line data
        line.set_data(x_data, y_data)

        # Adjust axes limits (optional)
        ax = line.axes
        ax.relim()  # Recompute limits
        ax.autoscale_view()  # Autoscale to fit new data

        # Redraw the figure
        plt.draw()
        plt.pause(0.1)  # Pause to update UI

    class SaveBatchLoss(Callback):
        def on_train_begin(self, logs={}):
            self.train_losses = []
            plt.ion()  # Turn on interactive mode
            # Create an initial plot
            fig, ax = plt.subplots()
            ax.set_title("Blue - train loss; Red - val loss")
            self.ax = ax
            self.line_train_loss, = ax.plot([], [], 'bo-',label="Training loss")  # Blue dots with line
            self.line_val_loss, = ax.plot([], [], 'ro-',label="Validation loss")  # Blue dots with line

        def on_epoch_end(self, epoch, logs=None):
            # Create an initial plot
            add_data_point(self.line_train_loss, epoch, logs.get('loss'))
            add_data_point(self.line_val_loss, epoch, logs.get('val_loss'))

        def on_train_batch_end(self, batch, logs={}):
            self.train_losses.append(logs.get('loss'))

        def on_train_end(self, logs={}):
            plt.ioff()  # Turn off interactive mode
            with open(save_path/"train_loss_per_batch", "wb") as fp:  # pickling
                pickle.dump(self.train_losses, fp)

    save_batch_loss = SaveBatchLoss()

    class LearningRateLogger(Callback):
        def __init__(self):
            super().__init__()
            self.learning_rates = []

            plt.ion()  # Turn on interactive mode
            # Create an initial plot
            fig, ax = plt.subplots()
            line, = ax.plot([], [], 'bo-',label="learning rate")
            ax.set_title("Learning rate")
            self.ax = ax
            self.line = line

        def on_epoch_end(self, epoch, logs=None):
            optimizer = self.model.optimizer
            lr = float(tensorflow.keras.backend.get_value(optimizer.learning_rate))
            self.learning_rates.append(lr)
            #print(" - learning Rate = {:.3e}".format(lr))
            add_data_point(self.line, epoch, lr)

        def on_train_end(self, logs={}):
            plt.ioff()  # Turn off interactive mode
            with open(save_path/"learning_rates", "wb") as fp:  # pickling
                pickle.dump(self.learning_rates, fp)

    lr_logger = LearningRateLogger()

    # Add callbacks to a list
    callbacks = [
        tensorboard,
        checkpoint,
        save_batch_loss,
        lr_logger
    ]

    # Create configuration dictionary
    config = {
        "epochs": epochs,
        "width": width,
        "height": height,
        "dim": channels,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "verbose": verbose,
        "stack_n": n,
        "initial_num_feature_maps": init_fm_dim,
        "training_ds_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "val_steps_per_epoch": val_steps_per_epoch,
        "num_epochs": epochs,
        "loss": loss,
        "optim": optimizer,
        "initializer": initializer,
        "callbacks": callbacks,
        "shortcut_type": shortcut_type
    }
    return config


def load_dataset():
    global global_dataset

    try:
        h5f = h5py.File(dataset_file, "r")

        if 'inputs' not in h5f or 'targets' not in h5f:
            raise KeyError("Missing 'inputs' or 'targets' datasets in the HDF5 file.")

        global_dataset = {"inputs": h5f['inputs'], "targets": h5f['targets']} # use lazy loading to create an object

        num_samples = global_dataset['inputs'].shape[0]
        print(f"Loaded dataset: {num_samples} samples into global_dataset.")

    except FileNotFoundError:
        raise FileNotFoundError("The dataset file was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected data key in the file: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")


def data_generator(inputs, targets, batch_size):
    total_samples = inputs.shape[0]

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        yield batch_inputs, batch_targets


def preprocessed_dataset(config):
    global global_dataset

    if global_dataset is None:
        load_dataset()

    inputs = global_dataset["inputs"]
    targets = global_dataset["targets"]
    validation_split = config["validation_split"]
    batch_size = config["batch_size"]

    # Calculate set size
    total_samples = len(inputs)
    val_test_size = int(total_samples * validation_split)
    val_size = val_test_size // 2
    train_size = total_samples - val_test_size

    # Calculate the indices for dataset splitting
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_samples)

    # Define generators
    train_gen = lambda: data_generator(inputs[train_indices], targets[train_indices], batch_size)
    val_gen = lambda: data_generator(inputs[val_indices], targets[val_indices], batch_size)
    test_gen = lambda: data_generator(inputs[test_indices], targets[test_indices], batch_size)

    # Create TensorFlow sets
    train_dataset = tensorflow.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
            tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
        )
    )

    validation_dataset = tensorflow.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
            tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
        )
    )

    test_dataset = tensorflow.data.Dataset.from_generator(
        test_gen,
        output_signature=(
            tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
            tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
        )
    )

    # Prepare datasets
    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().prefetch(tensorflow.data.AUTOTUNE)
    validation_dataset = validation_dataset.repeat().prefetch(tensorflow.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tensorflow.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def residual_block(x, number_of_filters, config):
    initializer = config["initializer"]

    # Create skip connection
    x_skip = x

    # Perform the original mapping
    x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1, 1),
               kernel_initializer=initializer, padding="same")(x_skip)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(number_of_filters, kernel_size=(3, 3),
               kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization(axis=3)(x)

    # Add the skip connection to the regular mapping
    x = Add()([x, x_skip])

    # Nonlinearly activate the result
    x = Activation("relu")(x)

    return x


def ResidualBlocks(x, config):
    # Set initial filter size
    filter_size = config.get("initial_num_feature_maps")

    # Paper: "Then we use a stack of 6n layers (...)
    #	with 2n layers for each feature map size."
    # 6n/2n = 3, so there are always 3 groups.
    for layer_group in range(4):
        # Each block in our code has 2 weighted layers,
        # and each group has 2n such blocks,
        # so 2n/2 = n blocks per group.
        for block in range(config.get("stack_n")):
            x = residual_block(x, filter_size, config)
    # Return final layer
    return x


def ResNetPath(x, config, scale=2):
    initializer = config.get("initializer")

    assert (scale >= 1 and isinstance(scale,int))
    if scale > 1:
        x = MaxPool2D(pool_size=(scale, scale), strides=scale, padding="same")(x)
    x = ResidualBlocks(x, config)
    if scale > 1:
        x = UpSampling2D(size=(scale, scale), interpolation="bilinear")(x)
        x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),
               strides=(1, 1), kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def model_base(config, path_num=2):
    initializer = config["initializer"]

    # Define model structure
    shp = (config.get("width"), config.get("height"),
                                  config.get("dim"))
    inputs = Input(shape=shp)
    x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),
               strides=(1, 1), kernel_initializer=initializer, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if path_num <1:
        print("error; number of path has to be at least 1")
    elif path_num == 1:
        x = ResNetPath(x, config, path_num)
    else:
        path_list=[]
        for path_no in range(1,path_num+1):
            path_list.append(ResNetPath(x, config, path_no))
        x = Add()(path_list)
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
               kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization()(x)
    outputs = Activation("relu")(x)

    return inputs, outputs


def init_model(config, path_num=2):
    inputs, outputs = model_base(config, path_num)

    # Initialize and compile model
    model = Model(inputs, outputs, name=config.get("name"))

    model.compile(loss=config.get("loss"),
                  optimizer=config.get("optim"),
                  metrics=config.get("optim_additional_metrics"))
    model.summary()

    return model


def train_model(model, train_batches, validation_batches, config):
    hist_obj = model.fit(train_batches,
                  validation_data=validation_batches,
                  batch_size=config.get("batch_size"),
                  epochs=config.get("num_epochs"),
                  verbose=config.get("verbose"),
                  callbacks=config.get("callbacks"),
                  steps_per_epoch=config.get("steps_per_epoch"),
                  validation_steps=config.get("val_steps_per_epoch")
                         )
    return hist_obj


def evaluate_model(model, test_batches, config):
    score = model.evaluate(test_batches, batch_size=1, steps=config["val_steps_per_epoch"]//2, verbose=1)
    print(f'Test loss: {round(score, 4)}')


def report_training_progress(hist_obj, config, save_data=False):
    train_loss = hist_obj.history['loss']
    val_loss = hist_obj.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # Create a figure with 2 subplots (2 row, 1 columns)
    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))

    try:
        # Try to access and plot loss per iteration
        with open(save_path / "train_loss_per_batch", "rb") as fp:
            train_loss_per_batch = pickle.load(fp)
        iters = range(1, len(train_loss_per_batch) + 1)
        ax1[0].plot(iters, train_loss_per_batch,'b', label='Training Loss')

    except Exception as e:
        print(f"Error occurred: {e}")

    # --- Subplot 1: Training and Validation Loss ---
    ax1[0].plot(epochs * config["steps_per_epoch"], train_loss, 'bo', label='Training Loss per epoch')
    ax1[0].plot(epochs * config["steps_per_epoch"], val_loss, 'ro', label='Validation Loss per epoch')
    ax1[0].legend(loc='upper right')
    ax1[0].set_ylabel('Loss')
    ax1[0].set_title('Training and Validation Loss')
    ax1[0].set_xlabel('Iteration')

    try:
        # Try to access and plot loss per iteration
        with open(save_path / "learning_rates", "rb") as fp:
            learning_rates = pickle.load(fp)

        # --- Subplot 2: Learning Rate per Epoch ---
        ax1[1].plot(epochs, learning_rates, 'g-', marker='o', label='Learning Rate')
        ax1[1].legend(loc='upper right')
        ax1[1].set_ylabel('Learning Rate')
        ax1[1].set_title('Learning Rate per Epoch')
        ax1[1].set_xlabel('Epoch')

    except Exception as e:
        print(f"Error occurred: {e}")

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path / "training_and_LR_curve.png")
    plt.show()

    if save_data:
        try:
            with open(save_path / "loss.pkl", 'wb') as file:
                pickle.dump(train_loss, file)

            with open(save_path / "val_loss.pkl", 'wb') as file:
                pickle.dump(val_loss, file)

        except Exception as e:
            print(f"Error occurred: {e}")

def training_process():
    config = model_configuration()
    train_batches, validation_batches, test_batches = preprocessed_dataset(config)
    resnet = init_model(config, path_num=1)
    history = train_model(resnet, train_batches, validation_batches, config)
    report_training_progress(history, config, save_data=True)
    evaluate_model(resnet, test_batches, config)

if __name__ == "__main__":
    training_process()
"""" -------------------------------------------------------------------
    Tensorflow support
    Version 0.5
   -------------------------------------------------------------------
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pickle
import re
import string
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from logging_lib import console_logger

"""" -------------------------------------------------------------------
    Constants
   -------------------------------------------------------------------
"""

ROOT_DIRECTORY = "C/Users/thomasbayer/data/"
MODEL_ROOT = os.path.join(ROOT_DIRECTORY, "data_science/models/")
IMAGE_ROOT = os.path.join(ROOT_DIRECTORY, "data_science/images/")

NAME_TEMPLATE = "{}_{}_w{}_ns{}"
DS_NAME_TEMPLATE = "ds_{}x{}x{}x{}"
DS_W2V_NAME_TEMPLATE = "ds_{}x{}"
DATA_PATH_TEMPLATE = "../data/{}"
DEFAULT_DATA_DIR = "../data/"
DEFAULT_DATASET_DIR = "ds/"
DEFAULT_ROOT_DIR = "../stored/"
DEFAULT_MODEL_DIR = DEFAULT_ROOT_DIR + "/models/"
DEFAULT_VECTORIZER_DIR = DEFAULT_MODEL_DIR + "vectorizer/"
MODEL_PATH_TEMPLATE = "../../models/{}.keras"

"""" -------------------------------------------------------------------
    Init
   -------------------------------------------------------------------
"""


def init_gpu(enable=True):
    print("Tensorflow v", tf.__version__)
    if tf.__version__ > "2.17.0":
        print("Keras v", tf.keras.__version__)

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    print(gpus)

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(f"setting memory growth for {gpu} to {enable}")
                tf.config.experimental.set_memory_growth(gpu, enable)
            # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration
            # (memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return gpus


"""" -------------------------------------------------------------------
    Functions
   -------------------------------------------------------------------
"""


def get_adam(learning_rate=0.001):
    if tf.__version__ <= "2.16.0":
        print("Using legacy keras")
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        print("Using new keras")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer


def eval_models_opt(
        model, X_train, y_train, X_test, y_test, num_epochs=10, batch_size=128,
        optimizer=tf.keras.optimizers.Adam(name="Default: Adam")
        ):
    models = model if isinstance(model, list) else [model]
    model_logs = {}
    for (i, current_model) in enumerate(models):
        print("-- {}: model {}".format(i, current_model.name))
        model_log = eval_model_opt(
                current_model, X_train, y_train, X_test, y_test,
                num_epochs, batch_size, optimizer
                )
        model_logs[i] = model_log
    return model_logs


def eval_model_opt(
        model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics='mse',
        optimizer=tf.keras.optimizers.Adam(name="Adam"), **kwargs
        ):
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    metric_list = metrics if isinstance(metrics, list) else [metrics]
    model_logs = {}
    model.build(input_shape=X_train.shape)
    init_weights = model.get_weights()
    opti_weights = None
    opti_score = -1
    for current_optimizer in optimizers:
        print("-- optimizer {}".format(current_optimizer._name))
        if init_weights is not None:
            model.set_weights(init_weights)
        model.reset_metrics()
        model.compile(
                loss=loss,
                optimizer=current_optimizer,
                metrics=metric_list
                )
        current_model_log = eval_model(
                model, X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size, **kwargs
                )
        current_weights = model.get_weights()
        current_score = current_model_log[1][1]
        # current_model_log.append(current_weights)
        if current_score > opti_score:
            opti_score = current_score
            opti_weights = current_weights
        model_logs[current_optimizer._name] = current_model_log
    model.set_weights(opti_weights)
    return model_logs


def eval_model(
        model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, validation_data=None, plt=None, **kwargs
        ):
    # dataset = dataset.shuffle(1000).batch(batch_size)
    # model training
    tf.keras.backend.clear_session()
    # ds = dataset_train.unbatch().batch(batch_size)
    model.reset_states()
    model.reset_metrics()
    print("model {}, batch_size = {:,.0f}".format(model.name, batch_size))
    start = timer()
    model_log = model.fit(
            X_train, y_train, epochs=epochs,
            batch_size=batch_size, verbose=1, validation_data=validation_data, **kwargs
            )
    end = timer()
    print("-----")
    if validation_data is not None:
        print('Validation loss: {:.3f}'.format(model_log.history['val_loss'][-1]))
        print("-----")
    print("Time: {:,.2f} seconds".format(end - start))
    score = model.evaluate(X_test, y_test, verbose=0)
    print("-----")
    print('Test loss: {:.3f}'.format(score[0]))
    print('Test accuracy: {:.3f}'.format(score[1]))
    if plt is not None:
        plt.plot(model_log.history['loss'])
    return model_log, score


def eval_model_ds(
        model, ds_train, ds_test=None, validation_data=None, epochs=10, batch_size=128, verbose=1,
        logger=console_logger.SimpleConsoleLogger("eval_model_ds"),
        steps_per_epoch=None,
        model_action=None,
        save=False,
        filename=None,
        **kwargs
        ):
    steps_per_epoch_str = str(steps_per_epoch) if steps_per_epoch is not None else "undefined"
    if batch_size is not None:
        logger.info(
                f"train model {model.name}, batch_size = {batch_size:,.0f} , steps_per_epoch = {steps_per_epoch_str}"
                )
        dataset_train = ds_train.unbatch().batch(batch_size)
    else:
        dataset_train = ds_train
        logger.info(f"train model {model.name}, steps_per_epoch = {steps_per_epoch_str}")
    start = timer()
    model_log = model.fit(
            dataset_train, validation_data=validation_data,
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            steps_per_epoch=steps_per_epoch,
            **kwargs
            )
    end = timer()
    logger.info("Time: {:,.2f} seconds".format(end - start))
    score = None
    if ds_test is not None:
        logger.info("evaluate test set")
        score = model.evaluate(ds_test, verbose=verbose)
        if isinstance(score, (float, int)):
            logger.info('Test loss: {:.3f}'.format(score))
        else:
            logger.info('Test loss: {:.3f}'.format(score[0]))
            logger.info('Test accuracy: {:.3f}'.format(score[1]))
    if model_action is not None:
        logger.info("apply model action")
        model_action(model, **kwargs)
    if save is True:
        filename = MODEL_PATH_TEMPLATE.format(f"{model.name}_{epochs}") if filename is None else filename
    if filename is not None:
        model.save(filename)
        print(f"model saved to {filename}")
    return model_log, score


def eval_models_ds(
        model, ds_train, ds_test=None, ds_validate=None, epochs=10, batch_size=128, verbose=1,
        logger=console_logger.SimpleConsoleLogger("eval_models_ds"), model_action=None, **kwargs
        ):
    models = model if isinstance(model, list) else [model]
    model_logs = {}
    for (i, current_model) in enumerate(models):
        logger.info(f"-- {i}: model {current_model.name}")
        model_log = eval_model_ds(
                current_model, ds_train, ds_test, ds_validate, epochs=epochs, batch_size=batch_size, verbose=verbose,
                logger=logger, model_action=model_action, **kwargs
                )
        model_logs[i] = model_log
    return model_logs


def history_printer(history):
    history_dict = history.history
    for key in history_dict.keys():
        print("{} = {:.4f}".format(key, history_dict[key][-1]))


def run(
        model, X_train, y_train, X_test=None, y_test=None, epochs=2, batch_size=128,
        model_saver=None, model_name=None, verbose=False,
        **kwargs
        ):
    print("--- Model {} ---".format(model.name))
    if model_saver is not None:
        print("Weights will be saved!")
    else:
        print("Weights will not be saved!")
    # train_size = X_train.size
    # steps_per_epoch = train_size // batch_size
    # print("steps per epoch = {:,.0f}".format(steps_per_epoch))

    start = timer()
    model_log = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, **kwargs)
    end = timer()
    print("Time: {:,.0f} seconds".format(end - start))
    if model_saver is not None:
        model_saver.save_weights(model, verbose=verbose)
    print("---")
    history_printer(model_log)
    score = 0
    if X_test is not None:
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss: {:.3f}'.format(score[0]))
        print('Test accuracy: {:.3f}'.format(score[1]))
    return model_log, score


def rename_df(dataframe, suffix):
    rename_dict = {}
    for column in dataframe.columns:
        rename_dict[column] = "{}-{}".format(column, suffix)
    return dataframe.rename(columns=rename_dict), rename_dict


def join_df_optimizers(results_model):
    df = None
    for opti_name in results_model.keys():
        print(opti_name)
        history = results_model[opti_name][0].history
        print(history)
        opti_df, _ = rename_df(pd.DataFrame(history), opti_name)
        df = opti_df if df is None else df.join(opti_df)
    return df


def plot_history(model_logs):
    df = join_df_optimizers(model_logs)
    df.plot(figsize=(16, 10))
    plt.grid(True)


def run_ds(
        model, dataset, epochs, train_size=0, model_saver=None, test_function=None, model_name=None, verbose=False,
        logger=console_logger.SimpleConsoleLogger("model"), **kwargs
        ):
    models = model if isinstance(model, list) else [model]
    if model_saver is not None:
        logger.info("Weights will be saved!")
    else:
        logger.info("Weights will not be saved!")
    batch_size = next(iter(dataset))[0].shape[0]
    logger.info("batch_size = {}".format(batch_size))
    if train_size > 0:
        steps_per_epoch = train_size // batch_size
        logger.info("steps per epoch = {:,.0f}".format(steps_per_epoch))
    else:
        steps_per_epoch = None
        logger.info("steps per epoch undefined")
    model_logs = {}
    for (i, current_model) in enumerate(models):
        logger.info("--- {} of {}: model {} ---".format(i + 1, len(models), current_model.name))
        start = timer()
        model_log = current_model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)
        end = timer()
        logger.info("Time: {:,.0f} seconds".format(end - start))
        model_logs[i] = model_log
        if model_saver is not None:
            model_saver.save_weights(current_model, verbose=verbose)
        if test_function is not None:
            logger.info("evaluate tests function")
            try:
                text = test_function(current_model)
                logger.info(text)
            except Exception as err:
                logger.error("cannot apply tests-function: {}".format(err))
        logger.info('Loss: {:.3f}'.format(model_log.history['loss'][-1]))
    return model_logs


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


LOSS_THRESHOLD = 0.1


class OnThresholdStop(tf.keras.callbacks.Callback):
    def __init__(self, threshold=LOSS_THRESHOLD):
        self.threshold = threshold

    def on_train_batch_end(self, epoch, logs={}):
        current_loss = logs.get('loss')
        if current_loss < self.threshold:
            print("\n Reached loss = {:.3f}, so stopping training!!".format(current_loss))
            self.model.stop_training = True


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def plot_percent_hist(ax, data, bins):
    counts, _ = np.histogram(data, bins=bins)
    widths = bins[1:] - bins[:-1]
    x = bins[:-1] + widths / 2
    ax.bar(x, counts / len(data), width=widths * 0.8)
    ax.xaxis.set_ticks(bins)
    ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(
                    lambda y, position: "{}%".format(int(np.round(100 * y)))
                    )
            )
    ax.grid(True)


def plot_activations_histogram(encoder, X, height=1, n_bins=10):
    X_valid_codings = encoder(X[:n_bins]).numpy()
    activation_means = X_valid_codings.mean(axis=0)
    mean = activation_means.mean()
    bins = np.linspace(0, 1, n_bins + 1)

    fig, [ax1, ax2] = plt.subplots(figsize=(10, 3), nrows=1, ncols=2, sharey=True)
    plot_percent_hist(ax1, X_valid_codings.ravel(), bins)
    ax1.plot([mean, mean], [0, height], "k--", label="Overall Mean = {:.2f}".format(mean))
    ax1.legend(loc="upper center", fontsize=14)
    ax1.set_xlabel("Activation")
    ax1.set_ylabel("% Activations")
    ax1.axis([0, 1, 0, height])
    plot_percent_hist(ax2, activation_means, bins)
    ax2.plot([mean, mean], [0, height], "k--")
    ax2.set_xlabel("Neuron Mean Activation")
    ax2.set_ylabel("% Neurons")
    ax2.axis([0, 1, 0, height])


# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------

class TextCompletion:

    def __init__(self, tokenizer, max_id) -> None:
        super().__init__()
        self.max_id = max_id
        self.tokenizer = tokenizer

    def preprocess(self, texts, one_hot=True):
        X = np.array(self.tokenizer.texts_to_sequences(texts)) - 1
        result = tf.one_hot(X, self.max_id) if one_hot is True else X
        return result

    def next_char(self, text, model, temperature=1, one_hot=True):
        X_new = self.preprocess([text], one_hot)
        y_proba = model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return self.tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text(self, model, text, n_chars=100, temperature=1, one_hot=True):
        for _ in range(n_chars):
            text += self.next_char(text, model, temperature, one_hot)
        return text


# -----------------------------------------------------------------------------
# Model support
# -----------------------------------------------------------------------------

import datetime


def get_path(root, directory):
    path = os.path.join(root, directory)
    if not (os.path.exists(path)):
        os.mkdir(path)
    return path


class GeneralSaver:
    def __init__(
            self, root_dir=DEFAULT_ROOT_DIR, project_directory=None,
            logger=console_logger.SimpleConsoleLogger("ModelSaver")
            ) -> None:
        self.logger = logger
        root_dir = root_dir if root_dir is not None else DEFAULT_ROOT_DIR
        if not (os.path.exists(root_dir)):
            os.mkdir(root_dir)
        if project_directory is None:
            project_directory = "default"
        self.project_directory = get_path(root_dir, project_directory)

    def get_path(self, model_name=None):
        path = self.project_directory if model_name is None else get_path(self.project_directory, model_name)
        return path

    def get_default_name(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def augment_filename(self, a, b):
        return "{}_{}_{}".format(self.get_path(), a, b)

    def save_fun(self, save_function, name=None, directory=None):
        filename = self.get_filename(name, directory)
        save_function(filename)

    def load_fun(self, load_function, name=None, directory=None):
        filename = self.get_filename(name, directory)
        return load_function(filename)


class ModelSaver(GeneralSaver):
    """

    """

    def __init__(self, root_dir=DEFAULT_ROOT_DIR, logger=console_logger.SimpleConsoleLogger("ModelSaver")) -> None:
        super().__init__(root_dir, project_directory="models/", logger=logger)

    def save_weights(self, model, name=None, verbose=False):
        """

        :param model:
        :param name:
        :param verbose:
        :return:
        """
        model_name = name if name is not None else model.name
        model_path = self.get_path(model_name)
        model.save_weights(model_path)
        if verbose is True:
            print("Weights of Model {} saved to {}".format(model_name, model_path))

    def save_weights_numpy(self, array, name=None, directory=None, verbose=False):
        """

        :param array:
        :param name:
        :param directory:
        :param verbose:
        :return:
        """
        model_name = name if name is not None else "weights"
        path = self.get_path(model_name=directory)
        filename = os.path.join(path, model_name)
        np.save(file=filename, arr=array, allow_pickle=False)
        self.logger.info("Weights saved to {}".format(filename))

    def load_weights(self, model, name=None, verbose=False):
        model_name = name if name is not None else model.name
        model_path = os.path.join(self.project_directory, model_name)
        model.load_weights(model_path)
        self.logger.info("Weights of Model {} loaded from {}".format(model_name, model_path))

    def load_weights_numpy(self, name=None, directory=None, verbose=False):
        model_name = name if name is not None else "weights"
        if not model_name.__contains__(".npy"):
            model_name = model_name + ".npy"
        path = self.project_directory if directory is None else os.path.join(self.project_directory, directory)
        filename = os.path.join(path, model_name)
        try:
            weights = np.load(file=filename, allow_pickle=False)
            self.logger.info("Weights loaded from {}".format(filename))
            return weights
        except FileNotFoundError as e:
            self.logger.error(e)
            return None


class VectorizationSaver(GeneralSaver):

    def __init__(
            self, root_dir=DEFAULT_ROOT_DIR, logger=console_logger.SimpleConsoleLogger(
                    "VectorizationSaver"
                    )
            ) -> None:
        super().__init__(root_dir, project_directory="vectorizer/", logger=logger)

    def get_filename(self, name=None, directory=None):
        model_name = name if name is not None else "vectorizer"
        path = self.get_path(model_name=directory)
        filename = os.path.join(path, model_name) + ".pkl"
        return filename

    def save(self, vectorizer, name=None, directory=None):
        filename = self.get_filename(name, directory=directory)
        pickle.dump({'config': vectorizer.get_config(), 'weights': vectorizer.get_weights()}, open(filename, "wb"))
        self.logger.info("Vectorizer saved to {}".format(filename))

    def load(self, name=None, directory=None, verbose=False):
        filename = self.get_filename(name, directory)
        try:
            from_disk = pickle.load(open(filename, "rb"))
            vectorizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
            vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))  # bug in keras
            vectorizer.set_weights(from_disk['weights'])
            self.logger.info("Vectorizer loaded from {}".format(filename))
            return vectorizer
        except FileNotFoundError as e:
            self.logger.error(e)
            return None


class DatasetSaver(GeneralSaver):

    def __init__(
            self, root_dir=DEFAULT_DATA_DIR, logger=console_logger.SimpleConsoleLogger(
                    "DatasetSaver"
                    )
            ) -> None:
        super().__init__(root_dir, project_directory="ds/", logger=logger)

    def get_filename(self, name, directory=None):
        model_name = name if name is not None else "dataset" + self.get_default_name()
        path = self.get_path(model_name=directory)
        filename = os.path.join(path, model_name)
        return filename

    def save(self, dataset, name=None, directory=None):
        filename = self.get_filename(name, directory=directory)
        self.logger.info(f"save dataset to {filename}")
        tf.data.experimental.save(dataset, filename)
        element_spec_path = f"{filename}/dataset_spec"
        with open(element_spec_path, "w") as f:
            f.write(str(dataset.element_spec))
            f.close()

    def load(self, name, element_spec=None, directory=None):
        """
        load dataset from file. Configuration must be provided, either as path_spec file or as parameter
        :param path: path to dataset
        :param logger: optianal
        :return: dataset (None, if something went wrong)
        """
        path = self.get_filename(name, directory=directory)
        self.logger.debug(f"load dataset from {path}")
        if element_spec is None:
            try:
                spec_path = f"{path}/dataset_spec"
                with open(spec_path, "r") as file:
                    self.logger.debug(f"load spec from {spec_path}")
                    element_spec_str = file.read()
                    file.close()
                element_spec = eval(element_spec_str)
            except FileNotFoundError as e:
                self.logger.error(e)
                return None
        try:
            train_word2vec_ds = tf.data.experimental.load(path, element_spec=element_spec)
            self.logger.info("dataset loaded from {}".format(path))
            return train_word2vec_ds
        except FileNotFoundError as e:
            self.logger.error(e)
            return None


# def load(self, name=None, directory=None, verbose=False):
#     filename = self.get_filename(name, directory)
#
#     from_disk = pickle.load(open(filename, "rb"))
#     vectorizer = TextVectorization.from_config(from_disk['config'])
#     vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))  # bug in keras
#     vectorizer.set_weights(from_disk['weights'])
#     if verbose is True: print("Vectorizer loaded from {}".format(filename))
#     return vectorizer


class ImageSaver(GeneralSaver):
    def __init__(self, project_directory=None, resolution=300) -> None:
        super().__init__(project_directory)
        self.resolution = resolution

    def save_fig(self, name, tight_layout=True, fig_extension="png", resolution=0, verbose=False):
        resolution = resolution if resolution > 0 else self.resolution
        image_name = os.path.join(self.project_directory, name + "." + fig_extension)
        if verbose is True:
            ("Saving figure, resolution = {}, format = {}".format(resolution, fig_extension), name)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(image_name, format=fig_extension, dpi=resolution)


"""" -------------------------------------------------------------------
    Data preparation
   -------------------------------------------------------------------
"""


@tf.keras.utils.register_keras_serializable(package="Custom", name=None)
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

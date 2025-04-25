# -----------------------------------------------------------------------------
# Model support
# -----------------------------------------------------------------------------
import datetime
import logging
import os
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from logging_lib import console_logger

# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import TextVectorization

"""" -------------------------------------------------------------------
    Constants
   -------------------------------------------------------------------
"""
ROOT_DIRECTORY = "/Users/thomasbayer/data/"
MODEL_ROOT = os.path.join(ROOT_DIRECTORY, "data_science/models/")
IMAGE_ROOT = os.path.join(ROOT_DIRECTORY, "data_science/images")

NAME_TEMPLATE = "{}_{}_w{}_ns{}"
DS_NAME_TEMPLATE = "ds_{}_{}_w{}_ns{}"
DATA_PATH_TEMPLATE = "../data/{}"
DEFAULT_DATA_DIR = "../data/"
DEFAULT_DATASET_DIR = "ds/"
DEFAULT_ROOT_DIR = "../stored/"
DEFAULT_MODEL_DIR = DEFAULT_ROOT_DIR + "/models/"
DEFAULT_VECTORIZER_DIR = DEFAULT_MODEL_DIR + "vectorizer/"
MODEL_PATH_TEMPLATE = "../models/{}"
W2V_MODLE_TEMPLATE = "w2v_{}x{}"

""" -------------------------------------------------------------------
    Functions
   -------------------------------------------------------------------
"""


def get_path(root, directory):
    path = os.path.join(root, directory)
    if not (os.path.exists(path)):
        os.mkdir(path)
    return path


class GeneralSaver:
    def __init__(self,
                 root_dir=DEFAULT_ROOT_DIR,
                 project_directory=None,
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

    def save(self, model, name=None):
        model_name = name if name is not None else model.name
        model_path = self.get_path(model_name)
        model.save(model_path)
        self.logger.info("Model {} saved to {}".format(model_name, model_path))

    def load(self, name="default"):
        model_path = self.get_path(name)
        model = tf.keras.models.load_model(model_path)
        self.logger.info(f"Model {model.name} loaded from {model_path}")
        return model


def save_vectorizer(vectorizer, directory, name=None, verbose=False):
    weights = vectorizer.get_weights()[0]
    config = vectorizer.get_config()
    if "standardize" in config:
        config["standardize"] = None
    model_name = name if name is not None else "vectorizer"
    save_weights_numpy(weights, directory, model_name, verbose=verbose)
    filename = os.path.join(directory, model_name + "-config.json")
    with open(filename, 'w') as f:
        json.dump(config, f)


def save_weights_numpy(array, directory, name=None, logger=logging.Logger("save_weihgts"), verbose=False):
    """
    Saves the weights in a numpy file.

    :param array: The weights to save
    :param name: The name prefix for the file
    :param directory: The directory to save the file in
    :param get_path_func: Function to retrieve the path
    :param logger: The logger to use for logging info about save operation
    :param verbose: Whether to log verbosely (unused)
    """
    model_name = name if name is not None else "weights"
    filename = os.path.join(directory, model_name)
    np.save(file=filename, arr=array, allow_pickle=True)
    logger.info("Weights saved to {}".format(filename))


class VectorizationSaver(GeneralSaver):

    def __init__(self, root_dir=DEFAULT_ROOT_DIR, logger=console_logger.SimpleConsoleLogger(
        "VectorizationSaver"
        )
                 ) -> None:
        super().__init__(root_dir, project_directory="vectorizer/", logger=logger)

    def get_filename(self, name=None, directory=None):
        model_name = name if name is not None else "vectorizer"
        path = self.get_path(model_name=directory)
        filename = os.path.join(path, model_name) + ".pkl"
        return filename

    def save(self, vectorizer, name=None, directory=None, ds=None):
        filename = self.get_filename(name, directory=directory)
        config = vectorizer.get_config()
        # if "standardize" in config:
        #     config["standardize"] = None
        pickle.dump({'config': config, 'weights': vectorizer.get_weights()}, open(filename, "wb"))
        self.logger.info("Vectorizer saved to {}".format(filename))
        if ds is not None:
            ds_filename = os.path.join(self.get_path(), "ds/")
            ds.save(ds_filename, compression='GZIP')
            self.logger.info("Dataset saved to {}".format(ds_filename))

    def load_ds(self, name=None, directory=None, ds=True):
        filename = self.get_filename(name, directory)
        self.logger.info("load vectorizer from {}".format(filename))
        from_disk = pickle.load(open(filename, "rb"))
        # from_disk['config']['standardize'] = standardizer
        vectorizer = TextVectorization.from_config(from_disk['config'])
        vectorizer.set_weights(from_disk['weights'])
        self.logger.info("Vectorizer loaded from {}".format(filename))
        dataset = None
        if ds is True:
            ds_filename = os.path.join(self.get_path(), "ds/")
            dataset = tf.data.Dataset.load(ds_filename, compression='GZIP')
            self.logger.info("Dataset loaded from {}".format(ds_filename))
        return vectorizer, dataset

    def load(self, name=None, directory=None, verbose=False):
        vectorizer, _ = self.load_ds(name=name, directory=directory, ds=False)
        return vectorizer


def get_w2v_filename(directory, name, window_size, num_ns, suffix=""):
    postfix = f"{name}_{window_size}x{num_ns}{suffix}"
    return os.path.join(directory, postfix)


def get_vectorizer_directory(directory, vocab_size, sequence_length, suffix=""):
    postfix = f"dataset_{vocab_size}x{sequence_length}{suffix}"
    return os.path.join(directory, postfix)


def save_w2v_tset(directory, targets, contexts, labels, window_size, num_ns):
    np.save(
        file=get_w2v_filename(directory, "targets", window_size, num_ns), arr=targets, allow_pickle=False
        )
    np.save(
        file=get_w2v_filename(directory, "contexts", window_size, num_ns), arr=contexts, allow_pickle=False
        )
    np.save(
        file=get_w2v_filename(directory, "labels", window_size, num_ns), arr=labels, allow_pickle=False
        )


def load_w2v_tset(directory, window_size, num_ns):
    targets = np.load(file=get_w2v_filename(directory, "targets", window_size, num_ns, suffix=".npy"))
    contexts = np.load(file=get_w2v_filename(directory, "contexts", window_size, num_ns, suffix=".npy"))
    labels = np.load(file=get_w2v_filename(directory, "labels", window_size, num_ns, suffix=".npy"))
    return targets, contexts, labels


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
        super.__init__(project_directory)
        self.resolution = resolution

    def save_fig(self, name, tight_layout=True, fig_extension="png", resolution=0, verbose=False):
        resolution = resolution if resolution > 0 else self.resolution
        image_name = os.path.join(self.project_directory, name + "." + fig_extension)
        if verbose is True:
            ("Saving figure, resolution = {}, format = {}".format(resolution, fig_extension), name)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(image_name, format=fig_extension, dpi=resolution)


# -----------------------------------------------------------------------------
# Deprecated
# -----------------------------------------------------------------------------

class DatasetSaver(GeneralSaver):

    def __init__(self, root_dir=DEFAULT_ROOT_DIR, logger=console_logger.SimpleConsoleLogger(
        "DatasetSaver"
        )
                 ) -> None:
        super().__init__(root_dir, project_directory="ds/", logger=logger)

    def get_filename(self, name, directory=None):
        model_name = name if name is not None else "dataset"
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

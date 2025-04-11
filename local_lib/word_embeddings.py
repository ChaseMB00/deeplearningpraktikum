import os
import numpy as np
import re
import string

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from logging_lib import console_logger

import local_lib.lib_tf as lt
import local_lib.savers as lsv

SEED = 42
BATCH_SIZE = 512
BUFFER_SIZE = 10000


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class ModelMismatch(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# -----------------------------------------------------------------------------
# Preparation
# -----------------------------------------------------------------------------

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


HTML_PATTERN = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'
CLEANR = re.compile(HTML_PATTERN)


def cleanhtml(input):
    stripped_html = tf.strings.regex_replace(input, HTML_PATTERN, ' ')
    return stripped_html


# ------------------------------------------------------------------------------
# Useful functions for words and embeddings
# ------------------------------------------------------------------------------

def word_to_vector(word, vocabulary, weights):
    index = vocabulary.index(word)
    word_one_hot = tf.one_hot(index, len(vocabulary) + 1)
    vector = tf.matmul(tf.expand_dims(word_one_hot, 0), weights)
    return tf.reshape(vector, -1)


def get_similarity_vector(v1, v2, vocabulary, word2vec):
    cosine = np.dot(v1, v2) / (tf.norm(v1) * tf.norm(v2))
    print(f"cosine: {cosine}")
    print(tf.norm(v1 - v2))
    return cosine


def get_similarity(word1, word2, vocabulary, word2vec):
    print(f"Similarity between {word1} and {word2}:")
    v1 = word_to_vector(word1, vocabulary, word2vec)
    v2 = word_to_vector(word2, vocabulary, word2vec)
    return get_similarity_vector(v1, v2, vocabulary, word2vec)


def similar_word_vector(vector, vocabulary, weights):
    cosine_list = tf.tensordot(weights, vector, axes=1)
    sim_index = np.array(cosine_list).argmax()
    return vocabulary[sim_index]


def similar_word(word, vocabulary, word2vec):
    weights = word2vec.layers[0].weights
    vector = word_to_vector(word, vocabulary, weights)
    return similar_word_vector(vector, vocabulary, weights)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class Word2Vec(tf.keras.Model):
    """
    Model for word2vec
    """

    def __init__(self, vocab_size, embedding_dim, num_ns=4, name='word2vec'):
        super(Word2Vec, self).__init__(name=name)

        self.target_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=1,
            name="w2v_embedding"
            )
        self.context_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns + 1
            )

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots

    def custom_loss(x_logit, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


# -----------------------------------------------------------------------------
# Vectorization
# -----------------------------------------------------------------------------

def load_vectorizer_ds(directory, name=None, ds=True):
    vs = lsv.VectorizationSaver(directory)
    vectorize_layer = vs.load(name=name)
    if ds:
        ds_filename = os.path.join(directory, "vectorizer/ds/")
        dataset = tf.data.Dataset.load(ds_filename, compression='GZIP')
        return vectorize_layer, dataset
    else:
        return vectorize_layer


def load_vectorize_layer_ds(root_directory,
                            vocab_size=5000,
                            sequence_length=64,
                            dataset_name="dataset",
                            logger=console_logger.SimpleConsoleLogger("load_vectorize_layer_ds")
                            ):
    logger.info(
        f"load vectorize_layer-dataset, vocab_size = {vocab_size}, sequence_length = {sequence_length}"
        )
    vectorize_directory = get_vectorizer_directory(root_directory, 'dataset', vocab_size, sequence_length)
    logger.info(f"looking for vectorize_layer in {vectorize_directory}")
    vectorize_layer, ds = load_vectorizer_ds(vectorize_directory, ds=True)
    return vectorize_layer, ds


def get_vectorizer(text_ds, standardizer=None, vocab_size=5000, sequence_length=64, batch_size=1024):
    """
    Create vectorizer from a text-dataset
    :param text_ds:
    :param standardizer:
    :param vocab_size:
    :param sequence_length:
    :return:
    """
    vectorizer = TextVectorization(
        standardize=standardizer,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
        )
    vectorizer.adapt(text_ds.batch(batch_size))
    inverse_vocab = vectorizer.get_vocabulary()
    print("size of vocab = {:,.0f}".format(len(inverse_vocab)))
    # Vectorize the data in text_ds.
    new_ds = text_ds
    text_vector_ds = new_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorizer).unbatch()
    return vectorizer, text_vector_ds


def get_vectorizer_directory(root_directory, dataset_name, vocab_size, sequence_length, sub_dir="", name=""):
    return os.path.join(root_directory, f"{dataset_name}_{vocab_size}x{sequence_length}", sub_dir, name)


def create_vectorizer_ds(text_ds,
                         standardizer=None,
                         vocab_size=5000,
                         sequence_length=64,
                         batch_size=1024,
                         root_directory=None,
                         dataset_name="dataset",
                         logger=console_logger.SimpleConsoleLogger("vectorize_ds")
                         ):
    logger.info(
        f"create or load vectorize_layer-dataset, vocab_size = {vocab_size}, sequence_length = {sequence_length}"
        )
    if root_directory is not None:
        vectorize_directory = get_vectorizer_directory(root_directory, 'dataset', vocab_size, sequence_length)
        logger.info(f"looking for vectorize_layer in {vectorize_directory}")
        vectorize_layer, ds = load_vectorizer_ds(vectorize_directory, ds=True)
        return vectorize_layer, ds

    vectorize_layer, ds = get_vectorizer(
        text_ds, standardizer=standardizer, vocab_size=vocab_size,
        sequence_length=sequence_length, batch_size=batch_size
        )
    if root_directory is not None:
        directory = get_vectorizer_directory(root_directory, dataset_name, vocab_size, sequence_length)
        vs = lsv.VectorizationSaver(directory, logger=logger)
        vs.save(vectorize_layer, ds=ds)
    return vectorize_layer, ds


def vectorizer_ds(text_ds,
                  standardizer=None,
                  vocab_size=5000,
                  sequence_length=64,
                  batch_size=1024,
                  root_directory=None,
                  dataset_name="dataset",
                  logger=console_logger.SimpleConsoleLogger("vectorize_ds")
                  ):
    logger.info("get vectorizer")
    vectorize_layer, ds = get_vectorizer(
        text_ds, standardizer=standardizer, vocab_size=vocab_size,
        sequence_length=sequence_length, batch_size=batch_size
        )
    if root_directory is not None:
        directory = get_vectorizer_directory(root_directory, dataset_name, vocab_size, sequence_length)
        vs = lsv.VectorizationSaver(directory, logger=logger)
        vs.save(vectorize_layer, ds=ds)
    return vectorize_layer, ds


def translate_vector(vectorize_layer, vector, crop=True):
    inverse_vocab = vectorize_layer.get_vocabulary()
    v = tf.boolean_mask(vector, tf.greater(vector, 0)) if crop else vector
    return tf.map_fn(lambda i: inverse_vocab[i], v, dtype=tf.string)


# -----------------------------------------------------------------------------
# Word2Vec
# -----------------------------------------------------------------------------


import tqdm


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed=SEED):
    """
    Create skip-gram of intput-text (sequences) for word2vec
    :param sequences: input text as sequence of sequences
    :param window_size:
    :param num_ns: number of negative samples
    :param vocab_size:
    :param seed:
    :return:
    """

    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    print("-- generate training data --")
    print(f"number of sequences: {len(sequences)}")
    print(f"vocab size         : {vocab_size}")
    print(f"window_size        : {window_size}")
    print(f"negative samples   : {num_ns}")

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0
            )

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling"
                )

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


# work in progress

def pretrained_embeddings(file_path, embedding_dim, vocab_size, word2idx):
    # 1.load in pre-trained word vectors     #feature vector for each word
    print("graph in function", tf.get_default_graph())
    print('Loading word vectors...')
    word2vec = {}
    with open(os.path.join(file_path + '.%sd.txt' % embedding_dim), errors='ignore', encoding='utf8') as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec

    print('Found %s word vectors.' % len(word2vec))

    # 2.prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = vocab_size
    # initialization by zeros
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word2idx.items():
        if i < vocab_size:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


from timeit import default_timer as timer


def generate_w2v_ds(targets, contexts, labels, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def generate_training_data_ds(text_vector_ds, window_size=2, num_ns=4, vocab_size=5000, seed=42):
    sequences = list(text_vector_ds.as_numpy_iterator())
    print("number of lines {:,.0f}".format(len(sequences)))
    return generate_training_data(sequences, window_size, num_ns, vocab_size, seed=seed)


# def save_dataset(dataset, path, logger=console_logger.SimpleConsoleLogger("save_ds")):
#     logger.info(f"save dataset to {path}")
#     tf.data.experimental.save(dataset, path)
#     element_spec_path = f"{path}_spec"
#     with open(element_spec_path, "w") as f:
#         f.write(str(dataset.element_spec))
#         f.close()


def create_word2vec_model(dataset, vocab_size, sequence_length, embedding_dim, num_ns=4, epochs=20, name='word2vec',
                          root_directory=None,
                          dataset_name="dataset",
                          logger=console_logger.SimpleConsoleLogger("get_word2vec_model"),
                          **kwargs
                          ):
    """
    Create word2vec-Model from prepared dataset
    :param dataset:
    :param vocab_size:
    :param embedding_dim:
    :param num_ns:
    :param epochs:
    :param kwargs:
    :return:
    """
    logger.info(f"fit word2vec-model, embedding-dim = {embedding_dim}")
    word2vec = Word2Vec(vocab_size + 1, embedding_dim, num_ns=num_ns, name=name)
    word2vec.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    model_log, _ = lt.eval_model_ds(word2vec, dataset, num_epochs=epochs, batch_size=None, logger=logger, **kwargs)
    if root_directory is not None:
        directory = get_vectorizer_directory(
            root_directory, dataset_name, vocab_size, sequence_length, sub_dir="w2v", name="weights"
            )
        logger.info(f"save weights to {directory}")
        filename = directory + "/embedding"
        word2vec.save_weights(filename)
        np.save(file=filename, arr=word2vec.layers[0].weights, allow_pickle=True)
        logger.info("Weights saved to {}".format(filename))
    return word2vec, model_log


def load_word2vec_weights(vocab_size, sequence_length,
                          root_directory=None,
                          dataset_name="dataset",
                          logger=console_logger.SimpleConsoleLogger("get_word2vec_model"),
                          **kwargs
                          ):
    directory = get_vectorizer_directory(
        root_directory, dataset_name, vocab_size, sequence_length, sub_dir="w2v", name="weights"
        )
    filename = directory + "/embedding.npy"
    logger.info(f"load weights from {filename}")
    try:
        weights = np.load(file=filename, allow_pickle=False)
        logger.info("Weights loaded from {}".format(filename))
        return weights
    except FileNotFoundError as e:
        logger.error(e)
        return None


# -----------------------------------------------------------------------------
# Combined Vectorization + word2vec
# -----------------------------------------------------------------------------


# def get_w2v_filename(directory, name, vocab_size, window_size, num_ns, suffix=""):
#     postfix = f"{name}_{vocab_size}x{window_size}x{num_ns}{suffix}"
#     return os.path.join(directory, postfix)


def get_w2v_ds_directory(root_directory, dataset_name, vocab_size, sequence_length, window_size, num_ns):
    name = lt.DS_W2V_NAME_TEMPLATE.format(window_size, num_ns)
    directory = get_vectorizer_directory(
        root_directory, dataset_name, vocab_size, sequence_length, sub_dir="w2v", name=name
        )
    return directory


def vector_ds_to_w2v_ds(text_vector_ds,
                        vocab_size=5000,
                        sequence_length=64,
                        window_size=2,
                        num_ns=4,
                        root_directory=None,
                        dataset_name="dataset",
                        augment_path=False,
                        seed=42,
                        logger=console_logger.SimpleConsoleLogger("vectorize_ds")
                        ):
    """

    """
    if root_directory is not None:
        # todo load target, context, labels
        directory = get_w2v_ds_directory(root_directory, dataset_name, vocab_size, sequence_length, window_size, num_ns)
        logger.info(f"looking for w2v-dataset in {directory}")
        try:
            w2v_ds = tf.data.Dataset.load(directory, compression='GZIP')
            logger.info(f"dataset already exists - nothing to do")
            return w2v_ds, None, None, None
        except:
            logger.info("dataset not found")
    start = timer()
    targets, contexts, labels = generate_training_data_ds(
        text_vector_ds, window_size=window_size, num_ns=num_ns, vocab_size=vocab_size,
        seed=seed
        )
    end = timer()
    logger.info("Time: {:,.2f} seconds".format(end - start))
    logger.info("generate w2v-dataset")
    w2v_ds = generate_w2v_ds(targets, contexts, labels)
    if root_directory is not None:
        directory = get_w2v_ds_directory(root_directory, dataset_name, vocab_size, sequence_length, window_size, num_ns)
        logger.info(f"save w2v-dataset to {directory}")
        w2v_ds.save(directory, compression='GZIP')
    return w2v_ds, targets, contexts, labels


def vectorize_w2v_ds(text_ds,
                     standardizer=None,
                     vocab_size=5000,
                     sequence_length=64,
                     window_size=2,
                     num_ns=4,
                     dataset_name=None,
                     augment_path=False,
                     seed=42,
                     logger=console_logger.SimpleConsoleLogger("vectorize_ds")
                     ):
    """
    Create a dataset for a word2vec-embedding (generate skip-grams). Warning: May take a long time
    :param text_ds: input text
    :param standardizer:
    :param vocab_size:
    :param sequence_length:
    :param window_size:
    :param num_ns: Number of negative samples (4)
    :param dataset_name: path to directory, if dataset shall be saved
    :param seed: 42
    :return: dataset, vectorizer, targets, contexts, labels (for skip-gram)
    """
    logger.info("get vectorizer")
    vectorizer, text_vector_ds = get_vectorizer(text_ds, standardizer, vocab_size, sequence_length)

    logger.info("generate training data")
    start = timer()
    targets, contexts, labels = generate_training_data_ds(
        text_vector_ds, window_size=window_size, num_ns=num_ns, vocab_size=vocab_size,
        seed=seed
        )
    end = timer()
    logger.info("Time: {:,.2f} seconds".format(end - start))
    w2v_ds = generate_w2v_ds(targets, contexts, labels)
    if dataset_name is not None:
        name = lt.DS_NAME_TEMPLATE.format(vocab_size, sequence_length, window_size, num_ns)
        current_path = "{}/{}".format(dataset_name, name) if augment_path is True else dataset_name
        ds = lt.DatasetSaver(logger=logger)
        ds.save(w2v_ds, name=current_path)
        # save_dataset(dataset, current_path, logger=logger)
    return vectorizer, text_vector_ds, w2v_ds, targets, contexts, labels


def word2vec_embedding(text_ds, standardizer=None, vocab_size=5000, sequence_length=64, embedding_dim=64, num_ns=4,
                       batch_size=BATCH_SIZE, **kwargs
                       ):
    dataset, vectorizer = vectorize_w2v_ds(text_ds, standardizer, vocab_size, sequence_length)
    word2vec = Word2Vec(vocab_size + 1, embedding_dim, num_ns=num_ns)
    word2vec.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )

    word2vec, history = get_word2vec_model(
        dataset=dataset, vocab_size=vocab_size, embedding_dim=embedding_dim, num_ns=num_ns, **kwargs
        )
    return word2vec, dataset, vectorizer, history


# -----------------------------------------------------------------------------
# load vectorizer and data
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Predefined functions
# -----------------------------------------------------------------------------

def generate_w2v_datasets(text_vector_ds,
                          vocab_size=5000,
                          sequence_length=64,
                          window_size=2,
                          num_ns=4,
                          root_directory='./',
                          dataset_name="dataset",
                          augment_path=False,
                          seed=42,
                          logger=console_logger.SimpleConsoleLogger("vectorize_ds")
                          ):
    window_sizes = np.array([window_size]).flatten()
    num_neg_samples = np.array([num_ns]).flatten()
    print(f"window_sizes: {window_sizes}")
    print(f"num_ns: {num_neg_samples}")
    for window_size in window_sizes:
        for num_ns in num_neg_samples:
            w2v_ds, targets, contexts, labels = vector_ds_to_w2v_ds(
                text_vector_ds, vocab_size, sequence_length, window_size=window_size, num_ns=num_ns,
                root_directory=root_directory, dataset_name=dataset_name, logger=logger
                )


def prepare_dataset_vectorizer(input_ds,
                               standardizer=lambda x: x,
                               vocab_size=5000,
                               sequence_length=64,
                               window_size=2,
                               num_ns=4,
                               path_ds=None,
                               augment_path=False,
                               directory=None,
                               seed=42,
                               logger=console_logger.SimpleConsoleLogger("prepare_ds")
                               ):
    dataset_name = directory + '/' if directory is not None else None
    dataset, vectorizer, targets, contexts, labels = vectorizer_ds(
        text_ds=input_ds,
        standardizer=standardizer,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        window_size=window_size,
        num_ns=num_ns,
        dataset_name=dataset_name,
        augment_path=augment_path,
        seed=seed,
        logger=logger
        )
    ms = lt.VectorizationSaver(root_dir=path_ds, logger=logger)
    name = lt.DS_NAME_TEMPLATE.format(vocab_size, sequence_length, window_size, num_ns)
    ms.save(vectorizer, name=name, directory=directory)
    logger.info("finished")
    return dataset, vectorizer, targets, contexts, labels

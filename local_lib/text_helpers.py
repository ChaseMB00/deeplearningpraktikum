import os
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

ROOT_DIRECTORY = "/Users/thomasbayer/data/"
TEXT_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data_science/text/")
MODEL_ROOT = os.path.join(ROOT_DIRECTORY, "data_science/models/")
SEED = 42


# -------------------------------------------------------------------
# Dataset Preparation
# -------------------------------------------------------------------

def prepare_text(text,
                 nr_unused_chars=2
                 ):
    text_vec_layer = tf.keras.layers.TextVectorization(
            split="character",
            standardize="lower"
            )
    text_vec_layer.adapt([text])
    encoded_raw = text_vec_layer([text])
    print(f"encoded_raw shape: {encoded_raw.shape}")
    encoded = encoded_raw[0]
    print(f"encoded sample: {encoded[:20]}")
    encoded -= nr_unused_chars  # drop tokens 0 (pad) and 1 (unknown), which we will not use
    n_tokens = text_vec_layer.vocabulary_size() - nr_unused_chars  # number of distinct chars = 39
    print(f"n_tokens: {n_tokens}, dataset_size: {len(encoded)}")
    return text_vec_layer, encoded, n_tokens


@tf.autograph.experimental.do_not_convert
@tf.function(reduce_retracing=True)
def one_hot_xy_function(X_batch,
                        Y_batch,
                        n_tokens
                        ):
    return (tf.one_hot(X_batch, depth=n_tokens), Y_batch)


def dataset_from_text_t(text,
                        n_steps=100,
                        batch_size=32,
                        one_hot=False,
                        repeat=False,
                        transform=None
                        ):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(text)
    config = tokenizer.get_config()

    print(tokenizer.texts_to_sequences(["First"]))
    print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))

    n_tokens = len(tokenizer.word_index)  # number of distinct characters
    dataset_size = tokenizer.document_count  # total number of characters

    print("max_id = {}, dataset_size = {:,.0f}".format(n_tokens, dataset_size))

    [encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
    train_size = dataset_size * 90 // 100
    if one_hot:
        transform = lambda x, y: one_hot_xy_function(x, y, n_tokens)
    dataset = seq_to_shifted_dataset(encoded, length=n_steps, batch_size=batch_size, transform=transform, repeat=repeat)
    return tokenizer, dataset, n_tokens, dataset_size


# todo: dataset_from_text_tv for text_vectorization

def seq_to_shifted_dataset(sequence,
                           length,
                           shift=1,
                           shuffle=0,
                           seed=None,
                           batch_size=32,
                           transform=None,
                           reshuffle_each_iteration=False,
                           repeat=False,
                           prefetch=tf.data.AUTOTUNE
                           ):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    if repeat:
        ds = ds.repeat()
    ds = ds.window(length + 1, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle > 0:
        ds = ds.shuffle(buffer_size=shuffle, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda window: (window[:, :-1], window[:, 1:])).cache().prefetch(prefetch)  # generate X,y pairs
    if transform:
        ds = ds.map(transform)
    return ds


def to_dataset_for_stateful_rnn(sequence,
                                length
                                ):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=length, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(length + 1)).batch(1)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


def to_non_overlapping_windows(sequence,
                               length
                               ):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=length, drop_remainder=True)
    return ds.flat_map(lambda window: window.batch(length + 1))


def to_batched_dataset_for_stateful_rnn(sequence,
                                        length,
                                        batch_size=32
                                        ):
    parts = np.array_split(sequence, batch_size)
    datasets = tuple(to_non_overlapping_windows(part, length) for part in parts)
    ds = tf.data.Dataset.zip(datasets).map(lambda *windows: tf.stack(windows))
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# -------------------------------------------------------------------
# Predictions using TextTokenizer
# -------------------------------------------------------------------

def next_token_id(sequence,
                  model,
                  temperature=1,
                  transform=lambda x: x,
                  argmax=False
                  ):
    sequence_vector = transform(sequence)
    # shape must be (batch_size, sequence_length, embedding-dimension)
    y_proba = model.predict(sequence_vector, verbose=0)[0, -1:]
    if argmax:
        char_id = np.argmax(y_proba)
    else:
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return char_id


def complete_text_abs(text,
                      model,
                      n_chars,
                      tokenize,
                      detokenize,
                      temperature=1,
                      transform=lambda x: x,
                      argmax=False,
                      nr_unused_chars=1
                      ):
    sequence = tokenize(text) - nr_unused_chars
    sequence = np.reshape(sequence, (1, len(sequence)))
    new_text = ""
    for _ in range(n_chars):
        next_token = next_token_id(
                sequence,
                model,
                temperature=temperature,
                transform=transform,
                argmax=argmax
                )
        new_text += detokenize(next_token + nr_unused_chars)
        sequence = np.append(sequence, next_token)
        sequence = np.reshape(sequence, (1, len(sequence)))
    return text + new_text.upper()


class TextPredictor:
    def __init__(self,
                 model,
                 tokenize,
                 detokenize,
                 nr_unused_chars,
                 transform=None
                 ):
        self.model = model
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.nr_unused_chars = nr_unused_chars
        self.transform = transform if transform else lambda x: x

    def complete_text(self,
                      text,
                      n_chars=10,
                      temperature=0.3,
                      argmax=False
                      ):
        return complete_text_abs(
                text,
                self.model,
                n_chars,
                self.tokenize,
                self.detokenize,
                temperature,
                transform=self.transform,
                nr_unused_chars=self.nr_unused_chars,
                argmax=argmax
                )


class TextPredictorTV(TextPredictor):
    def __init__(self,
                 model,
                 tokenizer,
                 nr_unused_chars=2,
                 transform=None
                 ):
        super().__init__(model, tokenizer, lambda x: tokenizer.get_vocabulary()[x], nr_unused_chars, transform)


class TextPredictorT(TextPredictor):
    def __init__(self,
                 model,
                 tokenizer,
                 nr_unused_chars=2
                 ):
        super().__init__(
                model,
                lambda x: np.array(tokenizer.texts_to_sequences(x)),
                lambda x: tokenizer.sequences_to_texts([[int(x)]])[0],
                nr_unused_chars,
                transform=lambda x: tf.one_hot(x, len(tokenizer.word_index))
                )


def complete_text_t(text,
                    n_chars,
                    model,
                    tokenizer,
                    temperature=1,
                    transform=lambda x: x,
                    nr_unused_chars=1,
                    argmax=False
                    ):
    sequence = np.array(tokenizer.texts_to_sequences(text)) - nr_unused_chars
    sequence = np.reshape(sequence, (1, len(sequence)))
    new_text = ""
    for _ in range(n_chars):
        next_token = next_token_id(
                sequence,
                model,
                temperature=temperature,
                transform=transform,
                argmax=argmax
                )
        new_text += tokenizer.sequences_to_texts([[int(next_token) + nr_unused_chars]])[0]
        sequence = np.append(sequence, next_token)
        sequence = np.reshape(sequence, (1, len(sequence)))
    print(tokenizer.sequences_to_texts(sequence + nr_unused_chars))
    return text + new_text.upper()

# def preprocess_t(tokenizer, max_id, texts):
#     X = np.array(tokenizer.texts_to_sequences(texts)) - 1
#     return tf.one_hot(X, max_id)
#
#
# def next_char_t(text, model, tokenizer, max_id, temperature=1):
#     X_new = preprocess_t(tokenizer, max_id, [text])
#     y_proba = model.predict(X_new)[0, -1:, :]
#     rescaled_logits = tf.math.log(y_proba) / temperature
#     char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
#     return tokenizer.sequences_to_texts(char_id.numpy())[0]

#
# def preprocess(tokenizer, max_id, texts):
#     X = np.array(tokenizer.texts_to_sequences(texts)) - 1
#     return tf.one_hot(X, max_id)

#
# class TextPredictorT:
#     def __init__(self, model, tokenizer, n_tokens, transform=None):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.max_id = n_tokens
#         self.transform = transform if transform else lambda x: tf.one_hot(x, n_tokens)
#
#     def complete_text(self, text, n_chars=10, temperature=1):
#         return complete_text_t(text, n_chars, self.model, self.tokenizer, temperature, transform=self.transform)


# -------------------------------------------------------------------
# Predictions using TextEmbedding
# -------------------------------------------------------------------

# def next_char_e(text, model, embedding, temperature=1, transform=lambda x: x):
#     text_embedding = embedding([text])
#     print(text_embedding.shape)
#     text_vector = transform(text_embedding)
#     y_proba_raw = model.predict(text_vector, verbose=0)
#     print(y_proba_raw.shape)
#     y_proba = y_proba_raw[0, -1:]
#     rescaled_logits = tf.math.log(y_proba) / temperature
#     char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
#     # print(char_id)
#     return embedding.get_vocabulary()[char_id + 2]
#
#
# def extend_text(text, model, embedding, n_chars=50, temperature=1):
#     full_text = text.upper()
#     for _ in range(n_chars):
#         full_text += next_char_e(full_text.lower(), model, embedding, temperature)
#     return full_text

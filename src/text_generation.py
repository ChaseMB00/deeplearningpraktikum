import numpy as np
import tensorflow as tf


def next_token(sequence, model, temperature=1, transform=None, argmax=True, n_tokens=39):
    inp = transform(sequence) if transform else sequence
    inp_tf = tf.reshape(inp, (1, len(sequence), n_tokens))
    y = model.predict(inp_tf)
    prediction = y[0, -1, :]
    print(f"prediction: {prediction.shape}")
    if argmax:
        char_id = np.argmax(prediction)
    else:
        prediction = prediction / temperature
        prediction = tf.nn.softmax(prediction)
        char_id = np.random.choice(range(0, n_tokens), p=prediction.numpy())
    return char_id


def generate(text, model, tokenizer, n_chars=50, temperature=1, argmax=True,
             n_tokens=39
             ):
    """
    Generate text using a trained model.
    :param text:
    :param model:
    :param tokenizer:
    :param n_chars:
    :param temperature:
    :param argmax:
    :param n_tokens:
    :return:
    """
    sequence = np.array(tokenizer.texts_to_sequences([text])[0])
    sequence = sequence - 1
    one_hot_transform = lambda x: tf.one_hot(x, depth=n_tokens)
    for _ in range(n_chars):
        char_id = next_token(
                sequence,
                model,
                transform=one_hot_transform,
                temperature=temperature,
                argmax=argmax,
                n_tokens=n_tokens
                )
        sequence = np.append(sequence, char_id)
    sequence = sequence + 1
    return tokenizer.sequences_to_texts([sequence])[0]

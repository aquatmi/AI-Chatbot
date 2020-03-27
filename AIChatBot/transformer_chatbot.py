# CODE ADAPTED FROM TUTORIAL AT:
# https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import pickle
from transformer_functions import *

tf.random.set_seed(1234)

print('@console: loaded paths')

# CONSTANTS
MAX_SAMPLES = 50000
MAX_LENGTH = 40

questions, answers = load_conversations(MAX_SAMPLES)
print('@console: questions and answers loaded')

# noinspection PyBroadException
try:
    # if possible to open saved model, do.
    # this is because building a new one is slow
    with open('input/transformer_model/tokenizer.pick', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print('@console: tokenizer loaded')
except:
    # build tokenizer
    CORPUS_SENTENCES = questions + answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2 ** 13)

    with open('input/transformer_model/tokenizer.pick', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('@console: tokenizer built and saved')

# define start and end token
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# vocab size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

questions, answers = tokenize_and_filter(questions, answers, tokenizer, MAX_LENGTH, START_TOKEN, END_TOKEN)

print('@console: tokenized and filtered sentences')

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print("@console: dataset created")

tf.keras.backend.clear_session()

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1


def train_transformer():
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    EPOCHS = 20
    model.fit(dataset, epochs=EPOCHS)
    model.save_weights('input/transformer_model/transformer_weights.h5')


def trans_load():
    t = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    t.load_weights('input/transformer_model/transformer_weights.h5')
    return t


def trans_evaluate(sentence, t):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = t(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def trans_predict(sentence, t):
    prediction = trans_evaluate(sentence, t)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    # print('Input: {}'.format(sentence))
    # print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

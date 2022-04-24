# Base libs
import traceback
import argparse
from functools import partial
from typing import *
# ML libs
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import TensorBoard
import keras_tuner as kt
from sklearn import metrics
# Local libs
from src import *


def get_args() -> argparse.Namespace:
    """Set-up the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='Project 4 for the Deep Learning class (COSC 525). '
                    'Involves the development of a Convolutional Neural Network.',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-m', '--model', type=str, required=True,
                                choices= [ "lstm","simpleRNN"], help="The model to train on.")
    required_args.add_argument('-e', '--epochs', type=int, required=True,
                                help="The numbe of epochs to run.")
    required_args.add_argument('-w', '--window', type=int, required=True,
                                help="The window size.")
    required_args.add_argument('-l', '--hidden', type=int, required=True,
                                help="The Hiddens size.")
    required_args.add_argument('-s', '--stride', type=int, required=True,
                                help="The stride.")
    required_args.add_argument('-t', '--temperature', type=float, required=True,
                                help="The Sampling temperature for the text generation.")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    args = parser.parse_args()
    return args


def build_model_RNN(model_type: str, hidden_size: int, window_size: int, sampling_temp: int,
                    vocab_size: int, lr: int = .01) -> Model:
    """ Build The Model"""
    model = Sequential()

    model.add(layers.Input(shape=(window_size, vocab_size), name='encoder_input'))

    if (model_type == "lstm"):
        model.add(layers.LSTM(hidden_size, return_sequences=True))
    else:
        model.add(layers.SimpleRNN(hidden_size, return_sequences=True))

    model.add(layers.Dense(vocab_size))
    #model.add(layers.Lambda(lambda x: x / sampling_temp))
    #model.add(layers.Softmax())

    opt = optimizers.RMSprop(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt)
    model.summary()
    return model


def select_char(prob_vec, sampling_temp):
    #prob_vec = prob_vec / sampling_temp
    #soft_max = layers.Softmax()
    #prob_vec = soft_max(prob_vec).numpy()
    for i in range(len(prob_vec))[1:]:
        prob_vec[i] = prob_vec[i] + prob_vec[i - 1]
    rand_val = np.random.random()
    char_vec = np.zeros(len(prob_vec))
    if (rand_val < prob_vec[0]):
        char_vec[0] = 1
        return char_vec
    for i in range(len(prob_vec))[1:]:
        if (rand_val < prob_vec[i] and rand_val > prob_vec[i - 1]):
            char_vec[i] = 1
            return char_vec


def predict_chars(initial_chars, model, sampling_temp, num_chars_produce):
    predicted_string = np.copy(initial_chars).tolist()
    current_vec = initial_chars

    for i in range(num_chars_produce):
        prob_vec = model.predict(np.array([current_vec]))[0, len(initial_chars) - 1]
        next_char = select_char(prob_vec, sampling_temp)
        predicted_string.append(next_char.tolist())
        current_vec = np.append(current_vec[1:], np.array([next_char]), axis=0)
    return predicted_string


def train_model(model, x, y, number_epochs, output_rate, callbacks, sampling_temp =1, lr = .01 ) -> list:
    generated_strings = []

    train_model = Sequential()
    train_model.add(model)
    opt = optimizers.RMSprop(learning_rate=lr)
    train_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt)

    predict_model = Sequential()
    predict_model.add(model)
    predict_model.add(layers.Lambda(lambda x: x / sampling_temp))
    predict_model.add(layers.Softmax())



    train_model.add(layers.Softmax())
    for i in range(number_epochs):
        if (i % output_rate == 0):
            random_start = np.random.randint(0, len(x))
            generated_strings.append(predict_chars(x[random_start], predict_model, 1, 5))
            print(''.join(decode_chars(generated_strings[len(generated_strings)-1])))
        train_model.fit(x, y, epochs=i+1, initial_epoch = i, batch_size = 10, callbacks=callbacks)
    return generated_strings



def decode_chars(encoded_values):
    one_hot_dict = load_pickle('one_hot_dict.pkl')
    reverse_dict = {''.join(str(int(e)) for e in values):keys for keys, values in one_hot_dict.items()}
    decoded_chars = []
    for e_letter in encoded_values:
        decoded_chars.append(reverse_dict[''.join(str(int(e)) for e in e_letter)])
    return decoded_chars


def main():
    """This is the main function of train.py

        Run "tensorboard --logdir logs/fit" in terminal and open http://localhost:6006/
    """
    args = get_args()
    # ---------------------- Hyperparameters ---------------------- #

    # ---------------------- Initialize variables ---------------------- #
    print("####### Initializing variables #######")

    # ---------------------- Load and prepare Dataset ---------------------- #
    print("####### Loading Dataset #######")
    # Load the dataset

    # Parameters
    window_size = args.window
    hidden_state = args.hidden
    stride = args.stride
    sampling_temp = args.temperature
    vocab_size = 37
    model_type = args.model
    epochs = args.epochs
    batch_size = 50

    x, y = create_train_data(file_name='beatles.txt', window_size=window_size, stride=stride)
    print("####### Building/Loading the Model #######")
    # ---------------------- Build/Load the Model ---------------------- #

    callbacks = []
    log_folder = "logs/fit/model_" + str(model_type) + \
                 "/hidden_" + str(hidden_state) + \
                 "/stride_" + str(stride) + \
                 "/window_" + str(window_size)

    callbacks.append(TensorBoard(log_dir=log_folder,
                                 histogram_freq=1,
                                 write_graph=True,
                                 write_images=False,
                                 update_freq='epoch',
                                 profile_batch=2,
                                 embeddings_freq=1))

    model = build_model_RNN(model_type, hidden_state, window_size, sampling_temp, vocab_size)
    generated_strings = train_model(model, x, y, epochs, 1, callbacks=callbacks)

    with open(log_folder+"/generated_strings.txt","w+") as f:
        for line in generated_strings:
            f.write(''.join(decode_chars(line))+"\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e

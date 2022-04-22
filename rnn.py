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
    # required_args.add_argument('-t', '--task', type=int, required=True,
    #                            choices=[1, 2, 3, 4, 5], help="The task/model to train on.")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.set_defaults(feature=False)
    optional_args.add_argument("--tuning", action='store_true', required=False,
                               help="Whether to use the validation or training set for training.")
    optional_args.add_argument("--n-rows", default=-1, type=int, required=False,
                               help="How many rows of the dataset to read.")
    optional_args.add_argument("--load-checkpoint", action='store_true', required=False,
                               help="Whether to load model from a checkpoint.")
    optional_args.add_argument("--plot-only", action='store_true', required=False,
                               help="No training, only plot results. "
                                    "Requires the use of --load-checkpoint.")
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    args = parser.parse_args()
    if args.plot_only and not args.load_checkpoint:
        raise ValueError("--plot-only requires --load-checkpoint")
    return args


def build_model_RNN(model_type: str, hidden_size: int, window_size: int, sampling_temp: int, vocab_size: int, lr:int = .1) -> Model:
    """ Build The Model"""
    model = Sequential()

    model.add(layers.Input(shape=(window_size, vocab_size), name='encoder_input'))

    if(model_type =="lstm"):
        model.add(layers.LSTM(hidden_size, return_sequences=True))
    else:
        model.add(layers.SimpleRNN(hidden_size, return_sequences = True))



    model.add(layers.Dense(vocab_size))
    #model.add(layers.Lambda(lambda x: x / sampling_temp))
    #model.add(layers.Softmax())


    opt = optimizers.RMSprop(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt)
    model.summary()
    return model

def select_char(prob_vec, sampling_temp):
    prob_vec = prob_vec/sampling_temp
    soft_max = layers.Softmax()
    prob_vec = soft_max(prob_vec).numpy()
    for i in range(len(prob_vec))[1:]:
        prob_vec[i] = prob_vec[i]+prob_vec[i-1]
    rand_val = np.random.random()
    char_vec = np.zeros(len(prob_vec))
    if(rand_val<prob_vec[0]):
        char_vec[0]= 1
        return char_vec
    for i in range(len(prob_vec))[1:]:
        if ( rand_val<prob_vec[i] and rand_val>prob_vec[i-1]):
            char_vec[i] = 1
            return char_vec

def predict_chars(initial_chars, model, sampling_temp, num_chars_produce):
    predicted_string = np.copy(initial_chars).tolist()
    current_vec = initial_chars

    for i in range(num_chars_produce):
        prob_vec = model.predict(np.array([current_vec]))[0,len(initial_chars)-1]
        next_char = select_char(prob_vec,sampling_temp)
        predicted_string.append(next_char.tolist())
        current_vec = np.append(current_vec[1:], np.array([next_char]),axis=0)
    return predicted_string




def train_model(model, x,y , number_epochs, output_rate) -> Model:

    generated_strings = []
    for i in range(number_epochs):
        if(i%output_rate==0):
            random_start = np.random.randint(0,len(x))
            generated_strings.append(predict_chars(x[random_start],model,1,5))
        model.fit(x,y,epochs =1)





def main():
    """This is the main function of train.py

        Run "tensorboard --logdir logs/fit" in terminal and open http://localhost:6006/
    """
    args = get_args()
    # ---------------------- Hyperparameters ---------------------- #
    # epochs = 70
    # lr = 0.00032
    # batch_size = 32
    # chkp_epoch_to_load = 30
    # chkp_additional_epochs = 30
    # tuning_epochs = 20
    # validation_set_perc = 0.2  # Percentage of the train dataset to use for validation
    # max_conv_layers = 4  # Only for tuning

    # ---------------------- Initialize variables ---------------------- #
    print("####### Initializing variables #######")
    # callbacks = []
    # log_folder = "logs/fit/t-" + str(args.task) + \
    #              "/a-" + args.attr + \
    #              "/b-" + str(batch_size) + \
    #              "/lr-" + str(lr)
    # # Create a validation set suffix if needed
    # val_set_suffix = ''
    # if args.tuning:
    #     val_set_suffix = '_tuning'
    # # Save model path
    # model_name = f'model_{epochs}epochs_{batch_size}batch-size_{lr}lr'
    # if args.n_rows != -1:
    #     model_name += f'_{args.n_rows}rows'
    # model_name += f'{val_set_suffix}.h5'
    # save_dir_path = os.path.join(model_path, f'{args.attr}_attr', f'task_{args.task}')
    # save_file_path = os.path.join(save_dir_path, model_name)
    # chkp_filename = os.path.join(save_dir_path,
    #                              model_name[:-3] + f'_epoch{chkp_epoch_to_load:02d}.ckpt')
    # if not args.tuning:
    #     build_model = build_model_RNN
    # else:
    #     build_model = tune_model_RNN

    # ---------------------- Load and prepare Dataset ---------------------- #
    print("####### Loading Dataset #######")
    # Load the dataset

    # Parameters
    window_size = 20
    hidden_state = 100
    stride = 5
    sampling_temp = 1
    vocab_size = 37
    model_type = "lstm"
    epochs = 5
    batch_size = 50;


    x, y = create_train_data(file_name='beatles.txt', window_size=20, stride=6)


    print("####### Building/Loading the Model #######")
    # ---------------------- Build/Load the Model ---------------------- #

    model = build_model_RNN(model_type, hidden_state, window_size,sampling_temp, vocab_size)
    train_model(model,x,y,epochs,1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e

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
        model.add(layers.lstm(hidden_size, return_sequences=True))
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
    prob_vec = layers.Softmax(prob_vec)
    for i in range(len(prob_vec))[1:]:
        prob_vec[i] = prob_vec[i]+prob_vec[i-1]
    rand_val = np.random.random()
    char_vec = np.zeros(len(prob_vec))
    if(rand_val<prob_vec[0]):
        char_vec[0]= 1
        return char_vec
    for i in range(len(prob_vec))[:1]:
        if ( rand_val<prob_vec[i] and rand_val>prob_vec[i-1]):
            char_vec[i] = 1
            return char_vec

def predict_chars(initial_chars, model, sampling_temp, num_chars_produce):
    predicted_string = np.copy(initial_chars).tolist()
    current_vec = initial_chars

    for i in range(num_chars_produce):
        prob_vec = model.predict(current_vec)
        next_char = select_char(prob_vec,sampling_temp)
        predicted_string.append(next_char.tolist())
        current_vec = np.append(current_vec[1:], predicted_string)
    return predicted_string




def train_model(model, training_data, number_epochs, output_rate) -> Model:

    for i in range(number_epochs):
        if(i%output_rate==0):
            print("")




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

    model = build_model_RNN(model_type, hidden_state, window_size,sampling_temp, vocab_size)
    test = model.fit(x,y,epochs = epochs, batch_size = batch_size)
    # ---------------------- Build/Load the Model ---------------------- #
    print("####### Building/Loading the Model #######")
    # Prepare images for training
    # images_train = images_train.reshape(*images_train.shape, 1)
    # encoded_train_labels = images_train

    # Training/Tuning
    # if not args.tuning:
    #     n_classes = (encoded_train_labels.shape[1], encoded_train_labels_2.shape[1])
    #     encoded_train_labels = [encoded_train_labels, encoded_train_labels_2]
    #     if args.load_checkpoint:
    #         model = tf.keras.models.load_model(chkp_filename)
    #         if chkp_new_lr is not None:
    #             K.set_value(model.optimizer.learning_rate, chkp_new_lr)
    #             print("#### CHANGING LEARNING RATE TO:", chkp_new_lr, " ####")
    #     else:
    #         model = build_model(input_shape=images_train.shape[1:],
    #                             n_classes=n_classes,
    #                             lr=lr)
    #     print(model.summary())
    # else:
    #     print("####### Tuning #######")
    #     build_model = partial(build_model, input_shape=images_train.shape[1:],
    #                           n_classes=encoded_train_labels.shape[1],
    #                           lr=lr, max_conv_layers=max_conv_layers)
    #     model = kt.Hyperband(build_model,
    #                          objective='val_accuracy',
    #                          factor=3,
    #                          directory=os.path.join(model_path,
    #                                                 f'{args.attr}_attr',
    #                                                 f'task_{args.task}'),
    #                          project_name=f'tuning_{epochs}epochs_{batch_size}batchsize_{lr}lr_max_conv_layers{max_conv_layers}')
    #     stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #     callbacks.append(stop_early)
    #     model.search(images_train,
    #                  encoded_train_labels,
    #                  epochs=tuning_epochs,
    #                  batch_size=batch_size,
    #                  validation_split=validation_set_perc,
    #                  callbacks=callbacks)
    #     # Get the optimal hyperparameters
    #     # best_hps = model.get_best_hyperparameters(num_trials=1)[0]
    #     print("Best Model:")
    #     print(model.results_summary())
    #     print(model.search_space_summary())
    #     print("####### Tuning Done #######")
    #     return

    # ---------------------- Fit the Model ---------------------- #
    # if not args.plot_only:
    #     print("####### Fitting the Model #######")
    #     callbacks.append(TensorBoard(log_dir=log_folder,
    #                                  histogram_freq=1,
    #                                  write_graph=True,
    #                                  write_images=False,
    #                                  update_freq='epoch',
    #                                  profile_batch=2,
    #                                  embeddings_freq=1))
    #     callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    #         filepath=os.path.join(save_dir_path, model_name[:-3] + '_epoch{epoch:02d}.ckpt'),
    #         save_weights_only=False,
    #         monitor='val_loss',
    #         mode='auto',
    #         save_best_only=True))
    #
    #     model.fit(images_train,
    #               encoded_train_labels,
    #               initial_epoch=chkp_epoch_to_load if args.load_checkpoint else 0,
    #               epochs=epochs + chkp_additional_epochs if args.load_checkpoint else epochs,
    #               batch_size=batch_size,
    #               validation_split=validation_set_perc,
    #               callbacks=callbacks)

    # ---------------------- Plots ---------------------- #
    # file_writer = tf.summary.create_file_writer(log_folder)

    # ---------------------- Save Model ---------------------- #
    # If we want to save every few epochs:
    # if not args.plot_only:
    #     model.save(save_file_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e

# Base libs
import traceback
import argparse
from functools import partial
from typing import *
# ML libs
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

    model.add(layers.SimpleRNN(hidden_size, return_sequences = True))



    model.add(layers.Dense(vocab_size))
    model.add(layers.Lambda(lambda x: x / sampling_temp))
    model.add(layers.Softmax())


    opt = optimizers.RMSprop(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt)
    model.summary()
    return model

def tune_model_RNN(hp, input_shape: Tuple[int, int], n_classes: int,
                   lr: float = 0.001, max_conv_layers: int = 3) -> Model:
    """ Build a feed-forward conv neural network"""
    # # Tuning Params
    # hp_cnn_activation = [hp.Choice(f'cnn_activation_{i}', values=['relu'], default='relu')
    #                      for i in range(max_conv_layers)]  # Only relu for now
    # hp_dense_activation = hp.Choice('dense_activation', values=['relu'], default='relu')  # Only relu
    # hp_filters = [hp.Choice(f'num_filters_{i}', values=[32, 64, 128], default=32)
    #               for i in range(max_conv_layers)]
    # hp_dense_units = hp.Int('dense_units', min_value=100, max_value=200, step=25)
    # hp_lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)
    # model = Sequential()
    # # Add the layers
    # for i in range(1, hp.Int("num_layers", 2, max_conv_layers + 1)):
    #     model.add(Conv2D(filters=hp_filters[i - 1], kernel_size=3,
    #                      activation=hp_cnn_activation[i - 1], input_shape=input_shape))
    #     model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))
    # model.add(Flatten())
    # model.add(Dense(hp_dense_units, activation=hp_dense_activation))
    # model.add(Dense(n_classes, activation='softmax'))
    # # Select the optimizer and the loss function
    # opt = optimizers.Adam(learning_rate=hp_lr)
    # # opt = optimizers.SGD(learning_rate=hp_lr)
    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
    #               optimizer=opt, metrics=['accuracy', 'mse'])
    # return model
    pass


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
    window_size = 20
    hidden_state = 100
    stride = 5
    sampling_temp = 1
    vocab_size = 37
    model_type = "lstm"



    x, y = create_train_data(file_name='beatles.txt', window_size=20, stride=6)

    model = build_model_RNN(model_type, hidden_state, window_size,sampling_temp, vocab_size)
    test = model.fit(x,y,epochs = 1, batch_size = 50)
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

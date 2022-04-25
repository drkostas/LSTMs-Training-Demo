# Base libs
import traceback
import argparse
from typing import *
# ML libs
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
from tensorflow.keras import backend as K
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import TensorBoard
# Local libs
from src import *
import os


def get_args() -> argparse.Namespace:
    """Set-up the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='Project 4 for the Deep Learning class (COSC 525). '
                    'Involves the development of a Convolutional Neural Network.',
        add_help=False)
    # Core args
    core_args = parser.add_argument_group('Core Arguments')
    core_args.set_defaults(feature=False)
    core_args.add_argument("-d", "--dataset", type=str, required=False, default='beatles.txt',
                           help="The dataset to use.")
    core_args.add_argument('-m', '--model', type=str, required=False, default='rnn',
                           choices=['rnn', 'lstm'], help="The model to use.")
    core_args.add_argument('-h', '--hidden-state', type=int, required=False, default=100,
                           help="The size of the hidden state.")
    core_args.add_argument('-w', '--window', type=int, required=False, default=10,
                           help="The window size.")
    core_args.add_argument('-s', '--stride', type=int, required=False, default=6,
                           help="The stride size.")
    core_args.add_argument('-t', '--temperature', type=int, required=False, default=1,
                           help="The sampling temperature.")
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.set_defaults(feature=False)
    optional_args.add_argument("--tuning", action='store_true', required=False,
                               help="Whether to use the validation or training set for training.")
    optional_args.add_argument("--load-checkpoint", action='store_true', required=False,
                               help="Whether to load model from a checkpoint.")
    optional_args.add_argument("--plot-only", action='store_true', required=False,
                               help="No training, only plot results. "
                                    "Requires the use of --load-checkpoint.")
    optional_args.add_argument("--help", action="help", help="Show this help message and exit")
    args = parser.parse_args()
    if args.plot_only and not args.load_checkpoint:
        raise ValueError("--plot-only requires --load-checkpoint")
    return args


def build_model(model_type: str, hidden_size: int, window_size: int, vocab_size: int,
                lr: float = .01) -> Model:
    """ Build The Model"""
    model = Sequential()

    model.add(layers.Input(shape=(window_size, vocab_size), name='encoder_input'))

    if model_type == "lstm":
        model.add(layers.LSTM(hidden_size, return_sequences=True))
    else:
        model.add(layers.SimpleRNN(hidden_size, return_sequences=True))

    model.add(layers.Dense(vocab_size))
    # model.add(layers.Lambda(lambda x: x / sampling_temp))
    # model.add(layers.Softmax())

    opt = optimizers.RMSprop(learning_rate=lr)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt)
    model.summary()
    return model


def select_char(prob_vec):
    # prob_vec = prob_vec / sampling_temp
    # soft_max = layers.Softmax()
    # prob_vec = soft_max(prob_vec).numpy()
    for i in range(len(prob_vec))[1:]:
        prob_vec[i] = prob_vec[i] + prob_vec[i - 1]
    rand_val = np.random.random()
    char_vec = np.zeros(len(prob_vec))
    if rand_val < prob_vec[0]:
        char_vec[0] = 1
        return char_vec
    for i in range(len(prob_vec))[1:]:
        if prob_vec[i] > rand_val > prob_vec[i - 1]:
            char_vec[i] = 1
            return char_vec


def predict_chars(initial_chars, model, num_chars_produce):
    predicted_string = np.copy(initial_chars).tolist()
    current_vec = initial_chars

    for i in range(num_chars_produce):
        prob_vec = model.predict(np.array([current_vec]))[0, len(initial_chars) - 1]
        next_char = select_char(prob_vec)
        predicted_string.append(next_char.tolist())
        current_vec = np.append(current_vec[1:], np.array([next_char]), axis=0)
    return predicted_string


def train_model(model, x, y, number_epochs, batch_size, callbacks, sampling_temp=1,
                lr=.01, output_rate=1, validation_split=0.01, first_epoch=0) -> Model:
    generated_strings = []

    train_model = Sequential()
    train_model.add(model)
    opt = optimizers.RMSprop(learning_rate=lr)
    train_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt)

    predict_model = Sequential()
    predict_model.add(model)
    predict_model.add(layers.Lambda(lambda j: j / sampling_temp))
    predict_model.add(layers.Softmax())

    train_model.add(layers.Softmax())
    losses = []
    for i in range(first_epoch, number_epochs):
        if i % output_rate == 0:
            random_start = np.random.randint(0, len(x))
            predicted_chars = predict_chars(initial_chars=x[random_start],
                                            model=predict_model, num_chars_produce=5)
            generated_strings.append(predicted_chars)
            print(''.join(decode_chars(generated_strings[len(generated_strings) - 1])))
        history = train_model.fit(x, y, epochs=i + 1, initial_epoch=i, batch_size=batch_size,
                                  callbacks=callbacks, validation_split=validation_split)
        losses.append(history.history['loss'][-1])
    return generated_strings, losses[-1]


def decode_chars(encoded_values):
    one_hot_dict = load_pickle('one_hot_dict.pkl')
    reverse_dict = {''.join(str(int(e)) for e in values): keys for keys, values in
                    one_hot_dict.items()}
    decoded_chars = []
    for e_letter in encoded_values:
        decoded_chars.append(reverse_dict[''.join(str(int(e)) for e in e_letter)])
    return decoded_chars


def tune_model(model_type, tuning_epochs, batch_size, validation_set_perc,
               callbacks, dataset):
    """
    RNN (stride: 12, window_size: 15,lr: 0.001, hidden_state: 100, sampling_temp: 1, output_rate: 3):
    3.0023319721221924
    LSTM (stride: 12, window_size: 5,lr: 0.01, hidden_state: 100, sampling_temp: 1, output_rate: 1):
    3.111567735671997
    """

    for stride in (1, 3, 5):
        for window_size in (5, 15, 30):
            for lr in (0.1, 0.05, 0.001):
                for hidden_state in (100, 300, 500):
                    for sampling_temp in (1, 3, 5):
                        for output_rate in (1, 3):
                            try:
                                del callbacks
                                callbacks = []
                                x, y, vocab_size = create_train_data(window_size=window_size,
                                                                     stride=stride)
                                model_folder_struct = f"val" + \
                                                      f"/dataset_{dataset}" + \
                                                      f"/model_{model_type}" + \
                                                      f"/hidden_{hidden_state}" + \
                                                      f"/batch_{batch_size}" + \
                                                      f"/stride_{stride}" + \
                                                      f"/window_{window_size}" + \
                                                      f"/temperature_{sampling_temp}" + \
                                                      f"/lr_{lr}"
                                log_folder = f"logs/{model_folder_struct}"
                                callbacks.append(TensorBoard(log_dir=log_folder,
                                                             histogram_freq=1,
                                                             write_graph=True,
                                                             write_images=False,
                                                             update_freq='epoch',
                                                             profile_batch=2,
                                                             embeddings_freq=1))
                                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                              patience=5)
                                callbacks.append(stop_early)
                                model = build_model(model_type=model_type, hidden_size=hidden_state,
                                                    window_size=window_size,
                                                    vocab_size=vocab_size, lr=lr)
                                generated_strings, loss = train_model(model, x, y,
                                                                      number_epochs=tuning_epochs,
                                                                      sampling_temp=sampling_temp,
                                                                      output_rate=output_rate,
                                                                      batch_size=batch_size,
                                                                      validation_split=validation_set_perc,
                                                                      callbacks=callbacks)
                                # Get the optimal hyperparameters
                                print(f"Model "
                                      f"(stride: {stride}, "
                                      f"window_size: {window_size},"
                                      f"lr: {lr}, "
                                      f"hidden_state: {hidden_state}, "
                                      f"sampling_temp: {sampling_temp}, "
                                      f"output_rate: {output_rate}): {loss}")
                            except Exception as e:
                                print("lookf0rmehehe")
                                print(e)


def main():
    """This is the main function of rnn.py

        Run "tensorboard --logdir logs/fit" in terminal and open http://localhost:6006/
    """
    tf.random.set_seed(1)
    args = get_args()
    # ---------------------- Hyperparameters ---------------------- #
    epochs = 5
    batch_size = 512
    lr = 0.001
    output_rate = 3
    tuning_epochs = 5  # How many epochs to train for tuning
    chkp_epoch_to_load = 1  # load checkpoint from this epoch
    extra_epochs = 1  # extra epochs to train after loading checkpoint
    validation_set_perc = .1
    dataset = args.dataset
    model_type = args.model
    hidden_state = args.hidden_state
    window_size = args.window
    stride = args.stride
    sampling_temp = args.temperature

    # ---------------------- Initialize variables ---------------------- #
    print("####### Initializing variables #######")
    callbacks = []
    fit_or_val = 'fit' if not args.tuning else 'val'
    model_folder_struct = f"{fit_or_val}" + \
                          f"/dataset_{dataset}" + \
                          f"/model_{model_type}" + \
                          f"/hidden_{hidden_state}" + \
                          f"/batch_{batch_size}" + \
                          f"/stride_{stride}" + \
                          f"/window_{window_size}" + \
                          f"/temperature_{sampling_temp}" + \
                          f"/lr_{lr}"
    # Create a validation set suffix if needed
    # Save model path
    model_name = model_folder_struct.replace("/", "-") + "/model.h5"
    chkp_filename = os.path.join(model_path, model_name[:-3] + f'_epoch{chkp_epoch_to_load:02d}.ckpt')
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_path, model_name[:-3] + '_epoch{epoch:02d}.ckpt'),
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_best_only=True))
    # Setup Tensorboard
    log_folder = f"logs/{model_folder_struct}"
    callbacks.append(TensorBoard(log_dir=log_folder,
                                 histogram_freq=1,
                                 write_graph=True,
                                 write_images=False,
                                 update_freq='epoch',
                                 profile_batch=2,
                                 embeddings_freq=1))

    # ---------------------- Load and prepare Dataset ---------------------- #
    print("####### Loading Dataset #######")
    x, y, vocab_size = create_train_data(file_name=dataset, window_size=window_size, stride=stride)

    # ---------------------- Build/Load the Model ---------------------- #
    print("####### Building/Loading the Model #######")
    if not args.tuning:
        if args.load_checkpoint:
            model = tf.keras.models.load_model(chkp_filename)
        else:
            model = build_model(model_type=model_type, hidden_size=hidden_state,
                                window_size=window_size,
                                vocab_size=vocab_size, lr=lr)
    else:
        print("####### Tuning #######")
        tune_model(model_type=model_type, tuning_epochs=tuning_epochs, batch_size=batch_size,
                   validation_set_perc=validation_set_perc, callbacks=callbacks, dataset=dataset)
        print("####### Tuning Done #######")
        return
    # ---------------------- Train the Model ---------------------- #
    number_epochs = epochs + extra_epochs if args.load_checkpoint else epochs
    first_epoch = chkp_epoch_to_load if args.load_checkpoint else 0
    generated_strings, loss = train_model(model=model, x=x, y=y,
                                          number_epochs=number_epochs,
                                          batch_size=batch_size, output_rate=output_rate,
                                          callbacks=callbacks, sampling_temp=sampling_temp, lr=lr,
                                          validation_split=validation_set_perc,
                                          first_epoch=first_epoch)
    with open(log_folder + "/generated_strings.txt", "w+") as f:
        for line in generated_strings:
            f.write(''.join(decode_chars(line)) + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + '\n' + str(traceback.format_exc()))
        raise e

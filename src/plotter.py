import matplotlib.pyplot as plt
import io
import tensorflow as tf
import numpy as np
import itertools

#
# def visualize_encoder_results(model, images):
#     figure = plt.figure(figsize=(12, 8))
#     # Figure For Image input
#     predicted = model.predict(images[:5])
#     for i in range(5):
#         image = images[i].reshape(32, 32)
#         plt.subplot(2, 5, i + 1)
#         plt.grid(False)
#         plt.imshow(image)
#         plt.subplot(2, 5, i + 6)
#         plt.grid(False)
#         plt.imshow(predicted[i])
#     return figure
#
#
# def visualize_random_input(decoder):
#     figure = plt.figure(figsize=(12, 8))
#     randomInput = np.random.rand(10, 15, 1)
#     randPredict = decoder.predict(randomInput)
#     for i in range(10):
#         # image = images_train[i].reshape(32, 32)
#         plt.subplot(2, 5, i + 1)
#         plt.grid(False)
#         plt.imshow(randPredict[i])
#
#
# def plot_to_image(figure):
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close(figure)
#     buf.seek(0)
#
#     face = tf.image.decode_png(buf.getvalue(), channels=4)
#     face = tf.expand_dims(face, 0)
#
#     return face
#
#
# def plot_confusion_matrix(cm, class_names):
#     figure = plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
#     plt.title("Confusion matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)
#
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
#     threshold = cm.max() / 2.
#
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         color = "white" if cm[i, j] > threshold else "black"
#         plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     return figure

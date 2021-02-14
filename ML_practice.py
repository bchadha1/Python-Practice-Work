import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub

print("TF version: ", tf.__version__)
print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

model_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = model_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32

data_dir = tf.keras.utils.get_file('flower_photos',
                                   'https://storage.googleapis.com/download.tensorflow.org/example_images'
                                   '/flower_photos.tgz',
                                   untar=True)

datagen_kwargs = dict(rescale=1. / 255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = False
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
        zoom_range=0.2, **datagen_kwargs)
else:
    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(data_dir, subset="training", shuffle=True, **dataflow_kwargs)

do_fine_tuning = False
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential(
    [tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)), hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
     tf.keras.layers.Dropout(rate=0.2),
     tf.keras.layers.Dense(train_generator.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))])
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

# Training Model

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=['accuracy'])
steps_per_epoch = train_generator.samples
validation_steps = valid_generator.samples
hist = model.fit(train_generator, epochs=5, steps_per_epoch=steps_per_epoch, validation_data=valid_generator,
                 validation_steps=validation_steps).history

plt.figure()
plt.ylabel("Loss (Training and Validaton)")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (Training and Validaton)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])

def get_class_string_from_index(index):
    for class_string, class_index in valid_generator.class_indices.items():
        if
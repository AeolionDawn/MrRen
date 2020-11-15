# -*- coding: utf-8 -*-
'''
    deepfool对抗训练model_cifar_deepfool
'''
import json
import os
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.datasets import cifar10
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D, Reshape)
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# from tensorflow.python.client import device_lib

# model = keras.applications.InceptionV3(weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels.1.h5')

# os.environ['KERAS_BACKEND'] = 'tensorflow'

# device_lib.list_local_devices()

label_cifar10 = ['airplane', 'automobile', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def getCifar10Label(idx):
    if idx > len(label_cifar10) or idx < 0: return 'error'
    return label_cifar10[idx]

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


input_shape = X_train.shape[1:]
print(input_shape)

M = 1
for i in input_shape:
    M *= i

X_train = X_train.reshape(X_train.shape[0], M)
X_test = X_test.reshape(X_test.shape[0], M)

num_classes = 10
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X_train_new = np.load("new_train_deepfool.npy").astype(np.float32)
y_train_new = y_train

print(X_train.shape, X_train_new.shape)

#对抗训练出来了,数据集合起来
X_train = np.concatenate((X_train, X_train_new))
y_train = np.concatenate((y_train, y_train_new))

Model_Name = 'model_cifar_deepfool.h5'

def buildModel():
    # # Model_Name = 'model_cifar_new.h5'
    # if os.path.exists(Model_Name):
    #     model = keras.models.load_model(Model_Name)
    # else:   # vgg16
    model = keras.Sequential()
    model.add(Reshape((32, 32, 3), input_shape=(M, )))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 224
    model.add(Conv2D(64, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 224
    model.add(Conv2D(64, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 224
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

    model.add(Conv2D(128, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 112
    model.add(Conv2D(128, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 112
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

    model.add(Conv2D(256, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 56
    model.add(Conv2D(256, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 56
    model.add(Conv2D(256, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 56
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

    model.add(Conv2D(512, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 28
    model.add(Conv2D(512, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 28
    model.add(Conv2D(512, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 28
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 14
    model.add(Conv2D(512, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 14
    model.add(Conv2D(512, kernel_size=(3, 3),
                     padding='same', activation='relu'))  # 14
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 7

    # straightening the output
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(Dense(4096, activation='relu'))  # the link layera
    model.add(Dense(4096, activation='relu'))  # the link layer

    model.add(Dense(num_classes, activation='softmax'))
    # define loss function,optimization
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(
                      lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.summary()
    return model

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #              optimizer=keras.optimizers.adadelta(),
    #              metrics=['accuracy'])

     # feature_layers = [
        #     Reshape((32, 32, 3), input_shape=(M, )),
        #     BatchNormalization(),
        #     Conv2D(64, 3, 3, border_mode="same"),
        #     Activation("relu"),
        #     BatchNormalization(),
        #     Conv2D(64, 3, 3, border_mode="same"),
        #     Activation("relu"),
        #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #     BatchNormalization(),
        #     Conv2D(128, 3, 3, border_mode="same"),
        #     Activation("relu"),
        #     BatchNormalization(),
        #     Dropout(0.5),
        #     Conv2D(128, 3, 3, border_mode="same"),
        #     Activation("relu"),
        #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #     BatchNormalization(),
        #     Dropout(0.5),
        #     Conv2D(128, 3, 3, border_mode="same"),
        #     Activation("relu"),
        #     Dropout(0.5),
        #     Conv2D(128, 3, 3, border_mode="same"),
        #     Activation("relu"),
        #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        #     BatchNormalization()
        # ]

        # classification_layer = [
        #     Flatten(),
        #     Dense(512),
        #     Activation("relu"),
        #     Dropout(0.5),
        #     Dense(num_classes),
        #     Activation("softmax")
        # ]
        # model = keras.models.Sequential(feature_layers+classification_layer)

        # initiate RMSprop optimizer
        # opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
        # train the model using RMSprop
        # model.compile(loss='categorical_crossentropy',
        #   optimizer=opt, metrics=['accuracy'])
def train(model, init_epoch = 0):
    hist = model.fit(X_train, y_train, initial_epoch = init_epoch, epochs=20, shuffle=True, validation_data=(X_test, y_test))
    model.save(Model_Name)


model = buildModel()

if os.path.exists(Model_Name):
    model.load_weights(Model_Name)
train(model, 0)
print('accuracy rate')
print(model.evaluate(X_test, y_test, batch_size=128))
print(model.metrics_names)

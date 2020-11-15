# -*- coding: utf-8 -*-
import json
import os
import random

# import geatpy as ea  # Import geatpy
import time
from tensorflow.python import keras
# import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D, Reshape)
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# import cv2
# from MyProblem import MyProblem  # Import MyProblem class

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

Model_Name = 'model_cifar.h5'
# Model_Name = 'model_cifar_new.h5'

#如果原先有模型，直接读取
if os.path.exists(Model_Name):
    model = keras.models.load_model(Model_Name)
else:   # vgg16
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
    model.summary()
    # initiate RMSprop optimizer
    # opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    # train the model using RMSprop
    # model.compile(loss='categorical_crossentropy',
    #   optimizer=opt, metrics=['accuracy'])
    time_start = time.time()
    #epochs=80
    epochs=2
    hist = model.fit(X_train, y_train, epochs=epochs, shuffle=True, validation_data=(X_test, y_test))

    save_dir=os.path.join(os.getcwd(),'models_test')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    Model_Name=Model_Name.replace('.h5',str(epochs) + '.h5')
    Model_path = os.path.join(save_dir, Model_Name)
    model.save(Model_path)

    time_end = time.time()
    print("Took", round((time_end - time_start) / 60,2), "mins to run", epochs, "epochs.")

print('accuracy rate')
print(model.evaluate(X_test, y_test, batch_size=128))
print(model.metrics_names)
# exit(0)


"""=========================Instantiate your problem=========================="""

#种群遗传算法
def processSample(sample, save_data_path):
    sample_label = model.predict_classes(sample.reshape(1, M))[0]
    print('sample_label', getCifar10Label(sample_label))
    choice = [i for i in range(num_classes)]
    choice[sample_label] = choice[-1]
    # for target_label in np.random.choice(choice[:9], 3):
    for target_label in np.random.choice(choice[:9], 1):
    # target_label = sample_label
    # for i in range(1, 2):
        # sample.reshape(M)
        problem = MyProblem(M, model, sample, target_label)     # In
        # problem = MyProblem(1, model, X_train[0], 1)     # In
        """==================================种群设置================================"""
        Encoding = 'RI'           # 编码方式
        NIND = 32# 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes,
                        problem.ranges, problem.borders)  # 创建区域描述器
        # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        population = ea.Population(Encoding, Field, NIND)
        """=================================算法参数设置=============================="""
        myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = 200# 最大进化代数
        """============================调用算法模板进行种群进化========================="""
        NDSet = myAlgorithm.run()  # 执行算法模板，得到帕累托最优解集NDSet
        # NDSet.save(save_data_path)              # 把结果保存到文件中

        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        np.savez(os.path.join(save_data_path, 'Phen'), NDSet.Phen)
        np.savez(os.path.join(save_data_path, 'ObjV'), NDSet.ObjV)
        # np.savez(os.path.join(save_data_path, f'Phen_{target_label}'), NDSet.Phen[:2])
        # np.savez(os.path.join(save_data_path, f'Field_{target_label}'), NDSet.Field[:5])
        # np.savez(os.path.join(save_data_path, f'ObjV_{target_label}'), NDSet.ObjV[:2])

    # # 输出
    # print('用时：%s 秒' % (myAlgorithm.passTime))
    # print('非支配个体数：%s 个' % (NDSet.sizes))
    # print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // myAlgorithm.passTime)))

    # with open('Result/Phen.csv', 'r', encoding='utf-8') as fin:
    #     line = fin.readlines()[2]
    #     x = json.loads("[" + line + "]")
    # a = np.array(x)
    # a = a.astype(float)

    # a.resize(1, 32 * 32 * 3)
    # print(model.predict_classes(a))

    # a.resize(32, 32, 3)
    # # plt.imshow(a)
    # plt.savefig("Result/sample.png")
    # # plt.show()


# X_train_len = X_train.shape[0]
# for i in range(X_train_len)[:10]:
#     # if not os.path.exists(f"cifar_adversarial5/idx_{i}"):
#     processSample(X_train[i], f"cifar_adversarial_new/idx_{i}")
#     if i % 1000 == 0:
#         cur_percentage = i // (X_train_len // 100)
#         print('\r[' + '=' * cur_percentage + ">" +
#               "." * (100 - cur_percentage) + ']')


from six.moves import xrange
import os
import random
import datetime
import re
import math
import logging
import numpy as np
from math import ceil
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.activations as Activation
from keras.optimizers import Adam
# import cv2
import matplotlib.pyplot as plt
import multiprocessing
from keras import regularizers
import glob
from keras.datasets import cifar10

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

def unet():

    model_input = KL.Input(shape=(32, 32, 1), name="u_net_input")
    conv1 = KL.Conv2D(32, (3, 3), name='conv1',padding="same")(model_input)
    conv1 = KL.LeakyReLU(alpha=0.2)(conv1)
    conv1 = KL.Conv2D(32, (3, 3), name='conv1a',padding="same")(conv1)
    conv1 = KL.LeakyReLU(alpha=0.2)(conv1)
    pool1 = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_1")(conv1)

    conv2 = KL.Conv2D(64, (3, 3), name='conv2',padding="same")(pool1)
    conv2 = KL.LeakyReLU(alpha=0.2)(conv2)
    conv2 = KL.Conv2D(64, (3, 3), name='conv2a',padding="same")(conv2)
    conv2 = KL.LeakyReLU(alpha=0.2)(conv2)
    pool2 = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_2")(conv2)

    conv3 = KL.Conv2D(128, (3, 3), name='conv3',padding="same")(pool2)
    conv3 = KL.LeakyReLU(alpha=0.2)(conv3)
    conv3 = KL.Conv2D(128, (3, 3), name='conv3a',padding="same")(conv3)
    conv3 = KL.LeakyReLU(alpha=0.2)(conv3)
    pool3 = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_3")(conv3)

    conv4 = KL.Conv2D(256, (3, 3), name='conv4',padding="same")(pool3)
    conv4 = KL.LeakyReLU(alpha=0.2)(conv4)
    conv4 = KL.Conv2D(256, (3, 3), name='conv4a',padding="same")(conv4)
    conv4 = KL.LeakyReLU(alpha=0.2)(conv4)
    pool4 = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_4")(conv4)

    conv5 = KL.Conv2D(512, (3, 3), name='conv5',padding="same")(pool4)
    conv5 = KL.LeakyReLU(alpha=0.2)(conv5)
    conv5 = KL.Conv2D(512, (3, 3), name='conv5a',padding="same")(conv5)
    conv5 = KL.LeakyReLU(alpha=0.2)(conv5)

    up6   = KL.Deconvolution2D(nb_filter=256, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv0',border_mode='same')(conv5)
    up6   = KL.Concatenate()([conv4,up6])
    conv6 = KL.Conv2D(256, (3, 3), name='conv6',padding="same")(up6)
    conv6 = KL.LeakyReLU(alpha=0.2)(conv6)
    conv6 = KL.Conv2D(256, (3, 3), name='conv6a',padding="same")(conv6)
    conv6 = KL.LeakyReLU(alpha=0.2)(conv6)

    up7   = KL.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv1',border_mode='same')(conv6)
    up7   = KL.Concatenate()([conv3,up7])
    conv7 = KL.Conv2D(128, (3, 3), name='conv7',padding="same")(up7)
    conv7 = KL.LeakyReLU(alpha=0.2)(conv7)
    conv7 = KL.Conv2D(128, (3, 3), name='conv7a',padding="same")(conv7)
    conv7 = KL.LeakyReLU(alpha=0.2)(conv7)

    up8   = KL.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv2',border_mode='same')(conv7)
    up8   = KL.Concatenate()([conv2,up8])
    conv8 = KL.Conv2D(64, (3, 3), name='conv8',padding="same")(up8)
    conv8 = KL.LeakyReLU(alpha=0.2)(conv8)
    conv8 = KL.Conv2D(64, (3, 3), name='conv8a',padding="same")(conv8)
    conv8 = KL.LeakyReLU(alpha=0.2)(conv8)

    up9   = KL.Deconvolution2D(nb_filter=32, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv3',border_mode='same')(conv8)
    up9   = KL.Concatenate()([conv1,up9])
    conv9 = KL.Conv2D(32, (3, 3), name='conv9',padding="same")(up9)
    conv9 = KL.LeakyReLU(alpha=0.2)(conv9)
    conv9 = KL.Conv2D(32, (3, 3), name='conv9a',padding="same")(conv9)
    conv9 = KL.LeakyReLU(alpha=0.2)(conv9)

    conv10 = KL.Conv2D(12, (3, 3), name='conv10',padding="same")(conv9)
    conv10 = KL.LeakyReLU(alpha=0.2)(conv10)
    # model_output = KL.Lambda(lambda t:tf.depth_to_space(t, 2))(conv10)
    model_output = KL.Conv2D(3, (1, 1), name='output', padding='same')(conv10)
    model = KM.Model(inputs=model_input,outputs=model_output)
    return model

def unet_v2():
    "unet for image reconstruction"
    # model_input = KL.Input(shape=(None,None,4),name="u_net_input")
    model_input = KL.Input(shape=(32, 32, 3),name="u_net_input")
    conv1  = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1', use_bias=True,padding="same")(model_input)
    conv1  = KL.LeakyReLU(alpha=0.2)(conv1)
    conv1a = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1a', use_bias=True,padding="same")(conv1)
    conv1a = KL.LeakyReLU(alpha=0.2)(conv1a)
    P1     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_1")(conv1a)

    conv2  = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2', use_bias=True,padding="same")(P1)
    conv2  = KL.LeakyReLU(alpha=0.2)(conv2)
    conv2a = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2a', use_bias=True,padding="same")(conv2)
    conv2a = KL.LeakyReLU(alpha=0.2)(conv2a)
    P2     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_2")(conv2a)

    conv3  = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3', use_bias=True,padding="same")(P2)
    conv3  = KL.LeakyReLU(alpha=0.2)(conv3)
    conv3a = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3a', use_bias=True,padding="same")(conv3)
    conv3a = KL.LeakyReLU(alpha=0.2)(conv3a)
    P3     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_3")(conv3a)

    conv4  = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4', use_bias=True,padding="same")(P3)
    conv4  = KL.LeakyReLU(alpha=0.2)(conv4)
    conv4a = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4a', use_bias=True,padding="same")(conv4)
    conv4a = KL.LeakyReLU(alpha=0.2)(conv4a)
    P4     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_4")(conv4a)

    conv5  = KL.Conv2D(768, (3, 3), strides=(1, 1), name='conv5', use_bias=True,padding="same")(P4)
    conv5  = KL.LeakyReLU(alpha=0.2)(conv5)
    conv5a = KL.Conv2D(768, (3, 3), strides=(1, 1), name='conv5a', use_bias=True,padding="same")(conv5)
    conv5a = KL.LeakyReLU(alpha=0.2)(conv5a)
    P5     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_5")(conv5a)


    up4      = KL.Deconvolution2D(nb_filter=384, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv4',border_mode='same')(P5)
    up4      = KL.LeakyReLU(alpha=0.2)(up4)
    C4       = KL.Concatenate()([P4,up4])
    conv4_u  = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4_u', use_bias=True,padding="same")(C4)
    conv4_u  = KL.LeakyReLU(alpha=0.2)(conv4_u)
    conv4a_u = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4a_u', use_bias=True,padding="same")(conv4_u)
    conv4a_u = KL.LeakyReLU(alpha=0.2)(conv4a_u)

    up3      = KL.Deconvolution2D(nb_filter=192, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv3',border_mode='same')(conv4a_u)
    up3      = KL.LeakyReLU(alpha=0.2)(up3)
    C3       = KL.Concatenate()([P3,up3])
    conv3_u  = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3_u', use_bias=True,padding="same")(C3)
    conv3_u  = KL.LeakyReLU(alpha=0.2)(conv3_u)
    conv3a_u = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3a_u', use_bias=True,padding="same")(conv3_u)
    conv3a_u = KL.LeakyReLU(alpha=0.2)(conv3a_u)


    up2      = KL.Deconvolution2D(nb_filter=96, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv2',border_mode='same')(conv3a_u)
    up2      = KL.LeakyReLU(alpha=0.2)(up2)
    C2       = KL.Concatenate()([P2,up2])
    conv2_u  = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2_u', use_bias=True,padding="same")(C2)
    conv2_u  = KL.LeakyReLU(alpha=0.2)(conv2_u)
    conv2a_u = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2a_u', use_bias=True,padding="same")(conv2_u)
    conv2a_u = KL.LeakyReLU(alpha=0.2)(conv2a_u)

    up1      = KL.Deconvolution2D(nb_filter=48, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv1',border_mode='same')(conv2a_u)
    up1      = KL.LeakyReLU(alpha=0.2)(up1)
    C1       = KL.Concatenate()([P1,up1])
    conv1_u  = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1_u', use_bias=True,padding="same")(C1)
    conv1_u  = KL.LeakyReLU(alpha=0.2)(conv1_u)
    conv1a_u = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1a_u', use_bias=True,padding="same")(conv1_u)
    conv1a_u = KL.LeakyReLU(alpha=0.2)(conv1a_u)

    up0      = KL.Deconvolution2D(nb_filter=24, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv0',border_mode='same')(conv1a_u)
    up0      = KL.LeakyReLU(alpha=0.2)(up0)
    C0       = KL.Concatenate()([model_input,up0])
    conv0_u  = KL.Conv2D(24, (3, 3), strides=(1, 1), name='conv0_u', use_bias=True,padding="same")(C0)
    conv0_u  = KL.LeakyReLU(alpha=0.2)(conv0_u)
    conv0a_u = KL.Conv2D(24, (3, 3), strides=(1, 1), name='conv0a_u', use_bias=True,padding="same")(conv0_u)
    conv0a_u = KL.LeakyReLU(alpha=0.2)(conv0a_u)

    x = KL.Conv2D(12,(1,1),strides=(1,1),name='convr',use_bias=True,padding="same")(conv0a_u)
    x = KL.LeakyReLU(alpha=0.2)(x)
    # model_output = KL.Lambda(lambda t:tf.depth_to_space(t,2))(x)
    # model_output = KL.Lambda(lambda t:tf.depth_to_space(t,2))(x)
    model_output = KL.Conv2D(3, (1, 1), name='output', padding='same')(x)
    model = KM.Model(inputs=model_input, outputs=model_output)
    return model




class Dataset(object):

    def __init__(self, data_mode=None):
        if data_mode=='train':
            data_list = '../dataset/train_list.txt'
        elif data_mode=='dev':
            data_list = '../dataset/val_list.txt'
        elif data_mode=='test':
            data_list = '../dataset/test_list.txt'
        else:
            print("wrong parameters for data_mode")
            # exit()
        # with open(data_list) as f:
            # self.image_pairs = [line.strip().split(' ')[:2] for line in f.readlines()]

        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()


    def load_image(self):
        pass

# In[6]:


# def data_generator(data, batch_size):
#     # image_pairs = np.copy(dataset.image_pairs)

#     # (X_train, y_train), (X_test, y_test) = cifar10.load_data()

#     shape = data.shape[1:]

#     while True:
#         try:
#             image_pair = [data[i] / 255 for i in random.sample(range(len(data)), batch_size)]

#             batch_noise_image  = np.zeros((batch_size, *shape))
#             batch_target_image = np.zeros((batch_size, *shape))

#             noise_upbound = (80, 200)
#             noise_lowbound = (0, 10)

#             noise_upbound_count = (10, 50)
#             noise_lowbound_count = (10, 50)

#             for i in range(batch_size):
#                 target_img = image_pair[i]
#                 noise_img = target_img.copy()
#                 for i in np.random.randint(0, noise_lowbound_count):
#                     px = np.random.randint(0, shape[0])
#                     py = np.random.randint(0, shape[1])
#                     origin_pixel = noise_img[px][py]
#                     for ch in range(shape[-1]):
#                         randnoise = np.random.randint(low=noise_lowbound[0], high=noise_lowbound[1])
#                         if random.randint(0, 1) == 0:
#                                 noise_img[px, py, ch] = min(1.0, origin_pixel[ch] + randnoise / 255)
#                         else:
#                                 noise_img[px, py, ch] = max(0.0, origin_pixel[ch] - randnoise / 255)

#                 for i in np.random.randint(0, noise_upbound_count):
#                     px = np.random.randint(0, shape[0])
#                     py = np.random.randint(0, shape[1])
#                     origin_pixel = noise_img[px][py]
#                     for ch in range(shape[-1]):
#                     # randnoise = np.random.randint(noise_upbound)
#                         randnoise = np.random.randint(low=noise_upbound[0], high=noise_upbound[1])
#                         if random.randint(0, 1) == 0:
#                                 noise_img[px, py, ch] = min(1.0, origin_pixel[ch] + randnoise / 255)
#                         else:
#                                 noise_img[px, py, ch] = max(0.0, origin_pixel[ch] - randnoise / 255)

#                 batch_noise_image[i,:,:,:]  = noise_img
#                 batch_target_image[i,:,:,:] = target_img

#             inputs = [batch_target_image]
#             outputs = [batch_noise_image]

#             # inputs = [batch_noise_image]
#             # outputs = [batch_target_image]

#             yield inputs,outputs
#         except (GeneratorExit, KeyboardInterrupt):
#             raise



def data_generator(data, batch_size, cw_dict): # 一次产生data_size 个样本
    shape = data.shape[1:]
    # 按直方图进行低像素扰动生成



    choice_keys = list(cw_dict.keys())
    choice_pv = list(cw_dict.values())
    choice_pv[0] = 1.0 - sum(choice_pv[1:])
    
    while True:
        try:
            image_pair = [data[i] / 255 for i in random.sample(range(len(data)), batch_size)]       # 随机采样

            batch_noise_image  = np.zeros((batch_size, *shape))
            batch_target_image = np.zeros((batch_size, *shape))

            noise_upbound = (150, 250)      # 高像素扰动变化数字
            
            noise_upbound_count = (10, 50)  # 高像素扰动像素点数量
            noise_lowbound_count = (300, 400)   # 低像素点扰动数量

            for i in range(batch_size):
                
                target_img = image_pair[i]
   
                noise_img = np.zeros(target_img.shape)

                # 按照直方图数据产生扰动
                cw_noise = np.random.choice(choice_keys, size = shape, p = choice_pv) / 255
                noise_img += cw_noise

                # for _ in np.random.randint(0, noise_upbound_count):     # 随机添加高像素扰动
                #     px = np.random.randint(0, shape[0])
                #     py = np.random.randint(0, shape[1])
                #     origin_pixel = noise_img[px][py]
                #     randnoise = np.random.randint(low=noise_upbound[0], high=noise_upbound[1])

                #     ch = np.random.randint(shape[2])
                #     if random.randint(0, 1) == 0:                       # 随机变大或变小
                #         noise_img[px, py, ch] = min(1.0, origin_pixel[ch] + randnoise / 255)    
                #     else:
                #         noise_img[px, py, ch] = max(0.0, origin_pixel[ch] - randnoise / 255)

                batch_noise_image[i,:,:,:]  = (target_img + noise_img).clip(0.0, 1.0)
                batch_target_image[i,:,:,:] = target_img


            inputs = [batch_noise_image]
            outputs = [batch_target_image]

            yield inputs,outputs
        except (GeneratorExit, KeyboardInterrupt):
            raise

def train(model, batch_size, start_epoch=0):


    cw_dict2 = {-1.0: 0.3219401041666654, 0.0: 0.3258463541666652,  # 直方图分布概率
            -2.0: 0.08398437499999986, 1.0: 0.09082031249999976,
            -3.0: 0.03971354166666668, 2.0: 0.038411458333333336,
            3.0: 0.018229166666666668, -4.0: 0.022460937499999986,
            4.0: 0.011718750000000003, 6.0: 0.006835937499999997,
            5.0: 0.006510416666666664, -7.0: 0.006835937499999997,
            -6.0: 0.00390625, -5.0: 0.01139322916666667,
            -10.0: 0.0009765625, 7.0: 0.00390625,
            -8.0: 0.004882812499999999, -9.0: 0.0013020833333333333,
            -11.0: 0.0019531249999999998, 9.0: 0.0013020833333333333,
            8.0: 0.002278645833333333, 11.0: 0.0006510416666666666,
            12.0: 0.0006510416666666666, -13.0: 0.0006510416666666666,
            -12.0: 0.0009765625} 

    train_generator = data_generator(X_train, batch_size=batch_size, cw_dict2)
    val_generator = data_generator(X_test, batch_size=batch_size)

    #Train
    #log("\nStarting at epoch {}. LR={}\n".format(self.epoch,learning_rate))
    #log("Checkpoint Path: {}".format(self.checkpoint_path))
    learning_rate = 0.0001
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=0.0, amsgrad=False)
#     model.compile(optimizer=adam, loss='mean_squared_error',metrics=['mae'])
    model.compile(optimizer=adam, loss="mean_absolute_error", metrics=['mae'])
    workers = multiprocessing.cpu_count()
    # keras.models.Model.fit_generator()
    model.fit_generator(
        train_generator,
        initial_epoch = start_epoch,
        epochs = 20,
        steps_per_epoch = 1024,
        # callbacks = callbacks,
        validation_data = val_generator,
        validation_steps =  10,
        max_queue_size = 10,
        workers = workers,
        use_multiprocessing = True,
    )

    # self.epoch = max(self.epoch, epochs)


def dunet(samples):
    model = unet_v2()
    model.load_weights('model_unet.h5')
    res = model.predict(samples).clip(0.0, 1.0)
    return res

if __name__ == '__main__':
    model = unet_v2()
#     model = unet()
#     keras.utils.plot_model(model, show_shapes=True)
    model.summary()
    train(model, 128)
    model.load_weights('model_unet_2.h5')
    # adv = plt.imread('adv_img.jpg') / 255
    # res = model.predict(np.array([adv]))[0].clip(0.0, 1.0)
    # print(res.min(), res.max())
    # plt.imsave('adv_out.jpg', res)
    # diff = res - adv
    # import seaborn as sns
    # sns.distplot((diff * 255).astype(np.int))
    # plt.savefig('diff.png')
    # train(model, 128, 95)
    # model.save("model_unet.h5")
#     train_generator = data_generator(Dataset(), batch_size=128)

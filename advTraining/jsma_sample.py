'''
    ifgsm对抗训练的model_cifar和model_cifar_new，以及dunet的模型model_unet进行jsma攻击测试

'''
import keras
import matplotlib.pyplot as plt
import numpy as np
from cleverhans.attacks import LBFGS, SaliencyMapMethod, ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications.imagenet_utils import decode_predictions
from keras.datasets import cifar10
# import cv2

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))  # 此处不同

# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
if __name__ == '__main__':
    # model_keras = keras.models.load_model('../model_cifar.h5')  # 88.7%
    # model_denoise = keras.models.load_model('model_unet.h5')
    model_keras = keras.models.load_model('model_cifar_new.h5') #79
    # model_keras = keras.models.load_model('../../genicAlgorithm/demo_cifar2/model_cifar_new5.h5')
    batch_size = 64
    success = 0
    denoise_success = 0

    data_size = 64
    idx = 0
    for st in range(idx, data_size, batch_size):
        ed = min(data_size, st + batch_size)
        sample = np.array(
            X_test[st: ed].reshape(-1, 32 * 32 * 3) / 255, dtype=np.float)
        # sample = np.array([sample])
        sess = keras.backend.get_session()
        model = KerasModelWrapper(model_keras)
        attack = SaliencyMapMethod(model, sess=sess)
        # print(model.predict(panda.reshape(1, *panda.shape)))

        param = dict(
            theta=1.,
            gamma=1.,
            clip_min=0.,
            clip_max=1.,
            y_target=None,
            symbolic_impl=True,
        )
        advs = attack.generate_np(sample, **param)
        # plt.imsave("sample.png", advs[0])

        preb = model_keras.predict(advs).argmax(
            axis=1).reshape((sample.shape[0], ))
        y_sample = model_keras.predict(sample).argmax(
            axis=1).reshape((sample.shape[0], ))
        # y_sample = y_sample.reshape((y_sample.shape[0], ))

        success += (preb != y_sample).sum()
        print((preb != y_sample).sum())

        # denoise_adv = model_denoise.predict(advs.reshape(-1, 32, 32, 3))

        # denoise_pred = model_keras.predict(
        #     denoise_adv.reshape(-1, 32 * 32 * 3))
        # print('denoise', (denoise_pred.argmax(axis=1).reshape(
        #     (sample.shape[0], )) != y_sample).sum())

        # denoise_success += (denoise_pred.argmax(axis=1).reshape(
        #     (sample.shape[0], )) != y_sample).sum()
        # break
    #测试训练模型和dunet降噪模型
    print(success / data_size, denoise_success / data_size)
    with open ('result_jsma.txt', 'w+') as fout:
        fout.write(str(success / data_size))
    # adv_img = (advs[0].copy() * 255).astype(np.uint8).reshape((32, 32, 3))
    # origin_img = X_test[idx]
    # print(success / data_size)
    # plt.subplot(1, 2, 1)
    # plt.imshow(origin_img)
    # plt.imsave('origin_img.jpg', origin_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(adv_img)
    # plt.imsave('adv_img.jpg', adv_img)
    # plt.savefig('fig.jpg')

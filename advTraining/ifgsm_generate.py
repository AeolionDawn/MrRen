'''
    用原始模型model_cifar生成ifgsm对抗样本数据集npy文件new_train
'''
# import keras
import time
from tensorflow.python import keras
import matplotlib.pyplot as plt
import numpy as np
from cleverhans.attacks import LBFGS, FastGradientMethod, ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.datasets import cifar10
import cv2

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
if __name__ == '__main__':
    model_keras = keras.models.load_model('../models_test/model_cifar_2.h5')
    # model_keras = keras.models.load_model('../models_test/keras_cifar10_trained_model.h5')
    batch_size = 512
    success = 0

    data_size = X_train.shape[0]
    adv_train = []
    time_st=time.time()
    for st in range(0, data_size, batch_size):
        sample = np.array(X_train[st : st + batch_size].reshape(-1, 32 * 32 * 3) / 255, dtype=np.float)
        # sample = np.array([sample])
        sess = keras.backend.get_session()
        model = KerasModelWrapper(model_keras)
        attack = ProjectedGradientDescent(model, sess=sess)
        # print(model.predict(panda.reshape(1, *panda.shape)))

        param = dict(
                eps= 10 / 255,
                eps_iter= 10 / 255 / 40,
                nb_iter= 40,
                rand_init= True,
                )
        advs = attack.generate_np(sample, **param)
        # plt.imsave("sample.png", advs[0])
        adv_train.append(advs)
        preb = model_keras.predict(advs).argmax(axis=1).reshape((sample.shape[0], ))
        y_sample = model_keras.predict(sample).argmax(axis=1).reshape((sample.shape[0], ))
        success += (preb != y_sample).sum()
        print((preb != y_sample).sum())

    time_ed=time.time()
    print("Took", round((time_ed - time_st) / 60,2), "mins to run")

    print(success / data_size)
    new_train = np.concatenate(adv_train)
    #保存ifgsm对抗样本数据集文件
    np.save('new_train', new_train)
    # plt.imshow(adv[0])
    # plt.show()

# import numpy as np
# import keras
# from keras import backend
# from keras.models import load_model
# import tensorflow as tf
# from cleverhans.utils_keras import KerasModelWrapper
# from keras.applications import inception_v3
# from keras.applications import imagenet_utils
# import cleverhans.attacks
# from cleverhans.attacks import CarliniWagnerL2
# import scipy.misc
# import matplotlib.pyplot as plt
# import cv2
# from keras.datasets import cifar10
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# keras_model = keras.models.load_model('model_cifar.h5')

# # Set the learning phase to false, the model is pre-trained.
# backend.set_learning_phase(False)

# # Set TF random seed to improve reproducibility
# tf.set_random_seed(1234)

# # if keras.backend.image_dim_ordering() != 'tf':
# #     keras.backend.set_image_dim_ordering('tf')
# #     print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
# #           "'th', temporarily setting to 'tf'")

# # Retrieve the tensorflow session
# sess = backend.get_session()

# image = X_train[0] / 255
# # Resizing the image to be of size 299 * 299

# # converting each pixel to the range [0,1] (Normalization)

# image = np.array([image / 255.0])

# wrap = KerasModelWrapper(keras_model)

# cw = cleverhans.attacks.FastGradientMethod(wrap, sess = sess)

# # carlini and wagner
# cw_params = {'batch_size': 1,
#              'confidence': 10,
#              'learning_rate': 0.1,
#              'binary_search_steps': 5,
#              'max_iterations': 1000,
#              'abort_early': True,
#              'initial_const': 0.01,
#              'clip_min': 0,
#              'clip_max': 1}

# adv_cw = cw.generate_np(image)

# plt.imshow(adv_cw[0])
# plt.show()

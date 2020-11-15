'''
    model_cifar_new和model_cifar_deepfool进行fgsm攻击测试
'''
from tensorflow.python import keras
import matplotlib.pyplot as plt
import numpy as np
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.datasets import cifar10
import cv2

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
if __name__ == '__main__':
    model_keras=keras.models.load_model('../models_test/model_cifar_10.h5')

    # model_keras = keras.models.load_model('model_cifar_new.h5')  #76.95
    # model_keras = keras.models.load_model('model_cifar_deepfool.h5') # 87.26
    batch_size = 512
    success = 0

    data_size = 10000

    for st in range(0, data_size, batch_size):
        sample = np.array(X_test[st : st + batch_size].reshape(-1, 32 * 32 * 3) / 255, dtype=np.float)
        # sample = np.array([sample])
        sess = keras.backend.get_session()
        model = KerasModelWrapper(model_keras)
        attack = FastGradientMethod(model, sess=sess)
        # print(model.predict(panda.reshape(1, *panda.shape)))

        param = {
            'eps' : 10 / 255,
            'clip_min': 0,
            'clip_max': 1
        }
        advs = attack.generate_np(sample, **param)

        # plt.imsave("sample.png", advs[0])

        preb = model_keras.predict(advs).argmax(axis=1).reshape((sample.shape[0], ))
        y_sample = model_keras.predict(sample).argmax(axis=1).reshape((sample.shape[0], ))

        success += (preb != y_sample).sum()
        print((preb != y_sample).sum())
    print(success / data_size)
    # plt.imshow(advs[0])
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

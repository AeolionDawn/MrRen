'''
    用原始模型model_cifar生成deepfool对抗样本数据集npy文件new_train_deepfool
'''
import keras
import matplotlib.pyplot as plt
import numpy as np
from cleverhans.attacks import DeepFool, DeepFool, ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications.imagenet_utils import decode_predictions
from keras.datasets import cifar10
import cv2

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
if __name__ == '__main__':
    model_keras = keras.models.load_model('model_cifar.h5')
    batch_size = 512
    success = 0

    data_size = X_train.shape[0]
    adv_train = []
    for st in range(0, data_size, batch_size):
        sample = np.array(X_train[st : st + batch_size].reshape(-1, 32 * 32 * 3) / 255, dtype=np.float)
        # sample = np.array([sample])
        sess = keras.backend.get_session()
        model = KerasModelWrapper(model_keras)
        attack = DeepFool(model, sess=sess)
        # print(model.predict(panda.reshape(1, *panda.shape)))

        param = dict(nb_candidate=10,
                       overshoot=0.01,
                       #overshoot=0.0,
                       max_iter=20,
                       clip_min=0.,
                       clip_max=1.
            )
        advs = attack.generate_np(sample, **param)
        # plt.imsave("sample.png", advs[0])
        adv_train.append(advs)
        preb = model_keras.predict(advs).argmax(axis=1).reshape((sample.shape[0], ))
        y_sample = model_keras.predict(sample).argmax(axis=1).reshape((sample.shape[0], ))
        success += (preb != y_sample).sum()
        print((preb != y_sample).sum())

    print(success / data_size)
    new_train = np.concatenate(adv_train)
    np.save('new_train_deepfool', new_train)
    # plt.imshow(adv[0])
    # plt.show()

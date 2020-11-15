'''
    model_cifar_new和model_cifar进行deepfool攻击测试
'''
import keras
import matplotlib.pyplot as plt
import numpy as np
from cleverhans.attacks import DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications.imagenet_utils import decode_predictions
from keras.datasets import cifar10
import cv2

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
if __name__ == '__main__':
    # model_keras = keras.models.load_model('model_cifar.h5')
    # model_keras = keras.models.load_model('../model_cifar_new.h5') #79

    batch_size = 512
    data_size = 10000
    res = []
    for idx in range(1, 2):
        success = 0
        model_keras = keras.models.load_model('model_cifar_new.h5') #86
        # model_keras = keras.models.load_model('model_cifar.h5') #92.57
        # model_keras = keras.models.load_model(f'../../genicAlgorithm/demo_cifar2/model_cifar_new{idx}.h5')
        for st in range(0, data_size, batch_size):
            ed = min(data_size, st + batch_size)
            sample = np.array(X_test[st : ed].reshape(-1, 32 * 32 * 3) / 255, dtype=np.float)
            y_sample = model_keras.predict(sample).argmax(axis=1).reshape((sample.shape[0], ))
            # sample = np.array([sample])
            sess = keras.backend.get_session()
            model = KerasModelWrapper(model_keras)
            attack = DeepFool(model, sess=sess)
            # print(model.predict(panda.reshape(1, *panda.shape)))
            param = dict(nb_candidate=10,
                       #overshoot=0.02,
                       overshoot=0.0,
                       max_iter=10,
                       clip_min=0.,
                       clip_max=1.
            )

            advs = attack.generate_np(sample, **param)
            preb = model_keras.predict(advs).argmax(axis=1).reshape((sample.shape[0], ))

            success += (preb != y_sample).sum()
            print((preb != y_sample).sum())
        print(success / data_size)
        res.append(success / data_size)
    print(res)


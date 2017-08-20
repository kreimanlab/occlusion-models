import os

import PIL.Image as pil_image
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.engine import Layer
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model


# from https://github.com/yjn870/keras-caffenet

def CaffeNet(weights=None, input_shape=(3, 227, 227), classes=1000):
    inputs = Input(shape=input_shape)
    dim_ordering = 'th'
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv1', dim_ordering=dim_ordering)(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1', dim_ordering=dim_ordering)(x)
    x = LRN2D(name='norm1')(x)
    x = ZeroPadding2D((2, 2), dim_ordering=dim_ordering)(x)
    x = Conv2D(256, (5, 5), activation='relu', name='conv2', dim_ordering=dim_ordering)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2', dim_ordering=dim_ordering)(x)
    x = LRN2D(name='norm2')(x)
    x = ZeroPadding2D((1, 1), dim_ordering=dim_ordering)(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv3', dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D((1, 1), dim_ordering=dim_ordering)(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv4', dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D((1, 1), dim_ordering=dim_ordering)(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv5', dim_ordering=dim_ordering)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool5', dim_ordering=dim_ordering)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5, name='drop6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5, name='drop7')(x)
    x = Dense(classes, name='fc8')(x)
    x = Activation('softmax', name='loss')(x)

    model = Model(inputs, x, name='caffenet')

    model.load_weights(weights)

    return model


class LRN2D(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                "LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)

        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k

        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta

        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_image(path):
    image = pil_image.open(path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((256, 256))
    x1 = int(image.width / 2 - 227 / 2)
    x2 = int(x1 + 227)
    y1 = int(image.height / 2 - 227 / 2)
    y2 = int(y1 + 227)
    image = image.crop((x1, y1, x2, y2))

    im = np.array(image).astype('float32')
    im = im.transpose((2, 0, 1))

    im[0, :, :] -= 123.68
    im[1, :, :] -= 116.779
    im[2, :, :] -= 103.939

    r = im[0, :, :].copy()
    b = im[2, :, :].copy()
    im[0, :, :] = b
    im[2, :, :] = r

    return np.expand_dims(im, axis=0)


if __name__ == '__main__':
    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '../../../data/imagenet/ILSVRC2012_val_00019877.JPEG')
    im = load_image(image_path)

    model = CaffeNet(weights='caffenet_weights_th.h5')

    preds = model.predict(im)
    print(decode_predictions(preds)[0])

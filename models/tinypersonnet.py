# A fully convolutional person detector for small images and few training data.
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize
from models.googlenet import inception_module, pooling_module
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2


def prepare_data(img):
    img = imresize(img, (96, 160)).astype(np.float32)
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # rgb2bgr
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img


def tinypersonnet(weights_path=None, remove_classifier=False):
    x = Input(shape=(96, 160, 3))
    print("Input shape: ")
    print(x._keras_shape)

    conv1 = x
    conv1 = Conv2D(16, (3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = pool1
    conv2 = Conv2D(16, (3, 3), activation='relu')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = pool2
    conv3 = Conv2D(32, (3, 3), activation='relu')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = pool3
    conv4 = Conv2D(32, (3, 3), activation='relu')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = pool4
    conv5 = Conv2D(32, (3, 3), activation='relu')(conv5)
    pool5 = MaxPooling2D((2,2), strides=(2,2))(conv5)

    conv6 = Conv2D(32, (1, 3), activation='relu')(pool5)
    conv6 = Dropout(0.5)(conv6)

    conv7 = Conv2D(2, (1,1), activation='softmax')(conv6)

    output = Flatten()(conv7)
    print("Output shape: ")
    print(output._keras_shape)

    # Create a keras model
    model = Model(inputs=[x], outputs=[output])
    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == "__main__":
    img = prepare_data(imread('cat.jpg', mode="RGB"))
    model = tinypersonnet("models/weights/tinypersonnet.h5")

    # Test pretrained model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img)
    print(np.argmax(out))

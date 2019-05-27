# 93.79% accuracy

from keras.datasets import cifar10
import numpy as np
import cv2
from keras.activations import softmax
from keras.backend import softmax
from keras.backend import int_shape
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, AveragePooling2D, concatenate, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils

(X_train_base, Y_train_base), (X_test_base, Y_test_base) = cifar10.load_data()
print("LOADED")
X_train_base = np.array([cv2.resize(img, (128, 128)) for img in X_train_base[:15000,:,:,:]])
print("x_train OK")
X_test_base = np.array([cv2.resize(img, (128, 128)) for img in X_test_base[:3000,:,:,:]])
print("x_test OK")

X_train = X_train_base
print(X_train.shape)
X_test = X_test_base
Y_train = Y_train_base[:15000,:]
Y_test = Y_test_base[:3000,:]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = np.divide(X_train, 255.0)
X_test = np.divide(X_test, 255.0)

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)


def inception_node(data_in, filters):
    # all the conv layers' strides are (1, 1)
    a = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(data_in)
    b = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(data_in)
    b = Conv2D(filters, (3, 3), padding = "same", data_format = "channels_last", activation = 'relu')(b)
    c = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(data_in)
    c = Conv2D(filters, (5, 5), padding = "same", data_format = "channels_last", activation = 'relu')(c)
    d = MaxPooling2D((3, 3), strides = (1, 1), padding = "same")(data_in)
    d = Conv2D(filters, (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(d)
    
    merged = concatenate([a, b, c, d], axis = 3)
    return merged

def softmax_out(data_in):
    sf = AveragePooling2D((5, 5), strides = (3, 3), data_format = "channels_last")(data_in)
    sf = Conv2D(256, (1, 1), strides = (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(sf)
    sf = Flatten()(sf)
    sf = Dense(1024, activation = "relu")(sf)
    sf = Dropout(0.7)(sf)
    out = Dense(10, activation = "softmax")(sf)
    return out

in_images = Input(shape = (128, 128, 3))
images = Conv2D(64, (7, 7), strides = (2, 2), padding = "same", data_format = "channels_last", activation = "relu")(in_images)
images = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(images)
images = BatchNormalization()(images)
images = Conv2D(192, (1, 1), strides = (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(images)
images = Conv2D(192, (3, 3), strides = (1, 1), padding = "same", data_format = "channels_last", activation = 'relu')(images)
images = BatchNormalization()(images)
images = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(images)

inc_1 = inception_node(images, 256)
inc_2 = inception_node(inc_1, 480)
inc_2 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(inc_2)
inc_3 = inception_node(inc_2, 512)
sf_out0 = softmax_out(inc_3)

inc_4 = inception_node(inc_3, 512)
inc_5 = inception_node(inc_4, 512)
inc_6 = inception_node(inc_5, 528)
inc_7 = inception_node(inc_6, 832)
sf_out1 = softmax_out(inc_6)

inc_8 = inception_node(inc_7, 832)
inc_9 = inception_node(inc_8, 1024)
out = AveragePooling2D((7, 7), data_format = "channels_last")(inc_9)
out = Dropout(0.4)(out)
sf_out2 = Flatten()(out)
sf_out2 = Dense(10, activation = "softmax", name = "sf_out2")(sf_out2)

model = Model(in_images, [sf_out0, sf_out1, sf_out2])
# model.summary()

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer = 'sgd', metrics = ['accuracy'])
model.fit(X_train, [Y_train, Y_train, Y_train], validation_data=(X_test, [Y_test, Y_test, Y_test]), epochs=30, batch_size=64)

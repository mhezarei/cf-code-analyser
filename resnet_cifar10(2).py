#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd 
import keras 
from keras.layers import *
from keras.models import *
from keras.activations import *
from keras.layers import Conv2D
from keras.backend import int_shape 
from keras.datasets import *


# In[14]:


def identity_block (X,fil,filters,stage,block):
    F1,F2,F3 = filters 
    X_shortcut = X
    
    X = Conv2D(filters = F1,kernel_size = (1,1),strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization (axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2,kernel_size = (fil,fil),strides = (1,1), padding = 'same')(X)
    X = BatchNormalization (axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3,kernel_size = (1,1),strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization (axis = 3)(X)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X


# In[15]:


def convolutional_block (X,f,filters,stage,block,s=2):
    F1,F2,F3 = filters 
    X_shortcut = X
    
    X = Conv2D(filters = F1,kernel_size = (1,1),strides = (s,s), padding = 'valid')(X)
    X = BatchNormalization (axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2,kernel_size = (f,f),strides = (1,1), padding = 'same')(X)
    X = BatchNormalization (axis = 3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3,kernel_size = (1,1),strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization (axis = 3)(X)
    X = Activation('relu')(X)
    
    X_shortcut = Conv2D(filters = F3,kernel_size = (1,1), strides = (s,s), padding = 'valid')(X_shortcut)
    X_shortcut = BatchNormalization (axis =3)(X_shortcut)
    
    X = Add()([X,X_shortcut])
    print(int_shape(X))
    print(int_shape(X_shortcut))
    X = Activation('relu')(X)
    
    return X
    


# In[16]:


def ResNet50(input_shape = (64, 64, 3), classes = 10):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)
    print(int_shape(X))
    X = Flatten()(X)
    print(int_shape(X))
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model


# In[17]:


model = ResNet50(input_shape = (32, 32, 3), classes = 10)
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[18]:


from keras.utils import np_utils
(X_train ,Y_train),(X_test,Y_test) = cifar10.load_data()
Y_train=np_utils.to_categorical (Y_train)
Y_test=np_utils.to_categorical (Y_test)


# In[19]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[1]:


model.fit(X_train, Y_train, epochs = 50, batch_size = 64)


# In[10]:


preds = model.evaluate(X_test, Y_test)


# In[11]:


print ("loss"  + str(preds[0]))
print ("Test Accuracy ="  + str(preds[1]))


# In[ ]:





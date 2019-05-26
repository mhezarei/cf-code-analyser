# 99.17% accuracy

from keras.layers import LSTM, Bidirectional, Input, Dropout, Dense, Embedding, BatchNormalization
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen = 150, num_words=15000)

x_train = sequence.pad_sequences(x_train, maxlen=150)
x_test = sequence.pad_sequences(x_test, maxlen=150)

in_sentences = Input(shape = (150,))
x = Embedding(15000, 512, input_length=150)(in_sentences)
x = Bidirectional(LSTM(150, recurrent_dropout = 0.1, dropout = 0.2))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(32, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
out = Dense(1, activation = 'sigmoid')(x)

model = Model(in_sentences, out)
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=15, batch_size=64)

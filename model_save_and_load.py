import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os.path

batch_size = 128
nb_classes = 10
nb_epoch = 20

img_rows = 28
img_cols = 28

f_log = './log'
f_model = './model'
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test  = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode = 'valid', input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3, 3, border_mode = 'valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])

    tb_cb = keras.callbacks.TensorBoard(log_dir=f_log, histogram_freq=1)
    cp_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(f_model,'cnn_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    cbks = [tb_cb, cp_cb]

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(f_model,'cnn_model.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(f_model,'cnn_model.yaml'), 'w').write(yaml_string)
    print('save weights')
    model.save_weights(os.path.join(f_model,'cnn_model_weights.hdf5'))
KTF.set_session(old_session)
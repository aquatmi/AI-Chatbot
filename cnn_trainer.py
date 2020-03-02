import pickle
import numpy as np

from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras import regularizers

import chatbot_functions as chatbot

##### OPEN DATA-SET
with open('input/data_set/test.p', 'rb') as f:
    test_data = pickle.load(f, encoding='latin1')

with open('input/data_set/train.p', 'rb') as f:
    train_data = pickle.load(f, encoding='latin1')

with open('input/data_set/valid.p', 'rb') as f:
    valid_data = pickle.load(f, encoding='latin1')

##### MAKING DATA SET
y_train = to_categorical(train_data['labels'], num_classes=43)
y_valid = to_categorical(valid_data['labels'], num_classes=43)

x_train = train_data['features']
x_valid = valid_data['features']
x_test = test_data['features']

##### BUILDING MODEL
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(43, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##### TRAINING MODELS
LRS = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
epochs = 20

print('Beginning Training')
model_history = model.fit(x_train[:32768], y_train[:32768], batch_size=128, epochs=epochs,
                          validation_data=(x_valid, y_valid), callbacks=[LRS], verbose=1)

print('Epochs={0:d}, training accuracy={1:.5f}, validation accuracy={2:.5f}'.
      format(epochs, max(model_history.history['accuracy']), max(model_history.history['val_accuracy'])))

##### SAVING MODELS

chatbot.save(model)
chatbot.graph(model_history)

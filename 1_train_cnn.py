from __future__ import print_function

from keras import layers
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

import numpy as np
import random

from sklearn.externals import joblib
partition = joblib.load('partition_sample_cnn.pkl')

np.random.seed(3)
label_idx = {'feces': 0 , 'tongue': 1, 'skin': 2}
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, label_idx, batch_size=32, dim=(5000,125,4),
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = label_idx
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size) , dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x_tmp = joblib.load(ID)
            label_tmp = ID.split('_')[1].split('.')[0]
            idx = np.random.randint(x_tmp.shape[0], size=5000)
            X[i,] = x_tmp[idx,:,:]

            # Store class
            y[i] = self.labels[label_tmp]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
params = {'batch_size': 64,
          'n_classes': 3,
          'shuffle': True}

# Datasets

# Generators
training_generator = DataGenerator(partition['train'], label_idx, **params)
testing_generator = DataGenerator(partition['test'], label_idx, **params)

def cnn_model(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(4, (1,6), strides = (1, 1), name = 'conv0', data_format = 'channels_last')(X_input)
    X = BatchNormalization(axis = -1, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((1,4), name='max_pool_0', data_format = 'channels_last')(X)
    
    X = Conv2D(16, (1,11), strides = (1, 1), name = 'conv1', data_format = 'channels_last')(X)
    X = BatchNormalization(axis = -1, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((1,2), name='max_pool_1', data_format = 'channels_last')(X)
    
    X = Conv2D(32, (1,10), strides = (1, 1), name = 'conv2', data_format = 'channels_last')(X)
    X = BatchNormalization(axis = -1, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Flatten()(X)
    
    X = Dense(32, activation='softmax', name='fc0')(X)
    
    X = Dense(3, activation='softmax', name='fc1')(X)
    
    model = Model(inputs = X_input, output = X, name = 'DeepDNA')
    
    return model

model = cnn_model((5000,125,4))
model.summary()

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,epochs=20)
model.save('sample_cnn_model.pkl')

acc = model.evaluate_generator(generator=testing_generator)
print(acc)


import h5py
import os
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,SeparableConv2D,BatchNormalization, Activation, Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint

import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def model():
    nclass = 6
    inp = Input(shape=(224,224,3))  
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(64, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(64, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.4)(x)
    
    x = Convolution2D(128, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.3)(x)
    
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)
    
    out = Dense(nclass, activation=softmax)(x)
    
    opt = optimizers.Adam(0.001)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

if __name__ == "__main__":

    with h5py.File(os.path.join("dataset_RGB.h5"), "r") as hf:
        X_train = hf["X_tr"][:]
        y_train = hf["y_tr"][:]
        # X_test = np.expand_dims(hf["X_te"][:], axis=-1)

    model = model()
    model.summary()

    # Directory where the checkpoints will be saved
    checkpoint_dir = "ckpt_cnn"
    # Name of the checkpoint files
    mcp = ModelCheckpoint(f'{checkpoint_dir}/CNN.h5', 
                            monitor="val_acc",
                            save_best_only=True, 
                            save_weights_only=True,
                            verbose = 1)

    print("Training model ...")
    
    epochs = 100
    batch_size = 64
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[mcp])
    model.load_weights(f'{checkpoint_dir}/CNN.h5')

    epochs = 100
    batch_size = 32

    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2)
    datagen.fit(X_train)
    
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size, epochs=epochs, callbacks=[mcp])
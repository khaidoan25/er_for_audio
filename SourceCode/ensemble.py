from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
import h5py
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, Input, Softmax, Flatten, Dropout, Conv1D, MaxPool1D, LeakyReLU, Concatenate, Activation
from tensorflow.python.keras.initializers import TruncatedNormal

def model_stft(time_step, n_stft):

    inputs_stft = Input(shape=(time_step, n_stft,), name="input_stft")
    stft = LSTM(128, return_sequences=True, dropout=0.3)(inputs_stft)
    stft = Bidirectional(LSTM(256, dropout=0.3, return_sequence=True, kernel_regularizer=tf.keras.regularizers.l2, recurrent_regularizer=tf.keras.regularizers.l2), merge_mode="concat")(stft)
    
    scores_stft = Dense(1, use_bias=False, name="score_stft")(stft)
    att_w_stft = Softmax(name="att_weight_stft")(scores_stft)
    stft = stft * att_w_stft
    stft = tf.keras.backend.sum(stft, axis=1)

    intermediate_feature = Dense(256, activation='relu', name="intermediate_dense_1")(stft)
    intermediate_feature = Dropout(0.3)(intermediate_feature)
    intermediate_feature = Dense(128, activation='relu')(intermediate_feature)
    intermediate_feature = Dropout(0.3)(intermediate_feature)
    cls_feature = Dense(32, activation='relu', name="intermediate_dense_2")(intermediate_feature)
    logits = Dense(6, name="logits")(cls_feature)
    output = Softmax(name="output")(logits)
    model = Model(inputs=inputs_stft, outputs=output)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

def model_gray():
    inputs = Input(shape=(224, 224, 1,), name="input_stft")
    x = Conv2D(64, (7, 5))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.3)(x)

    res1 = Conv2D(64, (7, 5), padding="same")(x)
    res1 = BatchNormalization()(res1)
    res1 = LeakyReLU()(res1)
    res1 = Conv2D(64, (7, 5), padding="same")(res1)
    res1 = BatchNormalization()(res1)
    x = x + res1
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (7, 5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.3)(x)

    res2 = Conv2D(128, (7, 5), padding="same")(x)
    res2 = BatchNormalization()(res2)
    res2 = LeakyReLU()(res2)
    res2 = Conv2D(128, (7, 5), padding="same")(res2)
    res2 = BatchNormalization()(res2)
    x = x + res2
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (7, 5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(2, 2)(x)
    x = Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation=LeakyReLU())(x)
    stft = Bidirectional(LSTM(128, dropout=0.3, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2, recurrent_regularizer=tf.keras.regularizers.l2), merge_mode="concat")(x)
    
    scores_stft = Dense(1, use_bias=False, name="score_stft")(stft)
    att_w_stft = Softmax(name="att_weight_stft")(scores_stft)
    stft = stft * att_w_stft
    stft = tf.keras.backend.sum(stft, axis=1)

    intermediate_feature = Dropout(0.3)(stft)
    cls_feature = Dense(64, activation=ReLU(), name="intermediate_dense_2")(intermediate_feature)
    cls_feature = BatchNormalization()(cls_feature)
    cls_feature = Dropout(0.3)(cls_feature)
    logits = Dense(6, name="logits")(cls_feature)
    output = Softmax(name="output")(logits)
    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

if __name__ == "__main__":
    w_stft = np.array([])
    w_gray = np.array([])
    # Thêm weight ở đây

    softmax_stft = np.load("softmax_stft.npy")
    softmax_gray = np.load("softmax_gray.npy")
    # Thêm softmax ở đây

    y_pred = []
    for sm_stft, sm_gray, _ in zip(softmax_stft, softmax_gray, _):
        softmax = sm_stft*w_stft + sm_gray*w_gray + # Thêm softmax * weight ở đây
        pred = np.argmax(softmax)
        y_pred.append(pred)

    test_files = [file for file in glob(os.path.join("Dataset/TestSet", "*"))]
    df = pd.DataFrame()
    df["File"] = test_files
    df["Label"] = y_pred
    df.to_csv("submission.csv", index=False)
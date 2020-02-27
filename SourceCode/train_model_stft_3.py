import h5py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Attention, Dense, Bidirectional, LSTM, Input, Softmax, Dot, Flatten, Concatenate, Average, Dropout
from tensorflow.python.keras.initializers import TruncatedNormal
tf.random.set_seed(64)

def model_stft(time_step, n_stft):

    inputs_stft = Input(shape=(time_step, n_stft,), name="input_stft")
    stft = LSTM(128, return_sequences=True, dropout=0.3)(inputs_stft)
    stft = Bidirectional(LSTM(256, dropout=0.3, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2, recurrent_regularizer=tf.keras.regularizers.l2), merge_mode="concat")(stft)
    
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

if __name__ == "__main__":
    with h5py.File(os.path.join("dataset_stft_3.h5"), "r") as hf:
        X_train = hf["X_tr"][:]
        y_train = hf["y_tr"][:]
        X_test = hf["X_te"][:]
        y_test = hf["y_te"][:]

    model = model_stft(118, 257)
    model.summary()

    # Directory where the checkpoints will be saved
    checkpoint_dir = "ckpt_stft"
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "model_stft_3.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        filepath=checkpoint_prefix,
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )

    print("Training model ...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=150, callbacks=[checkpoint_callback])
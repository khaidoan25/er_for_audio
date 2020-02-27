import h5py
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Attention, Dense, Bidirectional, LSTM, Input, Softmax, Dot, Flatten, BatchNormalization, Dropout, Conv2D, LeakyReLU, MaxPool2D, Reshape, ReLU
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(0)

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

    with h5py.File(os.path.join("dataset_gray_1.h5"), "r") as hf:
        X_train = np.expand_dims(hf["X_tr"][:], axis=-1)
        y_train = hf["y_tr"][:]
        X_test = np.expand_dims(hf["X_te"][:], axis=-1)
        y_test = hf["y_te"][:]

    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2)
    datagen.fit(X_train)

    model = model_gray()
    model.summary()

    # Directory where the checkpoints will be saved
    checkpoint_dir = "ckpt_gray"
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "model_gray_1.h5")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        filepath=checkpoint_prefix,
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )

    print("Training model ...")
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), 
                        steps_per_epoch=len(X_train)/32,
                        validation_data=(X_test, y_test),
                        epochs=150, callbacks=[checkpoint_callback])
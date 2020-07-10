import enum

import tensorflow as tf
from tensorflow.keras import layers, Model, backend
from tensorflow.keras.layers import Input
from net_config import ArchitectureConfig

class CRNN_MODE(enum.Enum):
    training = 1
    inference = 2

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]

    res = backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return res

def CRNN_model(crnn_mode):
    batchSize = ArchitectureConfig.BATCH_SIZE
    maxTextLen = ArchitectureConfig.MAX_TEXT_LENGTH
    img_size = ArchitectureConfig.IMG_SIZE

    input_data = Input(name='the_input', shape=img_size + (1,), dtype='float32')

    conv_1 = layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu')(input_data)

    pool_1 = layers.MaxPool2D(pool_size=2, padding='same')(conv_1)

    conv_2 = layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu')(pool_1)

    pool_2 = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same')(conv_2)

    conv_3 = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(pool_2)

    batch_norm_3 = layers.BatchNormalization()(conv_3)

    batch_norm_3 = layers.Activation('relu')(batch_norm_3)

    pool_3 = layers.MaxPool2D(pool_size=2, strides=(2, 2), padding='same')(batch_norm_3)

    conv_4 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(pool_3)

    conv_5 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(conv_4)

    pool_5 = layers.MaxPool2D(pool_size=(1,2), strides=(1, 2), padding='same')(conv_5)

    conv_6 = layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(pool_5)

    batch_norm_6 = layers.BatchNormalization()(conv_6)

    batch_norm_6 = layers.Activation('relu')(batch_norm_6)

    pool_6 = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same')(batch_norm_6)

    conv_7 = layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(pool_6)

    pool_7 = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same')(conv_7)

    cnnOut = tf.squeeze(pool_7, axis=[2])

    blstm_8 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(cnnOut)

    blstm_9 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(blstm_8)

    # transforms RNN output to character activations:
    # no unique labels
    inner = layers.Dense(216, kernel_initializer='he_normal',
                  name='dense2')(blstm_9)
    y_pred = layers.Activation('softmax', name='softmax')(inner)

    if crnn_mode == CRNN_MODE.inference:
        return Model(inputs=input_data, outputs=y_pred)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name = 'the_labels', shape=[ArchitectureConfig.MAX_TEXT_LENGTH], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    #Loss function
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels,
                          input_length, label_length], outputs=loss_out)

    y_func = backend.function([input_data], [y_pred])
    return model, y_func
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D,\
     Flatten, Dense, Dropout, Activation, BatchNormalization, Reshape, Lambda, LSTM, InputLayer
from tensorflow.keras import backend as K


def CNN3D(inp_shape, nb_classes, k_size=(3,3,3), drop_rate=0):
    NB_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3
    data = Input(shape=inp_shape)

    x = Conv3D(filters=(64), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(data)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(2,2,2))(x)    
    x = Dropout(drop_rate)(x)    
    
    x = Conv3D(filters=(128), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    x = Dropout(drop_rate)(x)    
    
    x = Conv3D(filters=(256), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    x = Dropout(drop_rate)(x)    

    x = Conv3D(filters=(512), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    x = Dropout(drop_rate)(x)    

    block_shape = K.int_shape(x)
    x = AveragePooling3D(pool_size=(block_shape[NB_AXIS], block_shape[ROW_AXIS], block_shape[COL_AXIS]),strides=(1, 1, 1))(x)
    x = Flatten()(x)

    output = Dense(nb_classes, activation="softmax")(x)
    
    model = Model(data, output)

    return model


# Model inspired from https://arxiv.org/pdf/1412.0767.pdf
def CNN3D_dense(inp_shape, nb_classes, k_size=(3,3,3)):
    data = Input(shape=inp_shape)

    x = Conv3D(filters=(64), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(data)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(2,2,2))(x)    
    
    x = Conv3D(filters=(128), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    
    x = Conv3D(filters=(256), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = Conv3D(filters=(512), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = Flatten()(x)
    
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)

    output = Dense(nb_classes, activation="softmax")(x)
    
    model = Model(data, output)
    
    return model


def CNN3D_lite(inp_shape, nb_classes):
    """
    Lite C3D Model + LSTM
    L2 Normalisation of C3D Lite Feature Vectors 
    3M Parameters
    Able to Run on the Jetson Nano at 8FPS

    # From https://github.com/patrickjohncyh/ibm-waldo/blob/master/2-MachineLearning/server-training/Models.py
    """

    model = tf.keras.Sequential()

    model.add(InputLayer(input_shape=inp_shape))
    model.add(Conv3D(32, 3,strides=(1,2,2), activation='relu', padding='same', name='conv1', input_shape=inp_shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(3,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))

    model.add(Reshape((2,384)))

    model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    model.add(LSTM(512, return_sequences=False,
                   input_shape= (2,384),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
from keras import regularizers
import tensorflow as tf
from keras.layers import LayerNormalization,Reshape,Conv1D,Conv2D,BatchNormalization,Dense,Activation,Dropout,Flatten,GlobalMaxPooling2D,Lambda#,regularizers#GlobalMaxPooling2D
import numpy as np
# Setting hyper-parameters
k = 40
# L=regularizers.l2(0.01)
channel =9
# lenth = 5
drop_out = 0.5
# out_channel = 8
out_channel2 = 128

activation = 'gelu'
def tSSVEPformer(inputs):
    x = Conv2D(out_channel2,kernel_size = (channel, 1),strides = 1,padding = 'valid')(inputs)
    x = LayerNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(drop_out)(x)
    x = Reshape((x.shape[2],x.shape[3]))(x)

    x1 = LayerNormalization()(x)
    x1 = Conv1D(out_channel2,kernel_size = 31,strides = 1,padding = 'same')(x1)
    x1 = LayerNormalization()(x1)
    x1 = Activation(activation)(x1)
    x1 = Dropout(drop_out)(x1)

    x = x + x1

    x1 = LayerNormalization()(x)
    x1 = Dense(out_channel2)(x1)
    x1 = Activation(activation)(x1)
    x1 = Dropout(drop_out)(x1)

    x = x + x1  

    x1 = LayerNormalization()(x)
    x1 = Conv1D(out_channel2,kernel_size = 31,strides = 1,padding = 'same')(x1)
    x1 = LayerNormalization()(x1)
    x1 = Activation(activation)(x1)
    x1 = Dropout(drop_out)(x1)

    x = x + x1

    x1 = LayerNormalization()(x)
    x1 = Dense(out_channel2)(x1)
    x1 = Activation(activation)(x1)
    x1 = Dropout(drop_out)(x1)

    x = x + x1  

    x = Flatten()(x)
    # the fully connected layer and "softmax"
    x = Dropout(drop_out)(x)
    x = Dense(k*6)(x)
    x = LayerNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(drop_out)(x)

    x = Dense(k,activation='softmax')(x)#shape=(None, 1, 1, 4)
    
    return x

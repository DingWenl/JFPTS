from random import sample
import random
import numpy as np
import keras
from keras.utils import np_utils
# get the training sampels
def train_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,train_list, channel,f_list):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        target_list = list(range(40))
        index_list = list(range(batchsize))
        B = 0.5

        list_batchsize = list(range(batchsize))
        random.shuffle(list_batchsize)
        for i in range(int(batchsize)):
            index_list[i] = np.random.beta(B,B)
            k = sample(train_list, 1)[0]
            m = sample(target_list, 1)[0]
            
            # get the target frequency
            f_s = f_list[m]
            # get the period
            period_frame = 250/f_s
            # get the number of possible period-based window shifts within the stimulus duration
            period_total = (1250-win_train) // period_frame
            # random selecting a period-based shift
            period_n = sample(range(int(period_total+1)), 1)[0]
            # obtain the start point of the shifted window, round() is used to obtain a integer
            time_start = 35+125 + round(period_frame*period_n)
            # obtain the end point of the shifted window    
            time_end = time_start + win_train
            # get four sub-inputs
            x_11 = train_data1[k][m][:,time_start:time_end]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            
            
            x_12 = train_data2[k][m][:,time_start:time_end]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            

            x_13 = train_data3[k][m][:,time_start:time_end]
            x_23 = np.reshape(x_13,(channel, win_train, 1))

            x_concatenate = np.concatenate((x_21, x_22, x_23), axis=-1)
            x_train[i] = x_concatenate
            y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
                
        x_out = np.array(x_train)
        y_out = np.reshape(y_train,(batchsize,40))

        yield x_out, y_out

# # get the validation samples
# def val_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,val_list, channel):
#     while True:
#         x_train1, x_train2, x_train3, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
#         target_list = list(range(40))
#         # get training samples of batchsize trials
#         for i in range(batchsize):
#             k = sample(val_list, 1)[0]
#             m = sample(target_list, 1)[0]
#             # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
#             time_start = 35+125#random.randint(35+125,int(1250+35+125-win_train))
#             time_end = time_start + win_train
#             # get four sub-inputs
#             x_11 = train_data1[k][m][:,time_start:time_end]
#             x_21 = np.reshape(x_11,(channel, win_train, 1))
#             x_train1[i]=x_21
            
#             x_12 = train_data2[k][m][:,time_start:time_end]
#             x_22 = np.reshape(x_12,(channel, win_train, 1))
#             x_train2[i]=x_22

#             x_13 = train_data3[k][m][:,time_start:time_end]
#             x_23 = np.reshape(x_13,(channel, win_train, 1))
#             x_train3[i]=x_23

#             y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
            
#         x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
#         x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
#         x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))

#         # concatenate the four sub-input into one input to make it can be as the input of the FB-tCNN's network
#         x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
#         y_train = np.reshape(y_train,(batchsize,40))
        
#         yield x_train, y_train



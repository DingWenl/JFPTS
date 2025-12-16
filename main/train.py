from keras.callbacks import ModelCheckpoint
from net import tSSVEPformer
from data_generator_1st_wMixUp import train_datagenerator#,val_datagenerator
from data_generator_2nd import train_datagenerator1#,val_datagenerator1
import scipy.io as scio 
from scipy import signal
from keras.models import Model
from keras.layers import Input
import numpy as np
import os

# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
    # read the data
    data = scio.loadmat(path)
    data_1 = data['data']
    # get the EEG data of selected 9 electrodes
    c1 = [47,53,54,55,56,57,60,61,62]
    
    train_data = data_1[c1,:,:,:]
    # get the filtered EEG-data with six-order Butterworth filter of the first sub-filter
    block_data_list1 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn11,wn21], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list1.append(target_data_list)
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
    block_data_list2 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn12,wn22], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list2.append(target_data_list)
    # get the filtered EEG-data with six-order Butterworth filter of the third sub-filter
    block_data_list3 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn13,wn23], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list3.append(target_data_list) 
    return block_data_list1, block_data_list2, block_data_list3

if __name__ == '__main__':
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    #%% Setting hyper-parameters
    # ampling frequency after downsampling
    fs = 250
    # the number of the electrode channels
    channel = 9
    # the hyper-parameters of the training process
    batchsize = 256
    
    # the filter ranges of the four sub-filters in the filter bank
    f_down1 = 6
    f_up1 = 50
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 14
    f_up2 = 50
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 22
    f_up3 = 50
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs
    
    # obtain the frequency prior
    f1 = 8
    f2 = f1 +1
    f3 = f2 +1
    f4 = f3 +1
    f5 = f4 +1
    f6 = f5 +1
    f7 = f6 +1
    f8 = f7 +1
    f9 = f1 + 0.2
    f10 = f2 + 0.2
    f11 = f3 + 0.2
    f12 = f4 + 0.2
    f13 = f5 + 0.2
    f14 = f6 + 0.2
    f15 = f7 + 0.2
    f16 = f8 + 0.2
    f17 = f9 + 0.2
    f18 = f10 + 0.2
    f19 = f11 + 0.2
    f20 = f12 + 0.2
    f21 = f13 + 0.2
    f22 = f14 + 0.2
    f23 = f15 + 0.2
    f24 = f16 + 0.2
    f25 = f17 + 0.2
    f26 = f18 + 0.2
    f27 = f19 + 0.2
    f28 = f20 + 0.2
    f29 = f21 + 0.2
    f30 = f22 + 0.2
    f31 = f23 + 0.2
    f32 = f24 + 0.2
    f33 = f25 + 0.2
    f34 = f26 + 0.2
    f35 = f27 + 0.2
    f36 = f28 + 0.2
    f37 = f29 + 0.2
    f38 = f30 + 0.2
    f39 = f31 + 0.2
    f40 = f32 + 0.2
    f_list = [f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,  f11,  f12,  f13,  f14,  f15,  f16,  f17,  f18,  f19,  f20,  f21,  f22,  f23,  f24,  f25,  f26,  f27,  f28,  f29,  f30,  f31,  f32,  f33,  f34,  f35,  f36,  f37,  f38,  f39,  f40]


    # the lsit of data lengths, using 0.5 s as an example
    t_train_list = [0.5] # [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #%% Training the models of multi-subjects
    # selecting the training subject
    for sub_selelct in range(1, 36):
        # the path of the dataset and you need change it for your training
        path = '/data/dwl/ssvep/benchmark/S%d.mat'%sub_selelct
        # get the filtered EEG-data of three sub-input of the training data
        data1, data2, data3 = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
        # selecting the training time-window
        for t_train in t_train_list:
            # transfer time to frame
            win_train = int(fs*t_train)
            # the traing data is randomly divided in the traning dataset and validation set according to the radio of 9:1
            for block_n in range(5,6):
                total_list = list(range(6))
                test_list = [block_n]
                train_list = [i for i in total_list if (i not in test_list)]
                #%% setting the input of the network
                input_shape = (channel, win_train, 3)
                input_tensor = Input(shape=input_shape)
                # using the t-SSVEPformer model
                preds = tSSVEPformer(input_tensor)
                model = Model(input_tensor, preds)
                # the path of the saved model and you need to change it
                model_path = '/data/dwl/ssvep/model/benchmark_JPFTS_test/test/time%3.1fs_subject%d_block%d.h5'%(t_train, sub_selelct,block_n)
                # some hyper-parameters in the training process
                model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # training, using model.fit or model.fit_generator
                
                train_epoch =  50
                #### first stage
                train_gen = train_datagenerator(batchsize,data1, data2, data3,win_train,train_list, channel,f_list)#, t_train)
                # val_gen = val_datagenerator(batchsize,data1, data2, data3,win_train,val_list, channel)#, t_train)
                
                # some hyper-parameters in the training process
                model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # training
                history = model.fit_generator(
                        train_gen,
                        steps_per_epoch= 10,
                        epochs=train_epoch,
                        validation_data=None,
                        validation_steps=1,
                        callbacks=[model_checkpoint]
                        )
                ##### second stage
                train_gen = train_datagenerator1(batchsize,data1, data2, data3,win_train,train_list, channel)#, t_train)
                # val_gen = val_datagenerator1(batchsize,data1, data2, data3,win_train,val_list, channel)#, t_train)
                
                # some hyper-parameters in the training process
                model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # training
                history = model.fit_generator(
                        train_gen,
                        steps_per_epoch= 10,
                        epochs=train_epoch,
                        validation_data=None,
                        validation_steps=1,
                        callbacks=[model_checkpoint]
                        )








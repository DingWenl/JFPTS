import scipy.io as scio 
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
def fft_test(y,t,frame1,frame2):
    ylen = len(y)
    yf1=abs(fft(y))/ylen           #归一化处理
    yf2 = yf1[range(int(ylen/2))]  #由于对称性，只取一半区间
    xf = np.arange(ylen)        # 频率
    # xf2 = xf[range(int(ylen/2))]/t  #取一半区间
    xf2 = xf[range(int(ylen/2))]/t  #取一半区间
    # xf2,yf2 = xf2[:frame2],yf2[:frame2]
    return xf2,yf2

# get the filtered EEG-data, label and the start time of each trial of the dataset (test set), more details refer to the "get_train_data" in "FB-tCNN_train"
def get_test_data(wn11,wn21,path):
    # read the data
    data = scio.loadmat(path)
    # get the EEG-data of the selected electrodes and downsampling it
    data_1 = data['data']
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
        
    return block_data_list1

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # Setting hyper-parameters,
    # sampling rate
    fs = 250
    # channel count
    channel = 9
    # filter range
    f_down1 = 6
    f_up1 = 50
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    # data length
    dl = 1.0
    # transfer to frame indices
    dl_frame = int(dl * fs)
    # visual latency is set to 108 ms for this example,27 = 0.108 *250, 125 is the cue time
    start_point = 125+27
    # solid line width
    linewidth_1 = 4
    # dotted line width
    linewidth_2 = 2
    
    # creat plot-related list
    # freqency indicx list in benchmark EEG data, for stimulus of 12, 12.2,12.4 Hz
    freq_indicx_list = [4, 12, 20]
    # freqency list in benchmark EEG data, for stimulus of 12, 12.2,12.4 Hz
    freq_list = [12, 12.2, 12.4]
    # color_list
    color_list = ["blue","purple","green"]
    
    # EEG data path
    path = '/Users/dingwenlong/Desktop/dwl/USTC_PHD/data/benchmark/S%d.mat'%22
    # get the EEG data
    data1 = get_test_data(wn11,wn21,path)
    data_array = np.array(data1)
    data_mean = np.mean(data_array, axis=0)
    

    
    for i in range(3):
    
        fre_n = freq_indicx_list[i]
        target_fre = freq_list[i]
        color_plot = color_list[i]
        # select the fre_n_EEG_indicx EEG data of Oz, 7 denotes the 8-th eeeg channel in the 9 EEG channel setting
        plot_data1 = data_mean[fre_n,7,start_point:start_point+dl_frame]
        
    
        # zero padding to 5 seconds
        zero_t_train = dl*5
        frame1 = int(f_down1*zero_t_train)
        frame2 = int(f_up1*zero_t_train)
        
        zero_data = np.zeros([int(zero_t_train*fs),], np.float64)
        
        zero_data[:dl_frame] = plot_data1
        
        ffx1,ffy_ori1 = fft_test(zero_data,zero_t_train,frame1,frame2)
        ffx1,ffy_ori1 = ffx1[frame1:frame2],ffy_ori1[frame1:frame2]
        
        
        fig, ax = plt.subplots(figsize=(12,6))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        for spine in ax.spines.values():
    
            spine.set_linewidth(2)
        plt.tick_params(width=2)
        
        
        plt.plot(ffx1,ffy_ori1,linewidth=linewidth_1,color=color_plot)
        
        
        plt.axvline(target_fre,color='black',linestyle='dotted',linewidth=linewidth_2)
        plt.axvline(target_fre*2,color='black',linestyle='dotted',linewidth=linewidth_2)
        plt.axvline(target_fre*3,color='black',linestyle='dotted',linewidth=linewidth_2)
        plt.axvline(target_fre*4,color='black',linestyle='dotted',linewidth=linewidth_2)
    
        plt.savefig('./subject22_fft_%3.1fHz.png'%target_fre,bbox_inches='tight')
    
    
    
    
    

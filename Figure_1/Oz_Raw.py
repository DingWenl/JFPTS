# from random import sample
import scipy.io as scio 
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt




def fft_test(y,t,frame1,frame2):
    ylen = len(y)
    yf1=abs(fft(y))/ylen           
    yf2 = yf1[range(int(ylen/2))]  
    xf = np.arange(ylen)        
    xf2 = xf[range(int(ylen/2))]/t  
    
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
    # Setting hyper-parameters
    # down_sample = 4
    fs = 250
    channel = 9


    f_down1 = 11.5
    f_up1 = 13.5
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    # data length
    dl = 1.0
    # transfer to frame indices
    dl_frame = int(dl * fs)
    # visual latency is set to 108 ms for this example,27 = 0.108 *250, 125 is the cue time
    start_point = 125+27
    
    # creat plot-related list
    # freqency indicx list in benchmark EEG data, for stimulus of 12, 12.2,12.4 Hz
    freq_indicx_list = [4, 12, 20]
    # freqency list in benchmark EEG data, for stimulus of 12, 12.2,12.4 Hz
    freq_list = [12, 12.2, 12.4]
    # phase_list
    phase_list = [0,0.5,1.0]
    # color_list
    color_list = ["blue","purple","green"]

    # EEG data path. subject 22
    path = '/Users/dingwenlong/Desktop/dwl/USTC_PHD/data/benchmark/S%d.mat'%22
    # get the filtered EEG-data of the four sub-filters, label and the start time of all trials of the test data
    data1 = get_test_data(wn11,wn21,path)
    data_array = np.array(data1)
    data_mean = np.mean(data_array, axis=0)

    
    for i in range(3):
        
        fre_n_EEG_indicx = freq_indicx_list[i]
        frequency = freq_list[i]    
        phase = phase_list[i] * np.pi  
        color_plot = color_list[i]
        plt.figure(figsize=(10,2))
        # plot EEG data
        # select the fre_n_EEG_indicx EEG data of Oz, 7 denotes the 8-th eeeg channel in the 9 EEG channel setting
        plot_data1 = data_mean[fre_n_EEG_indicx,7,start_point:start_point+dl_frame]
        # obtain the standardized EEG signal
        plot_mean = np.mean(plot_data1)
        plot_std = np.std(plot_data1)
        plot_nor = (plot_data1-plot_mean)/plot_std
        
        plt.plot(list(range(dl_frame)),plot_nor,linewidth=1.5,color=color_plot)
        
        # plot sine curve
        # Generate frame numbers (0 to 249, a total of 250 sampling points)
        frames = np.arange(0, int(fs * dl))
        # Generation time point (used for calculating sine waves)
        t = frames / fs  # The time corresponding to each frame (in seconds)
        # Generate a sine wave
        signal_ = np.sin(2 * np.pi * frequency * t + phase)
        # Draw the graphic (the horizontal axis represents the frame number)
        plt.plot(frames, signal_,linestyle='--',color='black')
        
        plt.savefig('./subject22_fft_%3.1fHz.png'%frequency,bbox_inches='tight')





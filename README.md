# JFPTS
1. Here are the codes of the JFPTS-enhanced t-SSVEPformer in the paper ["Breaking the performance barrier in deep learning-based SSVEP-BCIs: a joint frequency-phase training strategy“](https://iopscience.iop.org/article/10.1088/1741-2552/ae36f6).
2. the main folder is used to reproduce JFPTS-enhanced t-SSVEPformer, and the Figure_1 folder is used to reproduce Figure 1 of the paper.
## The related version information
1. Python == 3.9.13
2. Keras-gpu == 2.6.0
3. tensorflow-gpu == 2.6.0
4. scipy == 1.9.3
5. numpy == 1.19.3
## Training CNN-Former with EEG-ME for the benchmark dataset
1. Download the code.
2. Download the [benchmark dataset](http://bci.med.tsinghua.edu.cn/download.html) and its [paper](https://ieeexplore.ieee.org/abstract/document/7740878).
3. Create a model folder to save the model.
4. Change the data and model folder paths in train and test files to your data and model folder paths.

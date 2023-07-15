# Detection of Transition from Combustion Noise to Thermoacoustic Instability in a Partially Premixed Turbulent Combustor using Convolutional Neural Networks

This repository contains the code implementation for a machine-learning model that classifies pressure-time data points as noise or instability. The model utilizes logistic regression, support vector machines (SVM), and random forest for classification on the basis of the pressure wave amplitudes- high amplitude corresponding to instability and low to noise. Classification on the basis of pressure amplitude itself means that the model is built for a single combustor and so is a combustor-dependent model. On the other hand, to utilize the model for any general combustor, a combustor-independent model was built using Convolutional Neural Networks which classifies data points on the basis of patterns of pressure-time plots.

Combustor1 dataset - Noise-        https://drive.google.com/file/d/1_EpPyRzdzToxxPlf2GOGJ2ZCYuvQ7KtC/view?usp=drive_link
                     Instability-  https://drive.google.com/file/d/1050DPRuIzoumNgAnzkfg2MVIGPZdPvUP/view?usp=drive_link

Combustor2 dataset - Noise-        https://drive.google.com/file/d/1CTeKge7S-BPFoqFfyJREx3xqeYyNqMpo/view?usp=sharing
                     Instability-  https://drive.google.com/file/d/1rE1ug9wIHS6fwWoyuRvBcuX80vOLshjM/view?usp=sharing

## LogReg1 , SVM1 and RandomForest1

Both datasets of combustor1 are  CSV files in which one column is separated from the other by a comma. They have 3276800 rows (pressure recorded at these many times) and 9 columns (1st column is time and the rest eight are pressures at different locations of the combustor). Extracted the first column and stored it in an array. Similarly, extracted the 7th column and stored it in another array. Used statistical features for building the model and calculated them (mean, sd, and rms) for every 500 pressure values at different times. Built a data frame with statistical features as columns and added a new column to it which had their label (0-noise, 1-unstable). Did this for both the stable(noise) and unstable datasets and merged the two data frames after shuffling them. Now extracted input(mean,rms, and sd) and output(labels) from the shuffled data frame. Split input and output datasets into training and testing data. Now for each algorithm - Logistic Regression, SVM, and Random Forest used the scikit learn inbuilt functions. 
Result-  
| Algorithm           | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.992    | 0.994     | 0.99   | 0.992    |
| SVM                 | 0.992    | 0.995     | 0.99   | 0.992    |
| Random Forest       | 0.991    | 0.993     | 0.989  | 0.991    |


## LogReg2, SVM2 and RandomForest2

Both datasets of combustor2 are text files in which one column is separated from the other by tab space. They have 491520 rows (pressure recorded at these many times) and 4 columns (1st column is time and the rest three are pressures at different locations of the combustor). Extracted the first column and stored it in an array. Similarly, extracted the 2nd column and stored it in another array. Used statistical features for building the model and calculated them (mean, sd, and rms) for every 75 pressure values at different times. Built a data frame with statistical features as columns and added a new column to it which had their label (0-noise, 1-unstable). Did this for both the stable(noise) and unstable datasets and merged the two data frames after shuffling them. Now we have a shuffled dataset that contains pressure values of combustor2. Following this the data frame corresponding to combustor1 and combustor2 are merged and shuffled from where inputs and labels are extracted and the dataset is split into training and testing data. Now for each algorithm - Logistic Regression, SVM, and Random Forest used the scikit learn inbuilt functions. 
Result-  
| Algorithm           | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.898    | 0.923     | 0.868  | 0.895    |
| SVM                 | 0.902    | 0.933     | 0.86   | 0.895    |
| Random Forest       | 0.899    | 0.921     | 0.872  | 0.896    |


## LogReg3, SVM3 and RandomForest3

Instead of splitting the dataset into training and testing data, I have used the dataframe of combustor1 for training and that of combustor2 for testing. Now for each algorithm - Logistic Regression, SVM, and Random Forest used the scikit learn inbuilt functions.
Result-
| Algorithm           | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.746    | 0.812     | 0.632  | 0.713    |
| SVM                 | 0.759    | 0.760     | 0.758  | 0.759    |
| Random Forest       | 0.762    | 0.725     | 0.844  | 0.78     |

## Combustor Independent model (Images_generation, CNN1, CNN2 and CNN3)

Extracted time and pressure arrays from all the 4 datasets( noise and unstable datasets from each of the 2 combustors) and sliced them into many sub-arrays, each of them used for plotting pressure versus time values. All the plots are converted into images and saved in png format. Pixel value of each image is 288*432. Link for the image files of all the 4 datasets are given below:-
Combustor1 Noise- https://drive.google.com/drive/folders/1kQYHcJNo2UeQu_M-uJ0Jq5-sj1I3Mg6h?usp=sharing

Combustor1 Instability- https://drive.google.com/drive/folders/1IqayKsCNk0L1nyDbCfMUozwrkrMfAzb0?usp=sharing

Combustor2 Noise- https://drive.google.com/drive/folders/1tpLiOirzc40_NpG4uXA_3985-4IvWS2o?usp=sharing

Combustor2 Instability- https://drive.google.com/drive/folders/1LyuacRl5bwkmqfVo1-q3dUL6PSY2zH7g?usp=sharing

### Working of the CNN model
The model contains 2 convolutional layers followed by 2 fully connected layers. Both the convolutional layers are passed through a 3*3 filter, a relu activation and a maxpooling layer of 2*2 dimension. The output generated from the convolutional layers are flattened and passed through 2 ANN layers, one with relu activation and the classifier layer with sigmoid activation as it does binary classification.

### CNN1
Used the image pixel arrays of pressure-time plots of combustor1 as input and their labels as output. Shuffled them in such a manner that each array is still linked to its label as their index is same in their respective arrays. Split datasets into training and testing data and fed into the cnn model built using keras. 

### CNN2
Merged the input and output array  corresponding to combustor1 with that of combustor2 separately. Shuffled them and split it into training and testing dataset. Fed datas into the cnn model built using keras.

### CNN3
Used the input and output array corresponding to combustor1 as training dataset and that of combustor2 as testing dataset. Fed datas into the cnn model built using keras.

### Result-

| Code  | Training Accuracy | Testing Accuracy | 
|-------|-------------------|------------------|
| CNN1  |    0.9998         |      0.994       | 
| CNN2  |    0.9997         |      0.913       | 
| CNN3  |    0.9985         |      0.76        | 

## Conclusion
This project aims at classifying the data points as noise or instability for any general combustor by pattern recognition of pressure-time plots using convolutional neural networks.

## Requirements

The code in this repository requires the following dependencies:
- Python (version X.X.X)
- Libraries: NumPy, Matplotlib, Pandas, Scikit-learn, TensorFlow, Keras

Please ensure that you have the necessary dependencies installed before running the code.

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file.
3. Navigate to the specific code file you wish to run.
4. Run the code using a Python interpreter.
5. The results and any generated models will be displayed or saved based on the implementation.

Please note that the provided datasets are for demonstration purposes only, and you may need to replace them with your own datasets or modify the code accordingly.

For any questions or issues, please feel free to [open an issue](https://github.com/aakarsh-1123/Surge/issues) in this repository.

Happy exploring and experimenting!

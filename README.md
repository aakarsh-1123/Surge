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


## Repository Structure

This repository is organized as follows:

- `logistic_regression.py`: Implementation of logistic regression algorithm for classification based on pressure wave amplitude.
- `svm.py`: Implementation of support vector machines (SVM) algorithm for classification based on pressure wave amplitude.
- `random_forest.py`: Implementation of random forest algorithm for classification based on pressure wave amplitude.
- `cnn.py`: Implementation of the convolutional neural network (CNN) algorithm for classification based on pattern recognition of pressure-time plots.
- `single_combustor_dataset`: Folder containing the single combustor dataset, split into training and testing subsets.
- `double_combustor_dataset`: Folder containing the double combustor dataset, combined and split into training and testing subsets.
- `new_dataset`: Folder containing a new dataset for testing the trained models.

Feel free to explore the code and datasets provided in this repository.

## Requirements

The code in this repository requires the following dependencies:
- Python (version X.X.X)
- Libraries: NumPy, Pandas, Scikit-learn, TensorFlow, Keras

Please ensure that you have the necessary dependencies installed before running the code.

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file.
3. Navigate to the specific code file you wish to run (e.g., `logistic_regression.py`).
4. Run the code using a Python interpreter.
5. The results and any generated models will be displayed or saved based on the implementation.

Please note that the provided datasets are for demonstration purposes only, and you may need to replace them with your own datasets or modify the code accordingly.

For any questions or issues, please feel free to [open an issue](https://github.com/aakarsh-1123/Surge/issues) in this repository.

Happy exploring and experimenting!

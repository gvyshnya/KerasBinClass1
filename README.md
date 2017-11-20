# Introduction
This repo contains a Keras-based neural network (Deep Learning) solution to tackle a binary classification problem. 

This challenge has been implemented under the constraints below

- It is not allowed to perform feature engineering of any kind
- The input to the neural network should be the original provided data

# Files and Folders
There are assets in this repo as follows

- simple_keras.py - the Python script implementing the entire DL prediction pipeline (best neural network model setup, training, evaluation, and making predictions)
- keras_cv.py - the Python script implementing a framework for n-fold cross-validation of various Keras-based NN models, using scikit-learn capabilities and Keras scikit-learn wrapper
- EDA.R - the R script implementing the brief EDA for the initial data (see details below)
- ReadMe.md - this readme document

# Exploratory Data Analysis (EDA)

Brief EDA highlighted the following findings

- correlation analysis of the numerical features in the training set did not detect any highly correlated variables
- neither training nor testing sets contain any missing values (NAs) in any of the variables
- the target variable has balanced class ratio in the trainig set thus it does not impose complexities related to imbalanced class handling
- feature selection experiment with Boruta algorithm identified 21 significant feature variables (see the list below)

Boruta-based feature selection was implemented in R (please refer to EDA.R), and it identified the most significant features per the list below

- V2
- V131
- V168
- V170
- V205
- V275
- V279
- V287
- V345
- V346
- V368
- V384
- V407
- V417
- V460
- V487
- V514
- V521
- V523
- V524
- V537

# Implementation Notes

This section explains what had been tried in the course of tackling the DL challenge, what worked and what didn't and why. 

Sequential NN architecture has been selected as a straight-forward option to go.

The initial layer composition was quite simple, per the code fragment below 

`model.add(layers.Dense(16, input_dim=559, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))`

Obviously, it did not work out well (the accuracy score at the model evaluation step displayed _50.1%_ in average only).

After a number of improvement iterations, the model with different layer composition and activation functions was selected, along with the different optimizer (namely, ‘rmsprop’) and fine-tuning the number of epochs and the batch size for it.

`model.add(layers.Dense(256, activation='tanh', input_dim=559))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='tanh'))`

This raised the accuracy score to _87.1%_ at the model evaluation step (in average).

Below is the brief summary of what worked and what did not work for this particular DL challenge

Within the finally selected NN architecture and design, the following parameter tweaks degraded accuracy score

- Setting higher drop-out rates in the dropout layer (0.15 and higher)
- Increasing the number of epochs above 110 or making it less than 95
- Setting batch size below 95 or above 105

Other findings of the respective machine learning/DL experiements demonstrated that

- Using sdg activator, although improved accuracy at the evaluation step up to 62%, still did not work out for this problem well vs. the finally selected NN model design (the best performance within SGD was demonstrated by the following setup: _sgd = opt.SGD(lr=0.017, decay=1e-6, momentum=0.75, nesterov=True)_ )
- Using 'adam' activator did not help to improve accuracy
- Using other combinations of activation functions at the dense layers (like “relu”-“relu”, “tanh”-“relu”, “relu”-“softmax” etc.) did not score well enough at the model evaluation step (although they performed better than the initial model described above)
- Setting different weights to the NN layers has not been tried

**Notes:** 

- as one of the constraints to this challenge was no feature engineering and no raw data pre-processing, the final python script misses the data standardizing step - the separate experiment was conducted to demonstrate applying data standardizing in the pipeline would improve the score of the model accuracy by 0.5% in 4-fold cross-validation setup
- TensorFlow was used as a Keras backend, during the actual model training and making predictions

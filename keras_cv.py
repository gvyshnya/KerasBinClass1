"""
    This script implements the cross-validation framework for the classification problem being tackled
"""

#!/usr/bin/python
import numpy as np
import datetime as dt
import keras.models as models # Sequential
import keras.layers as layers # Dense
import keras.wrappers.scikit_learn as keras_scikit # import KerasClassifier
import sklearn.model_selection as msel # cross_val_score, StratifiedKFold
import sklearn.preprocessing as preproc # StandardScaler
import sklearn.pipeline as pipe # Pipeline


def create_baseline():
    """
    This will create a baseline model
    """
    # create model
    input_dimentions = 559
    model = models.Sequential()
    model.add(layers.Dense(60, input_dim=input_dimentions, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# smaller model
def create_smaller():
    # create model
    input_dimentions = 559
    model = models.Sequential()

    model.add(layers.Dense(32, activation='tanh', input_dim=input_dimentions))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # binary_accuracy
    return model

# bigger model
def create_bigger():
    # create model
    input_dimentions = 559
    model = models.Sequential()

    model.add(layers.Dense(256, activation='tanh', input_dim=input_dimentions))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # binary_accuracy
    return model

################################################
# Main execution loop
################################################
start_time = dt.datetime.now()
print("Started at ", start_time)

# fix random seed for reproducibility
seed = 25
np.random.seed(seed)

# training and validation set file paths
fname_training = "input/obtrain.csv"  # 2000 rows, 560 cols

# number of features
feature_number = 559

# load training dataset
training_dataset = np.loadtxt(fname_training, delimiter=",")

# split into input (X) and output (Y) variables
X = training_dataset[:,0:feature_number]
Y = training_dataset[:,feature_number]  #last col of the training set is the target

# specify general CV settings
folds = 5

# specify general NN CV settings
n_epochs = 30
n_batch_size = 50

# evaluate model with the original dataset
estimator = keras_scikit.KerasClassifier(build_fn=create_baseline, nb_epoch=n_epochs,
                                         batch_size=n_batch_size, verbose=0)
kfold = msel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
results = msel.cross_val_score(estimator, X, Y, cv=kfold)
print("Evaluation Results (baseline model, raw dataset: mean score, std): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate baseline model with the standardized (normally scaled) dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', preproc.StandardScaler()))
estimators.append(('mlp', keras_scikit.KerasClassifier(build_fn=create_baseline, epochs=n_epochs,
                                                       batch_size=n_batch_size, verbose=0)))
pipeline = pipe.Pipeline(estimators)
kfold = msel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
results = msel.cross_val_score(pipeline, X, Y, cv=kfold)
print("Evaluation Results (baseline model, standardized dataset: mean score, std): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate a smaller model
np.random.seed(seed)
estimators = []
#estimators.append(('standardize', preproc.StandardScaler()))
estimators.append(('mlp', keras_scikit.KerasClassifier(build_fn=create_smaller, epochs=n_epochs,
                                                       batch_size=n_batch_size, verbose=0)))
pipeline = pipe.Pipeline(estimators)
kfold = msel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
results = msel.cross_val_score(pipeline, X, Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate a bigger model
np.random.seed(seed)
estimators = []
#estimators.append(('standardize', preproc.StandardScaler()))
estimators.append(('mlp', keras_scikit.KerasClassifier(build_fn=create_bigger, epochs=n_epochs,
                                                       batch_size=n_batch_size, verbose=0)))
pipeline = pipe.Pipeline(estimators)
kfold = msel.StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
results = msel.cross_val_score(pipeline, X, Y, cv=kfold)
print("Bigger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print ("Finished keras_cv.py ... ")
print("Elapsed time: ", elapsed_time)
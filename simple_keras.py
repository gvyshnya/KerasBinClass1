"""
    This script implements the NN training, evaluation, and prediction for a binary classification problem
"""
#!/usr/bin/python
import keras.models as models
import keras.layers as layers
import numpy as np
import pandas as pd
import datetime as dt

def normalize_prediction_value(value):
    result = 1.0
    if value < 0:
        result = -1.0
    return result

################################################
# Main execution loop
################################################
start_time = dt.datetime.now()
print("Started at ", start_time)

# fix random seed for reproducibility
np.random.seed(25)

# training and validation set file paths
fname_training = "input/obtrain.csv"  # 2000 rows, 560 cols
fname_testing = "input/obval.csv"     # 600 rows, 560 cols, the last one for targets contains the fake values

# output csv file with predicted targets
fname_out_predictions = "output/predictions.csv"

# number of features
feature_number = 559

# load training and testing datasets
training_dataset = np.loadtxt(fname_training, delimiter=",")
testing_dataset = np.loadtxt(fname_testing, delimiter=",")

# split into input (X) and output (Y) variables
X = training_dataset[:,0:feature_number]
Y = training_dataset[:,feature_number]  #last col of the training set is the target

# create a model
model = models.Sequential()

model.add(layers.Dense(256, activation='tanh', input_dim=feature_number))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation='tanh'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  # binary_accuracy

# Fit the model
model.fit(X, Y, epochs=110, batch_size=100)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# predict

# calculate predictions
X_testing = testing_dataset[:,0:feature_number]
predictions = model.predict(X_testing)

# round predictions to display exactly -1 or 1 as per the expected output convention
target = [normalize_prediction_value(x[0]) for x in predictions]

df = pd.DataFrame(target, columns=['target'])
df.to_csv(fname_out_predictions, index=False)

end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print ("Finished simple_keras.py ... ")
print("Elapsed time: ", elapsed_time)
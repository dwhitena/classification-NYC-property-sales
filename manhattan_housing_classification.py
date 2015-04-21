import pandas as pd
import matplotlib.pyplot as plt
import sqlite3 as lite
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

# ----------------------
# IMPORT AND CLEAN DATA 
# ----------------------

df = pd.read_csv('rollingsales_manhattan.csv', skiprows=4, low_memory=False)

# specify the features for k-NN.  In this case, geographic features.
features = [
			'BLOCK',
			'LOT'
			]
df = df[features + ['NEIGHBORHOOD']]

# ----------------------
# TRAIN/TEST K-NN 
# ----------------------

# split data into training and test sets
dfTrain, dfTest = train_test_split(df, test_size=0.3)

# number of features
fnum = len(features)

#model = KNeighborsClassifier(n_neighbors=3)
#model.fit(dfTrain[:,:fnum], dfTrain[:,fnum])

#determine k for K-NN
for k in range(1,20):
    model = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    model.fit(dfTrain[:,:fnum], dfTrain[:,fnum])
    # make predictions
    expected = dfTest[:,fnum]
    predicted = model.predict(dfTest[:,:fnum])
    # misclassification rate
    error_rate = (predicted != expected).mean()
    print('%d:, %.2f' % (k, error_rate))

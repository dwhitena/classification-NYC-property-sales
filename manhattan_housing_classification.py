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

# set sale date as index
#index = pd.PeriodIndex(df['SALE DATE'], freq='D')
#df = df.set_index(index)
#ts = df['ADDRESS'].groupby(df.index).count()

# convert price to integer
df['SALE PRICE'] = df['SALE PRICE'].map(lambda element: 
										int(element.replace(",", "").lstrip("$")))

# convert square feet to integer
df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].map(lambda element: 
													  int(element.replace(",","")))
df['LAND SQUARE FEET'] = df['LAND SQUARE FEET'].map(lambda element: 
													  int(element.replace(",","")))

# just consider residential properties with a non-zero sale price
df = df[df['TAX CLASS AT TIME OF SALE'] == 1]
df = df[df['SALE PRICE'] > 100]
df = df[df['GROSS SQUARE FEET'] > 0]
#df['logsaleprice'] = df['SALE PRICE'].map(lambda num: np.log(num))
df['class1_dummy'] = pd.Categorical(df['BUILDING CLASS CATEGORY']).codes
df['class2_dummy'] = pd.Categorical(df['BUILDING CLASS AT TIME OF SALE']).codes
df['logsaleprice'] = df['SALE PRICE'].map(lambda num: np.log(num))
features = [
			'logsaleprice',
			'TOTAL UNITS'
			]
df = df[features + ['NEIGHBORHOOD']]

# ----------------------
# TRAIN/TEST K-NN 
# ----------------------

# split data into training and test sets
dfTrain, dfTest = train_test_split(df, test_size=0.2)

# number of features
fnum = len(features)

#model = KNeighborsClassifier(n_neighbors=3)
#model.fit(dfTrain[:,:fnum], dfTrain[:,fnum])

#determine k for K-NN
for k in range(1,20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(dfTrain[:,:fnum], dfTrain[:,fnum])
    # make predictions
    expected = dfTest[:,fnum]
    predicted = model.predict(dfTest[:,:fnum])
    # misclassification rate
    error_rate = (predicted != expected).mean()
    print('%d:, %.2f' % (k, error_rate))

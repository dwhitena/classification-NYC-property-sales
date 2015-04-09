import pandas as pd
import matplotlib.pyplot as plt
import sqlite3 as lite
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf

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

# just consider residential properties with a non-zero sale price
df = df[df['TAX CLASS AT TIME OF SALE'] == 1]
df = df[df['SALE PRICE'] > 100]
df = df[df['GROSS SQUARE FEET'] > 0]
df['logsaleprice'] = df['SALE PRICE'].map(lambda num: np.log(num))


# -----------------------------------
# LINEAR REGRESSION MODEL AND VISUALIZATION
# -----------------------------------

# SALE PRICE vs. GROSS SQUARE FEET
# shape data
# The dependent variable
y = np.matrix(df['SALE PRICE']).transpose()
# The independent variables shaped as columns
x = np.matrix(df['GROSS SQUARE FEET']).transpose()

# add column of ones (constant)
X = sm.add_constant(x)

# Linear model
model1 = sm.OLS(y,X).fit()
print 'Square Feet Coefficient: ', model1.params[1]
print 'Intercept: ', model1.params[0]
print 'P-Values: ', model1.pvalues  
print 'R-Squared: ', model1.rsquared

plt.figure()
sns.jointplot("GROSS SQUARE FEET", "SALE PRICE", df, kind="reg")
plt.savefig('price_vs_gsf_regression_model.png')

# log(SALE PRICE) vs. GROSS SQUARE FEET
# shape data
# The dependent variable
y = np.matrix(df['logsaleprice']).transpose()
# The independent variables shaped as columns
x = np.matrix(df['GROSS SQUARE FEET']).transpose()

# add column of ones (constant)
X = sm.add_constant(x)

# Linear model
model2 = sm.OLS(y,X).fit()
print 'Square Feet Coefficient: ', model2.params[1]
print 'Intercept: ', model2.params[0]
print 'P-Values: ', model2.pvalues  
print 'R-Squared: ', model2.rsquared

plt.figure()
sns.jointplot("GROSS SQUARE FEET", "logsaleprice", df, kind="reg")
plt.savefig('logprice_vs_gsf_regression_model.png')


# -------------------------------------------
# MULTIPLE REGRESSION MODEL AND VISUALIZATION
# -------------------------------------------

df.rename(columns={
				   "SALE PRICE":"sale_price", 
				   "GROSS SQUARE FEET":"gross_square_feet",
				   "BUILDING CLASS CATEGORY":"building_class"
				   }, inplace=True)

# Multiple regression model, price vs. building class and sq feet
model3 = smf.ols(formula="sale_price ~ C(building_class) + gross_square_feet", 
				 data=df).fit()
print model3.params
print model3.summary()
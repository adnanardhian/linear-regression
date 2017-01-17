import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import random

#read data
dataframe = pd.read_csv('challenge_dataset.txt', header = None, names = ['X','Y'])
x_values = dataframe[['X']].values.tolist()
y_values = dataframe[['Y']].values.tolist()

#train model on data
challenge_reg = linear_model.LinearRegression()
challenge_reg.fit(x_values, y_values)

#All Values
#Returns the coefficient of determination R^2 of the prediction.
print "Score 		: ",
print challenge_reg.score(x_values,y_values)

#Assign the Prediction Test of One Sample from existing data
n = random.randint(0,len(x_values)-1)
one_pred = challenge_reg.predict([x_values[n]])[0]

#Error of One value :
print "Index of X 	: ", + (n)
print "X Sample 	: ", + (x_values[n][0])
print "Y Prediction	: ", + (one_pred[0])
print "Y Actual 	: ", + (y_values[n][0])
print "Error    	: ", + (abs(one_pred[0] - y_values[n][0]))

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, challenge_reg.predict(x_values))
plt.show()


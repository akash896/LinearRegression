import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# print the labels of the loaded data set
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
# data is an array with the following attributes
# - age     age in years
#      - sex
#      - bmi     body mass index
#      - bp      average blood pressure
#      - s1      tc, total serum cholesterol
#      - s2      ldl, low-density lipoproteins
#      - s3      hdl, high-density lipoproteins
#      - s4      tch, total cholesterol / HDL
#      - s5      ltg, possibly log of serum triglycerides level
#      - s6      glu, blood sugar level
# print(diabetes.data)

# taking the 3rd column i.e. bmi as X axis
diabetes_X = diabetes.data[:,np.newaxis]
diabetes_X_temp = diabetes_X[:,:,2]

# print(len(diabetes_X))

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-30]
diabetes_X_test  = diabetes_X_temp[-30:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test  = diabetes.target[-30:]
#print(diabetes_y_train)

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_X_test)


# End result of prediction
print("Predicted value = ", diabetes_y_predicted)
print("Actual value = ", diabetes_y_test)


print("Mean Squared Error = ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights = ", model.coef_)
print("Intercept = ", model.intercept_)
plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_predicted)
plt.show()

"""
another correct impleementation
import pylab as pl
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:,np.newaxis]
diabetes_X_temp = diabetes_X[:,:,2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test  = diabetes_X_temp[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit (diabetes_X_train, diabetes_y_train)

# The coefficients
 print ('Coefficients: \n', regr.coef_)
# The mean square error
print ("Residual sum of squares: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2))
# Explained variance score: 1 is perfect prediction
print ('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
pl.scatter(diabetes_X_test, diabetes_y_test,  color='black')
pl.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

pl.xticks(())
pl.yticks(())

pl.show()

"""

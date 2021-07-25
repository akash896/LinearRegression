# Predicting Diabetes value of a person using all the features present

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
diabetes_X = diabetes.data
diabetes_X_temp = diabetes_X

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
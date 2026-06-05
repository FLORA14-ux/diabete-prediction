import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the cvs data to a Pandas DataFrame
heart_data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

# Print the first 5 rows of the dataset
print(heart_data.head())

# print last 5 rows of the dataset
print(heart_data.tail())

# number of rows and columns in the dataset
print(heart_data.shape)
(303, 14)

# getting some info about the dataset
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the dataset
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

1 --> Defective Heart
0 --> Healthy Heart

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
(303, 13) (242, 13) (61, 13)

model = LogisticRegression()

# training the Logistic Regression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train) 
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

Accuracy on Training data :  0.8512396694214877

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)    

print('Accuracy on Test data : ', test_data_accuracy)

Accuracy on Test data :  0.819672131147541

input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.0, 1, 0, 2)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person does not have a heart disease')
else:    print('The person has a heart disease') 

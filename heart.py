import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Chargement des données cardiaques
heart_data = pd.read_csv('heart.csv')

# Affichage des premières et dernières lignes
print(heart_data.head())
print(heart_data.tail())

# Dimensions du dataset
print(heart_data.shape) # Résultat attendu : (303, 14)

# Informations et vérification des valeurs manquantes
heart_data.info()
print(heart_data.isnull().sum())

# Statistiques descriptives
print(heart_data.describe())

# Distribution de la variable cible
print(heart_data['target'].value_counts())
# 1 --> Cœur avec anomalie
# 0 --> Cœur sain

# Séparation des fonctionnalités et de la cible
X = heart_data.drop(columns='target')
Y = heart_data['target']
print(X)
print(Y)

# Division des données (Train / Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initialisation du modèle
model = LogisticRegression(max_iter=1000)

# Entraînement du modèle
model.fit(X_train, Y_train)

# Précision sur les données d'entraînement
X_train_prediction = model.predict(X_train) 
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# Précision sur les données de test
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)    
print('Accuracy on Test data : ', test_data_accuracy)

# Saisie de test pour une prédiction
input_data = (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.0, 1, 0, 2)

# Transformation en tableau numpy et redimensionnement
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prédiction finale
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person does not have a heart disease')
else:
    print('The person has a heart disease')
import numpy as np  # Correction : 'numpy' au lieu de 'numepy'
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm  # Correction : 'svm' au lieu de 'swm'
from sklearn.metrics import accuracy_score  # Correction : 'metrics' au lieu de 'model_selection'

# 1. Chargement du jeu de données
# Assure-toi que le fichier 'diabetes.csv' est bien téléchargé dans le même dossier !
diabetes_data = pd.read_csv('diabetes.csv')

# Affichage des 5 premières lignes
print("--- Les 5 premières lignes du dataset ---")
print(diabetes_data.head())

# Nombre de lignes et de colonnes
print("\n--- Dimensions du dataset (Lignes, Colonnes) ---")
print(diabetes_data.shape)

# Obtenir les mesures statistiques
print("\n--- Statistiques descriptives ---")
print(diabetes_data.describe())

print("\n--- Nombre de cas (0 = Non-diabétique, 1 = Diabétique) ---")
print(diabetes_data['Outcome'].value_counts())

print("\n--- Moyennes globales groupées par résultat ---")
# Correction : 'diabetes_data' au lieu de 'diabetic_dataset'
print(diabetes_data.groupby('Outcome').mean())

# 2. Séparation des fonctionnalités (X) et de la cible (Y)
X = diabetes_data.drop('Outcome', axis=1)
Y = diabetes_data['Outcome']

# 3. Standardisation des données
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_data['Outcome']

# 4. Séparation en données d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print("\n--- Dimensions après séparation (Total, Train, Test) ---")
print(X.shape, X_train.shape, X_test.shape)

# 5. Création et entraînement du modèle SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# 6. Évaluation du modèle
# Précision sur les données d'entraînement
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('\nPrécision sur les données d\'entraînement : ', training_data_accuracy)

# Précision sur les données de test
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Précision sur les données de test : ', test_data_accuracy)

# 7. Système de prédiction pour un nouveau patient
# Correction : Syntaxe de la variable 'input_data'
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Changement des données d'entrée en tableau numpy
# Correction : 'asarray' au lieu de 'asaaray'
input_data_as_numpy_array = np.asarray(input_data)

# Redimensionnement du tableau
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardisation des données d'entrée
std_data = scaler.transform(input_data_reshaped)

# Prédiction finale
prediction = classifier.predict(std_data)

print("\n--- Résultat de la prédiction ---")
if (prediction[0] == 0):
    print('Le patient n\'est pas diabétique.')
else:
    print('Le patient est diabétique.')
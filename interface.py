import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Configuration et Style CSS (Font Awesome)
st.set_page_config(page_title="Plateforme de Diagnostic IA", layout="centered")
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)

st.markdown("""
    <style>
        .main-title { font-size: 2.2rem; font-weight: bold; margin-bottom: 20px; }
        .icon-spacing { margin-right: 10px; color: #4A90E2; }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION LATERALE ---
with st.sidebar:
    st.title("Menu de Navigation")
    option = st.selectbox("Choisir le diagnostic :", ["Analyse du Diabète", "Santé Cardiaque"])
    st.info("Cette plateforme utilise des modèles de Machine Learning entraînés sur des données cliniques.")

# --- FONCTIONS D'ENTRAINEMENT ---
@st.cache_resource
def charger_diabete():
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    Y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = svm.SVC(kernel='linear')
    model.fit(X_scaled, Y)
    return scaler, model

@st.cache_resource
def charger_coeur():
    df = pd.read_csv('heart.csv')
    X = df.drop(columns='target')
    Y = df['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    return model

# --- PAGE DIABÈTE ---
if option == "Analyse du Diabète":
    st.markdown('<div class="main-title"><i class="fa-solid fa-stethoscope icon-spacing"></i>Analyse du Diabète</div>', unsafe_allow_html=True)
    scaler, model = charger_diabete()
    
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Nombre de grossesses", 0, 20, 1)
        gluc = st.number_input("Taux de glucose (mg/dL)", 0, 300, 100)
        bp = st.number_input("Pression artérielle (mm Hg)", 0, 200, 70)
        skin = st.number_input("Épaisseur du pli cutané (mm)", 0, 100, 20)
    with col2:
        ins = st.number_input("Insuline (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("IMC", 0.0, 70.0, 25.0)
        dpf = st.number_input("Indice généalogique", 0.0, 3.0, 0.5)
        age = st.number_input("Âge", 1, 120, 30)

    if st.button("Lancer l'analyse Diabète"):
        data = scaler.transform(np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]]))
        pred = model.predict(data)
        if pred[0] == 1: st.error("Résultat : Risque de diabète détecté")
        else: st.success("Résultat : Aucun signe de diabète")

# --- PAGE COEUR ---
elif option == "Santé Cardiaque":
    st.markdown('<div class="main-title"><i class="fa-solid fa-heart-pulse icon-spacing"></i>Analyse Cardiaque</div>', unsafe_allow_html=True)
    model_coeur = charger_coeur()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", 1, 120, 45)
        sex = st.selectbox("Sexe (1=M, 0=F)", [1, 0])
        cp = st.number_input("Type de douleur thoracique (0-3)", 0, 3, 1)
        trestbps = st.number_input("Pression artérielle au repos", 50, 250, 120)
        chol = st.number_input("Cholestérol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Sucre dans le sang > 120 (1=Oui, 0=Non)", [0, 1])
        restecg = st.number_input("Électrocardiogramme au repos (0-2)", 0, 2, 0)
    with col2:
        thalach = st.number_input("Fréquence cardiaque max", 50, 250, 150)
        exang = st.selectbox("Angine liée à l'exercice (1=Oui, 0=Non)", [0, 1])
        oldpeak = st.number_input("Dépression ST", 0.0, 10.0, 1.0)
        slope = st.number_input("Pente du segment ST (0-2)", 0, 2, 1)
        ca = st.number_input("Nombre de vaisseaux colorés (0-3)", 0, 3, 0)
        thal = st.number_input("Thalassémie (0-3)", 0, 3, 2)

    if st.button("Lancer l'analyse Cardiaque"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        pred = model_coeur.predict(input_data)
        if pred[0] == 1: st.error("Résultat : Risque de maladie cardiaque détecté")
        else: st.success("Résultat : Cœur sain")
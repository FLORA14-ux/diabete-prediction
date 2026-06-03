import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Configuration de la page
st.set_page_config(page_title="Analyse de données - Diabète", layout="centered")

# Chargement de Font Awesome pour avoir de vraies icônes CSS
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
    unsafe_allow_html=True
)

# Style CSS personnalisé pour harmoniser les titres avec les icônes
st.markdown("""
    <style>
        .main-title { font-size: 2.2rem; font-weight: bold; margin-bottom: 5px; }
        .section-title { font-size: 1.5rem; font-weight: 600; margin-top: 20px; margin-bottom: 15px; }
        .icon-spacing { margin-right: 10px; color: #4A90E2; }
        .icon-success { margin-right: 10px; color: #2ecc71; }
        .icon-error { margin-right: 10px; color: #e74c3c; }
    </style>
""", unsafe_allow_html=True)

# Titre principal avec icône CSS (fa-stethoscope)
st.markdown('<div class="main-title"><i class="fa-solid fa-stethoscope icon-spacing"></i>Application d\'Évaluation Clinique du Diabète</div>', unsafe_allow_html=True)
st.write("Ce système informatique s'appuie sur un modèle d'apprentissage supervisé (Support Vector Machine) pour analyser les paramètres physiologiques et évaluer les facteurs de risque.")

# Chargement et entraînement du modèle en arrière-plan
@st.cache_resource
def entrainer_modele():
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    Y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_scaled, Y)
    return scaler, classifier

scaler, classifier = entrainer_modele()

# Formulaire de saisie avec icône CSS (fa-clipboard-list)
st.markdown('<div class="section-title"><i class="fa-solid fa-clipboard-list icon-spacing"></i>Saisie des paramètres physiologiques du patient</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Taux de glucose plasmatique (mg/dL)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Pression artérielle diastolique (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Épaisseur du pli cutané (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Taux d'insuline sérique (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Indice de Masse Corporelle (IMC)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Fonction antécédents familiaux (Diabetes Pedigree)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Âge du patient", min_value=1, max_value=120, value=30)

# Bouton d'analyse
if st.button("Lancer l'analyse du profil"):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    
    prediction = classifier.predict(std_data)
    
    st.markdown("---")
    st.markdown('<div class="section-title">Diagnostic du modèle :</div>', unsafe_allow_html=True)
    
    if prediction[0] == 0:
        # Affichage avec icône CSS de succès (fa-circle-check)
        st.success("Résultat : Patient non-diabétique (Classe 0)")
    else:
        # Affichage avec icône CSS d'alerte (fa-triangle-exclamation)
        st.error("Résultat : Patient diabétique (Classe 1)")
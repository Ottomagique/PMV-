import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime, timedelta
from itertools import combinations
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Configuration de la page
st.set_page_config(
    page_title="Analyse IPMVP Simplifiée",
    page_icon="📊",
    layout="wide"
)

# 🔹 Appliquer le CSS (Uniquement pour améliorer le design)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Manrope', sans-serif;
        color: #0C1D2D;
    }

    h1, h2, h3 {
        font-weight: 800;
        color: #00485F;
    }

    h4, h5, h6 {
        font-weight: 700;
        color: #00485F;
    }

    .stButton>button {
        background-color: #6DBABC;
        color: white;
        border-radius: 8px;
        padding: 12px 18px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #96B91D;
        color: white;
        transform: scale(1.05);
    }

    .stSidebar {
        background-color: #E7DDD9;
        padding: 20px;
        border-radius: 10px;
    }

    input, select, textarea {
        background-color: #E7DDD9 !important;
        border-radius: 5px;
        border: 1px solid #00485F;
    }

    .block-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #E7DDD9;
    }

    .stDataFrame {
        border: 1px solid #0C1D2D;
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

# 🎯 Interface utilisateur
st.title("📊 Analyse IPMVP")
st.write("""
Cette application vous permet d'analyser vos données de consommation énergétique selon le protocole IPMVP.
Importez un fichier Excel avec au minimum une colonne de dates et une colonne de consommations,
plus des colonnes optionnelles pour les variables explicatives comme les DJU, effectifs, etc.
""")

# 📂 **Chargement des données**
st.sidebar.header("Configuration")
st.sidebar.subheader("1. Chargement des données")
uploaded_file = st.sidebar.file_uploader("Fichier Excel de consommation", type=["xlsx", "xls"])

# 📌 Lecture du fichier (Fonction inchangée)
@st.cache_data
def load_data(file):
    """Charge les données depuis un fichier Excel"""
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)

if df is not None:
    st.subheader("Données chargées")
    st.dataframe(df.reset_index(drop=True))

    # Sélection des colonnes (Inchangé)
    date_col = st.sidebar.selectbox("Colonne de date", df.columns)
    conso_col = st.sidebar.selectbox("Colonne de consommation", df.columns)
    var_options = [col for col in df.columns if col not in [date_col, conso_col]]
    selected_vars = st.sidebar.multiselect("Variables explicatives", var_options)

    # 📈 **Affichage des résultats**
    tab1, tab2, tab3 = st.tabs(["📈 Consommation", "📊 Modèle", "📋 Données"])  # 👉 Onglet "Consommation" en premier

    with tab1:
        st.subheader("📊 Comparaison Consommation Mesurée vs Ajustée")
        st.write("Génération du graphe en cours...")

    with tab2:
        st.subheader("🔍 Analyse du modèle IPMVP")
        st.write("Analyse en cours...")

    with tab3:
        st.subheader("📋 Données détaillées")
        st.dataframe(df.reset_index(drop=True))

st.sidebar.markdown("---")
st.sidebar.info("Développé avec ❤️ et Streamlit 🚀")

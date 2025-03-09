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

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Analyse IPMVP Simplifiée",
    page_icon="📊",
    layout="wide"
)

# 🔹 Appliquer le CSS personnalisé APRÈS `st.set_page_config`
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

# 📊 **Exemple de données**
if not uploaded_file:
    st.info("👆 Chargez un fichier Excel ou utilisez ces données d'exemple.")
    example_data = {
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='MS'),
        'Consommation': [570, 467, 490, 424, 394, 350, 320, 310, 370, 420, 480, 540],
        'DJU_Base_18': [460, 380, 320, 240, 150, 50, 20, 30, 130, 230, 350, 430],
        'Effectif': [100, 100, 100, 98, 98, 95, 90, 90, 95, 98, 100, 100]
    }
    example_df = pd.DataFrame(example_data)

    st.subheader("Exemple de données")
    st.dataframe(example_df.reset_index(drop=True))

    use_example = st.button("Utiliser ces données d'exemple")
    if use_example:
        st.session_state.df = example_df
        st.success("Données d'exemple chargées!")
        st.rerun()

# 📌 **Lecture du fichier**
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
elif hasattr(st.session_state, 'df'):
    df = st.session_state.df

if df is not None:
    st.subheader("Données chargées")
    st.dataframe(df.reset_index(drop=True))

    # Sélection des colonnes
    date_col = st.sidebar.selectbox("Colonne de date", df.columns)
    conso_col = st.sidebar.selectbox("Colonne de consommation", df.columns)

    var_options = [col for col in df.columns if col not in [date_col, conso_col]]
    selected_vars = st.sidebar.multiselect("Variables explicatives", var_options)

    # 📈 **Affichage des résultats**
    tab1, tab2, tab3 = st.tabs(["📈 Consommation", "📊 Modèle", "📋 Données"])

    with tab1:
        st.subheader("📊 Comparaison Consommation Mesurée vs Ajustée")
        st.write("Génération du graphe en cours...")

    with tab2:
        st.subheader("🔍 Analyse du modèle IPMVP")
        st.write("Analyse en cours...")

    with tab3:
        st.subheader("📋 Données détaillées")
        st.dataframe(df.reset_index(drop=True))

    # 📥 **Téléchargement des résultats**
    st.sidebar.subheader("📥 Télécharger le rapport")
    if df is not None:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Données", index=False)
        st.sidebar.download_button("📂 Télécharger", buffer.getvalue(), file_name="rapport_IPMVP.xlsx")

st.sidebar.markdown("---")
st.sidebar.info("Développé avec ❤️ et Streamlit 🚀")

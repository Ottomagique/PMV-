import streamlit as st
import pandas as pd
import numpy as np
import io
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# 📌 Configuration de la page
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

st.sidebar.header("⚙️ Configuration")

# 📂 **Sélection des colonnes avant chargement**
st.sidebar.subheader("1. Sélection des colonnes")
date_col = st.sidebar.text_input("Nom de la colonne de date", "Date")
conso_col = st.sidebar.text_input("Nom de la colonne de consommation", "Consommation")

# Sélection des variables explicatives
var_input = st.sidebar.text_area("Noms des variables explicatives (séparés par une virgule)", "DJU_Base_18, Effectif")
var_options = [col.strip() for col in var_input.split(",") if col.strip()]

# Sélection du nombre de variables à tester (1 à 4)
st.sidebar.subheader("2. Choix du modèle")
max_features = st.sidebar.slider("Nombre de variables à tester", 1, 4, 2)

# 📂 **Chargement des données**
st.sidebar.subheader("3. Chargement des données")
uploaded_file = st.sidebar.file_uploader("Fichier Excel de consommation", type=["xlsx", "xls"])

# 📌 Lecture du fichier
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

    # Vérifier que les colonnes existent
    if date_col not in df.columns or conso_col not in df.columns:
        st.error("Les noms de colonnes sélectionnés ne sont pas valides.")
        st.stop()

    selected_vars = [col for col in var_options if col in df.columns]

    # Bouton pour lancer le calcul
    if st.sidebar.button("🚀 Lancer le calcul"):
        st.subheader("Analyse en cours...")

        # 🔹 Sélection des colonnes
        X = df[selected_vars] if selected_vars else pd.DataFrame(index=df.index)
        y = df[conso_col]

        best_model = None
        best_r2 = -1
        best_features = []
        
        # 🔹 Test des combinaisons de variables (de 1 à max_features)
        for n in range(1, max_features + 1):
            for combo in combinations(selected_vars, n):
                X_subset = X[list(combo)]
                model = LinearRegression()
                model.fit(X_subset, y)
                y_pred = model.predict(X_subset)
                r2 = r2_score(y, y_pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_features = list(combo)

        # 🔹 Résultats du modèle sélectionné
        if best_model:
            st.success("✅ Modèle trouvé avec succès !")
            st.write(f"**Meilleures variables utilisées :** {', '.join(best_features)}")
            st.write(f"**R² du modèle :** {best_r2:.4f}")

            # 🔹 Graphique de consommation
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(y)), y, color="#6DBABC", label="Consommation mesurée")
            ax.plot(range(len(y)), best_model.predict(df[best_features]), color="#E74C3C", marker='o', label="Consommation ajustée")
            ax.set_title("Comparaison Consommation Mesurée vs Ajustée")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("⚠️ Aucun modèle valide n'a été trouvé.")

st.sidebar.markdown("---")
st.sidebar.info("Développé avec ❤️ et Streamlit 🚀")

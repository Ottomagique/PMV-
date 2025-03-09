import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 📌 Configuration de la page
st.set_page_config(
    page_title="Analyse IPMVP Simplifiée",
    page_icon="📊",
    layout="wide"
)

# 🔹 Appliquer le CSS (Maintien du design existant)
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

# 📌 **Description de l'application améliorée**
st.title("📊 Analyse IPMVP")
st.markdown("""
Bienvenue sur **l'Analyse IPMVP Simplifiée** 🔍 !  
Cette application vous permet d'analyser **vos données de consommation énergétique** et de trouver le **meilleur modèle d'ajustement** basé sur plusieurs variables explicatives.

### **🛠️ Instructions :**
1. **Importer un fichier Excel 📂** contenant les données de consommation sur plusieurs années (*exemple : 3 ans de consommation*).
2. **Sélectionner la colonne de date 📅, la consommation ⚡ et les variables explicatives 📊**.
3. **Choisir le nombre de variables à tester 🔢** (de 1 à 4).
4. **Lancer le calcul 🚀** pour identifier le **meilleur modèle d’ajustement sur une période de 12 mois glissants**.

📌 **Pourquoi 12 mois glissants ?**  
L’analyse est réalisée sur **plusieurs sous-périodes de 12 mois** pour trouver la meilleure corrélation avec vos variables explicatives et obtenir un modèle fiable.
""")

# 📂 **Import du fichier et lancement du calcul**
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Importer un fichier Excel", type=["xlsx", "xls"])

with col2:
    lancer_calcul = st.button("🚀 Lancer le calcul", use_container_width=True)

# 📂 **Sélection des données**
st.sidebar.header("🔍 Sélection des données")

df = None

if uploaded_file:
    df = pd.read_excel(uploaded_file)

# 📌 Sélection des colonnes avec explication des données
date_col = st.sidebar.selectbox("📅 Nom de la colonne date (ex : 'Date')", df.columns if df is not None else [""])
conso_col = st.sidebar.selectbox("⚡ Nom de la colonne consommation (ex : 'Consommation')", df.columns if df is not None else [""])
var_options = [col for col in df.columns if col not in [date_col, conso_col]] if df is not None else []
selected_vars = st.sidebar.multiselect("📊 Variables explicatives (ex : 'DJU', 'Effectif')", var_options)

# Nombre de variables à tester
max_features = st.sidebar.slider("🔢 Nombre de variables à tester", 1, 4, 2)

# 📌 **Lancement du calcul seulement si le bouton est cliqué**
if df is not None and lancer_calcul:
    st.subheader("⚙️ Analyse en cours...")

    df[date_col] = pd.to_datetime(df[date_col])
    X = df[selected_vars] if selected_vars else pd.DataFrame(index=df.index)
    y = df[conso_col]

    # Nettoyage des données
    if X.isnull().values.any() or np.isinf(X.values).any():
        st.error("❌ Les variables explicatives contiennent des valeurs manquantes ou non numériques.")
        st.stop()

    if y.isnull().values.any() or np.isinf(y.values).any():
        st.error("❌ La colonne de consommation contient des valeurs manquantes ou non numériques.")
        st.stop()

    X = X.apply(pd.to_numeric, errors='coerce').dropna()
    y = pd.to_numeric(y, errors='coerce').dropna()

    best_model = None
    best_r2 = -1
    best_features = []
    best_y_pred = None

    # 🔹 Test sur plusieurs périodes glissantes de 12 mois
    periodes = df[date_col].dt.to_period('M').unique()
    if len(periodes) >= 12:
        for i in range(len(periodes) - 11):
            periode_actuelle = periodes[i:i+12]
            df_subset = df[df[date_col].dt.to_period('M').isin(periode_actuelle)]

            X_subset = df_subset[selected_vars]
            y_subset = df_subset[conso_col]

            for n in range(1, max_features + 1):
                for combo in combinations(selected_vars, n):
                    X_temp = X_subset[list(combo)]
                    model = LinearRegression()
                    model.fit(X_temp, y_subset)
                    y_pred = model.predict(X_temp)
                    r2 = r2_score(y_subset, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_subset, y_pred))
                    cv = rmse / np.mean(y_subset) if np.mean(y_subset) != 0 else np.inf
                    bias = np.mean(y_pred - y_subset) / np.mean(y_subset) if np.mean(y_subset) != 0 else np.inf

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                        best_features = list(combo)
                        best_y_pred = y_pred
                        best_cv = cv
                        best_bias = bias

    # 🔹 Résultats du modèle sélectionné
    if best_model:
        st.success("✅ Modèle trouvé avec succès !")
        st.markdown("### 📋 Résumé du modèle")
        st.write(f"**📈 R² :** `{best_r2:.4f}`")
        st.write(f"**📉 CV(RMSE) :** `{best_cv:.4f}`")
        st.write(f"**⚠️ NMBE (Biais normalisé) :** `{best_bias:.6f}`")
        st.write(f"**🛠️ Type de modèle :** `Régression Linéaire`")

        st.markdown("### 📊 Comparaison Consommation Mesurée vs Ajustée")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(best_y_pred, color="#E74C3C", marker='o', label="Consommation ajustée")
        ax.legend()
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Développé avec ❤️ et Streamlit 🚀")

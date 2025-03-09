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

# 🔹 Appliquer le CSS **(Respecte la charte graphique d'origine)**
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Manrope', sans-serif;
        color: #0C1D2D;
        background-color: #F8F6F2;
    }

    h1, h2, h3 {
        font-weight: 800;
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
    </style>
    """, unsafe_allow_html=True)

# 📌 **Description de l'application**
st.title("📊 Analyse IPMVP")
st.markdown("""
Bienvenue sur **l'Analyse IPMVP Simplifiée** 🔍 !  
Cette application analyse votre **consommation énergétique** et ajuste un modèle en fonction des variables explicatives.

### **🛠️ Instructions :**
1. **Importer un fichier Excel 📂** avec au moins une colonne de dates et une colonne de consommation.
2. **Sélectionner la colonne de date 📅, la consommation ⚡ et les variables explicatives 📊**.
3. **Choisir le nombre de variables à tester 🔢** (de 1 à 4).
4. **Lancer le calcul 🚀** pour identifier le **meilleur modèle**.

📌 **Le modèle teste des périodes glissantes de 12 mois** pour trouver la meilleure corrélation.
""")

# 📂 **Import du fichier et lancement du calcul**
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Importer un fichier Excel", type=["xlsx", "xls"])

with col2:
    lancer_calcul = st.button("🚀 Lancer le calcul", use_container_width=True)

if "lancer_calcul" not in st.session_state:
    st.session_state.lancer_calcul = False

if lancer_calcul:
    st.session_state.lancer_calcul = True

# 📂 **Sélection des données**
st.sidebar.header("🔍 Sélection des données")

df = None
if uploaded_file:
    df = pd.read_excel(uploaded_file)

# 📌 Sélection des colonnes
date_col = st.sidebar.selectbox("📅 Nom de la colonne Date", df.columns if df is not None else [""])
conso_col = st.sidebar.selectbox("⚡ Nom de la colonne Consommation", df.columns if df is not None else [""])
var_options = [col for col in df.columns if col not in [date_col, conso_col]] if df is not None else []
selected_vars = st.sidebar.multiselect("📊 Variables Explicatives", var_options)

max_features = st.sidebar.slider("🔢 Nombre de variables à tester", 1, 4, 2)

# 📌 **Graphique amélioré : Consommation réelle vs Ajustée**
def plot_consumption(y_actual, y_pred, dates):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fig.patch.set_facecolor("#E7DDD9")  # Fond beige du thème
    ax.set_facecolor("#F8F6F2")  # Fond clair pour lisibilité

    ax.bar(dates, y_actual, color="#00485F", label="🔵 Consommation réelle", alpha=0.8, width=0.6)
    ax.plot(dates, y_pred, color="#E74C3C", marker='o', linestyle='-', linewidth=2.5, markersize=7, 
            label="🔴 Consommation ajustée")
    
    ax.set_xlabel("📆 Mois", fontsize=12, fontweight="bold", color="#0C1D2D")
    ax.set_ylabel("⚡ Consommation", fontsize=12, fontweight="bold", color="#0C1D2D")
    ax.set_title("📊 Comparaison Consommation Mesurée vs Ajustée", fontsize=14, fontweight="bold", color="#00485F")

    ax.grid(True, linestyle="--", alpha=0.5, color="#B0BEC5")

    ax.set_xticks(dates)
    ax.set_xticklabels([d.strftime("%b %Y") for d in dates], rotation=45, ha="right")

    ax.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=True, facecolor="#F8F6F2")

    return fig

# 📌 **Lancer le calcul après sélection des variables**
if df is not None and st.session_state.lancer_calcul:
    with st.spinner("⏳ Analyse en cours..."):
        df[date_col] = pd.to_datetime(df[date_col])
        df[conso_col] = pd.to_numeric(df[conso_col], errors='coerce')
        df = df.dropna(subset=[conso_col])  

        X = df[selected_vars] if selected_vars else pd.DataFrame(index=df.index)
        y = df[conso_col]

        best_model = None
        best_r2 = -1
        best_cv = None
        best_bias = None
        best_features = []
        best_y_pred = None

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
                        cv = rmse / np.mean(y_subset)
                        bias = np.mean(y_pred - y_subset) / np.mean(y_subset)

                        if r2 > best_r2:
                            best_r2 = r2
                            best_cv = cv
                            best_bias = bias
                            best_model = model
                            best_features = list(combo)
                            best_y_pred = y_pred
                            best_dates = df_subset[date_col]

    st.success("✅ Résultats de l'analyse")
    st.write(f"**📌 Meilleur Modèle :** `{'Régression Linéaire'}`")
    st.write(f"**📑 Équation d'ajustement :** `y = {best_model.intercept_:.4f} + {' + '.join([f'{coef:.4f} × {feat}' for coef, feat in zip(best_model.coef_, best_features)])}`")
    st.pyplot(plot_consumption(y_subset, best_y_pred, best_dates))

st.sidebar.write("💡 **Développé par Efficacité Energétique, Carbone & RSE Team | © 2025**")

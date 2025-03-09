import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ✅ Configuration de la page (inchangé)
st.set_page_config(page_title="Analyse IPMVP", page_icon="📊", layout="wide")

# ✅ Appliquer le CSS personnalisé existant (inchangé)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Manrope', sans-serif; color: #0C1D2D; }
    h1, h2, h3 { font-weight: 800; }
    h4, h5, h6 { font-weight: 700; }
    .stButton>button { background-color: #00485F; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #96B91D; }
    .stSidebar { background-color: #6DBABC; }
    </style>
    """, unsafe_allow_html=True)

# ✅ Message de bienvenue (inchangé)
st.markdown("### 🎯 **Bienvenue dans l'application d'analyse IPMVP !**")
st.write("Cette application vous permet d'analyser vos données de consommation énergétique en fonction de divers facteurs explicatifs selon la méthodologie IPMVP.")

# ✅ Section : Importation du fichier
st.sidebar.subheader("📂 **Importer un fichier Excel**")
uploaded_file = st.sidebar.file_uploader("", type=["xlsx", "xls"])

# ✅ Vérification du fichier
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"📄 Fichier chargé : **{uploaded_file.name}**")
else:
    st.warning("📌 Importez un fichier pour commencer l'analyse.")
    st.stop()

# ✅ Sélection des colonnes
st.sidebar.subheader("🔍 **Sélection des données**")
date_col = st.sidebar.selectbox("📅 **Nom de la colonne date**", df.columns)
conso_col = st.sidebar.selectbox("⚡ **Nom de la colonne consommation**", df.columns)
var_explicatives = st.sidebar.multiselect("📊 **Variables explicatives**", [col for col in df.columns if col not in [date_col, conso_col]])
nb_var = st.sidebar.slider("🔢 **Nombre de variables à tester**", 1, min(4, len(var_explicatives)), 1)

# ✅ Conversion de la colonne date et suppression des valeurs NaN
df[date_col] = pd.to_datetime(df[date_col])
df = df.dropna()

# ✅ Classe pour la modélisation IPMVP
class ModelIPMVP:
    def __init__(self):
        self.best_model = None
        self.best_r2 = 0
        self.best_cv = None
        self.best_bias = None
        self.best_features = []
        self.best_formula = ""
        self.best_model_type = ""

    def evaluer_combinaison(self, X, y, features):
        """Évalue un modèle avec les variables sélectionnées"""
        X_subset = X[features]
        model = LinearRegression()
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)

        r2 = r2_score(y, y_pred)
        cv = np.sqrt(mean_squared_error(y, y_pred)) / np.mean(y)
        bias = np.mean(y_pred - y) / np.mean(y)
        conforme = r2 > 0.75 and abs(cv) < 0.2 and abs(bias) < 0.01

        return {
            "r2": r2, "cv": cv, "bias": bias, "model": model, "conforme": conforme, "y_pred": y_pred
        }

    def trouver_meilleur_modele(self, X, y):
        """Teste toutes les combinaisons et trouve le meilleur modèle"""
        for n in range(1, nb_var + 1):
            for combo in combinations(var_explicatives, n):
                result = self.evaluer_combinaison(X, y, list(combo))
                if result["conforme"] and result["r2"] > self.best_r2:
                    self.best_model = result["model"]
                    self.best_r2 = result["r2"]
                    self.best_cv = result["cv"]
                    self.best_bias = result["bias"]
                    self.best_features = list(combo)
                    self.best_formula = f"y = {self.best_model.intercept_:.4f} + " + " + ".join(
                        [f"{coef:.4f} × {feat}" for coef, feat in zip(self.best_model.coef_, self.best_features)])
                    self.best_model_type = "Régression Linéaire"

# ✅ Bouton de lancement de l'analyse
if st.sidebar.button("🚀 Lancer le calcul"):
    st.info("⚙️ **Analyse en cours...**")
    
    # ✅ Création du modèle et exécution
    modele_ipmvp = ModelIPMVP()
    modele_ipmvp.trouver_meilleur_modele(df[var_explicatives], df[conso_col])

    # ✅ Affichage des résultats (sans changer le design)
    st.success("✅ **Résultats de l'analyse**")

    st.markdown(f"🏆 **Modèle sélectionné** : `{modele_ipmvp.best_model_type}`")
    st.markdown(f"📋 **Conformité IPMVP** : ✅ Conforme" if modele_ipmvp.best_r2 > 0.75 else "❌ Non conforme")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 R²", f"{modele_ipmvp.best_r2:.4f}")
    col2.metric("📉 CV(RMSE)", f"{modele_ipmvp.best_cv:.4f}")
    col3.metric("⚖ Biais (NMBE)", f"{modele_ipmvp.best_bias:.6f}")

    st.markdown(f"✏️ **Équation d’ajustement :** `{modele_ipmvp.best_formula}`")
    st.markdown(f"🔍 **Variables utilisées** : {', '.join(modele_ipmvp.best_features)}")

    # ✅ Affichage du graphique (design inchangé)
    st.markdown("### 📊 **Comparaison Consommation Mesurée vs Ajustée**")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(df[date_col], df[conso_col], color="#6DBABC", alpha=0.6, label="Consommation réelle")
    ax.plot(df[date_col], modele_ipmvp.best_model.predict(df[var_explicatives]), color="#D32F2F", marker="o", linestyle="-", label="Consommation ajustée", linewidth=2)

    ax.set_xlabel("Mois")
    ax.set_ylabel("Consommation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ✅ Pied de page
st.sidebar.markdown("---")
st.sidebar.info("💡 Développé avec ❤️ par **Efficacité Energétique, Carbone & RSE Team** | © 2025")

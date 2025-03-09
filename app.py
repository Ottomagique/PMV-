import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

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

    .metrics-card {
        background-color: #fff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .equation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #6DBABC;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 10px 10px 0;
        font-family: monospace;
    }
    
    .conformity-good {
        color: #27ae60;
        font-weight: bold;
    }
    
    .conformity-medium {
        color: #f39c12;
        font-weight: bold;
    }
    
    .conformity-bad {
        color: #e74c3c;
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

# 📌 **Description de l'application**
st.title("📊 Analyse IPMVP")
st.markdown("""
Bienvenue sur **l'Analyse IPMVP Simplifiée** 🔍 !  
Cette application vous permet d'analyser **vos données de consommation énergétique** et de trouver le meilleur modèle d'ajustement basé sur plusieurs variables explicatives.

### **🛠️ Instructions :**
1. **Importer un fichier Excel 📂** contenant les données de consommation.
2. **Sélectionner la colonne de date, la consommation et les variables explicatives 📊**.
3. **Choisir le nombre de variables à tester 🔢** (de 1 à 4).
4. **Lancer le calcul 🚀** et obtenir le **meilleur modèle** avec une analyse graphique.
""")

# 📂 **Import du fichier et lancement du calcul**
col1, col2 = st.columns([3, 1])  # Mise en page : Import à gauche, bouton à droite

with col1:
    uploaded_file = st.file_uploader("📂 Importer un fichier Excel", type=["xlsx", "xls"])

with col2:
    lancer_calcul = st.button("🚀 Lancer le calcul", use_container_width=True)

# 📂 **Sélection des données (toujours visible même sans fichier importé)**
st.sidebar.header("🔍 Sélection des données")

df = None  # Initialisation pour éviter des erreurs

if uploaded_file:
    df = pd.read_excel(uploaded_file)  # Chargement du fichier

# **Définition des colonnes pour la sélection AVANT import**
date_col = st.sidebar.selectbox("📅 Nom de la donnée date", df.columns if df is not None else [""])
conso_col = st.sidebar.selectbox("⚡ Nom de la donnée consommation", df.columns if df is not None else [""])

# **Variables explicatives (seulement après importation du fichier)**
var_options = [col for col in df.columns if col not in [date_col, conso_col]] if df is not None else []
selected_vars = st.sidebar.multiselect("📊 Variables explicatives", var_options)

# Nombre de variables à tester
max_features = st.sidebar.slider("🔢 Nombre de variables à tester", 1, 4, 2)

# Fonction pour évaluer la conformité IPMVP
def evaluer_conformite(r2, cv_rmse):
    if r2 >= 0.75 and cv_rmse <= 0.15:
        return "Excellente", "good"
    elif r2 >= 0.5 and cv_rmse <= 0.25:
        return "Acceptable", "medium"
    else:
        return "Insuffisante", "bad"

# 📌 **Lancement du calcul seulement si le bouton est cliqué**
if df is not None and lancer_calcul:
    st.subheader("⚙️ Analyse en cours...")

    X = df[selected_vars] if selected_vars else pd.DataFrame(index=df.index)
    y = df[conso_col]

    # Nettoyage des données avant entraînement
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
    best_metrics = {}
    all_models = []

    # 🔹 Test des combinaisons de variables (de 1 à max_features)
    for n in range(1, max_features + 1):
        for combo in combinations(selected_vars, n):
            X_subset = X[list(combo)]
            model = LinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            
            # Calcul des métriques
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            cv_rmse = rmse / np.mean(y) if np.mean(y) != 0 else float('inf')
            bias = np.mean(y_pred - y) / np.mean(y) * 100
            
            # Récupération des coefficients
            coefs = {feature: coef for feature, coef in zip(combo, model.coef_)}
            intercept = model.intercept_
            
            # Statut de conformité IPMVP
            conformite, classe = evaluer_conformite(r2, cv_rmse)
            
            # Stockage du modèle
            model_info = {
                'features': list(combo),
                'r2': r2,
                'rmse': rmse,
                'cv_rmse': cv_rmse,
                'mae': mae,
                'bias': bias,
                'coefficients': coefs,
                'intercept': intercept,
                'conformite': conformite,
                'classe': classe,
                'model': model
            }
            all_models.append(model_info)

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_features = list(combo)
                best_metrics = model_info

    # 🔹 Tri des modèles par R² décroissant
    all_models.sort(key=lambda x: x['r2'], reverse=True)

    # 🔹 Résultats du modèle sélectionné
    if best_model:
        st.success("✅ Modèle trouvé avec succès !")
        
        # Créer l'équation du modèle sous forme de texte
        equation = f"Consommation = {best_metrics['intercept']:.4f}"
        for feature in best_features:
            coef = best_metrics['coefficients'][feature]
            sign = "+" if coef >= 0 else ""
            equation += f" {sign} {coef:.4f} × {feature}"
        
        # Afficher les métriques dans un tableau
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Résultats du modèle")
            st.markdown(f"""
            <div class="metrics-card">
                <h4>Modèle sélectionné: Régression linéaire multiple</h4>
                <p>Variables utilisées: {', '.join(best_features)}</p>
                <p>Conformité IPMVP: <span class="conformity-{best_metrics['classe']}">{best_metrics['conformite']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="equation-box">
                <h4>Équation d'ajustement:</h4>
                <p>{equation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Tableau des métriques
            metrics_df = pd.DataFrame({
                'Métrique': ['R²', 'RMSE', 'CV(RMSE)', 'MAE', 'Biais (%)'],
                'Valeur': [
                    f"{best_metrics['r2']:.4f}",
                    f"{best_metrics['rmse']:.4f}",
                    f"{best_metrics['cv_rmse']:.4f}",
                    f"{best_metrics['mae']:.4f}",
                    f"{best_metrics['bias']:.2f}"
                ]
            })
            st.table(metrics_df)

        # 🔹 Graphique de consommation
        st.subheader("📈 Visualisation des résultats")
        
        # Prédictions du modèle
        X_best = df[best_features]
        y_pred = best_model.predict(X_best)
        
        # Graphique de comparaison
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(y)), y, color="#6DBABC", alpha=0.7, label="Consommation mesurée")
        ax.plot(range(len(y)), y_pred, color="#E74C3C", marker='o', linewidth=2, markersize=4, label="Consommation ajustée")
        ax.set_title("Comparaison Consommation Mesurée vs Ajustée")
        ax.set_xlabel("Observations")
        ax.set_ylabel("Consommation")
        ax.legend()
        st.pyplot(fig)
        
        # Graphique de dispersion (measured vs predicted)
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(y, y_pred, color="#6DBABC", alpha=0.7)
        
        # Ligne de référence y=x
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label="Référence y=x")
        
        ax2.set_title("Consommation Mesurée vs Prédite")
        ax2.set_xlabel("Consommation Mesurée")
        ax2.set_ylabel("Consommation Prédite")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)
        
        # Affichage des résidus
        residus = y - y_pred
        
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.scatter(range(len(residus)), residus, color="#96B91D", alpha=0.7)
        ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax3.set_title("Analyse des Résidus")
        ax3.set_xlabel("Observations")
        ax3.set_ylabel("Résidus")
        ax3.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig3)
        
        # 🔹 Tableau des résultats pour tous les modèles testés
        st.subheader("📋 Classement des modèles testés")
        models_summary = []
        
        for i, model in enumerate(all_models[:10]):  # Afficher les 10 meilleurs modèles
            models_summary.append({
                "Rang": i+1,
                "Variables": ", ".join(model['features']),
                "R²": f"{model['r2']:.4f}",
                "CV(RMSE)": f"{model['cv_rmse']:.4f}",
                "Biais (%)": f"{model['bias']:.2f}",
                "Conformité": model['conformite']
            })
        
        st.table(pd.DataFrame(models_summary))
        
    else:
        st.error("⚠️ Aucun modèle valide n'a été trouvé.")

st.sidebar.markdown("---")
st.sidebar.info("Développé avec ❤️ et Streamlit 🚀")

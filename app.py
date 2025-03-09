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
        background-color: #E7DDD9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #00485F;
    }
    
    .equation-box {
        background-color: #E7DDD9;
        border-left: 4px solid #6DBABC;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 10px 10px 0;
        font-family: monospace;
        border: 1px solid #00485F;
    }
    
    .conformity-good {
        color: #96B91D;
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
    
    .footer-credit {
        text-align: center;
        margin-top: 30px;
        padding: 15px;
        background-color: #00485F;
        color: white;
        border-radius: 10px;
        font-size: 14px;
    }
    
    .instruction-card {
        background-color: #E7DDD9;
        border-left: 4px solid #96B91D;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 10px 10px 0;
    }
    
    .table-header {
        background-color: #00485F;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

# 📌 **Description de l'application**
st.title("📊 Analyse IPMVP")
st.markdown("""
Bienvenue sur **l'Analyse IPMVP Professionnelle** 🔍 !  
Cette application vous permet d'analyser **vos données de consommation énergétique** et de trouver le meilleur modèle d'ajustement basé sur plusieurs variables explicatives selon la méthodologie IPMVP.
""")

st.markdown("""
<div class="instruction-card">
<h3>🛠️ Guide d'utilisation</h3>
<ol>
    <li><strong>Préparation du fichier Excel</strong> : Assurez-vous que votre fichier contient une colonne de dates, une colonne de consommation et des variables explicatives potentielles (degrés-jours, occupation, production, etc.)</li>
    <li><strong>Import du fichier</strong> : Utilisez le bouton d'import pour charger votre fichier Excel (.xlsx ou .xls)</li>
    <li><strong>Sélection des données</strong> : Dans le panneau latéral, sélectionnez :
        <ul>
            <li>La colonne de date</li>
            <li>La colonne de consommation énergétique</li>
            <li>Les variables explicatives potentielles (température, production, etc.)</li>
        </ul>
    </li>
    <li><strong>Configuration de l'analyse</strong> : Choisissez le nombre maximum de variables à combiner (1 à 4)</li>
    <li><strong>Lancement</strong> : Cliquez sur "Lancer le calcul" pour obtenir le meilleur modèle d'ajustement</li>
    <li><strong>Analyse des résultats</strong> : Examinez les métriques (R², CV, biais), l'équation d'ajustement et les visualisations générées</li>
</ol>
</div>
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
        
        # Configuration du style des graphiques pour correspondre au thème
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.facecolor'] = '#F5F5F5'
        plt.rcParams['figure.facecolor'] = '#E7DDD9'
        plt.rcParams['axes.edgecolor'] = '#00485F'
        plt.rcParams['axes.labelcolor'] = '#00485F'
        plt.rcParams['axes.titlecolor'] = '#00485F'
        plt.rcParams['xtick.color'] = '#0C1D2D'
        plt.rcParams['ytick.color'] = '#0C1D2D'
        plt.rcParams['grid.color'] = '#00485F'
        plt.rcParams['grid.alpha'] = 0.1
        
        # Graphique de comparaison
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(y)), y, color="#6DBABC", alpha=0.8, label="Consommation mesurée")
        ax.plot(range(len(y)), y_pred, color="#96B91D", marker='o', linewidth=2, markersize=4, label="Consommation ajustée", zorder=10)
        ax.set_title("Comparaison Consommation Mesurée vs Ajustée", fontweight='bold', fontsize=14)
        ax.set_xlabel("Observations", fontweight='bold')
        ax.set_ylabel("Consommation", fontweight='bold')
        ax.legend(frameon=True, facecolor="#E7DDD9", edgecolor="#00485F")
        ax.grid(True, linestyle='--', alpha=0.2)
        # Annotation du R²
        ax.annotate(f"R² = {best_metrics['r2']:.4f}", xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, fontweight='bold', color='#00485F',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E7DDD9", edgecolor="#00485F", alpha=0.8))
        st.pyplot(fig)
        
        # Création d'une mise en page en colonnes pour les deux derniers graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique de dispersion (measured vs predicted)
            fig2, ax2 = plt.subplots(figsize=(8, 7))
            scatter = ax2.scatter(y, y_pred, color="#6DBABC", alpha=0.8, s=50, edgecolor='#00485F')
            
            # Ligne de référence y=x
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            ax2.plot([min_val, max_val], [min_val, max_val], '--', color='#00485F', linewidth=1.5, label="Référence y=x")
            
            ax2.set_title("Consommation Mesurée vs Prédite", fontweight='bold', fontsize=14)
            ax2.set_xlabel("Consommation Mesurée", fontweight='bold')
            ax2.set_ylabel("Consommation Prédite", fontweight='bold')
            ax2.legend(frameon=True, facecolor="#E7DDD9", edgecolor="#00485F")
            ax2.grid(True, linestyle='--', alpha=0.2)
            # Annotation du CV(RMSE)
            ax2.annotate(f"CV(RMSE) = {best_metrics['cv_rmse']:.4f}", xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color='#00485F',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E7DDD9", edgecolor="#00485F", alpha=0.8))
            st.pyplot(fig2)
        
        with col2:
            # Affichage des résidus
            residus = y - y_pred
            
            fig3, ax3 = plt.subplots(figsize=(8, 7))
            ax3.scatter(range(len(residus)), residus, color="#96B91D", alpha=0.8, s=50, edgecolor='#00485F')
            ax3.axhline(y=0, color='#00485F', linestyle='-', alpha=0.5, linewidth=1.5)
            ax3.set_title("Analyse des Résidus", fontweight='bold', fontsize=14)
            ax3.set_xlabel("Observations", fontweight='bold')
            ax3.set_ylabel("Résidus", fontweight='bold')
            ax3.grid(True, linestyle='--', alpha=0.2)
            
            # Annotation du biais
            ax3.annotate(f"Biais = {best_metrics['bias']:.2f}%", xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color='#00485F',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E7DDD9", edgecolor="#00485F", alpha=0.8))
            st.pyplot(fig3)
            
        # Ajout d'un expander pour expliquer l'interprétation des graphiques
        with st.expander("📚 Comment interpréter ces graphiques ?"):
            st.markdown("""
            ### Interprétation des visualisations
            
            **1. Graphique Consommation Mesurée vs Ajustée**
            - Compare les valeurs réelles (barres bleues) avec les prédictions du modèle (ligne verte)
            - Un modèle idéal montre une ligne qui suit étroitement les sommets des barres
            
            **2. Graphique de dispersion**
            - Les points doivent s'aligner le long de la ligne diagonale
            - Des points éloignés de la ligne indiquent des prédictions moins précises
            - Plus les points sont proches de la diagonale, meilleur est le modèle
            
            **3. Analyse des Résidus**
            - Montre l'erreur pour chaque observation (valeur réelle - valeur prédite)
            - Idéalement, les résidus devraient:
              - Être répartis de façon aléatoire autour de zéro
              - Ne pas présenter de tendance ou de motif visible
              - Avoir une distribution équilibrée au-dessus et en-dessous de zéro
            """)

        
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

# Ajout d'informations sur la méthodologie IPMVP
st.sidebar.subheader("📘 Méthodologie IPMVP")
st.sidebar.markdown("""
La méthodologie IPMVP (International Performance Measurement and Verification Protocol) évalue la qualité d'un modèle de régression selon ces critères :

- **R² ≥ 0.75** : Excellente corrélation
- **CV(RMSE) ≤ 15%** : Excellente précision
- **Biais < 5%** : Ajustement équilibré
""")

# Pied de page amélioré
st.markdown("---")
st.markdown("""
<div class="footer-credit">
    <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
    <p>Outil professionnel d'analyse et de modélisation énergétique conforme IPMVP</p>
</div>
""", unsafe_allow_html=True)

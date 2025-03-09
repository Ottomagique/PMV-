import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
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
    
    /* Style des info-bulles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #00485F;
        font-size: 14px;
        margin-left: 4px;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #00485F;
        color: white;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #00485F transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .model-badge {
        display: inline-block;
        background-color: #6DBABC;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 8px;
    }

    </style>
    """, unsafe_allow_html=True)

# 📌 **Description de l'application**
st.title("📊 Calcul IPMVP")
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
            <li>Les variables explicatives potentielles (Ensoleillement, DJU, etc.)</li>
        </ul>
    </li>
    <li><strong>Choix de la période d'analyse</strong> : Deux options sont disponibles :
        <ul>
            <li>Recherche automatique : l'application trouve la meilleure période de 12 mois dans vos données</li>
            <li>Sélection manuelle : choisissez vous-même la période d'analyse en sélectionnant les dates de début et de fin</li>
        </ul>
    </li>
    <li><strong>Type de modèle</strong> : Par défaut, l'application teste tous les types de modèles (linéaire, Ridge, Lasso, polynomiale) et sélectionne le meilleur, ou vous pouvez choisir un type spécifique</li>
    <li><strong>Configuration de l'analyse</strong> : Choisissez le nombre maximum de variables à combiner (1 à 4)</li>
    <li><strong>Lancement</strong> : Cliquez sur "Lancer le calcul" pour obtenir le meilleur modèle d'ajustement</li>
    <li><strong>Analyse des résultats</strong> : Examinez les métriques (R², CV, biais), l'équation d'ajustement et les visualisations générées</li>
</ol>
</div>
""", unsafe_allow_html=True)

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

# **Option pour rechercher automatiquement la meilleure période de 12 mois ou choisir une période**
period_choice = st.sidebar.radio(
    "📅 Sélection de la période d'analyse",
    ["Rechercher automatiquement la meilleure période de 12 mois", "Sélectionner manuellement une période spécifique"]
)

# Variables pour stocker les informations de la meilleure période
best_period_start = None
best_period_end = None
best_period_name = None
best_period_r2 = -1

# Option de sélection manuelle de période
if period_choice == "Sélectionner manuellement une période spécifique" and df is not None and date_col in df.columns:
    # Convertir la colonne de date si elle ne l'est pas déjà
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            st.sidebar.warning("⚠️ La colonne de date n'a pas pu être convertie. Assurez-vous qu'elle contient des dates valides.")
    
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        # Obtenir les dates minimales et maximales
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        # Sélection de la date de début et de fin
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Date de début", 
                                       value=min_date,
                                       min_value=min_date, 
                                       max_value=max_date)
        with col2:
            # Calcul de la date par défaut (12 mois après la date de début si possible)
            default_end = min(max_date, (pd.to_datetime(start_date) + pd.DateOffset(months=11)).date())
            end_date = st.date_input("Date de fin", 
                                     value=default_end,
                                     min_value=start_date, 
                                     max_value=max_date)
        
        # Calculer la différence en mois
        months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        
        # Afficher des informations sur la période sélectionnée
        st.sidebar.info(f"Période sélectionnée: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')} ({months_diff} mois)")
        
        # Recommandation pour 12 mois
        if months_diff != 12:
            if months_diff < 12:
                st.sidebar.warning(f"⚠️ La période sélectionnée est de {months_diff} mois. La méthodologie IPMVP recommande 12 mois.")
            else:
                st.sidebar.warning(f"⚠️ La période sélectionnée est de {months_diff} mois. Pour une analyse standard IPMVP, 12 mois sont recommandés.")

# **Variables explicatives (seulement après importation du fichier)**
var_options = [col for col in df.columns if col not in [date_col, conso_col]] if df is not None else []
selected_vars = st.sidebar.multiselect("📊 Variables explicatives", var_options)

# Type de modèle à utiliser
model_type = st.sidebar.selectbox(
    "🧮 Type de modèle de régression",
    ["Automatique (meilleur modèle)", "Linéaire", "Ridge", "Lasso", "Polynomiale"],
    index=0,
    help="Sélectionnez 'Automatique' pour tester tous les types de modèles et choisir le meilleur, ou sélectionnez un type spécifique"
)

# Paramètres spécifiques aux modèles
if model_type == "Ridge":
    alpha_ridge = st.sidebar.slider(
        "Alpha (régularisation Ridge)", 
        0.01, 10.0, 1.0, 0.01,
        help="Le paramètre alpha contrôle l'intensité de la régularisation. Une valeur plus élevée réduit davantage les coefficients pour éviter le surapprentissage."
    )
elif model_type == "Lasso":
    alpha_lasso = st.sidebar.slider(
        "Alpha (régularisation Lasso)", 
        0.01, 1.0, 0.1, 0.01,
        help="Le paramètre alpha contrôle l'intensité de la régularisation. Lasso peut réduire certains coefficients à zéro, effectuant ainsi une sélection de variables."
    )
elif model_type == "Polynomiale":
    poly_degree = st.sidebar.slider(
        "Degré du polynôme", 
        2, 3, 2,
        help="Le degré du polynôme détermine la complexité des relations non linéaires. Un degré 2 inclut les termes quadratiques (x²), un degré 3 inclut également les termes cubiques (x³)."
    )

# Nombre de variables à tester
max_features = st.sidebar.slider("🔢 Nombre de variables à tester", 1, 4, 2)

# Fonction pour créer une info-bulle
def tooltip(text, explanation):
    return f'<span>{text} <span class="tooltip">ℹ️<span class="tooltiptext">{explanation}</span></span></span>'

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
    
    # Convertir la colonne de date si elle ne l'est pas déjà
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            # Trier le dataframe par date
            df = df.sort_values(by=date_col)
        except:
            st.error("❌ La colonne de date n'a pas pu être convertie. Assurez-vous qu'elle contient des dates valides.")
            st.stop()
    
    # Option 1: Recherche automatique de la meilleure période
    if period_choice == "Rechercher automatiquement la meilleure période de 12 mois":
        # Vérifier s'il y a suffisamment de données (au moins 12 mois)
        date_ranges = []
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        current_date = min_date
        
        while current_date + pd.DateOffset(months=11) <= max_date:
            end_date = current_date + pd.DateOffset(months=11)
            period_name = f"{current_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"
            date_ranges.append((period_name, current_date, end_date))
            current_date = current_date + pd.DateOffset(months=1)
        
        if not date_ranges:
            st.error("❌ Pas assez de données pour une analyse sur 12 mois. Assurez-vous d'avoir au moins 12 mois de données.")
            st.stop()
            
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        best_period_data = None
        best_period_model = None
        best_period_features = None
        best_period_metrics = None
        best_period_r2 = -1
        
        for idx, (period_name, period_start, period_end) in enumerate(date_ranges):
            progress_text.text(f"Analyse de la période {period_name} ({idx+1}/{len(date_ranges)})")
            
            # Filtrer les données pour cette période
            period_df = df[(df[date_col] >= period_start) & (df[date_col] <= period_end)]
            
            # Vérifier que les données sont suffisantes
            if len(period_df) < 10:  # Éviter les périodes avec trop peu de données
                continue
                
            X = period_df[selected_vars] if selected_vars else pd.DataFrame(index=period_df.index)
            y = period_df[conso_col]
            
            # Nettoyage des données avant entraînement
            if X.isnull().values.any() or np.isinf(X.values).any():
                continue
                
            if y.isnull().values.any() or np.isinf(y.values).any():
                continue
                
            X = X.apply(pd.to_numeric, errors='coerce').dropna()
            y = pd.to_numeric(y, errors='coerce').dropna()
            
            # Test des combinaisons de variables
            for n in range(1, max_features + 1):
                for combo in combinations(selected_vars, n):
                    X_subset = X[list(combo)]
                    
                    try:
                        # Si mode automatique, tester tous les types de modèles
                        if period_choice == "Rechercher automatiquement la meilleure période de 12 mois" and model_type == "Automatique (meilleur modèle)":
                            model_types_to_test = [
                                ("Linéaire", LinearRegression(), "Régression linéaire"),
                                ("Ridge", Ridge(alpha=1.0), f"Régression Ridge (α=1.0)"),
                                ("Lasso", Lasso(alpha=0.1), f"Régression Lasso (α=0.1)"),
                                ("Polynomiale", Pipeline([
                                    ('poly', PolynomialFeatures(degree=2)),
                                    ('linear', LinearRegression())
                                ]), f"Régression polynomiale (degré 2)")
                            ]
                            
                            for m_type, m_obj, m_name in model_types_to_test:
                                m_obj.fit(X_subset, y)
                                y_pred = m_obj.predict(X_subset)
                                r2 = r2_score(y, y_pred)
                                
                                if r2 > best_period_r2:
                                    best_period_r2 = r2
                                    best_period_start = period_start
                                    best_period_end = period_end
                                    best_period_name = period_name
                                    best_period_data = period_df
                                    best_period_model = m_obj
                                    best_period_features = list(combo)
                                    
                                    # Calcul des métriques
                                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                                    mae = mean_absolute_error(y, y_pred)
                                    cv_rmse = rmse / np.mean(y) if np.mean(y) != 0 else float('inf')
                                    bias = np.mean(y_pred - y) / np.mean(y) * 100
                                    
                                    # Récupération des coefficients selon le type de modèle
                                    if m_type == "Linéaire":
                                        coefs = {feature: coef for feature, coef in zip(combo, m_obj.coef_)}
                                        intercept = m_obj.intercept_
                                    elif m_type in ["Ridge", "Lasso"]:
                                        coefs = {feature: coef for feature, coef in zip(combo, m_obj.coef_)}
                                        intercept = m_obj.intercept_
                                    elif m_type == "Polynomiale":
                                        # Pour le modèle polynomial, nous gardons une représentation simplifiée
                                        linear_model = m_obj.named_steps['linear']
                                        poly = m_obj.named_steps['poly']
                                        feature_names = poly.get_feature_names_out(input_features=combo)
                                        coefs = {name: coef for name, coef in zip(feature_names, linear_model.coef_)}
                                        intercept = linear_model.intercept_
                                    
                                    # Statut de conformité IPMVP
                                    conformite, classe = evaluer_conformite(r2, cv_rmse)
                                    
                                    # Stockage des métriques
                                    best_period_metrics = {
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
                                        'model_type': m_type,
                                        'model_name': m_name
                                    }
                        else:
                            # Création du modèle selon le type sélectionné
                            if model_type == "Linéaire":
                                model = LinearRegression()
                                model_name = "Régression linéaire"
                            elif model_type == "Ridge":
                                model = Ridge(alpha=alpha_ridge)
                                model_name = f"Régression Ridge (α={alpha_ridge})"
                            elif model_type == "Lasso":
                                model = Lasso(alpha=alpha_lasso)
                                model_name = f"Régression Lasso (α={alpha_lasso})"
                            elif model_type == "Polynomiale":
                                model = Pipeline([
                                    ('poly', PolynomialFeatures(degree=poly_degree)),
                                    ('linear', LinearRegression())
                                ])
                                model_name = f"Régression polynomiale (degré {poly_degree})"
                            
                            model.fit(X_subset, y)
                            
                            # Prédictions selon le type de modèle
                            y_pred = model.predict(X_subset)
                            r2 = r2_score(y, y_pred)
                            
                            if r2 > best_period_r2:
                                best_period_r2 = r2
                                best_period_start = period_start
                                best_period_end = period_end
                                best_period_name = period_name
                                best_period_data = period_df
                                best_period_model = model
                                best_period_features = list(combo)
                                
                                # Calcul des métriques
                                rmse = np.sqrt(mean_squared_error(y, y_pred))
                                mae = mean_absolute_error(y, y_pred)
                                cv_rmse = rmse / np.mean(y) if np.mean(y) != 0 else float('inf')
                                bias = np.mean(y_pred - y) / np.mean(y) * 100
                                
                                # Récupération des coefficients selon le type de modèle
                                if model_type == "Linéaire":
                                    coefs = {feature: coef for feature, coef in zip(combo, model.coef_)}
                                    intercept = model.intercept_
                                elif model_type in ["Ridge", "Lasso"]:
                                    coefs = {feature: coef for feature, coef in zip(combo, model.coef_)}
                                    intercept = model.intercept_
                                elif model_type == "Polynomiale":
                                    # Pour le modèle polynomial, nous gardons une représentation simplifiée
                                    linear_model = model.named_steps['linear']
                                    poly = model.named_steps['poly']
                                    feature_names = poly.get_feature_names_out(input_features=combo)
                                    coefs = {name: coef for name, coef in zip(feature_names, linear_model.coef_)}
                                    intercept = linear_model.intercept_
                                
                                # Statut de conformité IPMVP
                                conformite, classe = evaluer_conformite(r2, cv_rmse)
                                
                                # Stockage des métriques
                                best_period_metrics = {
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
                                    'model_type': model_type,
                                    'model_name': model_name
                                }
                    except:
                        continue
            
            # Mise à jour de la barre de progression
            progress_bar.progress((idx + 1) / len(date_ranges))
        
        progress_bar.empty()
        progress_text.empty()
        
        if best_period_data is not None:
            st.success(f"✅ Meilleure période trouvée : {best_period_name}")
            st.info(f"Période : {best_period_start.strftime('%d/%m/%Y')} - {best_period_end.strftime('%d/%m/%Y')}")
            
            # Utiliser les meilleurs résultats trouvés
            df_filtered = best_period_data
            best_model = best_period_model
            best_features = best_period_features
            best_metrics = best_period_metrics
            
            # Afficher les détails sur les données
            st.markdown(f"**📊 Nombre de points de données :** {len(df_filtered)}")
        else:
            st.error("❌ Aucun modèle valide n'a été trouvé sur les périodes analysées.")
            st.stop()
    
    # Option 2: Période spécifique sélectionnée
    else:
        # Filtrer les données selon la période sélectionnée manuellement
        df_filtered = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
        
        # Afficher le nombre de points de données
        st.info(f"Analyse sur la période du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")
        st.markdown(f"**📊 Nombre de points de données :** {len(df_filtered)}")
        
        if len(df_filtered) < 10:
            st.warning("⚠️ Le nombre de points de données est faible pour une analyse statistique fiable.")
        
        X = df_filtered[selected_vars] if selected_vars else pd.DataFrame(index=df_filtered.index)
        y = df_filtered[conso_col]

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
                # Si mode automatique, tester tous les types de modèles
                if model_type == "Automatique (meilleur modèle)":
                    # Créer une liste pour stocker les résultats des différents modèles
                    temp_models = []
                    
                    # Tester chaque type de modèle
                    model_types_to_test = [
                        ("Linéaire", LinearRegression(), "Régression linéaire"),
                        ("Ridge", Ridge(alpha=1.0), f"Régression Ridge (α=1.0)"),
                        ("Lasso", Lasso(alpha=0.1), f"Régression Lasso (α=0.1)"),
                        ("Polynomiale", Pipeline([
                            ('poly', PolynomialFeatures(degree=2)),
                            ('linear', LinearRegression())
                        ]), f"Régression polynomiale (degré 2)")
                    ]
                    
                    for m_type, m_obj, m_name in model_types_to_test:
                        m_obj.fit(X_subset, y)
                        y_pred = m_obj.predict(X_subset)
                        
                        # Calcul des métriques
                        r2 = r2_score(y, y_pred)
                        rmse = np.sqrt(mean_squared_error(y, y_pred))
                        mae = mean_absolute_error(y, y_pred)
                        cv_rmse = rmse / np.mean(y) if np.mean(y) != 0 else float('inf')
                        bias = np.mean(y_pred - y) / np.mean(y) * 100
                        
                        # Récupération des coefficients selon le type de modèle
                        if m_type == "Linéaire":
                            coefs = {feature: coef for feature, coef in zip(combo, m_obj.coef_)}
                            intercept = m_obj.intercept_
                        elif m_type in ["Ridge", "Lasso"]:
                            coefs = {feature: coef for feature, coef in zip(combo, m_obj.coef_)}
                            intercept = m_obj.intercept_
                        elif m_type == "Polynomiale":
                            # Pour le modèle polynomial, nous prenons une représentation simplifiée
                            linear_model = m_obj.named_steps['linear']
                            poly = m_obj.named_steps['poly']
                            feature_names = poly.get_feature_names_out(input_features=combo)
                            coefs = {name: coef for name, coef in zip(feature_names, linear_model.coef_)}
                            intercept = linear_model.intercept_
                        
                        # Statut de conformité IPMVP
                        conformite, classe = evaluer_conformite(r2, cv_rmse)
                        
                        # Stockage des résultats du modèle
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
                            'model': m_obj,
                            'model_type': m_type,
                            'model_name': m_name
                        }
                        
                        temp_models.append(model_info)
                        all_models.append(model_info)
                    
                    # Trouver le meilleur modèle parmi ceux testés
                    best_temp_model = max(temp_models, key=lambda x: x['r2'])
                    
                    if best_temp_model['r2'] > best_r2:
                        best_r2 = best_temp_model['r2']
                        best_model = best_temp_model['model']
                        best_features = best_temp_model['features']
                        best_metrics = best_temp_model
                
                else:
                    # Création du modèle selon le type sélectionné
                    if model_type == "Linéaire":
                        model = LinearRegression()
                        model_name = "Régression linéaire"
                    elif model_type == "Ridge":
                        model = Ridge(alpha=alpha_ridge)
                        model_name = f"Régression Ridge (α={alpha_ridge})"
                    elif model_type == "Lasso":
                        model = Lasso(alpha=alpha_lasso)
                        model_name = f"Régression Lasso (α={alpha_lasso})"
                    elif model_type == "Polynomiale":
                        model = Pipeline([
                            ('poly', PolynomialFeatures(degree=poly_degree)),
                            ('linear', LinearRegression())
                        ])
                        model_name = f"Régression polynomiale (degré {poly_degree})"
                    
                    model.fit(X_subset, y)
                    
                    # Prédictions selon le type de modèle
                    y_pred = model.predict(X_subset)
                    
                    # Calcul des métriques
                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    mae = mean_absolute_error(y, y_pred)
                    cv_rmse = rmse / np.mean(y) if np.mean(y) != 0 else float('inf')
                    bias = np.mean(y_pred - y) / np.mean(y) * 100
                    
                    # Récupération des coefficients selon le type de modèle
                    if model_type == "Linéaire":
                        coefs = {feature: coef for feature, coef in zip(combo, model.coef_)}
                        intercept = model.intercept_
                    elif model_type in ["Ridge", "Lasso"]:
                        coefs = {feature: coef for feature, coef in zip(combo, model.coef_)}
                        intercept = model.intercept_
                    elif model_type == "Polynomiale":
                        # Pour le modèle polynomial, nous gardons une représentation simplifiée
                        linear_model = model.named_steps['linear']
                        poly = model.named_steps['poly']
                        feature_names = poly.get_feature_names_out(input_features=combo)
                        coefs = {name: coef for name, coef in zip(feature_names, linear_model.coef_)}
                        intercept = linear_model.intercept_
                    
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
                        'model': model,
                        'model_type': model_type,
                        'model_name': model_name
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
                <h4>Modèle sélectionné: <span class="model-badge">{best_metrics['model_name']}</span></h4>
                <p>Variables utilisées: {', '.join(best_features)}</p>
                <p>Conformité IPMVP: <span class="conformity-{best_metrics['classe']}">{best_metrics['conformite']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Créer l'équation adaptée selon le type de modèle
            if best_metrics['model_type'] in ["Linéaire", "Ridge", "Lasso"]:
                equation = f"Consommation = {best_metrics['intercept']:.4f}"
                for feature in best_features:
                    coef = best_metrics['coefficients'][feature]
                    sign = "+" if coef >= 0 else ""
                    equation += f" {sign} {coef:.4f} × {feature}"
            elif best_metrics['model_type'] == "Polynomiale":
                equation = f"Consommation = {best_metrics['intercept']:.4f}"
                for feature_name, coef in best_metrics['coefficients'].items():
                    sign = "+" if coef >= 0 else ""
                    equation += f" {sign} {coef:.4f} × {feature_name}"
            
            st.markdown(f"""
            <div class="equation-box">
                <h4>Équation d'ajustement:</h4>
                <p>{equation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Tableau des métriques avec info-bulles
            st.markdown(f"""
            <table style="width:100%">
                <tr>
                    <th>Métrique</th>
                    <th>Valeur</th>
                </tr>
                <tr>
                    <td>{tooltip("R²", "Coefficient de détermination : mesure la proportion de variance de la variable dépendante qui est prédite à partir des variables indépendantes. Plus cette valeur est proche de 1, meilleur est l'ajustement du modèle aux données.")}</td>
                    <td>{best_metrics['r2']:.4f}</td>
                </tr>
                <tr>
                    <td>{tooltip("RMSE", "Root Mean Square Error (Erreur quadratique moyenne) : mesure l'écart-type des résidus (erreurs de prédiction). Exprimée dans la même unité que la variable dépendante.")}</td>
                    <td>{best_metrics['rmse']:.4f}</td>
                </tr>
                <tr>
                    <td>{tooltip("CV(RMSE)", "Coefficient de Variation du RMSE : exprime le RMSE en pourcentage de la moyenne observée, permettant de comparer la précision entre différents modèles indépendamment de l'échelle.")}</td>
                    <td>{best_metrics['cv_rmse']:.4f}</td>
                </tr>
                <tr>
                    <td>{tooltip("MAE", "Mean Absolute Error (Erreur absolue moyenne) : moyenne des valeurs absolues des erreurs. Moins sensible aux valeurs extrêmes que le RMSE.")}</td>
                    <td>{best_metrics['mae']:.4f}</td>
                </tr>
                <tr>
                    <td>{tooltip("Biais (%)", "Représente l'erreur systématique du modèle en pourcentage. Un biais positif indique une surestimation, un biais négatif une sous-estimation.")}</td>
                    <td>{best_metrics['bias']:.2f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        # 🔹 Graphique de consommation
        st.subheader("📈 Visualisation des résultats")
        
        # Prédictions du modèle
        X_best = df_filtered[best_features]
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
            
        # Ajout d'un expander pour expliquer les différents modèles de régression
        with st.expander("📚 Interprétation des différents modèles de régression"):
            st.markdown("""
            ### Types de modèles de régression
            
            **Régression linéaire multiple**
            - Modèle le plus courant pour l'IPMVP
            - Établit une relation linéaire : Y = a₀ + a₁X₁ + a₂X₂ + ... + aₙXₙ
            - Forces : Simple à interpréter, rapide à calculer
            - Limites : Ne peut capturer que des relations linéaires
            - Statut IPMVP : Explicitement mentionné et recommandé dans le protocole
            
            **Régression Ridge**
            - Ajoute une pénalité à la somme des carrés des coefficients
            - Formule : Y = a₀ + a₁X₁ + a₂X₂ + ... + aₙXₙ, avec minimisation de (résidus² + α × somme des coefficients²)
            - Forces : Gère mieux les variables corrélées, réduit le risque de surapprentissage
            - Limites : Tous les coefficients sont réduits mais aucun n'est éliminé
            - Statut IPMVP : Acceptable si les critères statistiques sont respectés et si le modèle reste documentable
            
            **Régression Lasso**
            - Ajoute une pénalité à la somme des valeurs absolues des coefficients
            - Formule : Y = a₀ + a₁X₁ + a₂X₂ + ... + aₙXₙ, avec minimisation de (résidus² + α × somme des |coefficients|)
            - Forces : Peut éliminer complètement des variables non pertinentes (coefficients = 0)
            - Limites : Peut être instable si les variables sont très corrélées
            - Statut IPMVP : Acceptable et peut même être préférable pour des modèles plus simples et robustes
            
            **Régression polynomiale**
            - Introduit des termes non linéaires (carrés, cubes, produits croisés)
            - Formule : Y = a₀ + a₁X₁ + a₂X₁² + a₃X₂ + a₄X₂² + a₅X₁X₂ + ...
            - Forces : Peut capturer des relations non linéaires
            - Limites : Risque élevé de surapprentissage, interprétation plus complexe
            - Statut IPMVP : Acceptable si les relations physiques sont plausibles et documentées
            """)
            
            st.info("""
            **Note sur la conformité IPMVP**
            
            Le protocole IPMVP (Option C) n'impose pas une méthode statistique spécifique, mais plutôt des critères de qualité:
            
            1. Le modèle doit avoir un R² ≥ 0.75 et un CV(RMSE) ≤ 15%
            2. Le modèle doit être documentable et transparent
            3. Les variables explicatives doivent avoir une relation plausible avec la consommation
            4. L'erreur-type des coefficients doit être évaluée
            
            Les méthodes avancées (Ridge, Lasso, polynomiale) sont acceptables et peuvent même produire des modèles plus robustes dans certaines situations, tant qu'elles respectent ces critères.
            """)

        
        # 🔹 Tableau des résultats pour tous les modèles testés
        st.subheader("📋 Classement des modèles testés")
        models_summary = []
        
        for i, model in enumerate(all_models[:10]):  # Afficher les 10 meilleurs modèles
            models_summary.append({
                "Rang": i+1,
                "Type": model['model_name'],
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

# Ajout d'informations sur la méthodologie IPMVP avec infobulles
st.sidebar.markdown(f"""
### 📘 Méthodologie IPMVP
La méthodologie IPMVP évalue la qualité d'un modèle de régression selon ces critères :

- {tooltip("R² ≥ 0.75", "Le coefficient de détermination R² mesure la proportion de la variance expliquée par le modèle. Une valeur de 0.75 signifie que 75% de la variabilité des données est expliquée par le modèle.")} : Excellente corrélation
- {tooltip("CV(RMSE) ≤ 15%", "Le coefficient de variation de l'erreur quadratique moyenne représente la dispersion relative des résidus. Il est calculé en divisant le RMSE par la moyenne des observations.")} : Excellente précision
- {tooltip("Biais < 5%", "Le biais représente l'erreur systématique du modèle. Un biais faible indique que le modèle ne surestime ni ne sous-estime systématiquement les valeurs.")} : Ajustement équilibré
""", unsafe_allow_html=True)

# Information sur les types de régression
st.sidebar.markdown(f"""
### 📊 Types de modèles
- {tooltip("Régression linéaire", "Modèle standard qui établit une relation linéaire entre les variables indépendantes et la consommation. C'est le modèle le plus couramment utilisé et explicitement mentionné dans l'IPMVP.")}
- {tooltip("Régression Ridge", "Technique de régularisation qui réduit le risque de surapprentissage en pénalisant les coefficients élevés. Conforme à l'IPMVP tant que les critères de qualité statistique (R², CV) sont respectés et que le modèle reste documentable.")}
- {tooltip("Régression Lasso", "Méthode qui peut réduire certains coefficients à zéro, effectuant ainsi une sélection de variables. Conforme à l'IPMVP car elle simplifie le modèle tout en maintenant sa précision statistique.")}
- {tooltip("Régression polynomiale", "Permet de modéliser des relations non linéaires. L'IPMVP accepte les modèles non linéaires si les relations physiques sont plausibles et si les critères statistiques sont respectés.")}
""", unsafe_allow_html=True)

# Information sur la conformité IPMVP des modèles avancés
st.sidebar.markdown(f"""
### ✅ Conformité IPMVP
{tooltip("Modèles avancés et IPMVP", "Le protocole IPMVP ne prescrit pas de méthode statistique spécifique, mais établit des critères de qualité statistique (R² et CV(RMSE)). Les méthodes avancées comme Ridge, Lasso ou polynomiale sont acceptables si elles respectent ces critères et si le modèle reste transparent et documentable.")}

Les modèles sont évalués selon les critères IPMVP :
- R² ≥ 0.75 : Excellente corrélation
- CV(RMSE) ≤ 15% : Excellente précision
""", unsafe_allow_html=True)

# Pied de page amélioré
st.markdown("---")
st.markdown("""
<div class="footer-credit">
    <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
    <p>Outil d'analyse et de modélisation énergétique conforme IPMVP</p>
</div>
""", unsafe_allow_html=True)

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
import hashlib
import pickle
import os
from datetime import datetime, timedelta
import base64
import scipy.stats as stats  # Ajouté pour les calculs statistiques t-test

# 📌 Configuration de la page
st.set_page_config(
    page_title="Analyse IPMVP Simplifiée",
    page_icon="📊",
    layout="wide"
)

#####################################
# SYSTÈME D'AUTHENTIFICATION - DÉBUT
#####################################

# Configuration de la gestion des utilisateurs
USER_DB_FILE = 'users_db.pkl'  # Fichier de stockage des utilisateurs
ADMIN_USERNAME = 'admin'  # Username de l'administrateur par défaut
ADMIN_PASSWORD = 'admin'  # Mot de passe de l'administrateur par défaut (à changer !)

# Fonction pour hacher les mots de passe
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Fonction pour initialiser la base de données des utilisateurs
def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        users = {
            ADMIN_USERNAME: {
                'password': hash_password(ADMIN_PASSWORD),
                'full_name': 'Administrateur',
                'email': 'admin@example.com',
                'created_at': datetime.now(),
                'is_admin': True
            }
        }
        with open(USER_DB_FILE, 'wb') as f:
            pickle.dump(users, f)
        return users
    else:
        with open(USER_DB_FILE, 'rb') as f:
            return pickle.load(f)

# Fonction pour sauvegarder la base de données des utilisateurs
def save_user_db(users):
    with open(USER_DB_FILE, 'wb') as f:
        pickle.dump(users, f)

# Fonction pour ajouter ou modifier un utilisateur
def update_user(username, password=None, full_name=None, email=None, is_admin=False):
    users = init_user_db()
    
    if username in users:
        # Mise à jour d'un utilisateur existant
        if password:
            users[username]['password'] = hash_password(password)
        if full_name:
            users[username]['full_name'] = full_name
        if email:
            users[username]['email'] = email
        users[username]['is_admin'] = is_admin
    else:
        # Création d'un nouvel utilisateur
        users[username] = {
            'password': hash_password(password) if password else '',
            'full_name': full_name or username,
            'email': email or '',
            'created_at': datetime.now(),
            'is_admin': is_admin
        }
    
    save_user_db(users)
    return True

# Fonction pour supprimer un utilisateur
def delete_user(username):
    users = init_user_db()
    if username in users and username != ADMIN_USERNAME:  # Empêcher la suppression de l'admin
        del users[username]
        save_user_db(users)
        return True
    return False

# Fonction pour vérifier les identifiants
def check_credentials(username, password):
    users = init_user_db()
    if username in users and users[username]['password'] == hash_password(password):
        return True
    return False

# Fonction pour vérifier si un utilisateur est admin
def is_admin(username):
    users = init_user_db()
    return username in users and users[username]['is_admin']

def show_login_form():
    # Définir l'interface utilisateur avec les styles
    st.markdown("""
    <style>
    /* Tous vos styles CSS ici... */
    </style>
    
    <!-- Image de fond réaliste -->
    <div class="login-background"></div>
    <!-- Overlay pour améliorer la lisibilité -->
    <div class="login-overlay"></div>
    
    <div class="glass-panel">
        <div class="brand-logo">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 60" width="120">
                <rect x="20" y="15" width="80" height="30" rx="5" fill="rgba(255,255,255,0.9)"/>
                <text x="60" y="37" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle" fill="#00485F">IPMVP</text>
                <path d="M30 15 L30 45" stroke="#96B91D" stroke-width="3"/>
                <path d="M90 15 L90 45" stroke="#6DBABC" stroke-width="3"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)
    
    # Rendre le titre et sous-titre séparément pour éviter les problèmes
    st.markdown('<h1 class="login-title">CALCUL & ANALYSE </h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-subtitle">Outil d\'analyse et de modélisation énergétique conforme aux standards du protocol IPMVP</p>', unsafe_allow_html=True)
    
    # Utiliser la clé "login_status" pour stocker le résultat de la tentative de connexion
    if "login_status" not in st.session_state:
        st.session_state.login_status = None
    
    # Afficher un message d'erreur si nécessaire
    if st.session_state.login_status == "failed":
        st.error("Identifiants incorrects. Veuillez réessayer.")
    
    # Créer le formulaire de connexion
    with st.form("login_form"):
        st.markdown('<label for="username" class="login-label">Nom d\'utilisateur</label>', unsafe_allow_html=True)
        username = st.text_input("", key="username_input", label_visibility="collapsed")
        st.markdown('<label for="password" class="login-label">Mot de passe</label>', unsafe_allow_html=True)
        password = st.text_input("", type="password", key="password_input", label_visibility="collapsed")
        
        submitted = st.form_submit_button("Se connecter")
        
        # Ne traiter la soumission du formulaire que si le bouton est cliqué
        if submitted:
            # Vérifier les identifiants
            if check_credentials(username, password):
                # Si corrects, définir des variables pour les utiliser après le rendu
                st.session_state.login_successful = True
                st.session_state.logged_username = username
                st.session_state.logged_admin = is_admin(username)
            else:
                st.session_state.login_status = "failed"
                st.session_state.login_successful = False
    
    # Fermer les divs
    st.markdown("""
        </div>
    
    <div class="glass-footer">
        <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Après tout le rendu, mettre à jour l'état de session si connexion réussie
    if st.session_state.get('login_successful', False):
        st.session_state['authenticated'] = True
        st.session_state['username'] = st.session_state.logged_username
        st.session_state['is_admin'] = st.session_state.logged_admin
        
        # Nettoyer les variables temporaires
        del st.session_state['login_successful']
        del st.session_state['logged_username']
        del st.session_state['logged_admin']
        del st.session_state['login_status']
        
        # Recharger la page
        st.rerun()

# Interface d'administration des utilisateurs
def show_admin_panel():
    st.header("Administration des utilisateurs")
    
    users = init_user_db()
    
    # Afficher la liste des utilisateurs
    st.subheader("Utilisateurs existants")
    
    user_data = []
    for username, data in users.items():
        user_data.append({
            "Nom d'utilisateur": username,
            "Nom complet": data.get('full_name', ''),
            "Email": data.get('email', ''),
            "Date de création": data.get('created_at', '').strftime('%d/%m/%Y') if 'created_at' in data else '',
            "Admin": "✅" if data.get('is_admin', False) else "❌"
        })
    
    st.table(user_data)
    
    # Diviser l'interface en onglets
    tab1, tab2 = st.tabs(["Ajouter/Modifier un utilisateur", "Supprimer un utilisateur"])
    
    # Onglet 1: Ajouter/Modifier un utilisateur
    with tab1:
        with st.form("user_form"):
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("Nom d'utilisateur*")
            with col2:
                password = st.text_input("Mot de passe*", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Nom complet")
            with col2:
                email = st.text_input("Email")
            
            is_admin = st.checkbox("Administrateur")
            
            submit = st.form_submit_button("Enregistrer l'utilisateur", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Le nom d'utilisateur et le mot de passe sont obligatoires.")
                else:
                    update_user(username, password, full_name, email, is_admin)
                    st.success(f"Utilisateur '{username}' enregistré avec succès.")
                    st.rerun()
    
    # Onglet 2: Supprimer un utilisateur
    with tab2:
        with st.form("delete_user_form"):
            user_to_delete = st.selectbox(
                "Sélectionner un utilisateur à supprimer",
                [u for u in users.keys() if u != ADMIN_USERNAME]
            )
            
            delete_submit = st.form_submit_button("Supprimer l'utilisateur", type="primary", use_container_width=True)
            
            if delete_submit:
                if delete_user(user_to_delete):
                    st.success(f"Utilisateur '{user_to_delete}' supprimé avec succès.")
                    st.rerun()
                else:
                    st.error("Impossible de supprimer cet utilisateur.")
    
    # Bouton pour revenir à l'application principale
    if st.button("Retour à l'application", use_container_width=True):
        st.session_state['show_admin'] = False
        st.rerun()

# Barre de navigation avec informations utilisateur et déconnexion
def show_navbar():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"<div style='padding: 10px 0;'>Connecté en tant que: <b>{st.session_state['username']}</b></div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('is_admin', False):
            if st.button("Administration", key="admin_button", use_container_width=True):
                st.session_state['show_admin'] = not st.session_state.get('show_admin', False)
                st.rerun()
    
    with col3:
        if st.button("Déconnexion", key="logout_button", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Initialisation des variables de session
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'show_admin' not in st.session_state:
    st.session_state['show_admin'] = False

# Vérification de l'authentification
if not st.session_state['authenticated']:
    show_login_form()
    st.stop()  # Arrête l'exécution du reste de l'application si non authentifié

# Affichage du panneau d'administration si demandé
if st.session_state.get('show_admin', False) and st.session_state.get('is_admin', False):
    show_admin_panel()
    st.stop()  # Arrête l'exécution de l'application principale quand on est dans l'admin

###################################
# SYSTÈME D'AUTHENTIFICATION - FIN
###################################
# NOUVELLES FONCTIONS POUR AMÉLIORER L'AFFICHAGE ET CALCULER LES STATISTIQUES T

# Fonction pour calculer les valeurs t-stat pour les coefficients
def calculate_t_stats(X, y, model, coefs):
    """
    Calcule les valeurs t-stat pour les coefficients de régression.
    
    Parameters:
    X (pandas.DataFrame): Variables explicatives
    y (pandas.Series): Variable cible
    model: Modèle de régression ajusté
    coefs (dict): Dictionnaire des coefficients
    
    Returns:
    dict: Dictionnaire des valeurs t-stat et p-values pour chaque variable
    """
    # Ne s'applique qu'aux modèles linéaires standards
    if not hasattr(model, 'coef_'):
        # Pour les modèles non standards comme les polynomiaux via Pipeline
        return {feature: None for feature in coefs.keys()}
    
    # Calcul des prédictions et des résidus
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Degrés de liberté et MSE
    n = len(y)
    p = len(model.coef_)
    df = n - p - 1
    if df <= 0:  # Éviter division par zéro ou valeurs négatives
        return {feature: None for feature in coefs.keys()}
        
    mse = np.sum(residuals ** 2) / df
    
    # Calcul de la matrice (X'X)^-1
    try:
        # Pour les modèles de régression linéaire standard
        X_matrix = X.values
        XtX_inv = np.linalg.inv(np.dot(X_matrix.T, X_matrix))
        
        # Erreurs standard
        se = np.sqrt(np.diag(XtX_inv) * mse)
        
        # Calcul des valeurs t
        t_stats = model.coef_ / se
        
        # Calcul des p-values
        p_values = [2 * (1 - stats.t.cdf(abs(t), df)) for t in t_stats]
        
        # Créer un dictionnaire des valeurs t et p-values
        result = {}
        for i, feature in enumerate(X.columns):
            result[feature] = {
                't_value': t_stats[i],
                'p_value': p_values[i],
                'significant': p_values[i] < 0.05  # Significatif au niveau 5%
            }
        
        return result
    except:
        # En cas d'erreur, retourner None pour toutes les variables
        return {feature: None for feature in X.columns}

# Fonction pour formater l'équation en ignorant les coefficients proches de zéro
def format_equation(intercept, coefficients, threshold=1e-4):
    """
    Formate l'équation du modèle en ignorant les coefficients proches de zéro.
    
    Parameters:
    intercept (float): Terme constant du modèle
    coefficients (dict): Dictionnaire des coefficients
    threshold (float): Seuil en dessous duquel un coefficient est considéré comme nul
    
    Returns:
    str: Équation formatée
    """
    equation = f"Consommation = {intercept:.4f}"
    
    # Trier les coefficients par valeur absolue décroissante pour un meilleur affichage
    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in sorted_coefs:
        # Ne pas inclure les coefficients proches de zéro
        if abs(coef) < threshold:
            continue
            
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.4f} × {feature}"
    
    return equation

# Fonction pour détecter automatiquement les colonnes de date et de consommation
def detecter_colonnes(df):
    # Initialiser les résultats
    date_col_guess = None
    conso_col_guess = None
    
    if df is None or df.empty:
        return date_col_guess, conso_col_guess
    
    # 1. Détecter la colonne de date
    date_keywords = ['date', 'temps', 'période', 'period', 'time', 'jour', 'day', 'mois', 'month', 'année', 'year']
    
    # Essayer d'abord de trouver une colonne de type datetime
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        date_col_guess = datetime_cols[0]
    else:
        # Chercher par mots-clés dans les noms de colonnes
        for keyword in date_keywords:
            potential_cols = [col for col in df.columns if keyword.lower() in col.lower()]
            if potential_cols:
                # Essayer de convertir en datetime
                for col in potential_cols:
                    try:
                        pd.to_datetime(df[col])
                        date_col_guess = col
                        break
                    except:
                        continue
                if date_col_guess:
                    break
    
    # 2. Détecter la colonne de consommation
    conso_keywords = ['consommation', 'conso', 'énergie', 'energy', 'kwh', 'mwh', 'wh', 
                      'électricité', 'electricity', 'gaz', 'gas', 'chaleur', 'heat', 
                      'puissance', 'power', 'compteur', 'meter']
    
    # Exclure la colonne de date si elle a été trouvée
    cols_to_check = [col for col in df.columns if col != date_col_guess]
    
    # Chercher par mots-clés dans les noms de colonnes
    for keyword in conso_keywords:
        potential_cols = [col for col in cols_to_check if keyword.lower() in col.lower()]
        if potential_cols:
            # Vérifier que ce sont des valeurs numériques
            for col in potential_cols:
                try:
                    if pd.to_numeric(df[col], errors='coerce').notna().sum() > 0.8 * len(df):
                        conso_col_guess = col
                        break
                except:
                    continue
            if conso_col_guess:
                break
    
    # Si aucune correspondance par mot-clé, essayer de trouver une colonne numérique
    if not conso_col_guess:
        numeric_cols = [col for col in cols_to_check if 
                        pd.api.types.is_numeric_dtype(df[col]) or 
                        pd.to_numeric(df[col], errors='coerce').notna().sum() > 0.8 * len(df)]
        if numeric_cols:
            # Sélectionner la première colonne numérique non-index qui n'est pas une date
            for col in numeric_cols:
                if not (col.lower().startswith('id') or col.lower().startswith('index')):
                    conso_col_guess = col
                    break
            if not conso_col_guess and numeric_cols:
                conso_col_guess = numeric_cols[0]
    
    return date_col_guess, conso_col_guess

# Fonction pour créer une info-bulle (mise à jour pour décaler les bulles à droite)
def tooltip(text, explanation):
    return f'<span>{text} <span class="tooltip">ℹ️<span class="tooltiptext tooltip-right">{explanation}</span></span></span>'

# Fonction pour évaluer la conformité IPMVP
def evaluer_conformite(r2, cv_rmse):
    if r2 >= 0.75 and cv_rmse <= 0.15:
        return "Excellente", "good"
    elif r2 >= 0.5 and cv_rmse <= 0.25:
        return "Acceptable", "medium"
    else:
        return "Insuffisante", "bad"

# Fonction sécurisée pour formater les valeurs numériques (ajoutée pour éviter les erreurs)
def format_value(value, fmt=".4f", default="N/A"):
    """
    Formate une valeur numérique de manière sécurisée.
    
    Parameters:
    value: Valeur à formater
    fmt (str): Format à appliquer (par défaut ".4f")
    default (str): Valeur par défaut si la conversion échoue
    
    Returns:
    str: Valeur formatée ou valeur par défaut
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, (int, float)):
            return f"{value:{fmt}}"
        elif isinstance(value, dict) and 't_value' in value and isinstance(value['t_value'], (int, float)):
            return f"{value['t_value']:{fmt}}"
        return default
    except:
        return default
        
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
    
    /* Style pour les info-bulles à droite */
    .tooltip-right {
        left: 100% !important;
        margin-left: 10px !important;
        bottom: 0 !important;
    }

    .tooltip-right::after {
        top: 50% !important;
        left: -5px !important;
        margin-left: 0 !important;
        margin-top: -5px !important;
        border-width: 5px !important;
        border-style: solid !important;
        border-color: transparent #00485F transparent transparent !important;
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

    /* Style pour les tableaux de données statistiques */
    .stats-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        border-radius: 5px;
        overflow: hidden;
    }
    
    .stats-table th {
        background-color: #00485F;
        color: white;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
    }
    
    .stats-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stats-table tr:nth-child(even) {
        background-color: rgba(109, 186, 188, 0.1);
    }
    
    .stats-table tr:hover {
        background-color: rgba(150, 185, 29, 0.1);
    }
    
    /* Style pour les badges de significativité */
    .significance-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: bold;
    }
    
    .significant {
        background-color: #96B91D;
        color: white;
    }
    
    .not-significant {
        background-color: #e74c3c;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 📌 **Description de l'application**
st.title("📊 Calcul IPMVP")
st.markdown("""
Bienvenue sur **l'Analyse & Calcul IPMVP ** 🔍 !  
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
    <li><strong>Type de modèle</strong> : 
        <ul>
            <li>Option "Automatique" (recommandée) : teste tous les types de modèles et sélectionne celui qui offre le meilleur R²</li>
            <li>Options spécifiques : vous pouvez choisir manuellement un type de modèle (linéaire, Ridge, Lasso, polynomiale) et ses paramètres</li>
        </ul>
    </li>
    <li><strong>Configuration de l'analyse</strong> : Choisissez le nombre maximum de variables à combiner (1 à 4)</li>
    <li><strong>Lancement</strong> : Cliquez sur "Lancer le calcul" pour obtenir le meilleur modèle d'ajustement</li>
    <li><strong>Analyse des résultats</strong> : 
        <ul>
            <li>L'équation d'ajustement montre la relation mathématique entre les variables</li>
            <li>Les métriques (R², CV(RMSE), biais) permettent d'évaluer la conformité IPMVP</li>
            <li>Les graphiques visualisent l'ajustement du modèle aux données réelles</li>
            <li>Le tableau de classement compare tous les modèles testés</li>
        </ul>
    </li>
</ol>
</div>
""", unsafe_allow_html=True)

# 📂 **Import du fichier et lancement du calcul**
col1, col2 = st.columns([3, 1])  # Mise en page : Import à gauche, bouton à droite

with col1:
    uploaded_file = st.file_uploader("📂 Importer un fichier Excel", type=["xlsx", "xls"])

with col2:
    lancer_calcul = st.button("🚀 Lancer le calcul", use_container_width=True)
 # Traitement du fichier importé
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)  # Chargement du fichier
        
        # Détecter automatiquement les colonnes de date et de consommation
        date_col_guess, conso_col_guess = detecter_colonnes(df)
        
        # Informer l'utilisateur des colonnes détectées automatiquement
        if date_col_guess and conso_col_guess:
            st.success(f"✅ Détection automatique : Colonne de date = '{date_col_guess}', Colonne de consommation = '{conso_col_guess}'")
        elif date_col_guess:
            st.info(f"ℹ️ Colonne de date détectée : '{date_col_guess}'. Veuillez sélectionner manuellement la colonne de consommation.")
        elif conso_col_guess:
            st.info(f"ℹ️ Colonne de consommation détectée : '{conso_col_guess}'. Veuillez sélectionner manuellement la colonne de date.")
        else:
            st.warning("⚠️ Impossible de détecter automatiquement les colonnes date et consommation. Veuillez les sélectionner manuellement.")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier Excel : {str(e)}")
        df = None
        date_col_guess = None
        conso_col_guess = None
else:
    df = None
    date_col_guess = None
    conso_col_guess = None

# 📂 **Sélection des données (toujours visible même sans fichier importé)**
st.sidebar.header("🔍 Sélection des données")

# **Définition des colonnes pour la sélection AVANT import**
date_col = st.sidebar.selectbox(
    "📅 Nom de la donnée date", 
    df.columns if df is not None else [""],
    index=list(df.columns).index(date_col_guess) if df is not None and date_col_guess in df.columns else 0
)

conso_col = st.sidebar.selectbox(
    "⚡ Nom de la donnée consommation", 
    df.columns if df is not None else [""],
    index=list(df.columns).index(conso_col_guess) if df is not None and conso_col_guess in df.columns else 0
)

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

st.sidebar.markdown("---")

# Ajouter les contrôles d'administration et de profil dans la barre latérale
if st.session_state['authenticated']:
    st.sidebar.header("👤 Gestion du compte")
    st.sidebar.markdown(f"Connecté en tant que: **{st.session_state['username']}**")
    
    # Section d'administration (visible uniquement pour les administrateurs)
    if st.session_state.get('is_admin', False):
        st.sidebar.markdown("#### 🔐 Administration")
        if st.sidebar.button("Gérer les utilisateurs", use_container_width=True):
            st.session_state['show_admin'] = True
            st.rerun()
    
    # Option de changement de mot de passe pour tous les utilisateurs
    st.sidebar.markdown("#### 🔑 Changer de mot de passe")
    with st.sidebar.form("change_password_form"):
        current_password = st.text_input("Mot de passe actuel", type="password")
        new_password = st.text_input("Nouveau mot de passe", type="password")
        confirm_password = st.text_input("Confirmer le mot de passe", type="password")
        submit_password = st.form_submit_button("Modifier le mot de passe", use_container_width=True)
        
        if submit_password:
            # Vérifier l'ancien mot de passe
            if not check_credentials(st.session_state['username'], current_password):
                st.sidebar.error("Mot de passe actuel incorrect.")
            elif new_password != confirm_password:
                st.sidebar.error("Les nouveaux mots de passe ne correspondent pas.")
            elif not new_password:
                st.sidebar.error("Le nouveau mot de passe ne peut pas être vide.")
            else:
                # Mettre à jour le mot de passe
                update_user(st.session_state['username'], new_password)
                st.sidebar.success("Mot de passe modifié avec succès!")
    
    # Bouton de déconnexion
    if st.sidebar.button("Déconnexion", key="sidebar_logout", use_container_width=True):
        for key in list   

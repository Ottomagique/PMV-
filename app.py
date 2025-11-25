# =============================================================================
# PARTIE 1 : BASE + AUTHENTIFICATION
# Application IPMVP Améliorée - Version 2.1 - Visualisations enrichies
# =============================================================================

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
from sklearn.model_selection import train_test_split
import math
import hashlib
import pickle
import os
from datetime import datetime, timedelta
import base64
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 📌 Configuration de la page
st.set_page_config(
    page_title="Analyse IPMVP Améliorée",
    page_icon="📊",
    layout="wide"
)

#####################################
# SYSTÈME D'AUTHENTIFICATION - DÉBUT
#####################################

# Configuration de la gestion des utilisateurs
USER_DB_FILE = 'users_db.pkl'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

def hash_password(password):
    """Hache les mots de passe pour la sécurité"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_user_db():
    """Initialise la base de données des utilisateurs"""
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

def save_user_db(users):
    """Sauvegarde la base de données des utilisateurs"""
    with open(USER_DB_FILE, 'wb') as f:
        pickle.dump(users, f)

def update_user(username, password=None, full_name=None, email=None, is_admin=False):
    """Ajoute ou modifie un utilisateur"""
    users = init_user_db()
    
    if username in users:
        if password:
            users[username]['password'] = hash_password(password)
        if full_name:
            users[username]['full_name'] = full_name
        if email:
            users[username]['email'] = email
        users[username]['is_admin'] = is_admin
    else:
        users[username] = {
            'password': hash_password(password) if password else '',
            'full_name': full_name or username,
            'email': email or '',
            'created_at': datetime.now(),
            'is_admin': is_admin
        }
    
    save_user_db(users)
    return True

def delete_user(username):
    """Supprime un utilisateur (sauf admin)"""
    users = init_user_db()
    if username in users and username != ADMIN_USERNAME:
        del users[username]
        save_user_db(users)
        return True
    return False

def check_credentials(username, password):
    """Vérifie les identifiants de connexion"""
    users = init_user_db()
    if username in users and users[username]['password'] == hash_password(password):
        return True
    return False

def is_admin(username):
    """Vérifie si un utilisateur est administrateur"""
    users = init_user_db()
    return username in users and users[username]['is_admin']

def show_login_form():
    """Affiche le formulaire de connexion"""
    # Interface de connexion avec style moderne
    st.markdown("""
    <style>
    /* Styles de connexion */
    .login-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        z-index: -2;
    }
    
    .login-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.3);
        z-index: -1;
    }
    
    .glass-panel {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        margin: 50px auto;
        max-width: 400px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .brand-logo {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .login-title {
        text-align: center;
        color: #00485F;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .login-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-style: italic;
    }
    
    .login-label {
        font-weight: 600;
        color: #00485F;
        margin-bottom: 5px;
        display: block;
    }
    
    .glass-footer {
        text-align: center;
        margin-top: 30px;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    </style>
    
    <!-- Interface de connexion -->
    <div class="login-background"></div>
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
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="login-title">CALCUL & ANALYSE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-subtitle">Outil d\'analyse et de modélisation énergétique conforme aux standards IPMVP</p>', unsafe_allow_html=True)
    
    # Gestion de l'état de connexion
    if "login_status" not in st.session_state:
        st.session_state.login_status = None
    
    if st.session_state.login_status == "failed":
        st.error("Identifiants incorrects. Veuillez réessayer.")
    
    # Formulaire de connexion
    with st.form("login_form"):
        st.markdown('<label for="username" class="login-label">Nom d\'utilisateur</label>', unsafe_allow_html=True)
        username = st.text_input("", key="username_input", label_visibility="collapsed")
        st.markdown('<label for="password" class="login-label">Mot de passe</label>', unsafe_allow_html=True)
        password = st.text_input("", type="password", key="password_input", label_visibility="collapsed")
        
        submitted = st.form_submit_button("Se connecter")
        
        if submitted:
            if check_credentials(username, password):
                st.session_state.login_successful = True
                st.session_state.logged_username = username
                st.session_state.logged_admin = is_admin(username)
            else:
                st.session_state.login_status = "failed"
                st.session_state.login_successful = False
    
    # Pied de page
    st.markdown("""
    <div class="glass-footer">
        <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mise à jour de l'état après rendu
    if st.session_state.get('login_successful', False):
        st.session_state['authenticated'] = True
        st.session_state['username'] = st.session_state.logged_username
        st.session_state['is_admin'] = st.session_state.logged_admin
        
        # Nettoyage des variables temporaires
        del st.session_state['login_successful']
        del st.session_state['logged_username']
        del st.session_state['logged_admin']
        del st.session_state['login_status']
        
        st.rerun()

def show_admin_panel():
    """Interface d'administration des utilisateurs"""
    st.header("🔐 Administration des utilisateurs")
    
    users = init_user_db()
    
    # Liste des utilisateurs existants
    st.subheader("👥 Utilisateurs existants")
    
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
    
    # Onglets pour la gestion
    tab1, tab2 = st.tabs(["➕ Ajouter/Modifier", "🗑️ Supprimer"])
    
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
            
            is_admin_checkbox = st.checkbox("Administrateur")
            
            submit = st.form_submit_button("💾 Enregistrer l'utilisateur", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("❌ Le nom d'utilisateur et le mot de passe sont obligatoires.")
                else:
                    update_user(username, password, full_name, email, is_admin_checkbox)
                    st.success(f"✅ Utilisateur '{username}' enregistré avec succès.")
                    st.rerun()
    
    with tab2:
        with st.form("delete_user_form"):
            user_to_delete = st.selectbox(
                "Sélectionner un utilisateur à supprimer",
                [u for u in users.keys() if u != ADMIN_USERNAME]
            )
            
            delete_submit = st.form_submit_button("🗑️ Supprimer l'utilisateur", type="primary", use_container_width=True)
            
            if delete_submit:
                if delete_user(user_to_delete):
                    st.success(f"✅ Utilisateur '{user_to_delete}' supprimé avec succès.")
                    st.rerun()
                else:
                    st.error("❌ Impossible de supprimer cet utilisateur.")
    
    # Retour à l'application
    if st.button("🔙 Retour à l'application", use_container_width=True):
        st.session_state['show_admin'] = False
        st.rerun()

def show_navbar():
    """Barre de navigation avec informations utilisateur"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"<div style='padding: 10px 0;'>👤 Connecté en tant que: <b>{st.session_state['username']}</b></div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('is_admin', False):
            if st.button("🔐 Administration", key="admin_button", use_container_width=True):
                st.session_state['show_admin'] = not st.session_state.get('show_admin', False)
                st.rerun()
    
    with col3:
        if st.button("🚪 Déconnexion", key="logout_button", use_container_width=True):
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
    st.stop()

# Affichage du panneau d'administration si demandé
if st.session_state.get('show_admin', False) and st.session_state.get('is_admin', False):
    show_admin_panel()
    st.stop()

###################################
# SYSTÈME D'AUTHENTIFICATION - FIN
###################################
# =============================================================================
# PARTIE 2 : FONCTIONS MÉTIER IPMVP
# Fonctions de calcul statistique, validation et scoring améliorées
# =============================================================================

# NOUVELLES FONCTIONS POUR L'ANALYSE IPMVP AMÉLIORÉE

def detecter_colonnes(df):
    """
    Détecte automatiquement les colonnes de date et de consommation
    """
    date_col_guess = None
    conso_col_guess = None
    
    if df is None or df.empty:
        return date_col_guess, conso_col_guess
    
    # 1. Détecter la colonne de date
    date_keywords = ['date', 'temps', 'période', 'period', 'time', 'jour', 'day', 'mois', 'month', 'année', 'year']
    
    # Essayer d'abord les colonnes datetime
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        date_col_guess = datetime_cols[0]
    else:
        # Chercher par mots-clés
        for keyword in date_keywords:
            potential_cols = [col for col in df.columns if keyword.lower() in col.lower()]
            if potential_cols:
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
    
    cols_to_check = [col for col in df.columns if col != date_col_guess]
    
    # Chercher par mots-clés
    for keyword in conso_keywords:
        potential_cols = [col for col in cols_to_check if keyword.lower() in col.lower()]
        if potential_cols:
            for col in potential_cols:
                try:
                    if pd.to_numeric(df[col], errors='coerce').notna().sum() > 0.8 * len(df):
                        conso_col_guess = col
                        break
                except:
                    continue
            if conso_col_guess:
                break
    
    # Si aucune correspondance, chercher une colonne numérique
    if not conso_col_guess:
        numeric_cols = [col for col in cols_to_check if 
                        pd.api.types.is_numeric_dtype(df[col]) or 
                        pd.to_numeric(df[col], errors='coerce').notna().sum() > 0.8 * len(df)]
        if numeric_cols:
            for col in numeric_cols:
                if not (col.lower().startswith('id') or col.lower().startswith('index')):
                    conso_col_guess = col
                    break
            if not conso_col_guess and numeric_cols:
                conso_col_guess = numeric_cols[0]
    
    return date_col_guess, conso_col_guess

def calculate_t_stats(X, y, model, coefs):
    """
    Calcule les valeurs t-stat pour les coefficients de régression
    """
    if not hasattr(model, 'coef_'):
        return {feature: None for feature in coefs.keys()}
    
    try:
        # Calcul des prédictions et résidus
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Degrés de liberté et MSE
        n = len(y)
        p = len(model.coef_)
        df = n - p - 1
        if df <= 0:
            return {feature: None for feature in coefs.keys()}
            
        mse = np.sum(residuals ** 2) / df
        
        # Calcul de la matrice (X'X)^-1
        X_matrix = X.values
        XtX_inv = np.linalg.inv(np.dot(X_matrix.T, X_matrix))
        
        # Erreurs standard
        se = np.sqrt(np.diag(XtX_inv) * mse)
        
        # Calcul des valeurs t
        t_stats = model.coef_ / se
        
        # Calcul des p-values
        p_values = [2 * (1 - stats.t.cdf(abs(t), df)) for t in t_stats]
        
        # Créer un dictionnaire des résultats
        result = {}
        for i, feature in enumerate(X.columns):
            result[feature] = {
                't_value': t_stats[i],
                'p_value': p_values[i],
                'significant': p_values[i] < 0.05
            }
        
        return result
    except:
        return {feature: None for feature in X.columns}

def detect_overfitting_intelligent(model_info, nb_observations):
    """
    Détection intelligente de l'overfitting selon le contexte
    """
    r2 = model_info['r2']
    nb_variables = len(model_info['features'])
    model_type = model_info['model_type']
    
    # Calcul du ratio observations/variables
    ratio = nb_observations / nb_variables if nb_variables > 0 else float('inf')
    
    # Critères d'overfitting adaptatifs
    is_overfitted = False
    warning_msg = ""
    severity = "info"
    
    # 1. R² extrême (toujours suspect)
    if r2 > 0.995:
        is_overfitted = True
        warning_msg = "🚨 R² extrême (>99.5%) - Overfitting quasi certain"
        severity = "error"
    
    # 2. R² très élevé avec contexte dangereux
    elif r2 > 0.98:
        if ratio < 5:  # Moins de 5 observations par variable
            is_overfitted = True
            warning_msg = f"🚨 R² = {r2:.3f} avec ratio obs/var = {ratio:.1f} - Overfitting probable"
            severity = "error"
        elif model_type == "Polynomiale":
            is_overfitted = True
            warning_msg = f"⚠️ Modèle polynomial avec R² = {r2:.3f} - Risque overfitting élevé"
            severity = "warning"
        elif nb_variables > 3:
            warning_msg = f"⚠️ R² = {r2:.3f} avec {nb_variables} variables - Vérifier la robustesse"
            severity = "warning"
    
    # 3. Ratio dangereux même avec R² modéré
    elif ratio < 3:
        is_overfitted = True
        warning_msg = f"🚨 Ratio observations/variables = {ratio:.1f} - Données insuffisantes"
        severity = "error"
    elif ratio < 5:
        warning_msg = f"⚠️ Ratio observations/variables = {ratio:.1f} - Risque overfitting"
        severity = "warning"
    
    return is_overfitted, warning_msg, severity

def calculate_ipmvp_score(model_info, nb_observations):
    """
    Calcule un score composite IPMVP de 0 à 100 points
    """
    r2 = model_info['r2']
    cv_rmse = model_info['cv_rmse']
    bias = abs(model_info['bias'])
    nb_variables = len(model_info['features'])
    model_type = model_info['model_type']
    
    # Score de base (60 points max)
    # R² : 30 points max
    r2_score = min(r2 / 0.75, 1.0) * 30 if r2 >= 0.5 else r2 * 20
    
    # CV(RMSE) : 20 points max (inversé - plus faible = mieux)
    cv_score = max(0, min((0.25 - cv_rmse) / 0.25, 1.0)) * 20
    
    # Biais : 10 points max
    bias_score = max(0, min((10 - bias) / 10, 1.0)) * 10
    
    base_score = r2_score + cv_score + bias_score
    
    # Bonus/Malus (40 points max)
    bonus_malus = 0
    
    # Bonus simplicité (15 points max)
    if nb_variables == 1:
        bonus_malus += 15
    elif nb_variables == 2:
        bonus_malus += 10
    elif nb_variables == 3:
        bonus_malus += 5
    
    # Bonus conformité IPMVP (15 points max)
    if r2 >= 0.75 and cv_rmse <= 0.15 and bias <= 5:
        bonus_malus += 15
    elif r2 >= 0.6 and cv_rmse <= 0.2 and bias <= 8:
        bonus_malus += 10
    elif r2 >= 0.5 and cv_rmse <= 0.25:
        bonus_malus += 5
    
    # Bonus significativité statistique (10 points max)
    if 't_stats' in model_info and model_type in ["Linéaire", "Ridge", "Lasso"]:
        significant_vars = 0
        total_vars = 0
        for feature in model_info['features']:
            if (feature in model_info['t_stats'] and 
                model_info['t_stats'][feature] is not None):
                total_vars += 1
                t_val = model_info['t_stats'][feature]
                if isinstance(t_val, dict) and 't_value' in t_val:
                    if abs(t_val['t_value']) > 2:
                        significant_vars += 1
                elif isinstance(t_val, (int, float)) and abs(t_val) > 2:
                    significant_vars += 1
        
        if total_vars > 0:
            sig_ratio = significant_vars / total_vars
            bonus_malus += sig_ratio * 10
    
    # Malus overfitting
    is_overfitted, _, severity = detect_overfitting_intelligent(model_info, nb_observations)
    if is_overfitted:
        if severity == "error":
            bonus_malus -= 30  # Gros malus
        else:
            bonus_malus -= 15  # Malus modéré
    
    # Malus modèle complexe
    if model_type == "Polynomiale":
        bonus_malus -= 5
    
    # Score final (0-100)
    final_score = max(0, min(100, base_score + bonus_malus))
    
    return final_score

def validate_data_quality(df, date_col, conso_col, selected_vars):
    """
    Valide la qualité des données avant l'analyse
    """
    issues = []
    warnings = []
    
    # 1. Vérification des données manquantes
    missing_dates = df[date_col].isnull().sum()
    missing_conso = df[conso_col].isnull().sum()
    
    if missing_dates > 0:
        issues.append(f"❌ {missing_dates} dates manquantes détectées")
    
    if missing_conso > 0:
        issues.append(f"❌ {missing_conso} valeurs de consommation manquantes")
    
    # 2. Vérification des variables explicatives
    for var in selected_vars:
        missing_var = df[var].isnull().sum()
        if missing_var > len(df) * 0.1:  # Plus de 10% manquant
            warnings.append(f"⚠️ Variable '{var}': {missing_var} valeurs manquantes ({missing_var/len(df)*100:.1f}%)")
    
    # 3. Vérification de la régularité temporelle
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        date_diff = df[date_col].diff().dropna()
        if date_diff.std().days > 5:  # Irrégularité > 5 jours
            warnings.append("⚠️ Espacement irrégulier entre les dates détecté")
    
    # 4. Vérification des valeurs aberrantes (consommation)
    if pd.api.types.is_numeric_dtype(df[conso_col]):
        q1 = df[conso_col].quantile(0.25)
        q3 = df[conso_col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[conso_col] < q1 - 1.5*iqr) | (df[conso_col] > q3 + 1.5*iqr)).sum()
        if outliers > len(df) * 0.05:  # Plus de 5% d'outliers
            warnings.append(f"⚠️ {outliers} valeurs aberrantes potentielles dans la consommation ({outliers/len(df)*100:.1f}%)")
    
    # 5. Vérification du nombre minimum de données
    if len(df) < 12:
        issues.append(f"❌ Données insuffisantes: {len(df)} points (minimum 12 requis)")
    elif len(df) < 24:
        warnings.append(f"⚠️ Données limitées: {len(df)} points (24+ recommandés pour train/test)")
    
    return issues, warnings

def check_variable_limits(nb_observations, nb_variables, model_type):
    """
    Vérifie les limitations de sécurité pour éviter l'overfitting
    """
    issues = []
    warnings = []
    
    # Règle des 10:1 pour les observations/variables
    max_vars_recommended = nb_observations // 10
    max_vars_minimum = nb_observations // 5  # Seuil critique
    
    if nb_variables > max_vars_minimum:
        issues.append(f"🚨 Trop de variables: {nb_variables} avec {nb_observations} observations (ratio {nb_observations/nb_variables:.1f}:1)")
        issues.append(f"Maximum critique: {max_vars_minimum} variables")
    elif nb_variables > max_vars_recommended:
        warnings.append(f"⚠️ Ratio observations/variables: {nb_observations/nb_variables:.1f}:1 (recommandé: ≥10:1)")
        warnings.append(f"Recommandation: maximum {max_vars_recommended} variables")
    
    # Limitations spécifiques aux modèles polynomiaux
    if model_type == "Polynomiale":
        if nb_observations < 20:
            issues.append(f"🚨 Modèle polynomial nécessite ≥20 observations (actuellement: {nb_observations})")
        elif nb_observations < 30:
            warnings.append(f"⚠️ Modèle polynomial avec {nb_observations} observations - Risque d'instabilité")
        
        # Estimation du nombre de paramètres générés
        estimated_params = nb_variables * 2 + nb_variables  # Approximation pour degré 2
        if estimated_params > nb_observations // 3:
            warnings.append(f"⚠️ Modèle polynomial générera ~{estimated_params} paramètres - Risque de complexité excessive")
    
    return issues, warnings

def format_equation(intercept, coefficients, threshold=1e-4):
    """
    Formate l'équation du modèle en ignorant les coefficients négligeables
    """
    equation = f"Consommation = {intercept:.4f}"
    
    # Trier les coefficients par valeur absolue décroissante
    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in sorted_coefs:
        # Ignorer les coefficients proches de zéro
        if abs(coef) < threshold:
            continue
            
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.4f} × {feature}"
    
    return equation

def tooltip(text, explanation):
    """
    Crée une info-bulle explicative
    """
    return f'<span>{text} <span class="tooltip">ℹ️<span class="tooltiptext tooltip-right">{explanation}</span></span></span>'

def evaluer_conformite(r2, cv_rmse, bias=None):
    """
    Évalue la conformité IPMVP avec critères enrichis
    """
    # Critères principaux
    r2_ok = r2 >= 0.75
    cv_ok = cv_rmse <= 0.15
    bias_ok = bias is None or abs(bias) <= 5
    
    if r2_ok and cv_ok and bias_ok:
        return "Excellente", "good"
    elif r2 >= 0.6 and cv_rmse <= 0.2 and (bias is None or abs(bias) <= 8):
        return "Bonne", "medium"
    elif r2 >= 0.5 and cv_rmse <= 0.25:
        return "Acceptable", "medium"
    else:
        return "Insuffisante", "bad"

def format_value(value, fmt=".4f", default="N/A"):
    """
    Formate une valeur numérique de manière sécurisée
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

def should_use_train_test_split(nb_observations):
    """
    Détermine si on doit utiliser un split train/test
    """
    if nb_observations >= 24:
        return True, "🚀 Mode validation robuste: Split train/test activé"
    elif nb_observations >= 18:
        return False, f"⚠️ {nb_observations} mois disponibles - Split train/test recommandé avec ≥24 mois"
    else:
        return False, f"📋 Mode IPMVP standard avec {nb_observations} mois de données"

def create_train_test_split(df, date_col, train_months=18):
    """
    Crée un split train/test temporel pour les données IPMVP
    """
    # Trier par date
    df_sorted = df.sort_values(by=date_col)
    
    # Calculer le point de coupure (18 premiers mois pour train)
    min_date = df_sorted[date_col].min()
    split_date = min_date + pd.DateOffset(months=train_months)
    
    # Split des données
    train_df = df_sorted[df_sorted[date_col] < split_date]
    test_df = df_sorted[df_sorted[date_col] >= split_date]
    
    return train_df, test_df, split_date
# =============================================================================
# PARTIE 3 : INTERFACE UTILISATEUR
# CSS, styling, sidebar et interface principale avec contrôles adaptatifs
# =============================================================================

# CSS AMÉLIORÉ AVEC NOUVEAUX STYLES POUR LES AMÉLIORATIONS IPMVP
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
    
    /* Styles pour les info-bulles */
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

    /* Styles pour les tableaux statistiques */
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
    
    /* Badges de significativité */
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

    /* NOUVEAUX STYLES POUR LES AMÉLIORATIONS IPMVP */
    
    /* Alertes et warnings */
    .alert-card {
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    .alert-error {
        background-color: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    
    .alert-warning {
        background-color: #fff8e1;
        border-color: #ff9800;
        color: #f57c00;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-color: #2196f3;
        color: #1565c0;
    }
    
    .alert-success {
        background-color: #e8f5e8;
        border-color: #4caf50;
        color: #2e7d32;
    }

    /* Scores et métriques améliorées */
    .score-card {
        background: linear-gradient(135deg, #6DBABC 0%, #96B91D 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .score-value {
        font-size: 2.5em;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .score-label {
        font-size: 1.1em;
        opacity: 0.9;
        margin-top: 5px;
    }

    /* Badges de statut */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .status-excellent {
        background-color: #96B91D;
        color: white;
    }
    
    .status-good {
        background-color: #6DBABC;
        color: white;
    }
    
    .status-warning {
        background-color: #ff9800;
        color: white;
    }
    
    .status-error {
        background-color: #f44336;
        color: white;
    }

    /* Mode train/test */
    .mode-indicator {
        background-color: rgba(109, 186, 188, 0.1);
        border: 2px solid #6DBABC;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        text-align: center;
    }
    
    .mode-title {
        font-weight: bold;
        color: #00485F;
        font-size: 1.2em;
        margin-bottom: 5px;
    }

    /* Limitations et warnings */
    .limitation-box {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .limitation-title {
        font-weight: bold;
        color: #e65100;
        margin-bottom: 8px;
    }

    /* Comparaison train/test */
    .comparison-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin: 15px 0;
    }
    
    .train-card {
        background-color: rgba(150, 185, 29, 0.1);
        border-left: 4px solid #96B91D;
        padding: 15px;
        border-radius: 0 8px 8px 0;
    }
    
    .test-card {
        background-color: rgba(109, 186, 188, 0.1);
        border-left: 4px solid #6DBABC;
        padding: 15px;
        border-radius: 0 8px 8px 0;
    }

    /* Progress bar personnalisée */
    .progress-container {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 20px;
        background: linear-gradient(90deg, #6DBABC 0%, #96B91D 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# TITRE ET DESCRIPTION PRINCIPALE
st.title("📊 Analyse IPMVP Améliorée")
st.markdown("""
Bienvenue dans **l'Analyse IPMVP Améliorée** ! 🚀  
Cette version inclut la **détection d'overfitting**, le **scoring composite**, et la **validation train/test** pour des analyses plus robustes.
""")

# GUIDE D'UTILISATION AMÉLIORÉ
st.markdown("""
<div class="instruction-card">
<h3>🛠️ Guide d'utilisation - Version Améliorée</h3>
<h4>📋 Nouveautés de cette version :</h4>
<ul>
    <li><strong>🛡️ Détection d'overfitting intelligente</strong> : Rejet automatique des modèles avec R² artificiellement gonflé</li>
    <li><strong>🎯 Score composite IPMVP</strong> : Sélection des modèles basée sur un score 0-100 points (R² + CV(RMSE) + simplicité + significativité)</li>
    <li><strong>🚀 Mode train/test adaptatif</strong> : Split automatique 18/6 mois si ≥24 mois de données</li>
    <li><strong>⚠️ Limitations sécurité</strong> : Contrôle du ratio observations/variables (règle 10:1)</li>
    <li><strong>📊 Métriques enrichies</strong> : Comparaison train/test, valeurs t de Student, warnings intelligents</li>
</ul>

<h4>🔄 Flux d'analyse intelligent :</h4>
<ol>
    <li><strong>Validation des données</strong> : Vérification qualité, détection anomalies</li>
    <li><strong>Mode adaptatif</strong> : 
        <ul>
            <li>≥24 mois → Mode "Validation robuste" avec train/test</li>
            <li>12-23 mois → Mode "IPMVP standard" avec protections renforcées</li>
        </ul>
    </li>
    <li><strong>Limitations automatiques</strong> : 
        <ul>
            <li>Variables max = nb_observations ÷ 10</li>
            <li>Polynôme seulement si ≥20 observations</li>
        </ul>
    </li>
    <li><strong>Sélection intelligente</strong> : Score composite privilégiant robustesse + simplicité</li>
    <li><strong>Résultats enrichis</strong> : Métriques avancées, warnings, recommandations</li>
</ol>

<h4>✅ Critères de qualité IPMVP renforcés :</h4>
<ul>
    <li><strong>R² ≥ 0.75</strong> : Corrélation excellente</li>
    <li><strong>CV(RMSE) ≤ 15%</strong> : Précision excellente</li>
    <li><strong>|Biais| < 5%</strong> : Ajustement équilibré</li>
    <li><strong>Variables significatives</strong> : |t| > 2 (p-value < 0.05)</li>
    <li><strong>Ratio obs/var ≥ 10:1</strong> : Données suffisantes</li>
</ul>
</div>
""", unsafe_allow_html=True)

# IMPORT DE FICHIER ET BOUTON DE CALCUL
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Importer un fichier Excel", type=["xlsx", "xls"])

with col2:
    lancer_calcul = st.button("🚀 Lancer l'analyse", use_container_width=True)

# TRAITEMENT DU FICHIER AVEC VALIDATION AMÉLIORÉE
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Détection automatique des colonnes
        date_col_guess, conso_col_guess = detecter_colonnes(df)
        
        # Messages d'information améliorés
        if date_col_guess and conso_col_guess:
            st.success(f"✅ **Détection automatique réussie**")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📅 **Date** : '{date_col_guess}'")
            with col2:
                st.info(f"⚡ **Consommation** : '{conso_col_guess}'")
        elif date_col_guess:
            st.info(f"📅 Colonne de date détectée : '{date_col_guess}'")
            st.warning("⚠️ Veuillez sélectionner manuellement la colonne de consommation.")
        elif conso_col_guess:
            st.info(f"⚡ Colonne de consommation détectée : '{conso_col_guess}'")
            st.warning("⚠️ Veuillez sélectionner manuellement la colonne de date.")
        else:
            st.error("❌ **Détection automatique échouée** - Sélection manuelle requise")
            
        # Affichage des informations sur le fichier
        st.markdown(f"""
        <div class="metrics-card">
            <h4>📊 Informations sur le fichier</h4>
            <ul>
                <li><strong>Nombre de lignes :</strong> {len(df)}</li>
                <li><strong>Nombre de colonnes :</strong> {len(df.columns)}</li>
                <li><strong>Colonnes disponibles :</strong> {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ **Erreur lors du chargement** : {str(e)}")
        df = None
        date_col_guess = None
        conso_col_guess = None
else:
    df = None
    date_col_guess = None
    conso_col_guess = None

# SIDEBAR - SÉLECTION DES DONNÉES AVEC CONTRÔLES ADAPTATIFS
st.sidebar.header("🔍 Configuration de l'analyse")

# Sélection des colonnes
date_col = st.sidebar.selectbox(
    "📅 Colonne de date", 
    df.columns if df is not None else [""],
    index=list(df.columns).index(date_col_guess) if df is not None and date_col_guess in df.columns else 0
)

conso_col = st.sidebar.selectbox(
    "⚡ Colonne de consommation", 
    df.columns if df is not None else [""],
    index=list(df.columns).index(conso_col_guess) if df is not None and conso_col_guess in df.columns else 0
)

# VALIDATION PRÉLIMINAIRE DES DONNÉES
if df is not None and date_col and conso_col:
    # Conversion et validation de base
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        
        # Validation de la qualité des données
        var_options = [col for col in df.columns if col not in [date_col, conso_col]]
        selected_vars = st.sidebar.multiselect("📊 Variables explicatives", var_options)
        
        if selected_vars:
            issues, warnings = validate_data_quality(df, date_col, conso_col, selected_vars)
            
            # Affichage des problèmes critiques
            if issues:
                st.sidebar.markdown("### 🚨 Problèmes détectés")
                for issue in issues:
                    st.sidebar.error(issue)
            
            # Affichage des avertissements
            if warnings:
                st.sidebar.markdown("### ⚠️ Avertissements")
                for warning in warnings:
                    st.sidebar.warning(warning)
                    
            # Vérification des limitations de variables
            if len(selected_vars) > 0:
                nb_obs = len(df)
                var_issues, var_warnings = check_variable_limits(nb_obs, len(selected_vars), "Général")
                
                if var_issues:
                    st.sidebar.markdown("### 🚫 Limitations dépassées")
                    for issue in var_issues:
                        st.sidebar.error(issue)
                        
                if var_warnings:
                    for warning in var_warnings:
                        st.sidebar.warning(warning)
                        
                # Affichage du ratio actuel
                ratio = nb_obs / len(selected_vars) if len(selected_vars) > 0 else float('inf')
                if ratio < 10:
                    status_color = "#f44336" if ratio < 5 else "#ff9800"
                    status_text = "Critique" if ratio < 5 else "Attention"
                else:
                    status_color = "#4caf50"
                    status_text = "Bon"
                    
                st.sidebar.markdown(f"""
                <div style="background-color: rgba(109, 186, 188, 0.1); padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>📊 Ratio Observations/Variables:</strong><br>
                    <span style="color: {status_color}; font-weight: bold; font-size: 1.2em;">{ratio:.1f}:1</span>
                    <span style="color: {status_color};">({status_text})</span><br>
                    <small>Recommandé: ≥10:1 | Minimum: ≥5:1</small>
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.sidebar.error(f"❌ Erreur dans la validation des données : {str(e)}")

# SÉLECTION DE LA PÉRIODE AVEC MODE ADAPTATIF
if df is not None and date_col:
    # Détermination du mode d'analyse
    nb_observations = len(df)
    use_train_test, mode_message = should_use_train_test_split(nb_observations)
    
    # Affichage du mode d'analyse
    if use_train_test:
        st.sidebar.markdown(f"""
        <div class="mode-indicator" style="background-color: rgba(150, 185, 29, 0.1); border-color: #96B91D;">
            <div class="mode-title">🚀 Mode Validation Robuste</div>
            <p>Split train/test automatique (18/6 mois)<br>
            Évaluation sur données non-vues</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div class="mode-indicator" style="background-color: rgba(109, 186, 188, 0.1); border-color: #6DBABC;">
            <div class="mode-title">📋 Mode IPMVP Standard</div>
            <p>Analyse sur toutes les données<br>
            Protections anti-overfitting renforcées</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.info(mode_message)

# SÉLECTION DE PÉRIODE
period_choice = st.sidebar.radio(
    "📅 Sélection de la période",
    ["Rechercher automatiquement la meilleure période de 12 mois", "Sélectionner manuellement une période spécifique"]
)

# Sélection manuelle de période avec validation améliorée
if period_choice == "Sélectionner manuellement une période spécifique" and df is not None and date_col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("📅 Début", 
                                     value=min_date,
                                     min_value=min_date, 
                                     max_value=max_date)
        with col2:
            default_end = min(max_date, (pd.to_datetime(start_date) + pd.DateOffset(months=11)).date())
            end_date = st.date_input("📅 Fin", 
                                   value=default_end,
                                   min_value=start_date, 
                                   max_value=max_date)
        
        # Calcul et validation de la période
        months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        
        # Validation de la période avec messages adaptatifs
        if months_diff < 12:
            st.sidebar.warning(f"⚠️ Période courte: {months_diff} mois (recommandé: ≥12)")
        elif months_diff == 12:
            st.sidebar.success(f"✅ Période IPMVP standard: {months_diff} mois")
        elif months_diff < 24:
            st.sidebar.info(f"ℹ️ Période étendue: {months_diff} mois")
        else:
            st.sidebar.success(f"✅ Période robuste: {months_diff} mois (train/test possible)")

# CONFIGURATION DU MODÈLE AVEC LIMITATIONS DYNAMIQUES
st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Configuration du modèle")

model_type = st.sidebar.selectbox(
    "Type de modèle de régression",
    ["Automatique (score composite)", "Linéaire", "Ridge", "Lasso", "Polynomiale"],
    index=0,
    help="Mode automatique recommandé : teste tous les modèles et sélectionne selon le score composite IPMVP"
)

# Limitations dynamiques selon les données
if df is not None and len(selected_vars) > 0:
    max_vars_safe = len(df) // 10
    max_vars_absolute = len(df) // 5
    
    if max_vars_safe < 1:
        st.sidebar.error("❌ Données insuffisantes pour l'analyse")
        max_features = st.sidebar.slider("🔢 Nombre de variables", 1, 1, 1, disabled=True)
    else:
        max_recommended = min(4, max_vars_safe)
        max_absolute = min(4, max_vars_absolute)
        
        max_features = st.sidebar.slider(
            "🔢 Nombre de variables à tester", 
            1, 
            max_absolute, 
            min(2, max_recommended),
            help=f"Recommandé: ≤{max_recommended} | Maximum absolu: {max_absolute}"
        )
        
        # Warning si au-dessus du seuil recommandé
        if max_features > max_recommended:
            st.sidebar.warning(f"⚠️ Au-dessus du seuil recommandé ({max_recommended})")
else:
    max_features = st.sidebar.slider("🔢 Nombre de variables à tester", 1, 4, 2)

# Paramètres spécifiques aux modèles avec validation
if model_type == "Ridge":
    alpha_ridge = st.sidebar.slider("Alpha (régularisation Ridge)", 0.01, 10.0, 1.0, 0.01)
elif model_type == "Lasso":
    alpha_lasso = st.sidebar.slider("Alpha (régularisation Lasso)", 0.01, 1.0, 0.1, 0.01)
elif model_type == "Polynomiale":
    # Vérification des limitations pour polynôme
    if df is not None and len(df) < 20:
        st.sidebar.error("❌ Modèle polynomial nécessite ≥20 observations")
        poly_degree = st.sidebar.slider("Degré du polynôme", 2, 2, 2, disabled=True)
    else:
        poly_degree = st.sidebar.slider("Degré du polynôme", 2, 3, 2)
        if df is not None and len(df) < 30:
            st.sidebar.warning("⚠️ Recommandation: ≥30 observations pour polynôme stable")

# INFORMATIONS SUR LA CONFORMITÉ IPMVP ENRICHIES
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### ✅ Critères IPMVP Améliorés
{tooltip("Score Composite", "Le nouveau système évalue les modèles sur un score 0-100 points combinant performance statistique, conformité IPMVP et simplicité. Fini le tri par R² seul !")}

**📊 Critères principaux :**
- **R² ≥ 0.75** : Corrélation excellente
- **CV(RMSE) ≤ 15%** : Précision excellente  
- **|Biais| < 5%** : Ajustement équilibré

**🎯 Nouveaux critères :**
- **Significativité** : |t| > 2 (p-value < 0.05)
- **Ratio obs/var** : ≥10:1 (protection overfitting)
- **Simplicité** : Moins de variables = meilleur score
""", unsafe_allow_html=True)

# INFORMATIONS SUR LES MODÈLES AVEC AMÉLIORATIONS
st.sidebar.markdown(f"""
### 🧮 Modèles disponibles

**🔄 Mode automatique (recommandé)**
- Teste tous les types de modèles
- Sélection par **score composite IPMVP**
- Ridge/Lasso retrouvent leur utilité !

**📈 Modèles individuels**
- {tooltip("Linéaire", "Modèle standard IPMVP. Relation linéaire simple et interprétable.")}
- {tooltip("Ridge", "Régularisation L2. Réduit l'overfitting, garde toutes les variables.")}
- {tooltip("Lasso", "Régularisation L1. Peut éliminer des variables non pertinentes.")}
- {tooltip("Polynomiale", "Relations non-linéaires. Attention au risque d'overfitting !")}
""", unsafe_allow_html=True)

# GESTION DU COMPTE UTILISATEUR DANS LA SIDEBAR
st.sidebar.markdown("---")
st.sidebar.header("👤 Gestion du compte")
st.sidebar.markdown(f"**Connecté :** {st.session_state['username']}")

# Panel d'administration pour les admins
if st.session_state.get('is_admin', False):
    st.sidebar.markdown("#### 🔐 Administration")
    if st.sidebar.button("👥 Gérer les utilisateurs", use_container_width=True):
        st.session_state['show_admin'] = True
        st.rerun()

# Changement de mot de passe
st.sidebar.markdown("#### 🔑 Sécurité")
with st.sidebar.expander("Changer de mot de passe"):
    with st.form("change_password_form"):
        current_password = st.text_input("Mot de passe actuel", type="password")
        new_password = st.text_input("Nouveau mot de passe", type="password")
        confirm_password = st.text_input("Confirmer le mot de passe", type="password")
        submit_password = st.form_submit_button("🔄 Modifier", use_container_width=True)
        
        if submit_password:
            if not check_credentials(st.session_state['username'], current_password):
                st.error("❌ Mot de passe actuel incorrect")
            elif new_password != confirm_password:
                st.error("❌ Les mots de passe ne correspondent pas")
            elif not new_password:
                st.error("❌ Le nouveau mot de passe ne peut pas être vide")
            else:
                update_user(st.session_state['username'], new_password)
                st.success("✅ Mot de passe modifié !")

# Bouton de déconnexion
if st.sidebar.button("🚪 Déconnexion", key="sidebar_logout", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# PIED DE PAGE AMÉLIORÉ
st.markdown("---")
st.markdown("""
<div class="footer-credit">
    <p><strong>📊 Analyse IPMVP Améliorée v2.0</strong></p>
    <p>✨ <strong>Nouveautés :</strong> Détection overfitting • Score composite • Train/Test split • Limitations sécurité</p>
    <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
</div>
""", unsafe_allow_html=True)
# =============================================================================
# PARTIE 4 : CALCUL ET RÉSULTATS
# Algorithme de calcul principal avec train/test et affichage des résultats
# =============================================================================

# LANCEMENT DU CALCUL PRINCIPAL AVEC AMÉLIORATIONS IPMVP
if df is not None and lancer_calcul and selected_vars:
    
    # Vérifications préliminaires
    if not date_col or not conso_col:
        st.error("❌ **Veuillez sélectionner les colonnes de date et de consommation**")
        st.stop()
    
    if not selected_vars:
        st.error("❌ **Veuillez sélectionner au moins une variable explicative**")
        st.stop()
    
    # Initialisation
    st.subheader("⚙️ Analyse en cours...")

    all_models = []
    
    # Conversion et tri des données
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
    except Exception as e:
        st.error(f"❌ **Erreur conversion date** : {str(e)}")
        st.stop()
    
    # OPTION 1: RECHERCHE AUTOMATIQUE DE LA MEILLEURE PÉRIODE
    if period_choice == "Rechercher automatiquement la meilleure période de 12 mois":
        
        # Génération des périodes candidates
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
            st.error("❌ **Données insuffisantes** pour une analyse sur 12 mois")
            st.stop()
        
        # Barre de progression améliorée
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            col1, col2, col3 = st.columns(3)
            with col1:
                current_period = st.empty()
            with col2:
                current_score = st.empty()
            with col3:
                best_so_far = st.empty()
        
        # Variables pour le meilleur modèle
        best_period_data = None
        best_period_model = None
        best_period_features = None
        best_period_metrics = None
        best_period_score = -1
        best_period_name = None
        
        # Analyse de chaque période
        for idx, (period_name, period_start, period_end) in enumerate(date_ranges):
            current_period.info(f"📅 **{period_name}**")
            progress_text.text(f"Analyse période {idx+1}/{len(date_ranges)}")
            
            # Filtrer les données
            period_df = df[(df[date_col] >= period_start) & (df[date_col] <= period_end)]
            
            if len(period_df) < 10:
                continue
            
            # Déterminer le mode d'analyse pour cette période
            use_train_test, _ = should_use_train_test_split(len(period_df))
            
            # Préparation des données
            X = period_df[selected_vars]
            y = period_df[conso_col]
            
            # Nettoyage des données
            if X.isnull().values.any() or np.isinf(X.values).any():
                continue
            if y.isnull().values.any() or np.isinf(y.values).any():
                continue
            
            X = X.apply(pd.to_numeric, errors='coerce').dropna()
            y = pd.to_numeric(y, errors='coerce').dropna()
            
            # Validation des limitations de sécurité
            var_issues, _ = check_variable_limits(len(period_df), len(selected_vars), model_type)
            if var_issues:
                continue
            
            period_best_score = -1
            period_best_model = None
            
            # Test des combinaisons de variables
            for n in range(1, min(max_features + 1, len(selected_vars) + 1)):
                for combo in combinations(selected_vars, n):
                    X_subset = X[list(combo)]
                    
                    # Split train/test si applicable
                    if use_train_test and len(period_df) >= 24:
                        train_df, test_df, split_date = create_train_test_split(period_df, date_col)
                        X_train = train_df[list(combo)]
                        y_train = train_df[conso_col]
                        X_test = test_df[list(combo)]
                        y_test = test_df[conso_col]
                        
                        # Nettoyage train/test
                        X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
                        y_train = pd.to_numeric(y_train, errors='coerce').dropna()
                        X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
                        y_test = pd.to_numeric(y_test, errors='coerce').dropna()
                        
                        if len(X_train) < 5 or len(X_test) < 3:
                            continue
                    else:
                        X_train, y_train = X_subset, y
                        X_test, y_test = None, None
                    
                    # Types de modèles à tester
                    if model_type == "Automatique (score composite)":
                        model_types_to_test = [
                            ("Linéaire", LinearRegression(), "Régression linéaire"),
                            ("Ridge", Ridge(alpha=1.0), "Régression Ridge (α=1.0)"),
                            ("Lasso", Lasso(alpha=0.1), "Régression Lasso (α=0.1)")
                        ]
                        
                        # Ajouter polynôme seulement si sécurisé
                        if len(period_df) >= 20:
                            model_types_to_test.append((
                                "Polynomiale", 
                                Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
                                "Régression polynomiale (degré 2)"
                            ))
                    else:
                        # Modèle spécifique sélectionné
                        if model_type == "Linéaire":
                            model_obj = LinearRegression()
                            model_name = "Régression linéaire"
                        elif model_type == "Ridge":
                            model_obj = Ridge(alpha=alpha_ridge)
                            model_name = f"Régression Ridge (α={alpha_ridge})"
                        elif model_type == "Lasso":  
                            model_obj = Lasso(alpha=alpha_lasso)
                            model_name = f"Régression Lasso (α={alpha_lasso})"
                        elif model_type == "Polynomiale":
                            if len(period_df) < 20:
                                continue  # Skip si pas assez d'observations
                            model_obj = Pipeline([('poly', PolynomialFeatures(degree=poly_degree)), ('linear', LinearRegression())])
                            model_name = f"Régression polynomiale (degré {poly_degree})"
                        
                        model_types_to_test = [(model_type, model_obj, model_name)]
                    
                    # Test de chaque type de modèle
                    for m_type, m_obj, m_name in model_types_to_test:
                        try:
                            # Entraînement du modèle
                            m_obj.fit(X_train, y_train)
                            
                            # Prédictions et métriques
                            if X_test is not None and y_test is not None:
                                # Mode train/test
                                y_pred_train = m_obj.predict(X_train)
                                y_pred_test = m_obj.predict(X_test)
                                
                                # Métriques sur le test set (priorité)
                                r2_test = r2_score(y_test, y_pred_test)
                                rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_test))
                                cv_rmse_test = rmse_test / np.mean(y_test) if np.mean(y_test) != 0 else float('inf')
                                bias_test = np.mean(y_pred_test - y_test) / np.mean(y_test) * 100
                                
                                # Métriques sur le train set
                                r2_train = r2_score(y_train, y_pred_train)
                                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_train))
                                cv_rmse_train = rmse_train / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                                bias_train = np.mean(y_pred_train - y_train) / np.mean(y_train) * 100
                                
                                # Utiliser les métriques de test pour l'évaluation
                                r2, cv_rmse, bias = r2_test, cv_rmse_test, bias_test
                                mae = mean_absolute_error(y_test, y_pred_test)
                                
                                # Détection d'overfitting par comparaison train/test
                                overfitting_detected = False
                                if abs(r2_train - r2_test) > 0.2:  # Écart R² > 20%
                                    overfitting_detected = True
                                if cv_rmse_test > cv_rmse_train * 1.5:  # CV(RMSE) test >> train
                                    overfitting_detected = True
                                    
                            else:
                                # Mode standard (toutes les données)
                                y_pred = m_obj.predict(X_train)
                                r2 = r2_score(y_train, y_pred)
                                
                                # Calcul RMSE corrigé selon IPMVP
                                n = len(y_train)
                                p = len(combo)
                                ssr = np.sum((y_train - y_pred) ** 2)
                                df_res = n - p - 1 if (n - p - 1) > 0 else 1
                                rmse = math.sqrt(ssr / df_res)
                                
                                cv_rmse = rmse / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                                bias = np.mean(y_pred - y_train) / np.mean(y_train) * 100
                                mae = mean_absolute_error(y_train, y_pred)
                                
                                overfitting_detected = False
                                r2_train = r2_test = r2
                                cv_rmse_train = cv_rmse_test = cv_rmse
                                bias_train = bias_test = bias
                            
                            # Détection d'overfitting intelligent
                            model_info_temp = {
                                'r2': r2,
                                'cv_rmse': cv_rmse,
                                'bias': bias,
                                'features': list(combo),
                                'model_type': m_type
                            }
                            
                            is_overfitted, warning_msg, severity = detect_overfitting_intelligent(model_info_temp, len(period_df))
                            
                            # Rejeter si overfitting détecté
                            if is_overfitted and severity == "error":
                                continue
                            
                            # Récupération des coefficients
                            if m_type in ["Linéaire", "Ridge", "Lasso"]:
                                coefs = {feature: coef for feature, coef in zip(combo, m_obj.coef_)}
                                intercept = m_obj.intercept_
                            elif m_type == "Polynomiale":
                                linear_model = m_obj.named_steps['linear']
                                poly = m_obj.named_steps['poly']
                                feature_names = poly.get_feature_names_out(input_features=combo)
                                coefs = {name: coef for name, coef in zip(feature_names, linear_model.coef_)}
                                intercept = linear_model.intercept_
                            
                            # Calcul des valeurs t
                            t_stats = calculate_t_stats(X_train, y_train, m_obj, coefs) if m_type in ["Linéaire", "Ridge", "Lasso"] else {feature: None for feature in combo}
                            
                            # Évaluation conformité IPMVP
                            conformite, classe = evaluer_conformite(r2, cv_rmse, bias)
                            
                            # Création du modèle info
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
                                'model_type': m_type,
                                'model_name': m_name,
                                'period': period_name,
                                't_stats': t_stats,
                                'overfitting_warning': warning_msg if not is_overfitted else "",
                                'overfitting_severity': severity if not is_overfitted else ""
                            }
                            
                            # Ajouter métriques train/test si disponibles
                            if X_test is not None:
                                model_info.update({
                                    'train_r2': r2_train,
                                    'test_r2': r2_test,
                                    'train_cv_rmse': cv_rmse_train,
                                    'test_cv_rmse': cv_rmse_test,
                                    'train_bias': bias_train,
                                    'test_bias': bias_test,
                                    'overfitting_train_test': overfitting_detected,
                                    'mode': 'train_test'
                                })
                            else:
                                model_info['mode'] = 'standard'
                            
                            # Calcul du score composite IPMVP
                            ipmvp_score = calculate_ipmvp_score(model_info, len(period_df))
                            model_info['ipmvp_score'] = ipmvp_score
                            
                            all_models.append(model_info)
                            
                            # Mise à jour du meilleur modèle selon le score composite
                            if ipmvp_score > period_best_score:
                                period_best_score = ipmvp_score
                                period_best_model = model_info
                            
                            # Mise à jour affichage en temps réel
                            current_score.metric("Score actuel", f"{ipmvp_score:.1f}/100")
                            
                        except Exception as e:
                            continue
            
            # Mise à jour du meilleur modèle global
            if period_best_model and period_best_score > best_period_score:
                best_period_score = period_best_score
                best_period_data = period_df
                best_period_model = period_best_model
                best_period_features = period_best_model['features']
                best_period_metrics = period_best_model
                best_period_name = period_name
                
                best_so_far.metric("Meilleur score", f"{best_period_score:.1f}/100")
            
            # Mise à jour de la barre de progression
            progress_bar.progress((idx + 1) / len(date_ranges))
        
        # Nettoyage de l'affichage de progression
        progress_container.empty()
        
        if best_period_data is not None:
            st.success(f"✅ **Meilleure période trouvée** : {best_period_name}")
            
            # Warning si moins de 12 mois
            if len(best_period_data) < 12:
                st.warning(f"⚠️ **Attention :** Seulement {len(best_period_data)} observations disponibles. L'IPMVP recommande au minimum 12 mois de données pour une baseline fiable.")
            
            # Affichage du score composite
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-value">{best_period_score:.1f}</div>
                    <div class="score-label">Score IPMVP</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.info(f"📅 **Période** : {best_period_name}")
            with col3:
                st.info(f"📊 **Points de données** : {len(best_period_data)}")
            
            # Utiliser les meilleurs résultats
            df_filtered = best_period_data
            best_model_obj = None  # À reconstruire si nécessaire
            best_features = best_period_features
            best_metrics = best_period_metrics
            
        else:
            st.error("❌ **Aucun modèle valide trouvé** sur les périodes analysées")
            st.stop()
    
    # OPTION 2: PÉRIODE SPÉCIFIQUE SÉLECTIONNÉE
    else:
        # Filtrage selon la période sélectionnée
        df_filtered = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
        
        st.info(f"📊 **Analyse sur période sélectionnée** : {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
        
        # Warning si moins de 12 mois
        if len(df_filtered) < 12:
            st.warning(f"⚠️ **Attention :** Seulement {len(df_filtered)} observations disponibles. L'IPMVP recommande au minimum 12 mois de données pour une baseline fiable.")
        
        # Vérification données suffisantes
        if len(df_filtered) < 10:
            st.error("❌ **Données insuffisantes** pour l'analyse (minimum 10 points)")
            st.stop()
        
        # Détermination du mode d'analyse
        use_train_test, mode_message = should_use_train_test_split(len(df_filtered))
        st.info(mode_message)
        
        # Préparation des données
        X = df_filtered[selected_vars]
        y = df_filtered[conso_col]
        
        # Nettoyage des données
        if X.isnull().values.any() or np.isinf(X.values).any():
            st.error("❌ **Variables explicatives** contiennent des valeurs manquantes")
            st.stop()
        
        if y.isnull().values.any() or np.isinf(y.values).any():
            st.error("❌ **Colonne consommation** contient des valeurs manquantes")
            st.stop()
        
        X = X.apply(pd.to_numeric, errors='coerce').dropna()
        y = pd.to_numeric(y, errors='coerce').dropna()
        
        # Variables pour le meilleur modèle
        best_model_obj = None
        best_score = -1
        best_features = []
        best_metrics = {}
        
        # Barre de progression pour l'analyse
        total_combinations = sum(len(list(combinations(selected_vars, n))) for n in range(1, max_features + 1))
        progress_bar = st.progress(0)
        progress_counter = 0
        
        # Test des combinaisons de variables
        for n in range(1, min(max_features + 1, len(selected_vars) + 1)):
            for combo in combinations(selected_vars, n):
                progress_counter += 1
                progress_bar.progress(progress_counter / total_combinations)
                
                X_subset = X[list(combo)]
                
                # Split train/test si applicable
                if use_train_test and len(df_filtered) >= 24:
                    train_df, test_df, split_date = create_train_test_split(df_filtered, date_col)
                    X_train = train_df[list(combo)]
                    y_train = train_df[conso_col]
                    X_test = test_df[list(combo)]
                    y_test = test_df[conso_col]
                    
                    # Nettoyage train/test
                    X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
                    y_train = pd.to_numeric(y_train, errors='coerce').dropna()
                    X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
                    y_test = pd.to_numeric(y_test, errors='coerce').dropna()
                    
                    if len(X_train) < 5 or len(X_test) < 3:
                        continue
                else:
                    X_train, y_train = X_subset, y
                    X_test, y_test = None, None
                
                # Types de modèles à tester (même logique que précédemment)
                if model_type == "Automatique (score composite)":
                    model_types_to_test = [
                        ("Linéaire", LinearRegression(), "Régression linéaire"),
                        ("Ridge", Ridge(alpha=1.0), "Régression Ridge (α=1.0)"),
                        ("Lasso", Lasso(alpha=0.1), "Régression Lasso (α=0.1)")
                    ]
                    
                    if len(df_filtered) >= 20:
                        model_types_to_test.append((
                            "Polynomiale", 
                            Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
                            "Régression polynomiale (degré 2)"
                        ))
                else:
                    # Même logique que précédemment pour les modèles spécifiques
                    if model_type == "Linéaire":
                        model_obj = LinearRegression()
                        model_name = "Régression linéaire"
                    elif model_type == "Ridge":
                        model_obj = Ridge(alpha=alpha_ridge)
                        model_name = f"Régression Ridge (α={alpha_ridge})"
                    elif model_type == "Lasso":
                        model_obj = Lasso(alpha=alpha_lasso)
                        model_name = f"Régression Lasso (α={alpha_lasso})"
                    elif model_type == "Polynomiale":
                        if len(df_filtered) < 20:
                            continue
                        model_obj = Pipeline([('poly', PolynomialFeatures(degree=poly_degree)), ('linear', LinearRegression())])
                        model_name = f"Régression polynomiale (degré {poly_degree})"
                    
                    model_types_to_test = [(model_type, model_obj, model_name)]
                
                # Test de chaque type de modèle (même logique que l'analyse par période)
                for m_type, m_obj, m_name in model_types_to_test:
                    try:
                        # Entraînement et évaluation (même code que précédemment)
                        m_obj.fit(X_train, y_train)
                        
                        # Calcul des métriques (même logique)
                        if X_test is not None and y_test is not None:
                            # Mode train/test
                            y_pred_train = m_obj.predict(X_train)
                            y_pred_test = m_obj.predict(X_test)
                            
                            r2_test = r2_score(y_test, y_pred_test)
                            rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_test))
                            cv_rmse_test = rmse_test / np.mean(y_test) if np.mean(y_test) != 0 else float('inf')
                            bias_test = np.mean(y_pred_test - y_test) / np.mean(y_test) * 100
                            
                            r2_train = r2_score(y_train, y_pred_train)
                            rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_train))
                            cv_rmse_train = rmse_train / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                            bias_train = np.mean(y_pred_train - y_train) / np.mean(y_train) * 100
                            
                            r2, cv_rmse, bias = r2_test, cv_rmse_test, bias_test
                            mae = mean_absolute_error(y_test, y_pred_test)
                            rmse = rmse_test
                            
                            overfitting_detected = False
                            if abs(r2_train - r2_test) > 0.2 or cv_rmse_test > cv_rmse_train * 1.5:
                                overfitting_detected = True
                                
                        else:
                            # Mode standard
                            y_pred = m_obj.predict(X_train)
                            r2 = r2_score(y_train, y_pred)
                            
                            n = len(y_train)
                            p = len(combo)
                            ssr = np.sum((y_train - y_pred) ** 2)
                            df_res = n - p - 1 if (n - p - 1) > 0 else 1
                            rmse = math.sqrt(ssr / df_res)
                            
                            cv_rmse = rmse / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                            bias = np.mean(y_pred - y_train) / np.mean(y_train) * 100
                            mae = mean_absolute_error(y_train, y_pred)
                            
                            overfitting_detected = False
                            r2_train = r2_test = r2
                            cv_rmse_train = cv_rmse_test = cv_rmse
                            bias_train = bias_test = bias
                        
                        # Détection d'overfitting et rejet si nécessaire
                        model_info_temp = {
                            'r2': r2,
                            'cv_rmse': cv_rmse,
                            'bias': bias,
                            'features': list(combo),
                            'model_type': m_type
                        }
                        
                        is_overfitted, warning_msg, severity = detect_overfitting_intelligent(model_info_temp, len(df_filtered))
                        
                        if is_overfitted and severity == "error":
                            continue
                        
                        # Récupération des coefficients (même logique)
                        if m_type in ["Linéaire", "Ridge", "Lasso"]:
                            coefs = {feature: coef for feature, coef in zip(combo, m_obj.coef_)}
                            intercept = m_obj.intercept_
                        elif m_type == "Polynomiale":
                            linear_model = m_obj.named_steps['linear']
                            poly = m_obj.named_steps['poly']
                            feature_names = poly.get_feature_names_out(input_features=combo)
                            coefs = {name: coef for name, coef in zip(feature_names, linear_model.coef_)}
                            intercept = linear_model.intercept_
                        
                        # Calcul des valeurs t
                        t_stats = calculate_t_stats(X_train, y_train, m_obj, coefs) if m_type in ["Linéaire", "Ridge", "Lasso"] else {feature: None for feature in combo}
                        
                        # Conformité IPMVP
                        conformite, classe = evaluer_conformite(r2, cv_rmse, bias)
                        
                        # Création du modèle info complet
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
                            'model_type': m_type,
                            'model_name': m_name,
                            'period': 'selected',
                            't_stats': t_stats,
                            'overfitting_warning': warning_msg if not is_overfitted else "",
                            'overfitting_severity': severity if not is_overfitted else ""
                        }
                        
                        # Métriques train/test
                        if X_test is not None:
                            model_info.update({
                                'train_r2': r2_train,
                                'test_r2': r2_test,
                                'train_cv_rmse': cv_rmse_train,
                                'test_cv_rmse': cv_rmse_test,
                                'train_bias': bias_train,
                                'test_bias': bias_test,
                                'overfitting_train_test': overfitting_detected,
                                'mode': 'train_test'
                            })
                        else:
                            model_info['mode'] = 'standard'
                        
                        # Score composite IPMVP
                        ipmvp_score = calculate_ipmvp_score(model_info, len(df_filtered))
                        model_info['ipmvp_score'] = ipmvp_score
                        
                        all_models.append(model_info)
                        
                        # Mise à jour du meilleur modèle
                        if ipmvp_score > best_score:
                            best_score = ipmvp_score
                            best_model_obj = m_obj
                            best_features = list(combo)
                            best_metrics = model_info
                        
                    except Exception as e:
                        continue
        
        progress_bar.empty()
    
    # TRI DES MODÈLES PAR SCORE COMPOSITE (PAS PAR R² !)
    all_models.sort(key=lambda x: x['ipmvp_score'], reverse=True)

    # AFFICHAGE DES RÉSULTATS AVEC AMÉLIORATIONS
    if best_metrics:
        st.success("✅ **Analyse terminée avec succès !**")
        
        # SCORE COMPOSITE ET INFORMATIONS PRINCIPALES
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-value">{best_metrics['ipmvp_score']:.1f}</div>
                <div class="score-label">Score IPMVP</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            conformity_class = f"status-{best_metrics['classe']}" if best_metrics['classe'] != 'medium' else "status-warning"
            st.markdown(f"""
            <div class="metrics-card" style="text-align: center;">
                <h4>Conformité IPMVP</h4>
                <span class="status-badge {conformity_class}">{best_metrics['conformite']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mode_info = "🚀 Train/Test" if best_metrics.get('mode') == 'train_test' else "📋 Standard"
            st.markdown(f"""
            <div class="metrics-card" style="text-align: center;">
                <h4>Mode d'analyse</h4>
                <p><strong>{mode_info}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metrics-card" style="text-align: center;">
                <h4>Modèle sélectionné</h4>
                <span class="model-badge">{best_metrics['model_name']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # ALERTES OVERFITTING SI DÉTECTÉES
        if best_metrics.get('overfitting_warning'):
            severity = best_metrics.get('overfitting_severity', 'warning')
            alert_class = f"alert-{severity}" if severity in ['error', 'warning', 'info'] else "alert-warning"
            st.markdown(f"""
            <div class="alert-card {alert_class}">
                <strong>⚠️ Attention :</strong> {best_metrics['overfitting_warning']}
            </div>
            """, unsafe_allow_html=True)
        
        # MÉTRIQUES PRINCIPALES AVEC COMPARAISON TRAIN/TEST
        st.subheader("📊 Métriques détaillées")
        
        if best_metrics.get('mode') == 'train_test':
            # Affichage train/test côte à côte
            st.markdown("""
            <div class="comparison-grid">
                <div class="train-card">
                    <h4>🎯 Entraînement (18 mois)</h4>
                </div>
                <div class="test-card">
                    <h4>🧪 Test (6 mois)</h4>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                train_metrics = f"""
                <table class="stats-table">
                    <tr><th>Métrique</th><th>Valeur Train</th></tr>
                    <tr><td>{tooltip("R²", "Coefficient de détermination sur les données d'entraînement")}</td><td>{best_metrics.get('train_r2', 0):.4f}</td></tr>
                    <tr><td>{tooltip("CV(RMSE)", "Coefficient de variation du RMSE sur l'entraînement")}</td><td>{best_metrics.get('train_cv_rmse', 0):.4f}</td></tr>
                    <tr><td>{tooltip("Biais (%)", "Erreur systématique en pourcentage sur l'entraînement")}</td><td>{best_metrics.get('train_bias', 0):.2f}</td></tr>
                </table>
                """
                st.markdown(train_metrics, unsafe_allow_html=True)
            
            with col2:
                test_metrics = f"""
                <table class="stats-table">
                    <tr><th>Métrique</th><th>Valeur Test</th></tr>
                    <tr><td>{tooltip("R²", "Coefficient de détermination sur les données de test (validation)")}</td><td>{best_metrics['r2']:.4f}</td></tr>
                    <tr><td>{tooltip("CV(RMSE)", "Coefficient de variation du RMSE sur le test")}</td><td>{best_metrics['cv_rmse']:.4f}</td></tr>
                    <tr><td>{tooltip("Biais (%)", "Erreur systématique en pourcentage sur le test")}</td><td>{best_metrics['bias']:.2f}</td></tr>
                </table>
                """
                st.markdown(test_metrics, unsafe_allow_html=True)
            
            # Analyse des écarts train/test
            r2_gap = abs(best_metrics.get('train_r2', 0) - best_metrics['r2'])
            cv_gap = abs(best_metrics.get('train_cv_rmse', 0) - best_metrics['cv_rmse'])
            
            if best_metrics.get('overfitting_train_test'):
                st.warning(f"⚠️ **Écart train/test détecté** : R² gap = {r2_gap:.3f}, CV(RMSE) gap = {cv_gap:.3f}")
            else:
                st.info(f"✅ **Bonne stabilité train/test** : R² gap = {r2_gap:.3f}, CV(RMSE) gap = {cv_gap:.3f}")
        
        else:
            # Affichage standard
            col1, col2 = st.columns(2)
            
            with col1:
                standard_metrics = f"""
                <table class="stats-table">
                    <tr><th>Métrique</th><th>Valeur</th></tr>
                    <tr><td>{tooltip("R²", "Coefficient de détermination : proportion de variance expliquée par le modèle")}</td><td>{best_metrics['r2']:.4f}</td></tr>
                    <tr><td>{tooltip("RMSE", "Root Mean Square Error : écart-type des résidus")}</td><td>{best_metrics['rmse']:.4f}</td></tr>
                    <tr><td>{tooltip("CV(RMSE)", "Coefficient de variation du RMSE en pourcentage de la moyenne")}</td><td>{best_metrics['cv_rmse']:.4f}</td></tr>
                    <tr><td>{tooltip("MAE", "Mean Absolute Error : erreur absolue moyenne")}</td><td>{best_metrics['mae']:.4f}</td></tr>
                    <tr><td>{tooltip("Biais (%)", "Erreur systématique du modèle en pourcentage")}</td><td>{best_metrics['bias']:.2f}</td></tr>
                </table>
                """
                st.markdown(standard_metrics, unsafe_allow_html=True)
            
            with col2:
                # Informations sur le modèle
                model_info_html = f"""
                <div class="metrics-card">
                    <h4>🔍 Informations du modèle</h4>
                    <p><strong>Variables utilisées :</strong> {', '.join(best_features)}</p>
                    <p><strong>Nombre de variables :</strong> {len(best_features)}</p>
                    <p><strong>Observations :</strong> {len(df_filtered)}</p>
                    <p><strong>Ratio obs/var :</strong> {len(df_filtered)/len(best_features):.1f}:1</p>
                </div>
                """
                st.markdown(model_info_html, unsafe_allow_html=True)
        
        # ÉQUATION DU MODÈLE
        st.subheader("📝 Équation d'ajustement")
        
        if best_metrics['model_type'] in ["Linéaire", "Ridge", "Lasso"]:
            equation = format_equation(best_metrics['intercept'], 
                                     {feature: best_metrics['coefficients'][feature] for feature in best_features})
        elif best_metrics['model_type'] == "Polynomiale":
            equation = format_equation(best_metrics['intercept'], best_metrics['coefficients'])
        
        st.markdown(f"""
        <div class="equation-box">
            <h4>🧮 Équation mathématique :</h4>
            <p style="font-size: 16px; font-weight: bold;">{equation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VALEURS T DE STUDENT POUR MODÈLES LINÉAIRES
        if 't_stats' in best_metrics and best_metrics['model_type'] in ["Linéaire", "Ridge", "Lasso"]:
            st.subheader("📈 Analyse de significativité statistique")
            
            # Construction du tableau de significativité avec st.dataframe (natif Streamlit)
            st.subheader("📈 Analyse de significativité statistique")
            
            significant_count = 0
            total_count = 0
            sig_data = []
            
            for feature in best_features:
                coef = best_metrics['coefficients'][feature]
                
                if feature in best_metrics['t_stats'] and best_metrics['t_stats'][feature] is not None:
                    t_stat = best_metrics['t_stats'][feature]
                    
                    if isinstance(t_stat, dict):
                        t_value = t_stat.get('t_value', 0)
                        p_value = t_stat.get('p_value', 1)
                        significant = t_stat.get('significant', False)
                    else:
                        t_value = t_stat
                        p_value = "N/A"
                        significant = abs(t_value) > 2
                    
                    total_count += 1
                    if significant:
                        significant_count += 1
                    
                    sig_data.append({
                        "Variable": feature,
                        "Coefficient": round(coef, 4),
                        "Valeur t": round(t_value, 3) if isinstance(t_value, (int, float)) else t_value,
                        "p-value": round(p_value, 4) if isinstance(p_value, (int, float)) else p_value,
                        "Significatif": "✅ Oui" if significant else "❌ Non"
                    })
                else:
                    sig_data.append({
                        "Variable": feature,
                        "Coefficient": round(coef, 4),
                        "Valeur t": "N/A",
                        "p-value": "N/A",
                        "Significatif": "N/A"
                    })
            
            # Affichage avec st.dataframe (natif, toujours fonctionnel)
            st.dataframe(
                sig_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", help="Variable explicative"),
                    "Coefficient": st.column_config.NumberColumn("Coefficient", help="Coefficient de régression", format="%.4f"),
                    "Valeur t": st.column_config.TextColumn("Valeur t", help="Statistique t de Student"),
                    "p-value": st.column_config.TextColumn("p-value", help="Probabilité associée"),
                    "Significatif": st.column_config.TextColumn("Significatif", help="Significatif si |t| > 2 (p < 0.05)")
                }
            )
            
            # Résumé de la significativité
            if total_count > 0:
                sig_percentage = (significant_count / total_count) * 100
                if sig_percentage >= 100:
                    st.success(f"✅ **Excellente significativité** : {significant_count}/{total_count} variables significatives ({sig_percentage:.0f}%)")
                elif sig_percentage >= 70:
                    st.info(f"✅ **Bonne significativité** : {significant_count}/{total_count} variables significatives ({sig_percentage:.0f}%)")
                else:
                    st.warning(f"⚠️ **Significativité limitée** : {significant_count}/{total_count} variables significatives ({sig_percentage:.0f}%)")
        
        # VISUALISATIONS AMÉLIORÉES
        st.subheader("📊 Visualisations")
        
        # Préparation des données pour les graphiques
        if best_metrics.get('mode') == 'train_test':
            # Reconstituer les prédictions train/test
            train_df, test_df, split_date = create_train_test_split(df_filtered, date_col)
            
            # Reconstruire le modèle pour les prédictions
            if best_metrics['model_type'] == "Linéaire":
                model_for_viz = LinearRegression()
            elif best_metrics['model_type'] == "Ridge":
                model_for_viz = Ridge(alpha=1.0)
            elif best_metrics['model_type'] == "Lasso":
                model_for_viz = Lasso(alpha=0.1)
            elif best_metrics['model_type'] == "Polynomiale":
                model_for_viz = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
            
            X_train = train_df[best_features]
            y_train = train_df[conso_col]
            X_test = test_df[best_features]
            y_test = test_df[conso_col]
            
            model_for_viz.fit(X_train, y_train)
            y_pred_train = model_for_viz.predict(X_train)
            y_pred_test = model_for_viz.predict(X_test)
            
            # Concaténation pour l'affichage
            X_all = pd.concat([X_train, X_test])
            y_all = pd.concat([y_train, y_test])
            y_pred_all = np.concatenate([y_pred_train, y_pred_test])
            
            # Marqueurs pour train/test
            train_indices = range(len(y_train))
            test_indices = range(len(y_train), len(y_train) + len(y_test))
            
        else:
            # Mode standard
            X_all = df_filtered[best_features]
            y_all = df_filtered[conso_col]
            
            # Reconstruction du modèle
            if best_metrics['model_type'] == "Linéaire":
                model_for_viz = LinearRegression()
            elif best_metrics['model_type'] == "Ridge":
                model_for_viz = Ridge(alpha=1.0)
            elif best_metrics['model_type'] == "Lasso":
                model_for_viz = Lasso(alpha=0.1)
            elif best_metrics['model_type'] == "Polynomiale":
                model_for_viz = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
            
            model_for_viz.fit(X_all, y_all)
            y_pred_all = model_for_viz.predict(X_all)
            
            train_indices = range(len(y_all))
            test_indices = []
        
        # Configuration matplotlib avec thème IPMVP
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'axes.facecolor': '#F5F5F5',
            'figure.facecolor': '#E7DDD9',
            'axes.edgecolor': '#00485F',
            'axes.labelcolor': '#00485F',
            'axes.titlecolor': '#00485F',
            'xtick.color': '#0C1D2D',
            'ytick.color': '#0C1D2D',
            'grid.color': '#00485F',
            'grid.alpha': 0.1
        })
        
        # GRAPHIQUE 1: Comparaison temporelle avec distinction train/test
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if best_metrics.get('mode') == 'train_test':
            # Affichage train
            ax.bar(train_indices, y_all.iloc[train_indices], color="#96B91D", alpha=0.7, label="Consommation mesurée (Train)", width=0.8)
            ax.plot(train_indices, y_pred_all[train_indices], color="#2E7D32", marker='o', linewidth=2.5, markersize=5, label="Consommation ajustée (Train)")
            
            # Affichage test
            ax.bar(test_indices, y_all.iloc[test_indices], color="#6DBABC", alpha=0.7, label="Consommation mesurée (Test)", width=0.8)
            ax.plot(test_indices, y_pred_all[test_indices], color="#00485F", marker='s', linewidth=2.5, markersize=5, label="Consommation ajustée (Test)")
            
            # Ligne de séparation
            if len(train_indices) > 0:
                ax.axvline(x=max(train_indices), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Séparation Train/Test')
            
            title_suffix = f" (Train: {len(train_indices)} pts, Test: {len(test_indices)} pts)"
        else:
            ax.bar(train_indices, y_all, color="#6DBABC", alpha=0.8, label="Consommation mesurée")
            ax.plot(train_indices, y_pred_all, color="#96B91D", marker='o', linewidth=2.5, markersize=4, label="Consommation ajustée")
            title_suffix = f" ({len(train_indices)} points)"
        
        ax.set_title(f"📊 Comparaison Consommation Mesurée vs Ajustée{title_suffix}", fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel("Observations", fontweight='bold', fontsize=12)
        ax.set_ylabel("Consommation", fontweight='bold', fontsize=12)
        ax.legend(frameon=True, facecolor="#E7DDD9", edgecolor="#00485F", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Annotations enrichies
        # Score IPMVP
        ax.annotate(f"Score IPMVP = {best_metrics['ipmvp_score']:.1f}/100", 
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=13, fontweight='bold', color='#00485F',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#E7DDD9", edgecolor="#00485F", alpha=0.9),
                   verticalalignment='top')
        
        # R² et CV(RMSE)
        metrics_text = f"R² = {best_metrics['r2']:.3f} | CV(RMSE) = {best_metrics['cv_rmse']:.3f}"
        if best_metrics.get('mode') == 'train_test':
            metrics_text += "\n(Mesuré sur Test Set)"
        ax.annotate(metrics_text, 
                   xy=(0.02, 0.88), xycoords='axes fraction',
                   fontsize=11, color='#00485F',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#E7DDD9", edgecolor="#6DBABC", alpha=0.85),
                   verticalalignment='top')
        
        # Nombre total de valeurs
        ax.text(0.98, 0.02, f"Total: {len(y_all)} valeurs",
               transform=ax.transAxes,
               fontsize=10, color='#00485F',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#E7DDD9", edgecolor="#6DBABC", alpha=0.8),
               verticalalignment='bottom', horizontalalignment='right')
        
        st.pyplot(fig)
        
        # GRAPHIQUES 2 ET 3: Dispersion et résidus côte à côte
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique de dispersion
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            
            if best_metrics.get('mode') == 'train_test':
                scatter_train = ax2.scatter(y_all.iloc[train_indices], y_pred_all[train_indices], 
                                          color="#96B91D", alpha=0.8, s=60, edgecolor='#2E7D32', linewidth=1, label="Train")
                scatter_test = ax2.scatter(y_all.iloc[test_indices], y_pred_all[test_indices], 
                                         color="#6DBABC", alpha=0.8, s=60, edgecolor='#00485F', linewidth=1, label="Test")
                ax2.legend()
            else:
                scatter = ax2.scatter(y_all, y_pred_all, color="#6DBABC", alpha=0.8, s=60, edgecolor='#00485F', linewidth=1)
            
            # Ligne de référence y=x
            min_val = min(min(y_all), min(y_pred_all))
            max_val = max(max(y_all), max(y_pred_all))
            ax2.plot([min_val, max_val], [min_val, max_val], '--', color='#00485F', linewidth=2, alpha=0.8, label="Référence y=x")
            
            ax2.set_title("📈 Consommation Mesurée vs Prédite", fontweight='bold', fontsize=14)
            ax2.set_xlabel("Consommation Mesurée", fontweight='bold')
            ax2.set_ylabel("Consommation Prédite", fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            # Annotation
            # Annotation enrichie
        if best_metrics.get('mode') == 'train_test':
            metrics_text = f"R² (Test) = {best_metrics['r2']:.4f}\nCV(RMSE) = {best_metrics['cv_rmse']:.3f}"
        else:
            metrics_text = f"R² = {best_metrics['r2']:.4f}\nCV(RMSE) = {best_metrics['cv_rmse']:.3f}"
        ax2.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=11, fontweight='bold', color='#00485F',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E7DDD9", edgecolor="#00485F", alpha=0.8),
                        verticalalignment='top')
            
        st.pyplot(fig2)
        
        with col2:
            # Analyse des résidus
            residus = y_all - y_pred_all
            
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            
            if best_metrics.get('mode') == 'train_test':
                ax3.scatter(train_indices, residus[train_indices], color="#96B91D", alpha=0.8, s=60, 
                           edgecolor='#2E7D32', linewidth=1, label="Résidus Train")
                ax3.scatter(test_indices, residus[test_indices], color="#6DBABC", alpha=0.8, s=60, 
                           edgecolor='#00485F', linewidth=1, label="Résidus Test")
                ax3.legend()
            else:
                ax3.scatter(range(len(residus)), residus, color="#96B91D", alpha=0.8, s=60, 
                           edgecolor='#2E7D32', linewidth=1)
            
            ax3.axhline(y=0, color='#00485F', linestyle='-', alpha=0.8, linewidth=2)
            ax3.set_title("📉 Analyse des Résidus", fontweight='bold', fontsize=14)
            ax3.set_xlabel("Observations", fontweight='bold')
            ax3.set_ylabel("Résidus", fontweight='bold')
            ax3.grid(True, linestyle='--', alpha=0.3)
            
            # Annotation
            ax3.annotate(f"Biais = {best_metrics['bias']:.2f}%", xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color='#00485F',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E7DDD9", edgecolor="#00485F", alpha=0.8))
            
            st.pyplot(fig3)
        
        # TABLEAU DE CLASSEMENT DES MODÈLES AVEC SCORE COMPOSITE
        st.subheader("🏆 Classement des modèles (Score composite IPMVP)")
        
        if all_models:
            # Limiter à 15 modèles pour la lisibilité
            models_to_show = all_models[:15]
            
            models_summary = []
            for i, model in enumerate(models_to_show):
                # Indicateur de mode
                mode_icon = "🚀" if model.get('mode') == 'train_test' else "📋"
                
                # Classe de conformité pour le style
                conformity_class = f"conformity-{model['classe']}"
                
                model_row = {
                    "🏆": f"{i+1}",
                    "Score": f"**{model['ipmvp_score']:.1f}**/100",
                    "Mode": mode_icon,
                    "Type": model['model_name'][:20] + ("..." if len(model['model_name']) > 20 else ""),
                    "Variables": ", ".join(model['features'][:2]) + ("..." if len(model['features']) > 2 else ""),
                    "R²": f"{model['r2']:.3f}",
                    "CV(RMSE)": f"{model['cv_rmse']:.3f}",
                    "Biais(%)": f"{model['bias']:.1f}",
                    "Conformité": model['conformite']
                }
                
                # Ajouter warning si overfitting
                if model.get('overfitting_warning'):
                    model_row["⚠️"] = "⚠️"
                else:
                    model_row["⚠️"] = ""
                
                models_summary.append(model_row)
            
            # Affichage du tableau
            df_summary = pd.DataFrame(models_summary)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Statistiques du classement
            excellent_count = sum(1 for m in models_to_show if m['conformite'] == 'Excellente')
            good_count = sum(1 for m in models_to_show if m['conformite'] in ['Bonne', 'Acceptable'])
            overfitting_count = sum(1 for m in models_to_show if m.get('overfitting_warning'))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏆 Total modèles", len(models_to_show))
            with col2:
                st.metric("✅ Excellents", excellent_count)
            with col3:
                st.metric("✅ Acceptables", good_count)
            with col4:
                st.metric("⚠️ Avec warnings", overfitting_count)
        
        # EXPLICATIONS ET RECOMMANDATIONS
        with st.expander("📚 Comprendre le nouveau système de scoring"):
            st.markdown("""
            ### 🎯 Score Composite IPMVP (0-100 points)
            
            **🔄 Changement majeur :** Fini le tri par R² seul ! Le nouveau système utilise un score composite qui évalue :
            
            #### 📊 Score de base (60 points max)
            - **R² (30pts)** : Performance statistique, pondérée selon les seuils IPMVP
            - **CV(RMSE) (20pts)** : Précision du modèle (plus faible = mieux)
            - **Biais (10pts)** : Équilibre du modèle (proche de 0 = mieux)
            
            #### 🎁 Bonus/Malus (40 points max)
            - **Simplicité (15pts)** : Moins de variables = modèle plus robuste
            - **Conformité IPMVP (15pts)** : Respect des critères standard
            - **Significativité (10pts)** : Variables avec |t| > 2
            - **Malus overfitting** : -15 à -30pts selon la sévérité
            - **Malus complexité** : -5pts pour polynôme
            
            #### ✅ Avantages du nouveau système
            - **Ridge/Lasso retrouvent leur utilité** : Pas pénalisés pour leur R² plus faible
            - **Fin des R² artificiels** : Modèles avec 99% de R² mais instables sont rejetés
            - **Évaluation holistique** : Combine performance, robustesse et simplicité
            - **Conformité IPMVP renforcée** : Critères standard intégrés au scoring
            """)
            
        with st.expander("📚 Interprétation des résultats"):
            st.markdown(f"""
            ### 🔍 Analyse de votre modèle
            
            **🏆 Score obtenu :** {best_metrics['ipmvp_score']:.1f}/100
            - 90-100 : Excellent modèle, très robuste
            - 70-89 : Bon modèle, fiable pour IPMVP
            - 50-69 : Modèle acceptable, à surveiller
            - <50 : Modèle insuffisant, révision nécessaire
            
            **📊 Mode d'analyse :** {best_metrics.get('mode', 'standard').title()}
            {'- Validation sur données non-vues (train/test)' if best_metrics.get('mode') == 'train_test' else '- Analyse sur toutes les données disponibles'}
            {'- Plus robuste mais nécessite ≥24 mois' if best_metrics.get('mode') == 'train_test' else '- Standard IPMVP avec protections renforcées'}
            
            **🧮 Type de modèle :** {best_metrics['model_name']}
            - Linéaire : Simple et interprétable
            - Ridge : Régularisé, gère bien les corrélations
            - Lasso : Sélection automatique de variables
            - Polynomiale : Relations non-linéaires, attention à l'overfitting
            
            **⚡ Variables utilisées :** {', '.join(best_features)}
            - Chaque variable doit avoir une justification physique
            - Privilégier les variables significatives (|t| > 2)
            - Éviter la redondance entre variables
            """, unsafe_allow_html=True)

        
        with st.expander("📚 Recommandations d'amélioration"):
            recommendations = []
            
            if best_metrics['ipmvp_score'] < 70:
                recommendations.append("🎯 **Score faible** : Envisager d'autres variables explicatives ou une période différente")
            
            if best_metrics['r2'] < 0.75:
                recommendations.append("📊 **R² insuffisant** : Le modèle explique moins de 75% de la variance (seuil IPMVP)")
            
            if best_metrics['cv_rmse'] > 0.15:
                recommendations.append("🎯 **Précision limitée** : CV(RMSE) > 15% (seuil IPMVP)")
            
            if abs(best_metrics['bias']) > 5:
                recommendations.append("⚖️ **Biais élevé** : Le modèle surestime ou sous-estime systématiquement")
            
            if best_metrics.get('overfitting_warning'):
                recommendations.append("⚠️ **Risque d'overfitting** : " + best_metrics['overfitting_warning'])
            
            if best_metrics.get('mode') == 'standard' and len(df_filtered) >= 24:
                recommendations.append("🚀 **Amélioration possible** : Vous avez assez de données pour le mode train/test (validation robuste)")
            
            # Analyse de la significativité
            if 't_stats' in best_metrics and best_metrics['model_type'] in ["Linéaire", "Ridge", "Lasso"]:
                non_significant = []
                for feature in best_features:
                    if feature in best_metrics['t_stats'] and best_metrics['t_stats'][feature] is not None:
                        t_stat = best_metrics['t_stats'][feature]
                        if isinstance(t_stat, dict):
                            significant = t_stat.get('significant', False)
                        else:
                            significant = abs(t_stat) > 2
                        
                        if not significant:
                            non_significant.append(feature)
                
                if non_significant:
                    recommendations.append(f"📉 **Variables non significatives** : {', '.join(non_significant)} (envisager de les retirer)")
            
            if len(df_filtered) / len(best_features) < 10:
                recommendations.append("📊 **Ratio obs/variables faible** : Risque d'instabilité, considérer moins de variables")
            
            if not recommendations:
                recommendations.append("✅ **Excellent modèle** : Aucune amélioration majeure nécessaire !")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # RÉSUMÉ EXÉCUTIF
        st.markdown("---")
        st.subheader("📋 Résumé exécutif")
        
        # Détermination du statut global
        if best_metrics['ipmvp_score'] >= 80 and best_metrics['conformite'] == 'Excellente':
            status = "✅ **MODÈLE EXCELLENT**"
            status_color = "#4caf50"
            status_msg = "Modèle hautement fiable, conforme aux standards IPMVP les plus exigeants."
        elif best_metrics['ipmvp_score'] >= 60 and best_metrics['conformite'] in ['Excellente', 'Bonne']:
            status = "✅ **MODÈLE ACCEPTABLE**"
            status_color = "#2196f3"
            status_msg = "Modèle valide pour utilisation IPMVP avec quelques améliorations possibles."
        elif best_metrics['ipmvp_score'] >= 40:
            status = "⚠️ **MODÈLE À AMÉLIORER**"
            status_color = "#ff9800"
            status_msg = "Modèle présentant des limitations, révision recommandée avant utilisation."
        else:
            status = "❌ **MODÈLE INSUFFISANT**"
            status_color = "#f44336"
            status_msg = "Modèle non conforme aux standards IPMVP, révision majeure nécessaire."
        
        # Affichage du résumé avec composants natifs Streamlit (plus fiable)
        st.markdown(f"### {status}")
        st.info(status_msg)
        
        # Métriques en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🏆 Score IPMVP",
                value=f"{best_metrics['ipmvp_score']:.1f}/100",
                help="Score composite évaluant performance, conformité IPMVP et simplicité"
            )
            st.metric(
                label="📊 R²",
                value=f"{best_metrics['r2']:.3f}",
                help="Coefficient de détermination (≥0.75 pour excellente conformité IPMVP)"
            )
        
        with col2:
            st.metric(
                label="🎯 CV(RMSE)",
                value=f"{best_metrics['cv_rmse']:.3f}",
                help="Coefficient de variation RMSE (≤0.15 pour excellente conformité IPMVP)"
            )
            st.metric(
                label="⚖️ Biais",
                value=f"{best_metrics['bias']:.1f}%",
                help="Erreur systématique du modèle (|biais| < 5% recommandé)"
            )
        
        with col3:
            st.metric(
                label="🧮 Modèle",
                value=best_metrics['model_name'][:20],
                help="Type de régression utilisé"
            )
            st.metric(
                label="📋 Variables",
                value=f"{len(best_features)}",
                help=f"Variables: {', '.join(best_features)}"
            )
        
    else:
        st.error("❌ **Aucun modèle valide trouvé**")
        st.markdown("""
        ### 🔍 Causes possibles :
        - **Données insuffisantes** : Moins de 10 observations
        - **Variables non pertinentes** : Aucune corrélation avec la consommation
        - **Overfitting détecté** : Tous les modèles rejetés pour R² suspect
        - **Limitations dépassées** : Trop de variables par rapport aux observations
        
        ### 💡 Solutions :
        1. **Vérifier les données** : Qualité, complétude, cohérence
        2. **Revoir les variables** : Choisir des variables physiquement liées à la consommation
        3. **Ajuster les paramètres** : Réduire le nombre de variables ou changer la période
        4. **Améliorer les données** : Ajouter plus d'observations si possible
        """)

elif df is not None and lancer_calcul and not selected_vars:
    st.warning("⚠️ **Veuillez sélectionner au moins une variable explicative** pour lancer l'analyse.")

elif lancer_calcul and df is None:
    st.warning("⚠️ **Veuillez d'abord importer un fichier Excel** pour lancer l'analyse.")

# MESSAGE INFORMATIF SI AUCUNE ACTION
elif df is None:
    st.info("""
    ### 🚀 Pour commencer votre analyse IPMVP :
    
    1. **📂 Importez votre fichier Excel** contenant :
       - Une colonne de dates (format date reconnu)
       - Une colonne de consommation énergétique (valeurs numériques)
       - Des variables explicatives (DJU, température, occupation, production, etc.)
    
    2. **🔍 Configurez l'analyse** dans le panneau latéral :
       - Vérifiez la détection automatique des colonnes
       - Sélectionnez vos variables explicatives
       - Choisissez le mode d'analyse (automatique recommandé)
    
    3. **🚀 Lancez l'analyse** et découvrez :
       - Le score composite IPMVP (0-100 points)
       - La détection intelligente d'overfitting
       - La validation train/test si applicable
       - Les recommandations d'amélioration
    
    ### ✨ **Nouveautés de cette version :**
    - **🛡️ Protection anti-overfitting** : Fini les R² artificiels à 99% !
    - **🎯 Score composite** : Évaluation holistique remplaçant le tri par R² seul
    - **🚀 Mode train/test** : Validation robuste si ≥24 mois de données
    - **⚠️ Limitations intelligentes** : Contrôle automatique du ratio observations/variables
    - **📊 Métriques enrichies** : Significativité statistique, comparaisons train/test
    """)

# PIED DE PAGE FINAL
st.markdown("---")
st.markdown("""
<div class="footer-credit">
    <p><strong>🎉 Analyse IPMVP Améliorée v2.1 - Visualisations enrichies ! 🎉</strong></p>
    <p><strong>🔧 Améliorations intégrées :</strong></p>
    <ul style="text-align: left; display: inline-block;">
        <li>✅ Détection overfitting intelligente</li>
        <li>✅ Score composite IPMVP (0-100 points)</li>
        <li>✅ Mode train/test adaptatif</li>
        <li>✅ Limitations sécurité (règle 10:1)</li>
        <li>✅ Métriques enrichies et visualisations améliorées</li>
        <li>✅ Affichage détaillé des intervalles train/test avec dates</li>
        <li>✅ R² et CV(RMSE) visibles sur tous les graphiques</li>
        <li>✅ Ridge/Lasso retrouvent leur utilité</li>
    </ul>
    <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
    <p><em>"Plus de R² à 99% bidons, place aux modèles robustes !" 🚀</em></p>
</div>
""", unsafe_allow_html=True)

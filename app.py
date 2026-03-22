# =============================================================================
# PARTIE 1 : BASE + AUTHENTIFICATION
# Calculette IPMVP - Modélisation ligne de base énergétique - Conformité IPMVP
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
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
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
    page_title="Calculette IPMVP",
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
        <p style="font-size:0.85em; opacity:0.8;">Efficacité Energétique & Carbone team</p>
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
    Calcule le score IPMVP selon les nouveaux critères (70 points max)
    
    R² : 30 points max (0.75 = 1pt, 1.00 = 30pts)
    CV(RMSE) : 30 points max (0.20 = 1pt, 0.00 = 30pts)
    T-stats : 10 points max (|t| = 2 → 2pts, |t| ≥ 5 → 10pts)
    
    TOTAL : 70 points maximum
    """
    r2 = model_info['r2']
    cv_rmse = model_info['cv_rmse']
    model_type = model_info['model_type']
    
    # =================================================================
    # 1. SCORE R² (30 points max)
    # =================================================================
    # R² = 0.75 → 1 pt
    # R² = 1.00 → 30 pts
    # Échelle linéaire
    
    if r2 >= 1.0:
        r2_score = 30.0
    elif r2 <= 0.75:
        r2_score = max(0, (r2 / 0.75))  # En dessous de 0.75, score proportionnel
    else:
        # Interpolation linéaire entre 0.75 et 1.00
        r2_score = 1 + ((r2 - 0.75) / (1.0 - 0.75)) * 29
    
    # =================================================================
    # 2. SCORE CV(RMSE) (30 points max)
    # =================================================================
    # CV(RMSE) = 0.20 → 1 pt
    # CV(RMSE) = 0.00 → 30 pts
    # Échelle linéaire inversée
    
    if cv_rmse <= 0.0:
        cv_score = 30.0
    elif cv_rmse >= 0.20:
        cv_score = max(0, 1 - (cv_rmse - 0.20) * 2)  # Au-dessus de 0.20, décroissance rapide
    else:
        # Interpolation linéaire entre 0.00 et 0.20
        cv_score = 30 - (cv_rmse / 0.20) * 29
    
    # =================================================================
    # 3. SCORE SIGNIFICATIVITÉ T-STATS (10 points max)
    # =================================================================
    # |t| = 2 → 2 pts
    # |t| ≥ 5 → 10 pts
    # Échelle linéaire
    
    t_score = 0
    
    if 't_stats' in model_info and model_type in ["Linéaire", "Ridge", "Lasso"]:
        total_vars = 0
        total_t_score = 0
        
        for feature in model_info['features']:
            if (feature in model_info['t_stats'] and 
                model_info['t_stats'][feature] is not None):
                
                t_stat = model_info['t_stats'][feature]
                
                # Extraire la valeur t
                if isinstance(t_stat, dict) and 't_value' in t_stat:
                    t_value = abs(t_stat['t_value'])
                elif isinstance(t_stat, (int, float)):
                    t_value = abs(t_stat)
                else:
                    continue
                
                total_vars += 1
                
                # Calcul du score pour cette variable
                if t_value >= 5.0:
                    var_t_score = 10.0
                elif t_value <= 2.0:
                    var_t_score = max(0, (t_value / 2.0) * 2)  # En dessous de 2, score proportionnel
                else:
                    # Interpolation linéaire entre 2 et 5
                    var_t_score = 2 + ((t_value - 2.0) / (5.0 - 2.0)) * 8
                
                total_t_score += var_t_score
        
        # Moyenne des scores t
        if total_vars > 0:
            t_score = total_t_score / total_vars
    
    # =================================================================
    # SCORE FINAL (0-70 points)
    # =================================================================
    final_score = r2_score + cv_score + t_score
    final_score = max(0, min(70, final_score))  # Borner entre 0 et 60
    
    return final_score

def get_ipmvp_qualification(score):
    """
    Convertit le score IPMVP (0-70) en qualification textuelle
    
    Excellent : ≥ 55/60
    Très bon : 45-54/60
    Bon : 35-44/60
    Correct : 25-34/60
    Non conforme : < 25/60
    """
    if score >= 55:
        return "Excellent", "excellent", "#4caf50"  # Vert foncé
    elif score >= 45:
        return "Très bon", "very_good", "#96B91D"  # Vert clair
    elif score >= 35:
        return "Bon", "good", "#6DBABC"  # Bleu-vert
    elif score >= 25:
        return "Correct", "fair", "#ff9800"  # Orange
    else:
        return "Non conforme", "non_compliant", "#f44336"  # Rouge


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
        warnings.append(f"⚠️ Données limitées: {len(df)} points (18+ recommandés pour train/test)")
    
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

def calculate_bias_ipmvp(y_true, y_pred, decimal_places=2):
    """
    Calcule le biais IPMVP officiel sur un jeu de données (train OU test).
    
    Formule officielle IPMVP :
      Biais(%) = Σ(Ŷᵢ - Yᵢ) / (n × Ȳ) × 100
    
    Args:
        y_true : valeurs réelles (Series ou array)
        y_pred : valeurs prédites (array)
        decimal_places : nombre de décimales
    Returns:
        float : biais en %
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n = len(y_true)
        mean_y = np.mean(y_true)
        if mean_y == 0 or n == 0:
            return 0.0
        bias = np.sum(y_pred - y_true) / (n * mean_y) * 100
        return round(float(bias), decimal_places)
    except Exception:
        return 0.0


def calculate_bias_reel_cv(X, y, model, decimal_places=2):
    """
    Calcule le biais RÉEL en mode standard via validation croisée Leave-One-Out (LOO-CV).
    
    Problème du biais OLS sur train :
      Pour toute régression linéaire avec intercept, Σ(Ŷᵢ - Yᵢ) = 0 par construction
      → le biais calculé sur les données d'entraînement vaut TOUJOURS exactement 0
      → ce n'est pas un biais "réel", c'est une propriété mathématique
    
    Solution — Validation croisée Leave-One-Out :
      On retire une observation à la fois, on réentraîne sur le reste,
      on prédit la valeur retirée → les prédictions ne sont PLUS sur données d'entraînement
      → le biais calculé sur ces prédictions LOO est un vrai biais non-biaisé
    
    Formule :
      Pour chaque observation i :
        - Entraîner le modèle sur toutes les observations SAUF i
        - Prédire Ŷᵢ avec ce modèle
      Biais_réel(%) = Σ(Ŷᵢ_LOO - Yᵢ) / (n × Ȳ) × 100
    
    Args:
        X: variables explicatives (DataFrame)
        y: variable cible (Series)
        model: modèle sklearn (LinearRegression, Ridge, Lasso)
        decimal_places: nombre de décimales
    
    Returns:
        float: biais réel en % (non-nul, non-biaisé)
    """
    try:
        n = len(y)
        y_arr = np.array(y)
        
        # LOO uniquement si modèle linéaire (pas polynomial - trop lent et instable)
        # Pour n > 30, utiliser KFold(5) pour la rapidité
        if n <= 30:
            cv = KFold(n_splits=n, shuffle=False)  # Leave-One-Out exact
        else:
            cv = KFold(n_splits=min(10, n // 3), shuffle=False)  # K-Fold pour grands datasets
        
        # Prédictions cross-validées
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
        
        # Biais IPMVP sur prédictions CV (formule officielle)
        mean_y = np.mean(y_arr)
        if mean_y == 0:
            return 0.0
        bias_cv = np.sum(y_pred_cv - y_arr) / (n * mean_y) * 100
        return round(float(bias_cv), decimal_places)
    
    except Exception:
        # Fallback : si CV échoue (ex: trop peu de données), retourner 0
        return 0.0



def format_equation(intercept, coefficients, threshold=1e-4):
    """
    Formate l'équation du modèle de régression sous forme lisible.
    Ignore les coefficients négligeables (< threshold).
    """
    try:
        equation = f"Consommation = {intercept:.4f}"
        sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, coef in sorted_coefs:
            if abs(coef) < threshold:
                continue
            sign = "+" if coef >= 0 else ""
            equation += f" {sign} {coef:.4f} × {feature}"
        return equation
    except Exception:
        return f"Consommation = {intercept:.4f} (coefficients non disponibles)"

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
    Détermine si on doit utiliser un split train/test.
    Activé dès que ≥18 mois de données disponibles (conforme IPMVP : train=12 min + test≥6 mois).
    """
    if nb_observations >= 18:
        return True, f"🚀 Mode validation robuste: Split train/test activé ({nb_observations} mois)"
    elif nb_observations > 12:
        return False, f"📋 Mode IPMVP standard : {nb_observations} mois — split non activé (< 18 mois requis)"
    elif nb_observations == 12:
        return False, f"📋 Mode IPMVP standard : exactement 12 mois — pas de split possible (test = 0)"
    else:
        return False, f"📋 Mode IPMVP standard avec {nb_observations} mois de données"

def create_train_test_split(df, date_col, train_months=12):
    """
    Crée un split train/test temporel pour les données IPMVP.
    
    ⚠️ RÈGLES IPMVP :
    - Train ≥ 12 mois (couvrir toutes les saisons pour la baseline)
    - Test ≥ 1 mois (validation sur données non vues)
    - Train ≥ Test (le modèle doit être entraîné sur plus de données qu'il n'en valide)
    
    Mode manuel recommandé IPMVP :
    - 13 mois de données → 12 train + 1 test
    - 18 mois de données → 12 train + 6 test (ratio 2:1)
    - 24 mois de données → 18 train + 6 test (ratio 3:1)
    - 36 mois de données → 24 train + 12 test (ratio 2:1)
    
    Pour un split manuel : ajuster le slider "Mois d'entraînement" dans la sidebar.
    Si le test résultant est > le train → le split est automatiquement rééquilibré.
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    n_total = len(df_sorted)
    
    # Calculer le point de coupure selon train_months
    min_date = df_sorted[date_col].min()
    split_date = min_date + pd.DateOffset(months=train_months)
    
    train_df = df_sorted[df_sorted[date_col] < split_date]
    test_df = df_sorted[df_sorted[date_col] >= split_date]
    
    # RÈGLE 1 : test doit avoir au moins 1 point
    if len(test_df) < 1:
        n_test = max(1, n_total - 12)
        train_df = df_sorted.iloc[:-n_test]
        test_df = df_sorted.iloc[-n_test:]
        split_date = test_df[date_col].min()
    
    # RÈGLE 2 : train doit être >= test (sinon rééquilibrer au ratio 2:1)
    if len(test_df) > len(train_df):
        n_train = max(12, int(n_total * 2 / 3))
        n_test = n_total - n_train
        if n_test < 1:
            n_test = 1
            n_train = n_total - n_test
        train_df = df_sorted.iloc[:n_train]
        test_df = df_sorted.iloc[n_train:]
        split_date = test_df[date_col].min()
    
    # RÈGLE 3 : train doit avoir au minimum 12 points (12 mois)
    if len(train_df) < 12:
        n_train = min(12, n_total - 1)
        train_df = df_sorted.iloc[:n_train]
        test_df = df_sorted.iloc[n_train:]
        split_date = test_df[date_col].min()
    
    return train_df, test_df, split_date
# =============================================================================
# PARTIE 3 : INTERFACE UTILISATEUR
# CSS, styling, sidebar et interface principale avec contrôles adaptatifs
# =============================================================================

# CSS AMÉLIORÉ AVEC NOUVEAUX STYLES POUR LES AMÉLIORATIONS IPMVP
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&display=swap');

    html, body {
        font-family: 'Manrope', sans-serif;
        color: #0C1D2D;
    }
    
    p, div, span, li, td, th, label {
        font-family: 'Manrope', sans-serif;
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

    input[type="text"], input[type="password"], input[type="number"], select, textarea {
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
st.markdown("""
<div style="padding: 28px 0 8px 0;">
    <h1 style="font-size:2.2em; font-weight:800; color:#00485F; margin:0; letter-spacing:-0.5px;">
        Calculette IPMVP
    </h1>
    <p style="color:#555; font-size:1.05em; margin-top:6px; font-weight:400;">
        Modélisation de la ligne de base énergétique · Scoring IPMVP · Validation train/test
    </p>
</div>
""", unsafe_allow_html=True)

# GUIDE D'UTILISATION
st.markdown("""
<div class="instruction-card">
<h3>Guide d'utilisation</h3>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:10px;">
    <div>
        <h4>Etapes</h4>
        <ol style="margin:0; padding-left:18px; line-height:1.9;">
            <li>Importer un fichier Excel (date + consommation + variables)</li>
            <li>Vérifier la détection automatique des colonnes</li>
            <li>Sélectionner les variables explicatives (DJU, occupation…)</li>
            <li>Choisir le type de modèle — <em>Automatique</em> recommandé</li>
            <li>Lancer l'analyse et consulter le score IPMVP</li>
        </ol>
    </div>
    <div>
        <h4>Critères IPMVP</h4>
        <table style="width:100%; border-collapse:collapse; font-size:0.92em;">
            <tr><td style="padding:4px 8px;"><strong>R²</strong></td><td style="padding:4px 8px; color:#00485F;">≥ 0.75</td></tr>
            <tr style="background:rgba(109,186,188,0.08);"><td style="padding:4px 8px;"><strong>CV(RMSE)</strong></td><td style="padding:4px 8px; color:#00485F;">≤ 20%</td></tr>
            <tr><td style="padding:4px 8px;"><strong>Biais</strong></td><td style="padding:4px 8px; color:#00485F;">≤ 5%</td></tr>
            <tr style="background:rgba(109,186,188,0.08);"><td style="padding:4px 8px;"><strong>T-stat</strong></td><td style="padding:4px 8px; color:#00485F;">|t| > 2</td></tr>
            <tr><td style="padding:4px 8px;"><strong>Ratio obs/var</strong></td><td style="padding:4px 8px; color:#00485F;">≥ 10:1</td></tr>
        </table>
        <p style="margin-top:10px; font-size:0.88em; color:#555;">
            <strong>Mode train/test</strong> activé automatiquement dès ≥ 18 mois de données.<br>
            Train ≥ 12 mois · Test ≥ 1 mois · Train ≥ Test
        </p>
    </div>
</div>
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

# Initialisation des variables par défaut (évite les NameError si aucun fichier)
selected_vars = []

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
    # Détermination du mode d'analyse — nombre de MOIS UNIQUES
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        nb_observations = df[date_col].dt.to_period('M').nunique()
    except:
        nb_observations = len(df)
    use_train_test, mode_message = should_use_train_test_split(nb_observations)
    
    # Affichage du mode d'analyse
    if use_train_test:
        st.sidebar.markdown(f"""
        <div class="mode-indicator" style="background-color: rgba(150, 185, 29, 0.1); border-color: #96B91D;">
            <div class="mode-title">🚀 Mode Validation Robuste</div>
            <p>Split train/test activé ({nb_observations} mois de données)<br>
            Configurer le split ci-dessous ↓</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div class="mode-indicator" style="background-color: rgba(109, 186, 188, 0.1); border-color: #6DBABC;">
            <div class="mode-title">📋 Mode IPMVP Standard</div>
            <p>Analyse sur toutes les données ({nb_observations} mois)<br>
            {"Ajoutez au moins 1 mois supplémentaire pour activer le split train/test" if nb_observations == 12 else "Données insuffisantes pour le split train/test"}</p>
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
            # Fin par défaut = dernier jour du 12ème mois complet (couvre 12 mois entiers)
            _end_month_start = pd.to_datetime(start_date) + pd.DateOffset(months=12)
            _last_day = (_end_month_start - pd.Timedelta(days=1)).date()
            default_end = min(max_date, _last_day)
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
# Valeurs par défaut (utilisées en mode Automatique)
alpha_ridge = 1.0
alpha_lasso = 0.1
poly_degree = 2

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

# CHOIX DES DÉCIMALES POUR LE BIAIS
st.sidebar.markdown("---")
st.sidebar.subheader("🔢 Précision d'affichage")
bias_decimals = st.sidebar.selectbox(
    "Décimales pour le Biais (%)",
    [1, 2, 3, 4, 5, 6, 7],
    index=1,
    help="Nombre de décimales pour afficher le biais IPMVP. Formule : Σ(Ŷᵢ-Yᵢ)/(n×Ȳ)×100"
)

# SPLIT TRAIN/TEST MANUEL
st.sidebar.subheader("✂️ Paramètres Train/Test")

# Calcul dynamique du max de mois train selon les données
if df is not None:
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        nb_mois_total = df[date_col].dt.to_period('M').nunique()
    except:
        nb_mois_total = len(df)
    
    max_train_months_possible = max(12, nb_mois_total - 1)
    # Défaut intelligent : 2/3 des mois pour le train (ex: 24 mois → 16 train + 8 test)
    # Minimum 12 mois (IPMVP baseline), maximum = max_train_months_possible
    default_train = max(12, min(int(nb_mois_total * 2 / 3), max_train_months_possible))
else:
    nb_mois_total = 24
    max_train_months_possible = 23
    default_train = 12

# N'afficher le slider que si on a assez de mois pour un split (≥13 mois)
if nb_mois_total >= 13:
    _slider_min = 12
    _slider_max = max(_slider_min + 1, min(36, max_train_months_possible))
    _slider_val = max(_slider_min, min(default_train, _slider_max))

    train_months_manual = st.sidebar.slider(
        "Mois d'entraînement (train)",
        min_value=_slider_min,
        max_value=_slider_max,
        value=_slider_val,
        step=1,
    help="""Comment faire le split train/test manuellement :
    
    1. CHOISIR les mois de train = durée de la période de référence (baseline)
       → Minimum 12 mois IPMVP (couvrir toutes les saisons)
    2. Le reste des données devient le TEST (validation sur données non vues)
       → Minimum 1 mois de test requis
    3. RÈGLE CLEF : Train ≥ Test (sinon le modèle ne peut pas généraliser)
    
    Exemples :
    - 13 mois dispo → 12 train + 1 test ✓ (minimum)
    - 18 mois dispo → 12 train + 6 test (ratio 2:1) ✓
    - 24 mois dispo → 18 train + 6 test (ratio 3:1) ✓  
    - 32 mois dispo → 22 train + 10 test (ratio 2:1) ✓
    
    Si train < test → le split est automatiquement rééquilibré à 2:1
    """
)
else:
    # Moins de 13 mois → pas de split possible, slider masqué
    train_months_manual = 12
    st.sidebar.warning(f"⚠️ **{nb_mois_total} mois disponibles** — le split train/test nécessite ≥ 13 mois (12 train + 1 test minimum). Mode standard activé.")

# Affichage du split prévu (seulement si split possible)
if df is not None and nb_mois_total >= 13:
    n_total = nb_mois_total  # utiliser les mois, pas les lignes
    n_test_preview = n_total - train_months_manual
    if n_test_preview < 1:
        n_test_preview = 1
        n_train_preview = n_total - 1
    elif n_test_preview > train_months_manual:
        n_train_preview = int(n_total * 2 / 3)
        n_test_preview = n_total - n_train_preview
        st.sidebar.warning(f"⚠️ Rééquilibrage auto → Train: {n_train_preview} | Test: {n_test_preview}")
    else:
        n_train_preview = train_months_manual
    
    ratio_preview = n_train_preview / n_test_preview if n_test_preview > 0 else 0
    color = "🟢" if ratio_preview >= 2 else ("🟡" if ratio_preview >= 1 else "🔴")
    st.sidebar.info(f"{color} Split prévu → **Train: {n_train_preview} mois** | **Test: {n_test_preview} mois** | Ratio: {ratio_preview:.1f}:1")
    
    # Affichage des dates de split si données disponibles avec colonne de date
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df_sorted_preview = df.sort_values(by=date_col)
            min_date_prev = df_sorted_preview[date_col].min()
            split_date_prev = min_date_prev + pd.DateOffset(months=n_train_preview)
            max_date_prev = df_sorted_preview[date_col].max()
            st.sidebar.markdown(f"""
            <div style="background:rgba(109,186,188,0.1); padding:8px; border-radius:6px; font-size:0.85em; margin-top:4px;">
            🎯 <b>Train :</b> {min_date_prev.strftime('%b %Y')} → {(split_date_prev - pd.DateOffset(months=1)).strftime('%b %Y')}<br>
            🧪 <b>Test :</b> {split_date_prev.strftime('%b %Y')} → {max_date_prev.strftime('%b %Y')}
            </div>
            """, unsafe_allow_html=True)
        except:
            pass
else:
    st.sidebar.info(f"ℹ️ Train: {train_months_manual} mois | Test: données restantes (≥1 pt requis)")

# INFORMATIONS SUR LA CONFORMITÉ IPMVP ENRICHIES
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### ✅ Critères IPMVP v2.3
{tooltip("Qualification IPMVP", "Système de notation sur 70 points : R² (30pts) + CV(RMSE) (30pts) + Significativité (10pts)")}

**📊 Scoring (70 points max) :**
- **R²** : 0.75 = 1pt → 1.00 = 30pts
- **CV(RMSE)** : 0.20 = 1pt → 0.00 = 30pts  
- **T-stats** : |t|=2 = 2pts → |t|≥5 = 10pts

**🎯 Qualifications :**
- **Excellent** : ≥ 55/70 points
- **Très bon** : 45-54/70 points
- **Bon** : 35-44/70 points
- **Correct** : 25-34/70 points
- **Non conforme** : < 25/70 points

- Seuil recommandé : **|Biais| < 5%**
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
    <p style="font-size:0.85em; opacity:0.7; margin:0;">
        Calculette IPMVP &nbsp;·&nbsp; Détection overfitting &nbsp;·&nbsp; Score composite &nbsp;·&nbsp; Validation train/test
    </p>
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
            
            # Validation des limitations de sécurité (avertissement seulement, ne pas bloquer)
            var_issues, _ = check_variable_limits(len(period_df), len(selected_vars), model_type)
            # On ne bloque plus la période entière — les combos de 1 variable peuvent quand même être valides
            
            period_best_score = -1
            period_best_model = None
            
            # Test des combinaisons de variables
            for n in range(1, min(max_features + 1, len(selected_vars) + 1)):
                for combo in combinations(selected_vars, n):
                    X_subset = X[list(combo)]
                    
                    # Split train/test si applicable
                    if use_train_test and len(period_df) >= 18:
                        train_df, test_df, split_date = create_train_test_split(period_df, date_col, train_months_manual)
                        X_train = train_df[list(combo)]
                        y_train = train_df[conso_col]
                        X_test = test_df[list(combo)]
                        y_test = test_df[conso_col]
                        
                        # Nettoyage train/test
                        X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
                        y_train = pd.to_numeric(y_train, errors='coerce').dropna()
                        X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
                        y_test = pd.to_numeric(y_test, errors='coerce').dropna()
                        
                        if len(X_train) < 5 or len(X_test) < 1:
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
                                bias_test = calculate_bias_ipmvp(y_test, y_pred_test, bias_decimals)
                                
                                # Métriques sur le train set
                                r2_train = r2_score(y_train, y_pred_train)
                                rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_train))
                                cv_rmse_train = rmse_train / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                                bias_train = calculate_bias_ipmvp(y_train, y_pred_train, bias_decimals)
                                
                                # PROTECTION R² NÉGATIF SUR LE TEST
                                # R² < 0 = modèle pire qu'une simple moyenne → rejeté systématiquement
                                # Cause probable : split déséquilibré ou données non-stationnaires
                                if r2_test < -0.5:  # Seuil assoupli : rejet seulement si très négatif
                                    continue  # Rejeter ce modèle
                                
                                # Utiliser les métriques de test pour l'évaluation
                                r2, cv_rmse, bias = r2_test, cv_rmse_test, bias_test
                                rmse = rmse_test
                                mae = mean_absolute_error(y_test, y_pred_test)
                                # MAPE sur test
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
                                    mape = round(float(mape), bias_decimals) if np.isfinite(mape) else 0.0
                                
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
                                
                                # Calcul RMSE corrigé selon IPMVP (avec degrés de liberté)
                                n = len(y_train)
                                p = len(combo)
                                ssr = np.sum((y_train - y_pred) ** 2)
                                df_res = n - p - 1 if (n - p - 1) > 0 else 1
                                rmse = math.sqrt(ssr / df_res)
                                cv_rmse = rmse / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                                
                                # BIAIS RÉEL via validation croisée LOO/KFold
                                # BIAIS IPMVP officiel : Σ(Ŷᵢ - Yᵢ) / (n × Ȳ) × 100
                                # Appliqué sur les données d'ajustement (formule prescrite IPMVP)
                                # OLS linéaire → 0 par propriété algébrique (normal et attendu IPMVP)
                                # Ridge/Lasso → légèrement non nul (régularisation)
                                bias = calculate_bias_ipmvp(y_train, y_pred, bias_decimals)
                                
                                mape = 0.0  # Retiré de l'affichage
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
                            
                            # Récupération des coefficients - initialisation défensive
                            coefs = {}
                            intercept = 0.0
                            
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
                                'mape': mape,
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
                            current_score.metric("Score actuel", f"{ipmvp_score:.1f}/70")
                            
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
                
                best_so_far.metric("Meilleur score", f"{best_period_score:.1f}/70")
            
            # Mise à jour de la barre de progression
            progress_bar.progress((idx + 1) / len(date_ranges))
        
        # Nettoyage de l'affichage de progression
        progress_container.empty()
        
        if best_period_data is not None:
            st.success(f"✅ **Meilleure période trouvée** : {best_period_name}")
            
            # Warning si moins de 12 mois
            if len(best_period_data) < 12:
                st.warning(f"⚠️ **Attention :** Seulement {len(best_period_data)} observations disponibles. L'IPMVP recommande au minimum 12 mois de données pour une baseline fiable.")
            
            # Affichage de la qualification
            qualification, qual_class, qual_color = get_ipmvp_qualification(best_period_score)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="score-card" style="background: linear-gradient(135deg, {qual_color} 0%, {qual_color}dd 100%);">
                    <div class="score-value">{qualification}</div>
                    <div class="score-label">Qualification IPMVP</div>
                    <div style="font-size: 0.9em; margin-top: 5px; opacity: 0.9;">{best_period_score:.1f}/70 points</div>
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
        
        # Détermination du mode d'analyse — basé sur le nombre de MOIS UNIQUES
        # (len(df_filtered) = nb de lignes, pas forcément nb de mois)
        nb_mois_uniques = df_filtered[date_col].dt.to_period('M').nunique()
        use_train_test, mode_message = should_use_train_test_split(nb_mois_uniques)
        st.info(f"📊 **{nb_mois_uniques} mois** dans la période sélectionnée — {mode_message}")
        
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
        rejected_r2_neg = 0
        rejected_overfit = 0
        accepted = 0
        
        # Test des combinaisons de variables
        for n in range(1, min(max_features + 1, len(selected_vars) + 1)):
            for combo in combinations(selected_vars, n):
                progress_counter += 1
                progress_bar.progress(progress_counter / total_combinations)
                
                X_subset = X[list(combo)]
                
                # Split train/test si applicable (IPMVP : ≥18 mois requis)
                if use_train_test and nb_mois_uniques >= 18:
                    train_df, test_df, split_date = create_train_test_split(df_filtered, date_col, train_months_manual)
                    X_train = train_df[list(combo)]
                    y_train = train_df[conso_col]
                    X_test = test_df[list(combo)]
                    y_test = test_df[conso_col]
                    
                    # Nettoyage train/test
                    X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
                    y_train = pd.to_numeric(y_train, errors='coerce').dropna()
                    X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
                    y_test = pd.to_numeric(y_test, errors='coerce').dropna()
                    
                    if len(X_train) < 5 or len(X_test) < 1:
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
                            bias_test = calculate_bias_ipmvp(y_test, y_pred_test, bias_decimals)
                            
                            r2_train = r2_score(y_train, y_pred_train)
                            rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_train))
                            cv_rmse_train = rmse_train / np.mean(y_train) if np.mean(y_train) != 0 else float('inf')
                            bias_train = calculate_bias_ipmvp(y_train, y_pred_train, bias_decimals)
                            
                            # PROTECTION R² NÉGATIF SUR LE TEST
                            # Seuil très permissif : on garde tout sauf les cas catastrophiques
                            if r2_test < -1.0:
                                rejected_r2_neg += 1
                                continue
                            
                            r2, cv_rmse, bias = r2_test, cv_rmse_test, bias_test
                            mae = mean_absolute_error(y_test, y_pred_test)
                            rmse = rmse_test
                            # MAPE sur test
                            with np.errstate(divide='ignore', invalid='ignore'):
                                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
                                mape = round(float(mape), bias_decimals) if np.isfinite(mape) else 0.0
                            
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
                            
                            # BIAIS RÉEL via validation croisée LOO/KFold
                            # BIAIS IPMVP officiel : Σ(Ŷᵢ - Yᵢ) / (n × Ȳ) × 100
                            # Appliqué sur les données d'ajustement (formule prescrite IPMVP)
                            # OLS linéaire → 0 par propriété algébrique (normal et attendu IPMVP)
                            # Ridge/Lasso → légèrement non nul (régularisation)
                            bias = calculate_bias_ipmvp(y_train, y_pred, bias_decimals)
                            
                            mape = 0.0  # Retiré de l'affichage
                            mae = mean_absolute_error(y_train, y_pred)
                            
                            overfitting_detected = False
                            r2_train = r2_test = r2
                            cv_rmse_train = cv_rmse_test = cv_rmse
                            bias_train = bias_test = bias
                        
                        # Détection d'overfitting : utiliser nb_mois_uniques TOTAL (pas juste train)
                        # Sinon avec 12 mois de train + 2 vars → ratio=6 → rejet injuste
                        model_info_temp = {
                            'r2': r2,
                            'cv_rmse': cv_rmse,
                            'bias': bias,
                            'features': list(combo),
                            'model_type': m_type
                        }
                        
                        is_overfitted, warning_msg, severity = detect_overfitting_intelligent(model_info_temp, nb_mois_uniques)
                        
                        if is_overfitted and severity == "error":
                            rejected_overfit += 1
                            continue
                        
                        accepted += 1
                        
                        # Récupération des coefficients - initialisation défensive
                        coefs = {}
                        intercept = 0.0
                        
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
                            'mape': mape,
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
                                'mode': 'train_test',
                                'train_n': len(X_train),
                                'test_n': len(X_test),
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
        
        # Diagnostic de rejet (aide au débogage si aucun modèle trouvé)
        total_tested = accepted + rejected_r2_neg + rejected_overfit
        if total_tested > 0:
            st.caption(f"🔍 Modèles testés : {total_tested} | ✅ Acceptés : {accepted} | ❌ R² test trop négatif : {rejected_r2_neg} | ⚠️ Overfitting rejeté : {rejected_overfit}")
    
        # TRI DES MODÈLES PAR SCORE COMPOSITE (PAS PAR R² !)
        all_models.sort(key=lambda x: x['ipmvp_score'], reverse=True)
        
        # TABLEAU RÉCAPITULATIF DE TOUTES LES PÉRIODES TESTÉES
        if all_models:
            st.markdown("---")
            st.subheader("📋 Tableau récapitulatif de toutes les périodes testées")
            
            # Construire le tableau
            periods_summary = []
            # Regrouper par période pour avoir le meilleur modèle de chaque période
            periods_seen = {}
            for m in all_models:
                period_key = m.get('period', 'N/A')
                if period_key not in periods_seen or m['ipmvp_score'] > periods_seen[period_key]['ipmvp_score']:
                    periods_seen[period_key] = m
            
            for period_key, m in sorted(periods_seen.items(), key=lambda x: x[1]['ipmvp_score'], reverse=True):
                mode_label = "🚀 Train/Test" if m.get('mode') == 'train_test' else "📋 Standard"
                qualification_label, _, _ = get_ipmvp_qualification(m['ipmvp_score'])
                row = {
                    "Période": period_key,
                    "Mode": mode_label,
                    "Modèle": m.get('model_name', 'N/A'),
                    "Variables": ', '.join(m.get('features', [])),
                    "R²": round(m['r2'], 4),
                    "CV(RMSE)": round(m['cv_rmse'], 4),
                    "Biais (%)": round(m['bias'], 2),
                    "Score IPMVP": round(m['ipmvp_score'], 1),
                    "Qualification": qualification_label,
                }
                if m.get('mode') == 'train_test':
                    row["R² Train"] = round(m.get('train_r2', 0), 4)
                    row["R² Test"] = round(m.get('test_r2', m['r2']), 4)
                    row["CV Train"] = round(m.get('train_cv_rmse', 0), 4)
                    row["CV Test"] = round(m.get('test_cv_rmse', m['cv_rmse']), 4)
                periods_summary.append(row)
            
            if periods_summary:
                df_summary = pd.DataFrame(periods_summary)
                # Colonnes à afficher selon mode
                has_traintest = any(m.get('mode') == 'train_test' for m in periods_seen.values())
                
                if has_traintest:
                    cols_display = ["Période", "Mode", "Modèle", "Variables", "R² Train", "R² Test", "CV Train", "CV Test", "Biais (%)", "Score IPMVP", "Qualification"]
                    cols_display = [c for c in cols_display if c in df_summary.columns]
                else:
                    cols_display = ["Période", "Mode", "Modèle", "Variables", "R²", "CV(RMSE)", "Biais (%)", "Score IPMVP", "Qualification"]
                
                df_display = df_summary[cols_display] if all(c in df_summary.columns for c in cols_display) else df_summary
                
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score IPMVP": st.column_config.ProgressColumn(
                            "Score IPMVP /70",
                            help="Score composite IPMVP (0-70 points)",
                            min_value=0, max_value=70, format="%.1f"
                        ),
                        "R²": st.column_config.NumberColumn("R²", format="%.4f"),
                        "R² Train": st.column_config.NumberColumn("R² Train", format="%.4f"),
                        "R² Test": st.column_config.NumberColumn("R² Test", format="%.4f"),
                        "CV(RMSE)": st.column_config.NumberColumn("CV(RMSE)", format="%.4f"),
                        "CV Train": st.column_config.NumberColumn("CV Train", format="%.4f"),
                        "CV Test": st.column_config.NumberColumn("CV Test", format="%.4f"),
                    }
                )
                st.caption(f"🏆 {len(periods_summary)} période(s) évaluée(s) — triées par score IPMVP décroissant")


    # AFFICHAGE DES RÉSULTATS AVEC AMÉLIORATIONS
    if best_metrics:
        st.success("✅ **Analyse terminée avec succès !**")
        
        # SCORE COMPOSITE ET INFORMATIONS PRINCIPALES
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            qualification, qual_class, qual_color = get_ipmvp_qualification(best_metrics['ipmvp_score'])
            
            st.markdown(f"""
            <div class="score-card" style="background: linear-gradient(135deg, {qual_color} 0%, {qual_color}dd 100%);">
                <div class="score-value">{qualification}</div>
                <div class="score-label">Qualification IPMVP</div>
                <div style="font-size: 0.9em; margin-top: 5px; opacity: 0.9;">{best_metrics['ipmvp_score']:.1f}/70 points</div>
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
        
        # Affichage des périodes d'analyse AVEC MÉTRIQUES
        st.markdown("---")
        st.subheader("📅 Périodes d'analyse & Métriques")
        
        if best_metrics.get('mode') == 'train_test':
            # Mode Train/Test : afficher les deux périodes avec leurs métriques
            train_df_temp, test_df_temp, split_date_temp = create_train_test_split(df_filtered, date_col, train_months_manual)
            
            # ⚠️ ALERTE si le split est déséquilibré
            if len(test_df_temp) > len(train_df_temp):
                st.error(f"""⚠️ **Split déséquilibré détecté et corrigé automatiquement**
                
Avec {len(df_filtered)} mois de données et {train_months_manual} mois de train demandés, 
le test ({len(df_filtered) - train_months_manual} mois) était plus long que le train ({train_months_manual} mois).

**Règle IPMVP : Train ≥ Test obligatoire** — Le split a été rééquilibré automatiquement au ratio 2:1.
→ Ajustez le slider "Mois d'entraînement" pour un split cohérent avec vos données.
""")
            
            # Récupération des métriques train
            train_r2_val    = best_metrics.get('train_r2', 0)
            train_cv_val    = best_metrics.get('train_cv_rmse', 0)
            # Biais TRAIN : calculé via LOO-CV (validation croisée) sur la période d'entraînement
            _m_type_train = best_metrics.get('model_type', 'Linéaire')
            if _m_type_train in ["Linéaire", "Ridge", "Lasso"]:
                _X_tr = train_df_temp[best_features].apply(pd.to_numeric, errors='coerce').dropna()
                _y_tr = pd.to_numeric(train_df_temp[conso_col], errors='coerce').dropna()
                if _m_type_train == "Linéaire":
                    _m_bias = LinearRegression()
                elif _m_type_train == "Ridge":
                    _m_bias = Ridge(alpha=1.0)
                else:
                    _m_bias = Lasso(alpha=0.1)
                train_bias_val = calculate_bias_reel_cv(_X_tr, _y_tr, _m_bias, bias_decimals)
            else:
                train_bias_val = best_metrics.get('train_bias', 0)
            # Récupération des métriques test
            test_r2_val     = best_metrics.get('test_r2', best_metrics['r2'])
            test_cv_val     = best_metrics.get('test_cv_rmse', best_metrics['cv_rmse'])
            test_bias_val   = best_metrics.get('test_bias', best_metrics['bias'])
            
            # Statuts test
            r2_ok   = "✅" if test_r2_val  >= 0.75 else ("⚠️" if test_r2_val  >= 0.60 else "❌")
            cv_ok   = "✅" if test_cv_val  <= 0.20 else ("⚠️" if test_cv_val  <= 0.30 else "❌")
            bias_ok = "✅" if abs(test_bias_val) <= 5 else ("⚠️" if abs(test_bias_val) <= 10 else "❌")
            
            col_train, col_test = st.columns(2)
            with col_train:
                # Statuts train
                tr2_ok  = "✅" if train_r2_val  >= 0.75 else ("⚠️" if train_r2_val  >= 0.60 else "❌")
                tcv_ok  = "✅" if train_cv_val  <= 0.20 else ("⚠️" if train_cv_val  <= 0.30 else "❌")
                st.markdown(f"""
                <div style="background-color: rgba(150, 185, 29, 0.1); border-left: 4px solid #96B91D; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #96B91D; margin: 0 0 12px 0;">🎯 PÉRIODE D'ENTRAÎNEMENT (TRAIN)</h4>
                    <p style="margin: 4px 0;">📅 <strong>Du :</strong> {train_df_temp[date_col].min().strftime('%b %Y')} &nbsp;→&nbsp; <strong>Au :</strong> {train_df_temp[date_col].max().strftime('%b %Y')}</p>
                    <p style="margin: 4px 0;">📊 <strong>Observations :</strong> {len(train_df_temp)} mois</p>
                    <hr style="border:1px solid #96B91D44; margin:10px 0;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.95em;">
                        <tr style="background:#96B91D22;">
                            <th style="padding:6px 8px; text-align:left;">Métrique</th>
                            <th style="padding:6px 8px; text-align:center;">Valeur</th>
                            <th style="padding:6px 8px; text-align:center;">Seuil IPMVP</th>
                            <th style="padding:6px 8px; text-align:center;">Statut</th>
                        </tr>
                        <tr>
                            <td style="padding:6px 8px;">R²</td>
                            <td style="padding:6px 8px; text-align:center; font-weight:bold;">{train_r2_val:.4f}</td>
                            <td style="padding:6px 8px; text-align:center; color:#666;">≥ 0.75</td>
                            <td style="padding:6px 8px; text-align:center;">{tr2_ok}</td>
                        </tr>
                        <tr style="background:#96B91D11;">
                            <td style="padding:6px 8px;">CV(RMSE)</td>
                            <td style="padding:6px 8px; text-align:center; font-weight:bold;">{train_cv_val:.4f}</td>
                            <td style="padding:6px 8px; text-align:center; color:#666;">≤ 0.20</td>
                            <td style="padding:6px 8px; text-align:center;">{tcv_ok}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 8px;">Biais (%)</td>
                            <td style="padding:6px 8px; text-align:center; font-weight:bold;">{train_bias_val:.{bias_decimals}f}%</td>
                            <td style="padding:6px 8px; text-align:center; color:#666;">≤ 5%</td>
                            <td style="padding:6px 8px; text-align:center;">{"✅" if abs(train_bias_val) <= 5 else ("⚠️" if abs(train_bias_val) <= 10 else "❌")}</td>
                        </tr>
                    </table>
                    <p style="margin:8px 0 0 0; font-size:0.82em; color:#666;">ℹ️ Biais calculé par validation croisée LOO sur les données d'entraînement.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_test:
                st.markdown(f"""
                <div style="background-color: rgba(109, 186, 188, 0.1); border-left: 4px solid #6DBABC; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #6DBABC; margin: 0 0 12px 0;">🧪 PÉRIODE DE TEST (VALIDATION)</h4>
                    <p style="margin: 4px 0;">📅 <strong>Du :</strong> {test_df_temp[date_col].min().strftime('%b %Y')} &nbsp;→&nbsp; <strong>Au :</strong> {test_df_temp[date_col].max().strftime('%b %Y')}</p>
                    <p style="margin: 4px 0;">📊 <strong>Observations :</strong> {len(test_df_temp)} mois</p>
                    <hr style="border:1px solid #6DBABC44; margin:10px 0;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.95em;">
                        <tr style="background:#6DBABC22;">
                            <th style="padding:6px 8px; text-align:left;">Métrique</th>
                            <th style="padding:6px 8px; text-align:center;">Valeur</th>
                            <th style="padding:6px 8px; text-align:center;">Seuil IPMVP</th>
                            <th style="padding:6px 8px; text-align:center;">Statut</th>
                        </tr>
                        <tr>
                            <td style="padding:6px 8px; font-weight:bold; color:#00485F;">R² ⭐</td>
                            <td style="padding:6px 8px; text-align:center; font-weight:bold; font-size:1.1em;">{test_r2_val:.4f}</td>
                            <td style="padding:6px 8px; text-align:center; color:#666;">≥ 0.75</td>
                            <td style="padding:6px 8px; text-align:center; font-size:1.2em;">{r2_ok}</td>
                        </tr>
                        <tr style="background:#6DBABC11;">
                            <td style="padding:6px 8px; font-weight:bold; color:#00485F;">CV(RMSE) ⭐</td>
                            <td style="padding:6px 8px; text-align:center; font-weight:bold; font-size:1.1em;">{test_cv_val:.4f}</td>
                            <td style="padding:6px 8px; text-align:center; color:#666;">≤ 0.20</td>
                            <td style="padding:6px 8px; text-align:center; font-size:1.2em;">{cv_ok}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 8px; font-weight:bold; color:#00485F;">Biais (%) ⭐</td>
                            <td style="padding:6px 8px; text-align:center; font-weight:bold; font-size:1.1em;">{test_bias_val:.{bias_decimals}f}%</td>
                            <td style="padding:6px 8px; text-align:center; color:#666;">≤ 5%</td>
                            <td style="padding:6px 8px; text-align:center; font-size:1.2em;">{bias_ok}</td>
                        </tr>
                    </table>
                    <p style="margin:8px 0 0 0; font-size:0.82em; color:#00485F; font-weight:bold;">⭐ Métriques sur données NON VUES — indicateurs IPMVP de référence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparaison synthétique Train vs Test
            r2_gap = abs(train_r2_val - test_r2_val)
            gap_color = "#4caf50" if r2_gap < 0.10 else ("#ff9800" if r2_gap < 0.20 else "#f44336")
            gap_icon  = "✅" if r2_gap < 0.10 else ("⚠️" if r2_gap < 0.20 else "❌")
            gap_label = "Excellent - pas d'overfitting" if r2_gap < 0.10 else ("Acceptable" if r2_gap < 0.20 else "Overfitting probable")
            st.markdown(f"""
            <div style="background:#f5f5f5; border-radius:8px; padding:10px 16px; margin-top:12px; display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
                <span style="font-weight:bold; color:#00485F;">📊 Écart R² Train/Test :</span>
                <span style="font-size:1.2em; font-weight:bold; color:{gap_color};">{gap_icon} {r2_gap:.4f}</span>
                <span style="color:#666; font-size:0.9em;">({gap_label})</span>
                <span style="color:#888; font-size:0.85em;">R² Train={train_r2_val:.4f} | R² Test={test_r2_val:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Mode Standard : afficher la période complète avec ses métriques
            r2_std   = best_metrics['r2']
            cv_std   = best_metrics['cv_rmse']
            bias_std = best_metrics['bias']
            r2_ok_s  = "✅" if r2_std  >= 0.75 else ("⚠️" if r2_std  >= 0.60 else "❌")
            cv_ok_s  = "✅" if cv_std  <= 0.20 else ("⚠️" if cv_std  <= 0.30 else "❌")
            bias_ok_s= "✅" if abs(bias_std) <= 5 else ("⚠️" if abs(bias_std) <= 10 else "❌")
            
            st.markdown(f"""
            <div style="background-color: rgba(109, 186, 188, 0.1); border-left: 4px solid #6DBABC; padding: 15px; border-radius: 8px;">
                <h4 style="color: #00485F; margin: 0 0 12px 0;">📅 PÉRIODE ANALYSÉE (MODE STANDARD)</h4>
                <p style="margin: 4px 0;">📅 <strong>Du :</strong> {df_filtered[date_col].min().strftime('%b %Y')} &nbsp;→&nbsp; <strong>Au :</strong> {df_filtered[date_col].max().strftime('%b %Y')}</p>
                <p style="margin: 4px 0;">📊 <strong>Observations :</strong> {len(df_filtered)} mois</p>
                <hr style="border:1px solid #6DBABC44; margin:10px 0;">
                <table style="width:60%; border-collapse:collapse; font-size:0.95em;">
                    <tr style="background:#6DBABC22;">
                        <th style="padding:6px 8px; text-align:left;">Métrique</th>
                        <th style="padding:6px 8px; text-align:center;">Valeur</th>
                        <th style="padding:6px 8px; text-align:center;">Seuil IPMVP</th>
                        <th style="padding:6px 8px; text-align:center;">Statut</th>
                    </tr>
                    <tr>
                        <td style="padding:6px 8px;">R²</td>
                        <td style="padding:6px 8px; text-align:center; font-weight:bold;">{r2_std:.4f}</td>
                        <td style="padding:6px 8px; text-align:center; color:#666;">≥ 0.75</td>
                        <td style="padding:6px 8px; text-align:center;">{r2_ok_s}</td>
                    </tr>
                    <tr style="background:#6DBABC11;">
                        <td style="padding:6px 8px;">CV(RMSE)</td>
                        <td style="padding:6px 8px; text-align:center; font-weight:bold;">{cv_std:.4f}</td>
                        <td style="padding:6px 8px; text-align:center; color:#666;">≤ 0.20</td>
                        <td style="padding:6px 8px; text-align:center;">{cv_ok_s}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 8px;">Biais (%)</td>
                        <td style="padding:6px 8px; text-align:center; font-weight:bold;">{bias_std:.{bias_decimals}f}%</td>
                        <td style="padding:6px 8px; text-align:center; color:#666;">≤ 5%</td>
                        <td style="padding:6px 8px; text-align:center;">{bias_ok_s}</td>
                    </tr>
                </table>

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
        
        # INFORMATIONS COMPLÉMENTAIRES DU MODÈLE
        st.markdown(f"""
        <div class="metrics-card" style="margin-top:12px;">
            <h4>🔍 Informations du modèle sélectionné</h4>
            <p style="margin:4px 0;"><strong>Variables :</strong> {', '.join(best_features)}</p>
            <p style="margin:4px 0;"><strong>Nb variables :</strong> {len(best_features)} &nbsp;|&nbsp; <strong>Observations totales :</strong> {len(df_filtered)} &nbsp;|&nbsp; <strong>Ratio obs/var :</strong> {len(df_filtered)/len(best_features):.1f}:1</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ALERTE R² NÉGATIF SUR TEST
        if best_metrics.get('mode') == 'train_test' and best_metrics.get('test_r2', best_metrics['r2']) < 0:
            st.error(f"""⚠️ **R² TEST NÉGATIF** : Le modèle est moins performant qu'une simple moyenne sur les données de test.
            
**Actions recommandées :**
1. Augmenter la période d'entraînement (slider "Mois d'entraînement")
2. Vérifier la cohérence des variables sur toute la période
3. Réduire le nombre de variables (risque d'overfitting)
""")
        
        # ÉQUATION DU MODÈLE
        st.subheader("📝 Équation d'ajustement")
        
        try:
            intercept_val = best_metrics.get('intercept', 0.0)
            coefs_val = best_metrics.get('coefficients', {})
            model_type_val = best_metrics.get('model_type', 'Linéaire')
            
            if model_type_val in ["Linéaire", "Ridge", "Lasso"]:
                # Ne garder que les features présentes dans les coefficients
                coefs_display = {f: coefs_val[f] for f in best_features if f in coefs_val}
                equation = format_equation(intercept_val, coefs_display)
            elif model_type_val == "Polynomiale":
                equation = format_equation(intercept_val, coefs_val)
            else:
                equation = f"Consommation = {intercept_val:.4f} (équation non disponible pour ce type de modèle)"
        except Exception as e_eq:
            equation = f"⚠️ Équation non disponible : {str(e_eq)}"
        
        st.markdown(f"""
        <div class="equation-box">
            <h4>🧮 Équation mathématique :</h4>
            <p style="font-size: 16px; font-weight: bold;">{equation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VALEURS T DE STUDENT POUR MODÈLES LINÉAIRES
        if 't_stats' in best_metrics and best_metrics.get('model_type') in ["Linéaire", "Ridge", "Lasso"]:
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
            train_df, test_df, split_date = create_train_test_split(df_filtered, date_col, train_months_manual)
            
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
        
        # ── Configuration matplotlib globale ──────────────────────────────────
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'axes.facecolor': '#FAFAFA',
            'figure.facecolor': '#E7DDD9',
            'axes.edgecolor': '#CCCCCC',
            'axes.labelcolor': '#00485F',
            'axes.titlecolor': '#00485F',
            'xtick.color': '#444444',
            'ytick.color': '#444444',
            'grid.color': '#DDDDDD',
            'grid.alpha': 0.6,
        })

        # ── Conversion en numpy (évite bug indexation pandas Series) ─────────
        y_all_arr   = np.array(y_all)
        y_pred_arr  = np.array(y_pred_all)
        residus_arr = y_all_arr - y_pred_arr
        train_idx   = list(train_indices)
        test_idx    = list(test_indices)
        mode_tt     = best_metrics.get('mode') == 'train_test'

        # ── GRAPHIQUE 1 : Mesurée vs Ajustée (temporel) ──────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        x_all = np.arange(len(y_all_arr))

        if mode_tt:
            ax.bar(train_idx, y_all_arr[train_idx],  color="#96B91D", alpha=0.65, width=0.8, label="Mesurée — Train")
            ax.bar(test_idx,  y_all_arr[test_idx],   color="#6DBABC", alpha=0.65, width=0.8, label="Mesurée — Test")
            ax.plot(train_idx, y_pred_arr[train_idx], color="#2E7D32", marker='o', linewidth=2, markersize=4, label="Ajustée — Train")
            ax.plot(test_idx,  y_pred_arr[test_idx],  color="#005F7A", marker='s', linewidth=2, markersize=4, label="Ajustée — Test")
            if train_idx:
                ax.axvline(x=max(train_idx) + 0.5, color='#CC3333', linestyle='--', linewidth=1.5, alpha=0.8, label="Séparation Train / Test")
        else:
            ax.bar(x_all, y_all_arr, color="#6DBABC", alpha=0.7, width=0.8, label="Mesurée")
            ax.plot(x_all, y_pred_arr, color="#96B91D", marker='o', linewidth=2, markersize=4, label="Ajustée")

        ax.set_title("Consommation mesurée vs ajustée", fontweight='bold', fontsize=15, pad=14)
        ax.set_xlabel("Observations", fontsize=11)
        ax.set_ylabel("Consommation", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10, loc="upper left")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        suffix    = " (Test)" if mode_tt else ""
        annot_txt = f"R² = {best_metrics['r2']:.3f}{suffix}   CV(RMSE) = {best_metrics['cv_rmse']:.3f}{suffix}   Score IPMVP {best_metrics['ipmvp_score']:.1f}/70"
        ax.annotate(annot_txt, xy=(0.5, 0.97), xycoords='axes fraction',
                    fontsize=10, color='#00485F', ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── GRAPHIQUES 2 ET 3 : Dispersion + Résidus ─────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            fig2, ax2 = plt.subplots(figsize=(7, 6))

            if mode_tt:
                ax2.scatter(y_all_arr[train_idx], y_pred_arr[train_idx],
                            color="#96B91D", alpha=0.85, s=55, edgecolor='white', linewidth=0.5, label="Train", zorder=3)
                ax2.scatter(y_all_arr[test_idx],  y_pred_arr[test_idx],
                            color="#6DBABC", alpha=0.85, s=55, edgecolor='white', linewidth=0.5, label="Test",  zorder=4)
                ax2.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)
            else:
                ax2.scatter(y_all_arr, y_pred_arr,
                            color="#6DBABC", alpha=0.85, s=55, edgecolor='white', linewidth=0.5, zorder=3)

            _mn = min(y_all_arr.min(), y_pred_arr.min()) * 0.97
            _mx = max(y_all_arr.max(), y_pred_arr.max()) * 1.03
            ax2.plot([_mn, _mx], [_mn, _mx], '--', color='#00485F', linewidth=1.5, alpha=0.7, label="Droite ideale")

            r2_disp = f"R2 (Test) = {best_metrics['r2']:.4f}" if mode_tt else f"R2 = {best_metrics['r2']:.4f}"
            ax2.annotate(r2_disp, xy=(0.05, 0.95), xycoords='axes fraction',
                         fontsize=11, fontweight='bold', color='#00485F', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

            ax2.set_title("Mesurée vs Prédite", fontweight='bold', fontsize=13)
            ax2.set_xlabel("Consommation mesurée", fontsize=10)
            ax2.set_ylabel("Consommation prédite", fontsize=10)
            ax2.grid(linestyle='--', alpha=0.4)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with col2:
            fig3, ax3 = plt.subplots(figsize=(7, 6))

            if mode_tt:
                ax3.scatter(train_idx, residus_arr[train_idx],
                            color="#96B91D", alpha=0.85, s=55, edgecolor='white', linewidth=0.5, label="Train", zorder=3)
                ax3.scatter(test_idx,  residus_arr[test_idx],
                            color="#6DBABC", alpha=0.85, s=55, edgecolor='white', linewidth=0.5, label="Test",  zorder=4)
                ax3.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)
            else:
                ax3.scatter(range(len(residus_arr)), residus_arr,
                            color="#96B91D", alpha=0.85, s=55, edgecolor='white', linewidth=0.5, zorder=3)

            ax3.axhline(y=0, color='#CC3333', linestyle='--', linewidth=1.5, alpha=0.8)
            ax3.annotate(f"Biais = {best_metrics['bias']:.{bias_decimals}f}%",
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         fontsize=11, fontweight='bold', color='#00485F', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

            ax3.set_title("Residus", fontweight='bold', fontsize=13)
            ax3.set_xlabel("Observations", fontsize=10)
            ax3.set_ylabel("Residu (mesure - predit)", fontsize=10)
            ax3.grid(linestyle='--', alpha=0.4)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
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
                
                qual, _, _ = get_ipmvp_qualification(model['ipmvp_score'])
                model_row = {
                    "🏆": f"{i+1}",
                    "Score": f"**{model['ipmvp_score']:.1f}**/60",
                    "Qualification": qual,
                    "Mode": mode_icon,
                    "Type": model['model_name'][:20] + ("..." if len(model['model_name']) > 20 else ""),
                    "Variables": ", ".join(model['features'][:2]) + ("..." if len(model['features']) > 2 else ""),
                    "R²": f"{model['r2']:.3f}",
                    "CV(RMSE)": f"{model['cv_rmse']:.3f}",
                    "Biais(%)": f"{model['bias']:.{bias_decimals}f}",
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
            
            #### 📊 Score de base (70 points max)
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
            
            **🏆 Qualification obtenue :** {qualification} ({best_metrics['ipmvp_score']:.1f}/70 points)
            
            **Échelle de notation :**
            - **Excellent** (≥55/60) : Modèle très robuste, hautement conforme IPMVP
            - **Très bon** (45-54/60) : Modèle fiable et robuste pour M&V
            - **Bon** (35-44/60) : Modèle valide avec améliorations possibles
            - **Correct** (25-34/60) : Modèle acceptable, révision recommandée
            - **Non conforme** (<25/60) : Modèle insuffisant, révision majeure requise
            
            **Composition du score :**
            - R² : 30 points max (seuil IPMVP : 0.75)
            - CV(RMSE) : 30 points max (seuil IPMVP : 0.20 soit 20%)
            - Significativité (t-stats) : 10 points max (seuil : |t| ≥ 2)
            
            **📊 Mode d'analyse :** {best_metrics.get('mode', 'standard').title()}
            {'- Validation sur données non-vues (train/test)' if best_metrics.get('mode') == 'train_test' else '- Analyse sur toutes les données disponibles'}
            {'- Plus robuste mais nécessite ≥18 mois' if best_metrics.get('mode') == 'train_test' else '- Standard IPMVP avec protections renforcées'}
            
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
            
            if best_metrics.get("mode") == "standard" and len(df_filtered) > 12:
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
        qualification, qual_class, status_color = get_ipmvp_qualification(best_metrics['ipmvp_score'])
        
        # Messages selon la qualification
        status_messages = {
            "Excellent": ("✅ **MODÈLE EXCELLENT**", "Modèle hautement fiable, conforme aux standards IPMVP les plus exigeants."),
            "Très bon": ("✅ **MODÈLE TRÈS BON**", "Modèle fiable et robuste, parfaitement adapté aux calculs M&V selon IPMVP."),
            "Bon": ("✅ **MODÈLE BON**", "Modèle valide pour utilisation IPMVP avec quelques améliorations possibles."),
            "Correct": ("⚠️ **MODÈLE CORRECT**", "Modèle acceptable mais présentant des limitations, révision recommandée."),
            "Non conforme": ("❌ **MODÈLE NON CONFORME**", "Modèle ne respectant pas les standards IPMVP, révision majeure nécessaire.")
        }
        
        status, status_msg = status_messages.get(qualification, status_messages["Non conforme"])
        
        # Affichage du résumé avec composants natifs Streamlit (plus fiable)
        st.markdown(f"### {status}")
        st.info(status_msg)
        
        # Métriques en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            qualification, qual_class, qual_color = get_ipmvp_qualification(best_metrics['ipmvp_score'])
            st.metric(
                label="🏆 Qualification IPMVP",
                value=qualification,
                delta=f"{best_metrics['ipmvp_score']:.1f}/70 pts",
                help="Qualification basée sur R², CV(RMSE) et significativité statistique"
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
                value=f"{best_metrics['bias']:.{bias_decimals}f}%",
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
    - **🚀 Mode train/test** : Validation robuste si ≥18 mois de données
    - **⚠️ Limitations intelligentes** : Contrôle automatique du ratio observations/variables
    - **📊 Métriques enrichies** : Significativité statistique, comparaisons train/test
    """)

# PIED DE PAGE FINAL
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #00485F 0%, #005F7A 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-top: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
">
    <div>
        <p style="color:white; font-size:1.1em; font-weight:700; margin:0;">Calculette IPMVP</p>
        <p style="color:rgba(255,255,255,0.65); font-size:0.82em; margin:4px 0 0 0;">
            Modélisation · Scoring · Validation · Conformité IPMVP
        </p>
    </div>
    <div style="text-align:right;">
        <p style="color:rgba(255,255,255,0.5); font-size:0.78em; margin:0;">
            R² &nbsp;·&nbsp; CV(RMSE) &nbsp;·&nbsp; Biais &nbsp;·&nbsp; T-stats
        </p>
        <p style="color:rgba(255,255,255,0.35); font-size:0.72em; margin:4px 0 0 0;">
            Efficacité Energétique & Carbone team
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

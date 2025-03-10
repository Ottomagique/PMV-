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

# Image SVG d'efficacité énergétique pour la page de connexion
def get_energy_efficiency_svg():
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
      <!-- Fond du ciel -->
      <defs>
        <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#E7DDD9" />
          <stop offset="100%" stop-color="#6DBABC" />
        </linearGradient>
      </defs>
      <rect width="800" height="400" fill="url(#skyGradient)" />
      
      <!-- Soleil avec rayons énergétiques -->
      <circle cx="650" cy="100" r="50" fill="#FFD700" />
      <g opacity="0.7">
        <path d="M650 30 L650 10" stroke="#FFD700" stroke-width="4" />
        <path d="M650 190 L650 170" stroke="#FFD700" stroke-width="4" />
        <path d="M580 100 L560 100" stroke="#FFD700" stroke-width="4" />
        <path d="M740 100 L720 100" stroke="#FFD700" stroke-width="4" />
        <path d="M600 50 L580 30" stroke="#FFD700" stroke-width="4" />
        <path d="M700 150 L720 170" stroke="#FFD700" stroke-width="4" />
        <path d="M600 150 L580 170" stroke="#FFD700" stroke-width="4" />
        <path d="M700 50 L720 30" stroke="#FFD700" stroke-width="4" />
      </g>
      
      <!-- Bâtiments modernes et efficaces énergétiquement -->
      <!-- Tour principale avec panneaux solaires -->
      <rect x="100" y="120" width="120" height="280" fill="#00485F" rx="5" />
      <rect x="110" y="140" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="170" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="200" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="230" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="260" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="290" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="320" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="110" y="350" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="100" y="100" width="120" height="20" fill="#96B91D" rx="2" />
      
      <!-- Immeuble avec système de végétalisation -->
      <rect x="250" y="180" width="100" height="220" fill="#00485F" rx="5" />
      <rect x="260" y="200" width="80" height="15" fill="#6DBABC" rx="2" />
      <rect x="260" y="225" width="80" height="15" fill="#6DBABC" rx="2" />
      <rect x="260" y="250" width="80" height="15" fill="#6DBABC" rx="2" />
      <rect x="260" y="275" width="80" height="15" fill="#6DBABC" rx="2" />
      <rect x="260" y="300" width="80" height="15" fill="#6DBABC" rx="2" />
      <rect x="260" y="325" width="80" height="15" fill="#6DBABC" rx="2" />
      <rect x="260" y="350" width="80" height="15" fill="#6DBABC" rx="2" />
      <path d="M250 180 C250 170, 270 160, 300 160 C330 160, 350 170, 350 180" fill="#96B91D" />
      <path d="M260 160 C280 140, 320 140, 340 160" fill="#96B91D" />
      
      <!-- Immeuble de bureaux smart building -->
      <rect x="380" y="150" width="130" height="250" fill="#00485F" rx="5" />
      <rect x="395" y="170" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="395" y="200" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="395" y="230" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="395" y="260" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="395" y="290" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="395" y="320" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="395" y="350" width="100" height="20" fill="#6DBABC" rx="2" />
      <rect x="380" y="140" width="130" height="10" fill="#96B91D" />
      <rect x="420" y="110" width="50" height="30" fill="#00485F" rx="5" />
      
      <!-- Bâtiment avec design futuriste intelligent -->
      <path d="M540 400 L540 200 L600 150 L660 200 L660 400 Z" fill="#00485F" />
      <path d="M550 380 L550 210 L600 170 L650 210 L650 380 Z" fill="#E7DDD9" />
      <rect x="570" y="220" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="595" y="220" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="620" y="220" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="570" y="260" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="595" y="260" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="620" y="260" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="570" y="300" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="595" y="300" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="620" y="300" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="570" y="340" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="595" y="340" width="15" height="30" fill="#6DBABC" rx="2" />
      <rect x="620" y="340" width="15" height="30" fill="#6DBABC" rx="2" />
      
      <!-- Éléments d'énergie renouvelable -->
      <!-- Éoliennes -->
      <line x1="700" y1="400" x2="700" y2="280" stroke="#00485F" stroke-width="8" />
      <circle cx="700" cy="280" r="5" fill="#E7DDD9" />
      <path d="M700 280 L730 250" stroke="#6DBABC" stroke-width="4" />
      <path d="M700 280 L670 250" stroke="#6DBABC" stroke-width="4" />
      <path d="M700 280 L700 240" stroke="#6DBABC" stroke-width="4" />
      
      <!-- Panneaux solaires -->
      <rect x="40" y="380" width="40" height="20" fill="#00485F" />
      <rect x="40" y="360" width="40" height="20" fill="#6DBABC" transform="skewX(-15)" />
      
      <!-- Symbole d'efficacité énergétique -->
      <circle cx="445" cy="70" r="30" fill="#96B91D" opacity="0.9" />
      <path d="M445 50 L445 90 M425 70 L465 70" stroke="white" stroke-width="5" />
      <path d="M430 55 L460 85 M430 85 L460 55" stroke="white" stroke-width="3" />
      
      <!-- Flux de données / énergie entre les bâtiments -->
      <path d="M220 200 C250 180, 250 220, 280 200" stroke="#96B91D" stroke-width="2" stroke-dasharray="5,3" />
      <path d="M350 250 C365 230, 365 270, 380 250" stroke="#96B91D" stroke-width="2" stroke-dasharray="5,3" />
      <path d="M510 230 C525 210, 525 250, 540 230" stroke="#96B91D" stroke-width="2" stroke-dasharray="5,3" />
      
      <!-- Symboles d'énergie verte autour des bâtiments -->
      <circle cx="160" cy="80" r="15" fill="#96B91D" opacity="0.7" />
      <circle cx="300" cy="140" r="12" fill="#96B91D" opacity="0.7" />
      <circle cx="445" cy="100" r="10" fill="#96B91D" opacity="0.7" />
      <circle cx="600" cy="130" r="14" fill="#96B91D" opacity="0.7" />
      
      <!-- Titre de l'application -->
      <text x="400" y="50" font-family="Arial" font-size="30" fill="#00485F" text-anchor="middle" font-weight="bold">Analyse IPMVP</text>
    </svg>
    '''

# Fonction pour afficher le formulaire de connexion
def show_login_form():
    st.markdown("""
    <style>
    .login-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #E7DDD9;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #00485F;
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
    }
    .svg-container {
        max-width: 800px;
        margin: 0 auto 2rem auto;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    </style>
    
    <div class="svg-container">
        %s
    </div>
    
    <div class="login-container">
        <div class="login-header">
            <h2 style="color: #00485F;">Calcul IPMVP</h2>
            <p>Veuillez vous connecter pour accéder à l'application</p>
        </div>
    </div>
    """ % get_energy_efficiency_svg(), unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter", use_container_width=True)
        
        if submit:
            if check_credentials(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.session_state['is_admin'] = is_admin(username)
                st.rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")
    
    st.markdown("""
    <div class="login-footer">
        <p>Développé avec ❤️ par <strong>Efficacité Energétique, Carbone & RSE team</strong> © 2025</p>
        <p>Outil d'analyse et de modélisation énergétique conforme IPMVP</p>
    </div>
    """, unsafe_allow_html=True)

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
                    st.success(f"Utilisateur '{username}' en

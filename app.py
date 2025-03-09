import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime, timedelta
from itertools import combinations
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Configuration de la page
st.set_page_config(
    page_title="Analyse IPMVP Simplifiée",
    page_icon="📊",
    layout="wide"
)

# Ajouter du CSS personnalisé
st.markdown("""
<style>
    .reportview-container {
        background-color: #F5F7FA;
    }
    .main {
        background-color: #F5F7FA;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
    }
    .metric-container {
        background-color: #E0E8F0;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .formula-box {
        background-color: #E0E8F0;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 5px;
        border-radius: 3px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #E0E8F0;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    
    /* Amélioration des tableaux */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 14px;
        border-radius: 5px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
    }
    th {
        background-color: #1E88E5;
        color: white;
        text-align: left;
        padding: 12px 15px;
    }
    td {
        padding: 12px 15px;
    }
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    tr:hover {
        background-color: #e6f3ff;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("📊 Analyse IPMVP")
st.markdown("""
<div class="card">
    <p>Cette application vous permet d'analyser vos données de consommation énergétique selon le protocole IPMVP.</p>
    <p>Importez un fichier Excel avec au minimum une colonne de dates et une colonne de consommations,
    plus des colonnes optionnelles pour les variables explicatives comme les DJU, effectifs, etc.</p>
    <p>Le modèle analysera automatiquement 12 mois glissants et trouvera la meilleure combinaison de variables.</p>
</div>
""", unsafe_allow_html=True)

# Définition des fonctions d'analyse IPMVP
@st.cache_data
def evaluer_combinaison(X, y, features, _type="linear"):
    """Évalue une combinaison de variables et retourne les métriques"""
    X_subset = X[features]
    
    if _type == "poly":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_subset = poly.fit_transform(X_subset)
        model = LinearRegression()
    else:
        model = LinearRegression()
    
    model.fit(X_subset, y)
    y_pred = model.predict(X_subset)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    cv = rmse / np.mean(y) if np.mean(y) != 0 else np.inf
    bias = np.mean(y_pred - y) / np.mean(y) if np.mean(y) != 0 else np.inf
    
    conforme = r2 > 0.75 and abs(cv) < 0.2 and abs(bias) < 0.01
    
    return {
        'r2': r2,
        'cv': cv,
        'bias': bias,
        'model': model,
        'conforme': conforme,
        'y_pred': y_pred
    }

class ModelIPMVP:
    def __init__(self):
        self.best_model = None
        self.best_features = None
        self.best_formula = None
        self.best_r2 = 0
        self.best_cv = None
        self.best_bias = None
        self.best_model_type = None
        self.best_coefficients = None
        self.best_intercept = None
        self.best_y_pred = None
    
    def trouver_meilleur_modele(self, X, y, max_features=4, progress_callback=None):
        """Trouve le meilleur modèle en testant différentes combinaisons de variables"""
        # Si pas assez de données, retourner directement False
        if len(y) < 12:
            st.warning("L'analyse IPMVP nécessite au moins 12 mois de données.")
            return False
            
        # Recherche rapide: commencer par vérifier les colonnes DJU
        dju_colonnes = []
        for col in X.columns:
            if 'dju' in str(col).lower() or 'degre' in str(col).lower():
                dju_colonnes.append(col)
        
        # Tester d'abord les modèles DJU seuls
        for dju_col in dju_colonnes:
            result = evaluer_combinaison(X, y, [dju_col])
            if result['conforme']:
                self._update_best_model(result, [dju_col], "Linéaire (E = a×DJU + c)", X, y)
                return True
        
        # Ensuite, tester toutes les combinaisons possibles
        max_features = min(max_features, len(X.columns))
        models_tested = 0
        total_models = sum(len(list(combinations(X.columns, i))) for i in range(1, max_features + 1))
        
        for n_features in range(1, max_features + 1):
            feature_combos = list(combinations(X.columns, n_features))
            for i, feature_subset in enumerate(feature_combos):
                feature_subset = list(feature_subset)
                
                # Mettre à jour la progression
                models_tested += 1
                if progress_callback:
                    progress_callback(models_tested / total_models)
                
                # Évaluer le modèle linéaire
                result = evaluer_combinaison(X, y, feature_subset)
                if result['conforme'] and result['r2'] > self.best_r2:
                    self._update_best_model(result, feature_subset, "Linéaire", X, y)
                
                # Évaluer le modèle polynomial (uniquement pour 1-2 variables)
                if n_features <= 2:
                    result = evaluer_combinaison(X, y, feature_subset, _type="poly")
                    if result['conforme'] and result['r2'] > self.best_r2:
                        self._update_best_model(result, feature_subset, "Polynomiale (degré 2)", X, y)
        
        return self.best_model is not None
    
    def _update_best_model(self, result, features, model_type, X, y):
        """Met à jour le meilleur modèle avec les résultats"""
        self.best_r2 = result['r2']
        self.best_cv = result['cv']
        self.best_bias = result['bias']
        self.best_model = result['model']
        self.best_model_type = model_type
        self.best_features = features
        self.best_y_pred = result['y_pred']
        
        if hasattr(self.best_model, 'coef_'):
            self.best_coefficients = self.best_model.coef_
            self.best_intercept = self.best_model.intercept_
        
        self._construire_formule()
    
    def _construire_formule(self):
        """Construit la formule du meilleur modèle"""
        if self.best_model is None:
            self.best_formula = "Aucun modèle valide trouvé"
            return
        
        formula = f"{self.best_intercept:.4f}"
        for i, coef in enumerate(self.best_coefficients):
            feature_name = self.best_features[i]
            formula += f" + {coef:.4f} × ({feature_name})"
            
        self.best_formula = formula
    
    def generer_rapport(self):
        """Génère un rapport structuré sur le meilleur modèle"""
        if self.best_model is None:
            return """
            <div class="card" style="border-left: 5px solid #dc3545;">
                <h3 style="color: #dc3545;">❌ Aucun modèle valide</h3>
                <p>L'algorithme n'a pas pu trouver de modèle conforme aux critères IPMVP avec les données fournies.</p>
            </div>
            """
        
        # Formatage des métriques
        r2_formatted = f"{self.best_r2:.4f}"
        cv_formatted = f"{self.best_cv:.4f}"
        bias_formatted = f"{self.best_bias:.8f}"
        
        # Créer le statut des critères
        r2_status = "✅" if self.best_r2 > 0.75 else "❌"
        cv_status = "✅" if self.best_cv < 0.2 else "❌"
        bias_status = "✅" if abs(self.best_bias) < 0.01 else "❌"
        
        # Formater la liste des variables
        variables_list = "<ul>" + "".join([f"<li>{var}</li>" for var in self.best_features]) + "</ul>"
        
        rapport = f"""
        <div class="card" style="border-left: 5px solid #28a745;">
            <h3 style="color: #28a745;">✅ Modèle IPMVP conforme</h3>
            <p>Type de modèle: <span class="highlight">{self.best_model_type}</span></p>
            
            <h4>Variables sélectionnées:</h4>
            {variables_list}
            
            <h4>Formule d'ajustement:</h4>
            <div class="formula-box">
                {self.best_formula}
            </div>
            
            <h4>Métriques de performance:</h4>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #E0E8F0;">
                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Métrique</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Valeur</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Seuil IPMVP</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Statut</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">R² (coefficient de détermination)</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{r2_formatted}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">> 0.75</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{r2_status}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">CV(RMSE) (coefficient de variation)</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{cv_formatted}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">< 0.2</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{cv_status}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">NMBE (biais normalisé)</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{bias_formatted}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">< 0.01</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{bias_status}</td>
                </tr>
            </table>
        </div>
        """
        return rapport
    
    def visualiser_resultats(self, X, y, dates=None):
        """Crée des visualisations pour le meilleur modèle et retourne les figures"""
        if self.best_model is None:
            return None, None, None
        
        # Calculer les prédictions
        X_subset = X[self.best_features]
        if "Polynomiale" in self.best_model_type:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_subset = poly.fit_transform(X_subset)
        
        y_pred = self.best_model.predict(X_subset)
        
        # Créer un DataFrame pour l'analyse
        results_df = pd.DataFrame({
            'Valeur_Réelle': y,
            'Valeur_Prédite': y_pred,
            'Erreur': y - y_pred,
            'Erreur_Pourcentage': (y - y_pred) / y * 100 if np.mean(y) != 0 else np.zeros_like(y)
        })
        
        # Ajouter les dates si disponibles
        if dates is not None:
            results_df['Date'] = dates
        
        # Créer les graphiques
        fig1, fig2 = self._creer_graphiques(results_df)
        
        return results_df, fig1, fig2
    
    def _creer_graphiques(self, results_df):
        """Crée et retourne les visualisations"""
        # Utiliser un style plus moderne pour les graphiques
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Graphique 1: Analyse du modèle
        fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig1.suptitle('Analyse du modèle IPMVP', fontsize=16, y=0.98)
        
        # Valeurs réelles vs prédites
        axes[0, 0].scatter(results_df['Valeur_Réelle'], results_df['Valeur_Prédite'], alpha=0.7, 
                         color='#1E88E5', edgecolors='navy')
        min_val = min(results_df['Valeur_Réelle'].min(), results_df['Valeur_Prédite'].min())
        max_val = max(results_df['Valeur_Réelle'].max(), results_df['Valeur_Prédite'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        axes[0, 0].set_title('Valeurs Réelles vs Prédites', fontsize=12)
        axes[0, 0].set_xlabel('Valeurs Réelles')
        axes[0, 0].set_ylabel('Valeurs Prédites')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution des erreurs
        sns.histplot(results_df['Erreur'], kde=True, ax=axes[0, 1], color='#1E88E5', bins=10)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Distribution des Erreurs', fontsize=12)
        axes[0, 1].set_xlabel('Erreur')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Erreurs vs valeurs prédites
        axes[1, 0].scatter(results_df['Valeur_Prédite'], results_df['Erreur'], alpha=0.7,
                         color='#1E88E5', edgecolors='navy')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Erreurs vs Valeurs Prédites', fontsize=12)
        axes[1, 0].set_xlabel('Valeurs Prédites')
        axes[1, 0].set_ylabel('Erreur')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Importance des variables
        if len(self.best_features) > 0:
            coefs = pd.DataFrame({
                'Variable': self.best_features,
                'Coefficient': np.abs(self.best_coefficients)
            })
            coefs = coefs.sort_values('Coefficient', ascending=False)
            barplot = sns.barplot(x='Coefficient', y='Variable', data=coefs, ax=axes[1, 1], palette=['#1E88E5'])
            axes[1, 1].set_title('Importance des Variables', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(coefs['Coefficient']):
                axes[1, 1].text(v + 0.01, i, f"{v:.4f}", va='center')
        
        plt.tight_layout()
        
        # Graphique 2: Consommation mesurée vs calculée
        fig2 = plt.figure(figsize=(14, 7))
        
        if 'Date' in results_df.columns:
            # Trier par date
            results_df = results_df.sort_values('Date')
            
            # Créer un graphique plus élégant
            plt.bar(range(len(results_df)), results_df['Valeur_Réelle'], color='#4285F4', 
                   width=0.6, label='Consommation mesurée', alpha=0.7)
            
            plt.plot(range(len(results_df)), results_df['Valeur_Prédite'], color='#EA4335',
                    marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Consommation ajustée')
            
            # Formater l'axe des x avec les dates
            date_labels = [d.strftime('%b-%y') if hasattr(d, 'strftime') else d for d in results_df['Date']]
            plt.xticks(range(len(results_df)), date_labels, rotation=45)
        else:
            plt.bar(range(len(results_df)), results_df['Valeur_Réelle'], color='#4285F4', 
                   width=0.6, label='Consommation mesurée', alpha=0.7)
            plt.plot(range(len(results_df)), results_df['Valeur_Prédite'], color='#EA4335',
                    marker='o', linestyle='-', linewidth=2.5, markersize=8, label='Consommation ajustée')
        
        plt.title('Comparaison Consommation Mesurée vs Ajustée', fontsize=16)
        plt.ylabel('Consommation')
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Ajouter la formule d'ajustement
        plt.figtext(0.5, 0.01, f"Formule d'ajustement: {self.best_formula}", 
                   ha='center', fontsize=11, bbox={"facecolor":"#E0E8F0", "alpha":0.8, "pad":5, 
                                                 "boxstyle":"round,pad=0.5"})
        
        plt.tight_layout(pad=3)
        
        return fig1, fig2

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    """Charge les données depuis un fichier Excel"""
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

# Fonction pour vérifier qu'on a au moins 12 mois de données
def verifier_periode_12_mois(df, date_col):
    """Vérifie que les données couvrent au moins 12 mois"""
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    dates = df[date_col].dt.to_period('M').unique()
    if len(dates) < 12:
        st.warning(f"Les données ne couvrent que {len(dates)} mois. L'analyse IPMVP recommande au moins 12 mois.")
        return False
    return True

# Interface utilisateur
st.sidebar.header("Configuration")

# Chargement des données
st.sidebar.subheader("1. Chargement des données")
uploaded_file = st.sidebar.file_uploader("Fichier Excel de consommation", type=["xlsx", "xls"])

# Exemple de données
if not uploaded_file:
    st.info("👈 Chargez votre fichier Excel ou utilisez les données d'exemple ci-dessous.")
    
    # Créer des données d'exemple sur 12 mois
    example_data = {
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='MS'),
        'Consommation': [570, 467, 490, 424, 394, 350, 320, 310, 370, 420, 480, 540],
        'DJU_Base_18': [460, 380, 320, 240, 150, 50, 20, 30, 130, 230, 350, 430],
        'Effectif': [100, 100, 100, 98, 98, 95, 90, 90, 95, 98, 100, 100]
    }
    example_df = pd.DataFrame(example_data)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Exemple de données")
    st.dataframe(example_df)
    
    use_example = st.button("Utiliser ces données d'exemple")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if use_example:
        st.session_state.df = example_df
        st.success("Données d'exemple chargées!")
        st.rerun()

# Si des données sont chargées (fichier ou exemple)
df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
elif hasattr(st.session_state, 'df'):
    df = st.session_state.df

if df is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Données chargées")
    st.dataframe(df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Identification automatique des colonnes
    date_col = None
    conso_col = None
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in str(col).lower():
            date_col = col
        elif 'conso' in str(col).lower() or 'energy' in str(col).lower() or 'energie' in str(col).lower():
            conso_col = col
    
    # Sélection des colonnes via sidebar
    st.sidebar.subheader("2. Sélection des colonnes")
    date_col = st.sidebar.selectbox("Colonne de date", 
                                   options=df.columns, 
                                   index=df.columns.get_loc(date_col) if date_col else 0)
    
    conso_col = st.sidebar.selectbox("Colonne de consommation", 
                                    options=df.columns, 
                                    index=df.columns.get_loc(conso_col) if conso_col else min(1, len(df.columns)-1))
    
    # Vérifier que la colonne de date est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            st.error(f"La colonne {date_col} n'a pas pu être convertie en date.")
            st.stop()
    
    # Vérifier qu'on a au moins 12 mois de données
    verifier_periode_12_mois(df, date_col)
    
    # Préparation des variables explicatives
    var_options = [col for col in df.columns if col != date_col and col != conso_col]
    
    # Détecter et sélectionner les variables non-vides
    var_non_vides = []
    for col in var_options:
        if not df[col].isna().all() and not (df[col] == 0).all():
            var_non_vides.append(col)
    
    st.sidebar.subheader("3. Variables explicatives")
    selected_vars = st.sidebar.multiselect(
        "Sélectionnez les variables explicatives", 
        options=var_options,
        default=var_non_vides[:4]  # Prendre jusqu'à 4 variables non-vides par défaut
    )
    
    # Configuration du modèle
    st.sidebar.subheader("4. Configuration du modèle")
    
    # Correction du slider avec gestion des cas problématiques
    if selected_vars:
        max_vars = min(4, len(selected_vars))
        max_features = st.sidebar.slider(
            "Nombre maximum de variables à combiner", 
            min_value=1, 
            max_value=max(2, max_vars),  # Assurez-vous que max_value est au moins 2
            value=min(2, max_vars)
        )
    else:
        st.sidebar.warning("Aucune variable explicative sélectionnée. L'analyse sera limitée.")
        max_features = 1  # Valeur par défaut si aucune variable n'est sélectionnée
    
    # Bouton pour lancer l'analyse
    if st.sidebar.button("🚀 Lancer l'analyse IPMVP"):
        st.subheader("Analyse IPMVP en cours...")
        
        # Préparer les données (juste les 12 derniers mois si plus)
        df_sorted = df.sort_values(by=date_col)
        periodes = df_sorted[date_col].dt.to_period('M').unique()
        
        # Si plus de 12 mois, prendre les 12 derniers
        if len(periodes) > 12:
            st.info(f"Analyse sur les 12 derniers mois disponibles (sur {len(periodes)} mois au total)")
            latest_months = periodes[-12:]
            df_analysis = df_sorted[df_sorted[date_col].dt.to_period('M').isin(latest_months)]
        else:
            df_analysis = df_sorted
        
        # Afficher la période analysée
        start_date = df_analysis[date_col].min().strftime('%Y-%m-%d')
        end_date = df_analysis[date_col].max().strftime('%Y-%m-%d')
        st.markdown(f'<div class="card"><p><strong>Période analysée</strong>: du {start_date} au {end_date}</p></div>', 
                  unsafe_allow_html=True)
        
        # Préparation des données pour l'analyse
        X = df_analysis[selected_vars] if selected_vars else pd.DataFrame(index=df_analysis.index)
        y = df_analysis[conso_col]
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Si aucune variable explicative, on informe l'utilisateur
        if X.empty or X.shape[1] == 0:
            st.warning("Aucune variable explicative sélectionnée. Le modèle sera limité.")
            # Créer au moins une variable factice (mois de l'année)
            X['mois'] = df_analysis[date_col].dt.month
        
        # Création du modèle
        modele_ipmvp = ModelIPMVP()
        status_text.text("Recherche du modèle optimal...")
        
        # Lancer l'analyse
        success = modele_ipmvp.trouver_meilleur_modele(
            X, y, max_features=max_features,
            progress_callback=lambda p: progress_bar.progress(p)
        )
        
        if success:
            status_text.text("Analyse terminée avec succès!")
            progress_bar.progress(1.0)
            
            # Afficher le rapport
            st.subheader("Résultats de l'analyse IPMVP")
            rapport = modele_ipmvp.generer_rapport()
            st.markdown(rapport, unsafe_allow_html=True)
            
            # Ajout d'un résumé avec des métriques visuelles
            st.subheader("Résumé des performances")
            
            # Créer une mise en page en colonnes pour les métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="R² (Qualité d'ajustement)",
                    value=f"{modele_ipmvp.best_r2:.4f}",
                    delta="Bon" if modele_ipmvp.best_r2 > 0.8 else ("Acceptable" if modele_ipmvp.best_r2 > 0.75 else "Insuffisant")
                )
            
            with col2:
                st.metric(
                    label="CV(RMSE) (Précision)",
                    value=f"{modele_ipmvp.best_cv:.4f}",
                    delta="Bon" if modele_ipmvp.best_cv < 0.15 else ("Acceptable" if modele_ipmvp.best_cv < 0.2 else "Insuffisant"),
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    label="NMBE (Biais)",
                    value=f"{modele_ipmvp.best_bias:.6f}",
                    delta="Excellent" if abs(modele_ipmvp.best_bias) < 0.005 else ("Bon" if abs(modele_ipmvp.best_bias) < 0.01 else "Insuffisant"),
                    delta_color="inverse"
                )
            
            # Informations supplémentaires sur le modèle
            st.markdown(f"""
            <div class="card">
                <h4>Interprétation du modèle</h4>
                <p>Le modèle utilise <b>{len(modele_ipmvp.best_features)}</b> variables pour expliquer les variations de consommation.</p>
                <p>Type de modèle: <b>{modele_ipmvp.best_model_type}</b></p>
                <p>Ce modèle explique <b>{modele_ipmvp.best_r2*100:.1f}%</b> des variations de la consommation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualisations
            results_df, fig1, fig2 = modele_ipmvp.visualiser_resultats(X, y, dates=df_analysis[date_col])
            
            if fig1 and fig2:
                st.subheader("Visualisation des résultats")
                
                # Créer des onglets pour les différentes visualisations
                tab1, tab2, tab3 = st.tabs(["📊 Modèle", "📈 Consommation", "📋 Données"])
                
                with tab1:
                    st.pyplot(fig1)
                    st.markdown("**Analyse du modèle**: Ces graphiques montrent la qualité de l'ajustement du modèle.")
                
                with tab2:
                    st.pyplot(fig2)
                    st.markdown("**Comparaison des consommations**: Ce graphique compare la consommation réelle mesurée avec celle calculée par le modèle.")
                
                with tab3:
                    st.dataframe(results_df)
                    st.markdown("**Données détaillées**: Ce tableau présente les valeurs réelles, prédites et les erreurs du modèle.")
                
                # Section de téléchargement des résultats
                st.markdown("""
                <div class="card">
                    <h4>Téléchargement des résultats</h4>
                    <p>Vous pouvez télécharger les résultats au format Excel, incluant les données originales, 
                    les résultats du modèle et un résumé du rapport.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Préparation des données pour l'export
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name="Données d'origine", index=False)
                    results_df.to_excel(writer, sheet_name="Résultats du modèle", index=False)
                    
                    # Créer un résumé pour l'export
                    resume_data = {
                        "Métrique": ["Type de modèle", "Variables", "R²", "CV(RMSE)", "NMBE (Biais)", "Formule"],
                        "Valeur": [
                            modele_ipmvp.best_model_type,
                            ", ".join(modele_ipmvp.best_features),
                            f"{modele_ipmvp.best_r2:.4f}",
                            f"{modele_ipmvp.best_cv:.4f}",
                            f"{modele_ipmvp.best_bias:.8f}",
                            modele_ipmvp.best_formula
                        ]
                    }
                    pd.DataFrame(resume_data).to_excel(writer, sheet_name="Résumé", index=False)
                
                # Bouton de téléchargement avec une présentation améliorée
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Télécharger le rapport complet (Excel)",
                        data=buffer.getvalue(),
                        file_name=f"rapport_ipmvp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("Impossible de générer les visualisations.")
        else:
            status_text.text("Analyse terminée, mais aucun modèle conforme trouvé.")
            progress_bar.progress(1.0)
            st.markdown("""
            <div class="card" style="border-left: 5px solid #FFC107;">
                <h3 style="color: #FFC107;">⚠️ Aucun modèle conforme</h3>
                <p>Aucun modèle conforme aux critères IPMVP n'a pu être trouvé avec ces données.</p>
                <h4>Suggestions:</h4>
                <ul>
                    <li>Vérifiez que vos données couvrent au moins 12 mois complets</li>
                    <li>Ajoutez plus de variables explicatives (comme les DJU)</li>
                    <li>Assurez-vous que vos variables explicatives sont corrélées avec la consommation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Développé pour l'analyse IPMVP")

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 📌 **Graphique : Consommation réelle vs Ajustée**
def plot_consumption(y_actual, y_pred, dates):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Barres pour la consommation réelle
    ax.bar(dates, y_actual, color="#6DBABC", label="Consommation réelle", alpha=0.7)

    # Ligne pour la consommation ajustée
    ax.plot(dates, y_pred, color="#E74C3C", marker='o', linestyle='-', linewidth=2, label="Consommation ajustée")

    # Mise en forme du graphe
    ax.set_xlabel("Mois")
    ax.set_ylabel("Consommation")
    ax.set_title("📊 Comparaison Consommation Mesurée vs Ajustée")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig

# 📌 **Lancer le calcul après sélection des variables**
if df is not None and lancer_calcul:
    st.subheader("⚙️ Analyse en cours...")

    df[date_col] = pd.to_datetime(df[date_col])
    X = df[selected_vars] if selected_vars else pd.DataFrame(index=df.index)
    y = df[conso_col]

    best_model = None
    best_r2 = -1
    best_features = []
    best_y_pred = None

    # 🔹 Test sur plusieurs périodes glissantes de 12 mois
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

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                        best_features = list(combo)
                        best_y_pred = y_pred
                        best_dates = df_subset[date_col]

    # 🔹 Affichage des résultats et du graphe correct
    if best_model:
        st.success("✅ Modèle trouvé avec succès !")

        st.markdown("### 📊 Comparaison Consommation Mesurée vs Ajustée")
        fig = plot_consumption(y_subset, best_y_pred, best_dates)
        st.pyplot(fig)

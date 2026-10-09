import streamlit as st

st.set_page_config(page_title="Maintenance prédictive explicable", page_icon="⚙️", layout="wide")

st.title("⚙️ Maintenance prédictive explicable")
st.subheader("Cas 15 – AI4I 2020 : prédire et expliquer les pannes de machines")

st.markdown("""
Cette application prédit si une machine va tomber en panne **et explique pourquoi**.

**Suis les modules dans l'ordre (menu à gauche) :**
1. **Données** : téléchargement automatique, statistiques, visualisations
2. **Préparation** : variables, encodage, normalisation, séparation train/test
3. **Modélisation** : 5 modèles optimisés et comparés
4. **Évaluation** : courbes ROC, précision-rappel, matrice de confusion, seuil
6. **API** : interrogation du service FastAPI, avec explication SHAP de la prédiction
""")

st.info("Les données sont déséquilibrées (environ 3,4 % de pannes) : on évalue les modèles avec "
        "le rappel, le F1 et la PR-AUC, pas avec l'accuracy.")

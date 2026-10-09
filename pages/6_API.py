import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Module 6 – API", page_icon="🔌", layout="wide")
st.title("🔌 Module 6 – API d'interprétation (FastAPI)")

st.markdown("L'application envoie les caractéristiques d'une machine au service **FastAPI** (`api.py`), "
            "qui renvoie la probabilité de panne, la décision et l'explication SHAP. "
            "Lancement de l'API dans un autre terminal : `uvicorn api:app --reload`")

url = st.text_input("Adresse de l'API", os.environ.get("API_URL", "http://127.0.0.1:8000")).rstrip("/")
st.markdown(f"Documentation interactive (Swagger) : [{url}/docs]({url}/docs)")

# ---------- État de l'API ----------
try:
    info = requests.get(f"{url}/", timeout=5).json()
    st.success(f"API connectée – modèle : **{info['modele']}**, seuil de décision : **{info['seuil']}**")
except Exception:
    st.error("API injoignable. Vérifie qu'elle tourne (`uvicorn api:app --reload`) et l'adresse ci-dessus.")
    st.stop()

# ---------- Formulaire ----------
st.subheader("Caractéristiques de la machine")
with st.form("machine"):
    c = st.columns(3)
    machine = {
        "type": c[0].selectbox("Type de produit", ["L", "M", "H"]),
        "air_temperature": c[1].number_input("Température de l'air (K)", 290.0, 310.0, 302.5, 0.1),
        "process_temperature": c[2].number_input("Température du process (K)", 300.0, 320.0, 310.9, 0.1),
        "rotational_speed": c[0].number_input("Vitesse de rotation (rpm)", 1000.0, 3000.0, 1290.0, 10.0),
        "torque": c[1].number_input("Couple (Nm)", 0.0, 90.0, 65.0, 0.5),
        "tool_wear": c[2].number_input("Usure de l'outil (min)", 0.0, 260.0, 210.0, 1.0),
    }
    top_k = st.slider("Nombre de variables dans l'explication", 1, 10, 5)
    envoyer = st.form_submit_button("Envoyer à l'API", type="primary")

if not envoyer:
    st.info("Les valeurs par défaut décrivent une machine à risque (couple élevé, vitesse faible, outil usé). "
            "Essaie aussi une machine normale : couple 40, vitesse 1500, usure 20.")
    st.stop()

# ---------- Appel de l'API ----------
try:
    r = requests.post(f"{url}/explain", params={"top_k": top_k}, json=machine, timeout=60)
    r.raise_for_status()
    res = r.json()
except Exception as e:
    st.error(f"Échec de l'appel à l'API : {e}")
    st.stop()

a, b = st.columns(2)
a.metric("Probabilité de panne", f"{res['probabilite_panne']:.1%}")
b.metric("Décision", res["decision"])

if res["decision"] == "Panne probable":
    st.error(res["synthese"])
else:
    st.success(res["synthese"])

dc = pd.DataFrame(res["contributions"]).iloc[::-1]
dc["Variable"] = dc["variable"] + " = " + dc["valeur"]
fig = px.bar(dc, x="contribution", y="Variable", orientation="h", color="effet",
             color_discrete_map={"augmente le risque": "#E45756", "diminue le risque": "#4C78A8"},
             labels={"contribution": "Contribution SHAP"},
             title="Pourquoi le modèle a pris cette décision")
st.plotly_chart(fig)
st.caption("Barre rouge : la variable pousse vers la panne. Barre bleue : elle rassure le modèle.")

with st.expander("Réponse JSON brute renvoyée par l'API"):
    st.json(res)

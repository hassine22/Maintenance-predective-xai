import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Module 1 – Données", page_icon="📊", layout="wide")
st.title("📊 Module 1 – Données")
COULEURS = {"Non": "#4C78A8", "Oui": "#E45756"}


@st.cache_data(show_spinner="Téléchargement de AI4I 2020 depuis l'UCI…")
def charger_ai4i():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=601)
    df = d.data.features.copy()
    df["Machine failure"] = d.data.targets["Machine failure"].values
    return df


# ---------- Chargement ----------
source = st.radio("Source des données",
                  ["AI4I 2020 (téléchargement automatique UCI)", "Importer un fichier CSV"], horizontal=True)

if source.startswith("AI4I"):
    df = charger_ai4i()
    target = "Machine failure"
    st.success("Jeu AI4I 2020 téléchargé automatiquement (UCI, id = 601). Les colonnes TWF, HDF, PWF, "
               "OSF, RNF sont écartées : elles révèlent la panne (fuite de données).")
else:
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is None:
        st.info("Importe un fichier CSV pour continuer.")
        st.stop()
    df = pd.read_csv(fichier, sep=None, engine="python")
    target = st.selectbox("Variable cible", df.columns, index=len(df.columns) - 1)

valeurs = sorted(df[target].dropna().unique().tolist(), key=str)
if len(valeurs) < 2:
    st.error("La cible doit contenir au moins deux valeurs.")
    st.stop()
if set(valeurs) != {0, 1}:
    pos = st.selectbox("Classe positive (l'événement à prédire)", valeurs, index=len(valeurs) - 1)
    df = df.copy()
    df[target] = (df[target] == pos).astype(int)

# Mémoriser les données pour les autres modules
st.session_state["df"] = df
st.session_state["target"] = target
y = df[target]

# ---------- Chiffres clés ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", len(df))
c2.metric("Colonnes", df.shape[1])
c3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))
c4.metric("Classe positive", f"{y.mean():.1%}")

t1, t2, t3, t4, t5 = st.tabs(["Aperçu", "Statistiques", "Valeurs manquantes",
                              "Proportion des classes", "Visualisations"])

with t1:
    st.dataframe(df.head(50))
    st.dataframe(pd.DataFrame({"Type": df.dtypes.astype(str), "Valeurs distinctes": df.nunique()}))

with t2:
    st.dataframe(df.describe().T.round(2))
    cat = df.select_dtypes(exclude="number").columns
    if len(cat):
        st.dataframe(df[cat].describe().T)

with t3:
    st.dataframe(pd.DataFrame({"Manquantes": df.isnull().sum(), "%": (df.isnull().mean() * 100).round(2)}))
    st.write(f"Doublons : **{int(df.duplicated().sum())}**")

with t4:
    prop = pd.DataFrame({"Classe": ["Pas de panne (0)", "Panne (1)"],
                         "Nombre": [int((y == 0).sum()), int((y == 1).sum())]})
    prop["%"] = (prop["Nombre"] / len(y) * 100).round(2)
    a, b = st.columns(2)
    a.dataframe(prop, hide_index=True)
    b.plotly_chart(px.pie(prop, names="Classe", values="Nombre", hole=0.5,
                          color_discrete_sequence=[COULEURS["Non"], COULEURS["Oui"]]))
    if y.mean() < 0.2:
        st.warning("Classes déséquilibrées : l'accuracy serait trompeuse (un modèle qui dit toujours "
                   f"« pas de panne » aurait {1 - y.mean():.1%} d'accuracy sans détecter aucune panne).")

with t5:
    d = df.copy()
    d["Panne"] = np.where(y == 1, "Oui", "Non")
    num = [c for c in d.select_dtypes(include="number").columns if c != target]
    cat = [c for c in d.select_dtypes(exclude="number").columns if c != "Panne"]

    st.subheader("Distributions selon la classe")
    st.caption("histnorm = densité : chaque classe est ramenée à sa propre taille, sinon les pannes "
               "seraient invisibles.")
    col = st.selectbox("Variable", num, key="dist")
    st.plotly_chart(px.histogram(d, x=col, color="Panne", barmode="overlay", marginal="box",
                                 histnorm="probability density", color_discrete_map=COULEURS))

    st.subheader("Matrice de corrélation")
    corr = d[num + [target]].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, height=550))

    st.subheader("Taux de panne par variable")
    var = st.selectbox("Variable", num + cat, key="taux")
    if var in num:
        groupes = pd.qcut(d[var], q=10, duplicates="drop")
        taux = d.groupby(groupes, observed=True)[target].mean().reset_index()
        taux[var] = taux[var].astype(str)
    else:
        taux = d.groupby(var)[target].mean().reset_index()
    taux[target] *= 100
    fig = px.bar(taux, x=var, y=target, labels={target: "Taux de panne (%)"})
    fig.add_hline(y=y.mean() * 100, line_dash="dash", annotation_text=f"Taux moyen ({y.mean():.1%})")
    st.plotly_chart(fig)

    st.subheader("Nuage de points")
    c1, c2 = st.columns(2)
    vx = c1.selectbox("Axe X", num, index=num.index("Rotational speed") if "Rotational speed" in num else 0)
    vy = c2.selectbox("Axe Y", num, index=num.index("Torque") if "Torque" in num else min(1, len(num) - 1))
    st.plotly_chart(px.scatter(d.sort_values("Panne"), x=vx, y=vy, color="Panne", opacity=0.6,
                               color_discrete_map={"Non": "lightgray", "Oui": COULEURS["Oui"]}))

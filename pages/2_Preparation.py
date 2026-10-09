import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

st.set_page_config(page_title="Module 2 – Préparation", page_icon="🧹", layout="wide")
st.title("🧹 Module 2 – Préparation")

if "df" not in st.session_state:
    st.info("Ouvre d'abord la page **Donnees** (Module 1) pour charger les données.")
    st.stop()

df = st.session_state["df"].copy()
target = st.session_state["target"]
st.write(f"Variable cible : **{target}**  (1 = panne, 0 = pas de panne)")

# ---------- Variables physiques (AI4I uniquement) ----------
ai4i = {"Torque", "Rotational speed", "Air temperature", "Process temperature"} <= set(df.columns)
physique = False
if ai4i:
    physique = st.checkbox("Ajouter les variables physiques : Power (couple × vitesse) "
                           "et Temp diff (écart de température)", value=True)
    if physique:
        df["Power"] = df["Torque"] * df["Rotational speed"] * 2 * 3.1416 / 60
        df["Temp diff"] = df["Process temperature"] - df["Air temperature"]

# Colonnes exclues par défaut : fuite de données et identifiants
EXCLUES = ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]
candidats = [c for c in df.columns if c != target]
defaut = [c for c in candidats if c not in EXCLUES
          and not (df[c].dtype == object and df[c].nunique() > 0.5 * len(df))]

# ---------- Paramètres ----------
with st.form("preparation"):
    features = st.multiselect("Variables explicatives", candidats, default=defaut,
                              help="Les identifiants et les colonnes qui révèlent la cible sont exclus par défaut.")
    c1, c2, c3 = st.columns(3)
    num_impute = c1.selectbox("Valeurs manquantes (numériques)", ["median", "mean"],
                              format_func={"median": "Médiane", "mean": "Moyenne"}.get)
    c1.caption("Catégorielles : valeur la plus fréquente.")
    scaler_nom = c2.selectbox("Normalisation", ["StandardScaler", "MinMaxScaler", "Aucune"])
    c2.caption("Encodage des catégories : one-hot.")
    test_size = c3.slider("Part du jeu de test", 0.10, 0.40, 0.20, 0.05)
    c4, c5 = st.columns(2)
    stratify = c4.checkbox("Séparation stratifiée (même proportion de pannes dans train et test)", value=True)
    seed = c5.number_input("random_state", value=42, step=1)
    valider = st.form_submit_button("Appliquer la préparation", type="primary")

if valider:
    if not features:
        st.error("Choisis au moins une variable.")
        st.stop()

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if stratify else None, random_state=int(seed))

    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(),
              "Aucune": "passthrough"}[scaler_nom]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy=num_impute)), ("scaler", scaler)])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    # Mémoriser pour les modules suivants
    st.session_state["split"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                                 "features": features, "num_cols": num_cols, "cat_cols": cat_cols,
                                 "physique": physique}
    st.session_state["preprocessor"] = preprocessor   # non entraîné : il sera ajusté dans chaque pipeline
    for cle in ["results", "probas", "thresholds"]:   # les anciens modèles ne sont plus valables
        st.session_state.pop(cle, None)

# ---------- Résultat ----------
if "split" not in st.session_state:
    st.info("Règle les paramètres puis clique sur **Appliquer la préparation**.")
    st.stop()

sp = st.session_state["split"]
st.success("Préparation appliquée.")
c1, c2, c3 = st.columns(3)
c1.metric("Entraînement", f"{len(sp['X_train'])} lignes", f"{sp['y_train'].mean():.2%} de pannes", delta_color="off")
c2.metric("Test", f"{len(sp['X_test'])} lignes", f"{sp['y_test'].mean():.2%} de pannes", delta_color="off")
c3.metric("Variables", f"{len(sp['num_cols'])} num. + {len(sp['cat_cols'])} cat.")

from sklearn.base import clone
prep = clone(st.session_state["preprocessor"])
Xt = prep.fit_transform(sp["X_train"])          # appris sur le train uniquement
noms = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
apercu = pd.DataFrame(Xt, columns=noms)

st.write("**Train après prétraitement** (imputation, one-hot, normalisation) :")
st.dataframe(apercu.head(10).round(3))
st.dataframe(apercu.describe().T[["mean", "std", "min", "max"]].round(2))
st.caption("Le prétraitement est appris sur le train seulement (fit_transform), puis appliqué tel quel "
           "au test (transform) : le test ne doit rien « souffler » au modèle.")

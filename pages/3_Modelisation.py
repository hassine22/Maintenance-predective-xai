import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

st.set_page_config(page_title="Module 3 – Modélisation", page_icon="🤖", layout="wide")
st.title("🤖 Module 3 – Modélisation et comparaison")

if "split" not in st.session_state:
    st.info("Applique d'abord la préparation (page **Preparation**).")
    st.stop()

sp = st.session_state["split"]
X_train, X_test, y_train, y_test = sp["X_train"], sp["X_test"], sp["y_train"], sp["y_test"]

# ---------- Grilles d'hyperparamètres : (valeurs possibles, valeurs cochées par défaut) ----------
GRILLES = {
    "Régression logistique": {"C": ([0.01, 0.1, 1, 10], [0.1, 1])},
    "Arbre de décision": {"max_depth": ([3, 5, 8, None], [5, 8]),
                          "min_samples_leaf": ([1, 5, 20], [1, 5])},
    "KNN": {"n_neighbors": ([5, 11, 21, 31], [11, 21]),
            "weights": (["uniform", "distance"], ["distance"])},
    "Random Forest": {"n_estimators": ([100, 200, 400], [200]),
                      "max_depth": ([None, 10, 20], [None, 10]),
                      "min_samples_leaf": ([1, 3, 5], [1])},
    "XGBoost": {"n_estimators": ([100, 200, 400], [200, 400]),
                "max_depth": ([3, 5, 7], [3, 5]),
                "learning_rate": ([0.03, 0.05, 0.1, 0.2], [0.05, 0.1])},
}
modeles = list(GRILLES) if XGBClassifier else [m for m in GRILLES if m != "XGBoost"]
if XGBClassifier is None:
    st.warning("XGBoost n'est pas installé (pip install xgboost) : il est retiré de la liste.")


def creer_modele(nom, ponderation, ratio):
    cw = "balanced" if ponderation else None
    if nom == "Régression logistique":
        return LogisticRegression(class_weight=cw, max_iter=2000)
    if nom == "Arbre de décision":
        return DecisionTreeClassifier(class_weight=cw, random_state=42)
    if nom == "KNN":
        return KNeighborsClassifier()
    if nom == "Random Forest":
        return RandomForestClassifier(class_weight=cw, random_state=42, n_jobs=-1)
    return XGBClassifier(scale_pos_weight=ratio if ponderation else 1.0,
                         eval_metric="logloss", random_state=42, n_jobs=-1)


# ---------- Paramètres ----------
choix = st.multiselect("Modèles à entraîner", modeles, default=modeles)
c1, c2 = st.columns(2)
deseq = c1.radio("Gestion du déséquilibre",
                 ["Pondération des classes (class_weight / scale_pos_weight)",
                  "SMOTE pour tous les modèles", "Aucune"])
c1.caption("Avec la pondération, KNN (qui n'a pas ce paramètre) utilise SMOTE.")
folds = c2.slider("Validation croisée : nombre de plis", 3, 10, 5)
c2.caption("Critère d'optimisation : PR-AUC, adaptée aux classes déséquilibrées.")

st.subheader("Hyperparamètres testés (GridSearch)")
grilles = {}
for nom in choix:
    with st.expander(nom):
        grilles[nom] = {}
        for param, (options, defaut) in GRILLES[nom].items():
            vals = st.multiselect(param, options, default=defaut, key=f"{nom}_{param}",
                                  format_func=lambda v: "illimitée" if v is None else str(v))
            grilles[nom][f"clf__{param}"] = vals or [defaut[0]]
        nb = int(np.prod([len(v) for v in grilles[nom].values()]))
        st.caption(f"{nb} combinaison(s) × {folds} plis = {nb * folds} entraînements")

# ---------- Entraînement ----------
if st.button("Lancer l'entraînement", type="primary", disabled=not choix):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    ponderation = deseq.startswith("Pondération")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results, probas = {}, {}
    barre = st.progress(0.0)

    for i, nom in enumerate(choix):
        barre.progress(i / len(choix), text=f"Entraînement : {nom}…")
        prep = clone(st.session_state["preprocessor"])
        clf = creer_modele(nom, ponderation, ratio)
        smote = deseq.startswith("SMOTE") or (nom == "KNN" and ponderation)
        if smote:   # SMOTE appliqué uniquement aux données d'entraînement de chaque pli
            pipe = ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=42)), ("clf", clf)])
        else:
            pipe = Pipeline([("prep", prep), ("clf", clf)])

        search = GridSearchCV(pipe, grilles[nom], scoring="average_precision", cv=cv, n_jobs=-1)
        t0 = time.time()
        search.fit(X_train, y_train)
        results[nom] = {"model": search.best_estimator_, "cv": search.best_score_,
                        "params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                        "temps": time.time() - t0, "smote": smote}
        probas[nom] = search.best_estimator_.predict_proba(X_test)[:, 1]

    barre.progress(1.0, text="Terminé")
    st.session_state["results"] = results
    st.session_state["probas"] = probas
    st.session_state["thresholds"] = {}

if "results" not in st.session_state:
    st.info("Choisis les modèles puis clique sur **Lancer l'entraînement**.")
    st.stop()

# ---------- Tableau comparatif ----------
results, probas = st.session_state["results"], st.session_state["probas"]
lignes, matrices = [], []
for nom, r in results.items():
    p = probas[nom]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    lignes.append({"Modèle": nom, "PR-AUC": average_precision_score(y_test, p),
                   "ROC-AUC": roc_auc_score(y_test, p),
                   "Précision": precision_score(y_test, pred, zero_division=0),
                   "Rappel": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
                   "PR-AUC (CV)": r["cv"], "Temps (s)": r["temps"],
                   "Meilleurs hyperparamètres": str(r["params"])})
    matrices.append({"Modèle": nom, "TN": tn, "FP (fausses alertes)": fp,
                     "FN (pannes ratées)": fn, "TP (pannes détectées)": tp,
                     "Pannes détectées": f"{tp}/{tp + fn}"})

tab = pd.DataFrame(lignes).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
scores = ["PR-AUC", "ROC-AUC", "Précision", "Rappel", "F1", "PR-AUC (CV)"]

st.subheader("Tableau comparatif (jeu de test, seuil 0,5)")
st.dataframe(tab.style.format({**{c: "{:.3f}" for c in scores}, "Temps (s)": "{:.1f}"})
             .background_gradient(cmap="Greens", subset=["PR-AUC", "Rappel", "F1"]), hide_index=True)

st.plotly_chart(px.bar(tab.melt(id_vars="Modèle", value_vars=["PR-AUC", "ROC-AUC", "F1", "Rappel"]),
                       x="Modèle", y="value", color="variable", barmode="group",
                       labels={"value": "Score", "variable": "Métrique"}))

st.subheader("Détail des erreurs (seuil 0,5)")
st.dataframe(pd.DataFrame(matrices), hide_index=True)

meilleur = tab.iloc[0]
st.info(f"Meilleur modèle selon la PR-AUC : **{meilleur['Modèle']}** ({meilleur['PR-AUC']:.3f}). "
        "Si la PR-AUC en validation croisée et celle du test sont proches, le modèle généralise bien "
        "(pas de surapprentissage). Le seuil de 0,5 n'est pas forcément le bon : on l'ajuste au Module 4.")

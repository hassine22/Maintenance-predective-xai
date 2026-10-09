import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

st.set_page_config(page_title="Module 4 – Évaluation", page_icon="📈", layout="wide")
st.title("📈 Module 4 – Évaluation")

if "results" not in st.session_state:
    st.info("Entraîne d'abord des modèles (page **Modelisation**).")
    st.stop()

y_test = st.session_state["split"]["y_test"]
probas = st.session_state["probas"]
thresholds = st.session_state.setdefault("thresholds", {})

# ---------- Courbes ROC et précision-rappel ----------
c1, c2 = st.columns(2)

fig = go.Figure()
for nom, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nom} ({roc_auc_score(y_test, p):.3f})"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                         name="Hasard (0,5)"))
fig.update_layout(title="Courbes ROC", xaxis_title="Taux de fausses alertes (FPR)",
                  yaxis_title="Rappel (pannes détectées)", height=460)
c1.plotly_chart(fig)

fig = go.Figure()
for nom, p in probas.items():
    prec, rec, _ = precision_recall_curve(y_test, p)
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"{nom} ({average_precision_score(y_test, p):.3f})"))
fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
              annotation_text=f"Hasard ({y_test.mean():.3f})")
fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel (pannes détectées)",
                  yaxis_title="Précision (alertes justes)", height=460)
c2.plotly_chart(fig)

st.caption("Avec des classes déséquilibrées, les courbes ROC se ressemblent toutes et paraissent excellentes ; "
           "la courbe précision-rappel révèle les vraies différences entre modèles.")

# ---------- Curseur de seuil ----------
st.subheader("Curseur de seuil de décision")
st.caption("Le modèle donne une probabilité de panne ; au-dessus du seuil, il déclenche une alerte. "
           "Seuil bas : plus de pannes détectées mais plus de fausses alertes. Seuil haut : l'inverse.")

noms = list(probas)
meilleur = max(noms, key=lambda n: average_precision_score(y_test, probas[n]))
nom = st.selectbox("Modèle", noms, index=noms.index(meilleur))
seuil = st.slider("Seuil", 0.01, 0.99, float(thresholds.get(nom, 0.5)), 0.01, key=f"seuil_{nom}")
thresholds[nom] = seuil   # mémorisé pour les modules 5 et 6

p = probas[nom]
pred = (p >= seuil).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

k = st.columns(5)
k[0].metric("Rappel", f"{recall_score(y_test, pred, zero_division=0):.3f}", f"{tp}/{tp + fn} détectées",
            delta_color="off")
k[1].metric("Précision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
k[2].metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")
k[3].metric("Fausses alertes (FP)", int(fp))
k[4].metric("Pannes ratées (FN)", int(fn))

c1, c2 = st.columns([1, 1.4])

texte = [[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]]
fig = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Prédit : non", "Prédit : panne"],
                           y=["Réel : non", "Réel : panne"], text=texte, texttemplate="%{text}",
                           colorscale="Blues", showscale=False))
fig.update_yaxes(autorange="reversed")
fig.update_layout(title=f"Matrice de confusion – {nom} (seuil {seuil:.2f})", height=380)
c1.plotly_chart(fig)

# Précision, rappel et F1 pour tous les seuils (calcul vectorisé, instantané)
seuils = np.linspace(0.01, 0.99, 99)
yv = y_test.to_numpy()
P = p[None, :] >= seuils[:, None]
TP = (P & (yv == 1)).sum(axis=1)
FP = (P & (yv == 0)).sum(axis=1)
FN = (yv == 1).sum() - TP
prec = np.divide(TP, TP + FP, out=np.zeros(len(seuils)), where=(TP + FP) > 0)
rec = TP / (TP + FN)
f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(seuils)), where=(prec + rec) > 0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=seuils, y=prec, name="Précision"))
fig.add_trace(go.Scatter(x=seuils, y=rec, name="Rappel"))
fig.add_trace(go.Scatter(x=seuils, y=f1, name="F1", line=dict(dash="dot")))
fig.add_vline(x=seuil, line_color="red", annotation_text=f"seuil = {seuil:.2f}")
fig.update_layout(title="Effet du seuil", xaxis_title="Seuil", yaxis_title="Score", height=380)
c2.plotly_chart(fig)

# ---------- Meilleur seuil de chaque modèle ----------
st.subheader("Seuil qui maximise le F1 pour chaque modèle")
lignes = []
for n, pp in probas.items():
    pr, rc, th = precision_recall_curve(y_test, pp)
    f = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
    i = int(np.argmax(f))
    pr_ = (pp >= th[i]).astype(int)
    a, b, c, d = confusion_matrix(y_test, pr_, labels=[0, 1]).ravel()
    lignes.append({"Modèle": n, "Seuil optimal": round(float(th[i]), 2), "F1": f[i], "Rappel": rc[i],
                   "Précision": pr[i], "Pannes détectées": f"{d}/{d + c}", "Fausses alertes": int(b)})
tab = pd.DataFrame(lignes).sort_values("F1", ascending=False).reset_index(drop=True)
st.dataframe(tab.style.format({"F1": "{:.3f}", "Rappel": "{:.3f}", "Précision": "{:.3f}"}), hide_index=True)

if st.button("Appliquer ces seuils optimaux à tous les modèles"):
    for _, r in tab.iterrows():
        thresholds[r["Modèle"]] = float(r["Seuil optimal"])
        st.session_state.pop(f"seuil_{r['Modèle']}", None)   # remet le curseur sur la nouvelle valeur
    st.rerun()

st.caption("Limite : le seuil est choisi sur le jeu de test, ces chiffres sont donc légèrement optimistes. "
           "En maintenance, on peut préférer un seuil plus bas si une panne ratée coûte plus cher qu'une "
           "inspection inutile : c'est une décision métier.")

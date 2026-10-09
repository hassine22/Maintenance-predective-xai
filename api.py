from typing import Literal
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

bundle = joblib.load("model.joblib")
model = bundle["model"]
THRESHOLD = float(bundle["threshold"])
FEATURES = bundle["features"]
MODEL_NAME = bundle.get("model_name", "Modèle")

prep = model.named_steps["prep"]
clf = model.named_steps["clf"]
explainer = shap.TreeExplainer(clf)
FEATURE_NAMES = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

LIBELLES = {"Air temperature": "la température de l'air", "Process temperature": "la température du process",
            "Rotational speed": "la vitesse de rotation", "Torque": "le couple",
            "Tool wear": "l'usure de l'outil", "Power": "la puissance", "Temp diff": "l'écart de température",
            "Type_L": "le type de produit L", "Type_M": "le type de produit M", "Type_H": "le type de produit H"}
UNITES = {"Air temperature": "K", "Process temperature": "K", "Rotational speed": "rpm",
          "Torque": "Nm", "Tool wear": "min", "Power": "W", "Temp diff": "K"}

app = FastAPI(title="API Maintenance prédictive – AI4I 2020",
              description="Prédit le risque de panne d'une machine et explique la prédiction avec SHAP.",
              version="1.0")


class Machine(BaseModel):
    type: Literal["L", "M", "H"] = Field(..., description="Qualité du produit : L, M ou H", examples=["L"])
    air_temperature: float = Field(..., gt=0, description="Température de l'air (K)", examples=[302.5])
    process_temperature: float = Field(..., gt=0, description="Température du process (K)", examples=[310.9])
    rotational_speed: float = Field(..., gt=0, description="Vitesse de rotation (rpm)", examples=[1290])
    torque: float = Field(..., ge=0, description="Couple (Nm)", examples=[65.0])
    tool_wear: float = Field(..., ge=0, description="Usure de l'outil (min)", examples=[210])


def to_dataframe(m):
    df = pd.DataFrame([{"Type": m.type, "Air temperature": m.air_temperature,
                        "Process temperature": m.process_temperature,
                        "Rotational speed": m.rotational_speed, "Torque": m.torque, "Tool wear": m.tool_wear}])
    df["Power"] = df["Torque"] * df["Rotational speed"] * 2 * 3.1416 / 60
    df["Temp diff"] = df["Process temperature"] - df["Air temperature"]
    return df[FEATURES]


def valeur_lisible(f, df):
    if f.startswith("Type_"):
        return str(df["Type"].iloc[0])
    return f"{df[f].iloc[0]:.1f} {UNITES.get(f, '')}".strip()


@app.get("/", summary="Vérifier que l'API fonctionne")
def accueil():
    return {"statut": "ok", "modele": MODEL_NAME, "seuil": THRESHOLD, "variables": FEATURES}


@app.post("/predict", summary="Probabilité de panne et décision")
def predict(machine: Machine):
    proba = float(model.predict_proba(to_dataframe(machine))[0, 1])
    return {"probabilite_panne": round(proba, 4), "seuil": THRESHOLD,
            "decision": "Panne probable" if proba >= THRESHOLD else "Fonctionnement normal"}


@app.post("/explain", summary="Explication SHAP pour une machine")
def explain(machine: Machine, top_k: int = 5):
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k doit être >= 1")
    df = to_dataframe(machine)
    proba = float(model.predict_proba(df)[0, 1])
    exp = explainer(prep.transform(df), check_additivity=False)
    values, base = np.asarray(exp.values[0]), np.asarray(exp.base_values[0])
    if values.ndim == 2:
        values, base = values[:, 1], np.ravel(base)[1]
    base = float(np.ravel(base)[0])

    contributions = sorted(
        ({"variable": f, "valeur": valeur_lisible(f, df), "contribution": round(float(v), 4),
          "effet": "augmente le risque" if v > 0 else "diminue le risque"}
         for f, v in zip(FEATURE_NAMES, values)),
        key=lambda c: abs(c["contribution"]), reverse=True)

    if proba >= THRESHOLD:
        causes = [c for c in contributions if c["contribution"] > 0][:2]
        debut, lien = "Le risque de panne est élevé", "principalement à cause de"
    else:
        causes = [c for c in contributions if c["contribution"] < 0][:2]
        debut, lien = "Le risque de panne est faible", "principalement grâce à"
    details = " et ".join(f"{LIBELLES.get(c['variable'], c['variable'])} ({c['valeur']})" for c in causes)
    synthese = f"{debut} ({proba:.0%}), {lien} {details}." if causes else f"{debut} ({proba:.0%})."

    return {"probabilite_panne": round(proba, 4),
            "decision": "Panne probable" if proba >= THRESHOLD else "Fonctionnement normal",
            "valeur_de_base": round(base, 4), "contributions": contributions[:top_k],
            "synthese": synthese}

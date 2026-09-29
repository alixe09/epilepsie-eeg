"""Méthode classique : caractéristiques par canal + gradient boosting, par patient.

Sauvegarde les probabilités hors-pli (chaque fenêtre prédite par un modèle qui
ne l'a pas vue) dans resultats/probas/classique_<patient>.npz. Avec
--decoupage chrono : calibration puis surveillance dans l'ordre du temps
(classique_chrono_<patient>.npz ; NaN sur les enregistrements de calibration).

Usage : python src/classique.py [--canaux temporaux] [--decoupage chrono] [chb01 ...]
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from caracteristiques import NOMS_PAR_CANAL, charger_patient
from decoupage import iterations
from donnees import PATIENTS, RACINE

DOSSIER_PROBAS = RACINE / "resultats" / "probas"
# Montage réduit « portable » : 4 dérivations temporales, proches de ce que
# permettent des électrodes autour de l'oreille.
MONTAGES = {
    "complet": list(range(18)),
    "temporaux": [1, 2, 13, 14],  # F7-T7, T7-P7, F8-T8, T8-P8
}
SEUIL_ICTAL = 0.5  # une fenêtre est « crise » si au moins la moitié est annotée


def colonnes(montage):
    n = len(NOMS_PAR_CANAL)
    return np.concatenate([np.arange(c * n, (c + 1) * n) for c in MONTAGES[montage]])


def entrainer(X, y):
    modele = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, class_weight="balanced",
        random_state=0)
    return modele.fit(X, y)


def exemples_entrainement(frac):
    """Masque et étiquettes : on écarte les fenêtres à cheval sur le début/la fin."""
    garder = (frac >= SEUIL_ICTAL) | (frac == 0)
    return garder, (frac >= SEUIL_ICTAL).astype(int)


def main(patients, montage, decoupage="plis"):
    DOSSIER_PROBAS.mkdir(parents=True, exist_ok=True)
    cols = colonnes(montage)
    for p in patients:
        t0 = time.time()
        d = charger_patient(p)
        X = d["X"][:, cols]
        garder, y = exemples_entrainement(d["frac"])
        proba = np.full(len(X), np.nan, np.float32)
        its = iterations(d["crises"], decoupage)
        for app_f, test_f in its:
            app = np.isin(d["fichier"], app_f) & garder
            test = np.isin(d["fichier"], test_f)
            modele = entrainer(X[app], y[app])
            proba[test] = modele.predict_proba(X[test])[:, 1]
        suffixe = ("" if montage == "complet" else f"_{montage}") + ("" if decoupage == "plis" else "_chrono")
        np.savez(DOSSIER_PROBAS / f"classique{suffixe}_{p}.npz", proba=proba)
        print(f"{p} ({montage}, {decoupage}) : {len(its)} entraînements, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("patients", nargs="*", default=PATIENTS)
    a.add_argument("--canaux", default="complet", choices=list(MONTAGES))
    a.add_argument("--decoupage", default="plis", choices=["plis", "chrono"])
    args = a.parse_args()
    main(args.patients, args.canaux, args.decoupage)

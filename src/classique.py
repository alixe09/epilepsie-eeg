"""Méthode classique : caractéristiques par canal + gradient boosting, par patient.

Sauvegarde les probabilités hors-pli (chaque fenêtre prédite par un modèle qui
ne l'a pas vue) dans resultats/probas/classique_<patient>.npz.

Usage : python src/classique.py [--canaux temporaux] [chb01 ...]
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from caracteristiques import NOMS_PAR_CANAL, charger_patient
from decoupage import plis
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


def main(patients, montage):
    DOSSIER_PROBAS.mkdir(parents=True, exist_ok=True)
    cols = colonnes(montage)
    for p in patients:
        t0 = time.time()
        d = charger_patient(p)
        X = d["X"][:, cols]
        pli_fichier = plis(d["crises"])
        pli = pli_fichier[d["fichier"]]
        garder, y = exemples_entrainement(d["frac"])
        proba = np.zeros(len(X), np.float32)
        for k in np.unique(pli_fichier):
            app = (pli != k) & garder
            modele = entrainer(X[app], y[app])
            proba[pli == k] = modele.predict_proba(X[pli == k])[:, 1]
        suffixe = "" if montage == "complet" else f"_{montage}"
        np.savez(DOSSIER_PROBAS / f"classique{suffixe}_{p}.npz", proba=proba, pli=pli_fichier)
        print(f"{p} ({montage}) : {len(np.unique(pli_fichier))} plis, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("patients", nargs="*", default=PATIENTS)
    a.add_argument("--canaux", default="complet", choices=list(MONTAGES))
    args = a.parse_args()
    main(args.patients, args.canaux)

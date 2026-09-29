"""Extraits pour la démo : ~10 min d'EEG autour de crises, avec les probabilités hors-pli.

Les probabilités viennent des modèles qui n'ont PAS vu l'enregistrement (pli de
test), et le réglage d'alarme est celui choisi sur les 5 autres patients : la
démo montre exactement ce qu'a mesuré l'évaluation.

Usage : python src/make_demo_data.py
"""
import json

import numpy as np

from caracteristiques import charger_patient
from donnees import DOSSIER_DONNEES, RACINE
from pretraitement import FS, filtrer

SORTIE = RACINE / "app" / "demo_data"
EXEMPLES = {  # enregistrement : (début, fin) de l'extrait en secondes
    "chb01_03": (2700, 3300),  # crise de 40 s, cas typique
    "chb05_13": (900, 1500),  # crise de 115 s
    "chb16_17": (1500, 2400),  # 2 crises de 6 et 8 s : le cas difficile
    "chb24_04": (1000, 1850),  # 3 crises en 12 min
}
METHODES = {"classique": "classique", "cnn_int8": "cnn_int8"}


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    reglages = {m: {l["patient"]: {"seuil": l["seuil"], "lissage": l["lissage"]}
                    for l in json.loads((RACINE / "resultats" / f"resume_{m}.json")
                                        .read_text(encoding="utf-8"))["patients"]}
                for m in METHODES}
    for nom, (a, b) in EXEMPLES.items():
        patient = nom.split("_")[0]
        d = charger_patient(patient)
        i = list(d["noms"]).index(nom)
        sel = d["fichier"] == i
        t = d["t"][sel]
        garde = (t >= a) & (t < b)
        probas = {m: np.load(RACINE / "resultats" / "probas" / f"{m}_{patient}.npz")["proba"][sel][garde]
                  for m in METHODES}
        with np.load(DOSSIER_DONNEES / patient / f"{nom}.npz") as z:
            echelle = float(z["echelle_uv"])
            # filtré sur tout l'enregistrement (filtre causal déjà stabilisé au début de l'extrait)
            eeg = filtrer(z["eeg"].astype(np.float32) * echelle)[:, int(a * FS):int(b * FS)]
            canaux = z["canaux"]
        crises = d["crises"][i]
        crises = crises[(crises[:, 1] > a) & (crises[:, 0] < b)] - a
        np.savez_compressed(
            SORTIE / f"{nom}.npz",
            eeg=np.round(eeg / 0.5).astype(np.int16), echelle_uv=0.5, fs=FS, canaux=canaux,
            debut_s=a, crises=crises, t=t[garde] - a,
            **{f"proba_{m}": probas[m].astype(np.float16) for m in METHODES},
            reglages=json.dumps({m: reglages[m][patient] for m in METHODES}))
        print(nom, (SORTIE / f"{nom}.npz").stat().st_size // 1000, "Ko")


if __name__ == "__main__":
    main()

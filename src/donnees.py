"""Accès aux enregistrements CHB-MIT convertis (voir telecharger_chbmit.py)."""
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DONNEES = RACINE / "data" / "chbmit_128hz"
PATIENTS = ["chb01", "chb05", "chb08", "chb16", "chb23", "chb24"]


def lister_enregistrements(patient):
    """Fichiers .npz d'un patient, dans l'ordre chronologique (ordre des noms)."""
    return sorted(p for p in (DOSSIER_DONNEES / patient).glob("*.npz")
                  if not p.name.endswith(".part.npz"))


def charger(chemin):
    """Renvoie (eeg en µV float32 de forme (18, n), fs, crises (k, 2) en secondes)."""
    with np.load(chemin) as d:
        eeg = d["eeg"].astype(np.float32) * np.float32(d["echelle_uv"])
        return eeg, int(d["fs"]), d["crises"].astype(float).reshape(-1, 2)


def canaux(chemin):
    with np.load(chemin) as d:
        return [str(c) for c in d["canaux"]]

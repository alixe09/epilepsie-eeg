"""Méthode classique : caractéristiques spectrales et temporelles par canal.

Inspirée de Shoeb (2009) : énergie par bande de fréquence sur chaque
dérivation. Une crise se traduit typiquement par une activité rythmique qui
envahit une ou plusieurs bandes, d'abord sur quelques canaux puis ailleurs.

Usage : python src/caracteristiques.py [chb01 ...]   (met en cache dans data/caracteristiques/)
"""
import sys
import time

import numpy as np

from donnees import PATIENTS, RACINE, charger, lister_enregistrements
from pretraitement import FS, N_FENETRE, filtrer, fenetres, fraction_ictale, instants_decision

BANDES = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
NOMS_PAR_CANAL = [*BANDES, "longueur_ligne", "mobilite", "complexite"]
DOSSIER_CACHE = RACINE / "data" / "caracteristiques"

_FREQS = np.fft.rfftfreq(N_FENETRE, 1 / FS)
_HANN = np.hanning(N_FENETRE).astype(np.float32)


def calculer(f):
    """f : fenêtres (n, canaux, N_FENETRE) -> (n, canaux × 8) en float32."""
    spectre = np.abs(np.fft.rfft(f * _HANN, axis=-1)) ** 2
    puissances = [np.log(spectre[..., (_FREQS >= a) & (_FREQS < b)].sum(-1) + 1e-3)
                  for a, b in BANDES.values()]
    d1 = np.diff(f, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = f.var(-1) + 1e-6, d1.var(-1) + 1e-6, d2.var(-1) + 1e-6
    mobilite = np.sqrt(v1 / v0)  # Hjorth : fréquence moyenne
    complexite = np.sqrt(v2 / v1) / mobilite  # Hjorth : écart à une sinusoïde pure
    longueur = np.log(np.abs(d1).mean(-1) + 1e-3)  # « line length », sensible aux pointes
    x = np.stack([*puissances, longueur, mobilite, complexite], axis=-1)
    return x.reshape(len(f), -1).astype(np.float32)


def calculer_enregistrement(chemin, taille_lot=600):
    eeg, _, crises = charger(chemin)
    f = fenetres(filtrer(eeg))
    x = np.concatenate([calculer(f[i:i + taille_lot]) for i in range(0, len(f), taille_lot)])
    return x, fraction_ictale(len(f), crises), crises, eeg.shape[-1] / FS


def charger_patient(patient):
    """Caractéristiques d'un patient (calculées une fois puis lues depuis le cache).

    Renvoie un dict : X (n, 144), frac (part ictale de chaque fenêtre),
    fichier (index de l'enregistrement de chaque fenêtre), t (instant de décision
    dans l'enregistrement), noms (enregistrements), crises (liste par enregistrement),
    duree (s, par enregistrement).
    """
    cache = DOSSIER_CACHE / f"{patient}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    X, frac, fichier, t, noms, crises, duree = [], [], [], [], [], [], []
    for i, chemin in enumerate(lister_enregistrements(patient)):
        x, fr, cr, du = calculer_enregistrement(chemin)
        X.append(x)
        frac.append(fr)
        fichier.append(np.full(len(x), i, np.int16))
        t.append(instants_decision(len(x)))
        noms.append(chemin.stem)
        crises.append(cr)
        duree.append(du)
    tab_crises = np.empty(len(crises), dtype=object)  # np.array() empilerait les formes égales
    tab_crises[:] = crises
    d = dict(X=np.concatenate(X), frac=np.concatenate(frac), fichier=np.concatenate(fichier),
             t=np.concatenate(t), noms=np.array(noms), crises=tab_crises, duree=np.array(duree))
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **d)
    return d


if __name__ == "__main__":
    for p in sys.argv[1:] or PATIENTS:
        t0 = time.time()
        d = charger_patient(p)
        print(f"{p} : {len(d['X'])} fenêtres, {d['duree'].sum() / 3600:.1f} h, "
              f"{sum(len(c) for c in d['crises'])} crises ({time.time() - t0:.0f} s)")

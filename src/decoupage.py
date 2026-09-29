"""Validation croisée par enregistrement, spécifique à chaque patient.

Chaque fichier d'1 h est entièrement dans l'entraînement ou dans le test :
aucune fenêtre de test ne chevauche une fenêtre d'entraînement. Les fichiers
contenant des crises sont répartis à tour de rôle entre les plis pour que chaque
pli teste au moins une crise. Chaque heure d'enregistrement est testée
exactement une fois : les métriques couvrent toutes les données.
"""
import numpy as np

N_PLIS = 4


def plis(crises_par_fichier, n_plis=N_PLIS):
    """Renvoie pli[i] ∈ {0..k-1} pour chaque enregistrement i."""
    avec = [i for i, c in enumerate(crises_par_fichier) if len(c)]
    sans = [i for i, c in enumerate(crises_par_fichier) if not len(c)]
    k = max(1, min(n_plis, len(avec)))
    pli = np.empty(len(crises_par_fichier), int)
    for rang, i in enumerate(avec):
        pli[i] = rang % k
    for rang, i in enumerate(sans):
        pli[i] = rang % k
    return pli


N_CRISES_CALIBRATION = 3


def blocs_chronologiques(crises_par_fichier, n_calibration=N_CRISES_CALIBRATION):
    """Scénario « dispositif » : calibration puis surveillance, dans l'ordre du temps.

    Renvoie bloc[i] pour chaque enregistrement (dans l'ordre chronologique) :
      0  = calibration (du début jusqu'à la n-ième crise incluse) : jamais testé ;
      b≥1 = surveillance : testé par un modèle entraîné sur les blocs < b seulement.
    Un nouveau bloc commence après chaque enregistrement contenant une crise
    (le modèle est recalibré avec chaque nouvelle crise).
    """
    bloc = np.zeros(len(crises_par_fichier), int)
    vues, courant = 0, 0
    for i, c in enumerate(crises_par_fichier):
        if courant == 0 and vues >= n_calibration:
            courant = 1
        bloc[i] = courant
        vues += len(c)
        if courant > 0 and len(c):
            courant += 1
    return bloc


def iterations(crises_par_fichier, decoupage):
    """Liste de (enregistrements d'entraînement, enregistrements de test) selon le découpage."""
    if decoupage == "plis":
        pli = plis(crises_par_fichier)
        return [(np.flatnonzero(pli != k), np.flatnonzero(pli == k)) for k in np.unique(pli)]
    bloc = blocs_chronologiques(crises_par_fichier)
    return [(np.flatnonzero(bloc < b), np.flatnonzero(bloc == b)) for b in range(1, bloc.max() + 1)]

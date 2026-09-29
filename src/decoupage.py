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

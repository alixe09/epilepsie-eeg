"""Filtrage et découpage en fenêtres, identiques pour les deux méthodes.

Tout est causal (aucun échantillon futur) : c'est ce que pourrait faire un
dispositif portable qui analyse le signal au fil de l'eau.
"""
import numpy as np
from scipy.signal import butter, sosfilt

FS = 128
DUREE_FENETRE = 4.0  # s
PAS = 1.0  # s : une décision par seconde
N_FENETRE = int(DUREE_FENETRE * FS)
N_PAS = int(PAS * FS)
BANDE = (0.5, 40.0)  # Hz : retire la dérive lente et le secteur (60 Hz aux États-Unis)

_SOS = butter(4, BANDE, btype="bandpass", fs=FS, output="sos")


def filtrer(eeg):
    """Passe-bande causal 0,5–40 Hz, canal par canal. eeg : (canaux, n)."""
    return sosfilt(_SOS, eeg, axis=-1).astype(np.float32)


def fenetres(eeg):
    """Vue (n_fenetres, canaux, N_FENETRE) sans copie ; fenêtre k = [k·PAS, k·PAS + 4 s)."""
    n = (eeg.shape[-1] - N_FENETRE) // N_PAS + 1
    if n <= 0:
        return np.empty((0, eeg.shape[0], N_FENETRE), eeg.dtype)
    vue = np.lib.stride_tricks.sliding_window_view(eeg, N_FENETRE, axis=-1)[:, ::N_PAS][:, :n]
    return vue.transpose(1, 0, 2)


def instants_decision(n_fenetres):
    """Instant (s, depuis le début du fichier) où chaque décision est disponible : fin de fenêtre."""
    return np.arange(n_fenetres) * PAS + DUREE_FENETRE


def fraction_ictale(n_fenetres, crises):
    """Part de chaque fenêtre recouverte par une crise annotée (0 à 1)."""
    debut = np.arange(n_fenetres) * PAS
    fin = debut + DUREE_FENETRE
    frac = np.zeros(n_fenetres)
    for d, f in crises:
        frac += np.clip(np.minimum(fin, f) - np.maximum(debut, d), 0, None) / DUREE_FENETRE
    return np.clip(frac, 0, 1)

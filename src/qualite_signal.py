"""Contrôle de qualité du signal, en amont de la détection (exigence EX-07).

Deux défauts d'électrode fréquents sur un appareil porté :
  - électrode décollée ou câble coupé : signal plat ;
  - saturation / artefact massif (mouvement, contact) : amplitude non physiologique.
Si trop de dérivations sont touchées, la fenêtre est déclarée inexploitable :
pas de décision « crise », et le défaut est signalé à l'aidant (pas de silence).
"""
import numpy as np

PLAT_UV = 0.5  # écart-type sous lequel une dérivation est considérée muette
SATURATION_UV = 3000.0  # µV crête : non physiologique (crises mesurées jusqu'à 2 400 µV)
MAX_CANAUX_DEFECTUEUX = 4  # sur 18 (sur 4 en montage réduit : 1)


def canaux_defectueux(f):
    """f : fenêtres (n, canaux, échantillons) en µV -> masque (n, canaux)."""
    plat = f.std(axis=-1) < PLAT_UV
    sature = np.abs(f).max(axis=-1) > SATURATION_UV
    return plat | sature


def fenetres_exploitables(f, max_defectueux=MAX_CANAUX_DEFECTUEUX):
    """True si la fenêtre peut être analysée."""
    return canaux_defectueux(f).sum(axis=-1) <= max_defectueux


def appliquer(proba, exploitable):
    """Probabilité forcée à 0 (pas d'alarme de crise) sur les fenêtres inexploitables,
    et indicateur de défaut à remonter à l'aidant."""
    return np.where(exploitable, proba, 0.0), ~exploitable

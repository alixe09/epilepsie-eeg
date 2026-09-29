"""Lecteur EDF minimal en numpy (format European Data Format, Kemp et al. 1992).

Suffisant pour CHB-MIT : enregistrements continus, échantillons int16, même
fréquence sur tous les canaux. Évite une dépendance lourde (mne) pour
une seule fonction.
"""
from pathlib import Path

import numpy as np

UNITES_VERS_UV = {"uV": 1.0, "µV": 1.0, "mV": 1e3, "V": 1e6}


def _champs(bloc, n, largeur):
    return [bloc[i * largeur:(i + 1) * largeur].decode("latin-1").strip() for i in range(n)]


def lire_edf(chemin):
    """Renvoie (signaux en µV de forme (canaux, n), fs, noms des canaux)."""
    brut = Path(chemin).read_bytes()
    taille_entete = int(brut[184:192])
    n_records = int(brut[236:244])
    duree_record = float(brut[244:252])
    ns = int(brut[252:256])

    h = memoryview(brut)[256:taille_entete].tobytes()
    decalage = 0

    def lire(largeur):
        nonlocal decalage
        vals = _champs(h[decalage:decalage + ns * largeur], ns, largeur)
        decalage += ns * largeur
        return vals

    noms = lire(16)
    lire(80)  # transducteur
    unites = lire(8)
    phys_min = np.array(lire(8), float)
    phys_max = np.array(lire(8), float)
    dig_min = np.array(lire(8), float)
    dig_max = np.array(lire(8), float)
    lire(80)  # préfiltrage
    n_ech = np.array(lire(8), int)

    if len(set(n_ech)) != 1:
        raise ValueError("fréquences d'échantillonnage différentes selon les canaux")
    n = n_ech[0]
    donnees = np.frombuffer(brut, dtype="<i2", offset=taille_entete)
    if n_records < 0:  # durée inconnue dans l'entête : déduite de la taille du fichier
        n_records = donnees.size // (ns * n)
    donnees = donnees[:n_records * ns * n].reshape(n_records, ns, n)
    donnees = donnees.transpose(1, 0, 2).reshape(ns, n_records * n)

    # physique = (numérique - dig_min) * gain + phys_min, converti en µV
    vers_uv = np.array([UNITES_VERS_UV.get(u, 1.0) for u in unites])
    gain = (phys_max - phys_min) / (dig_max - dig_min) * vers_uv
    decal = phys_min * vers_uv - dig_min * gain
    x = donnees * gain[:, None] + decal[:, None]
    return x, n / duree_record, noms

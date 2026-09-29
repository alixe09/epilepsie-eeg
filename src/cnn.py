"""Réseau de neurones : CNN 1D sur l'EEG filtré brut (4 s × 18 canaux), par patient.

Même découpage et même post-traitement que la méthode classique. Les crises ne
représentent que ~0,3 % du temps : chaque lot d'entraînement est tiré à moitié
dans les crises (avec décalage temporel aléatoire, donc des fenêtres toujours
un peu différentes) et à moitié dans le reste de l'enregistrement.

Usage : python src/cnn.py [chb01 ...]
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from classique import SEUIL_ICTAL
from decoupage import plis
from donnees import PATIENTS, RACINE, charger, lister_enregistrements
from pretraitement import DUREE_FENETRE, FS, N_FENETRE, filtrer, fenetres

DOSSIER_PROBAS = RACINE / "resultats" / "probas"
DOSSIER_MODELES = RACINE / "resultats" / "modeles"
ECRETAGE = 10.0  # en écarts-types : borne l'entrée (utile pour la quantification int8)
TAILLE_LOT = 64
EPOQUES = 15
PAS_PAR_EPOQUE = 150


def construire_modele(n_canaux=18):
    L = tf.keras.layers
    reg = tf.keras.regularizers.l2(1e-4)
    entree = L.Input((N_FENETRE, n_canaux))
    x = entree
    # Pas de BatchNormalization : ses statistiques, apprises sur des lots équilibrés
    # 50/50, ne correspondent pas au flux réel (99,7 % hors crise) ; un modèle sur
    # quatre prédisait « crise » en permanence. L'entrée est déjà normalisée par canal.
    for filtres, noyau, pas in [(16, 7, 2), (32, 5, 1), (32, 5, 1), (64, 3, 1)]:
        x = L.Conv1D(filtres, noyau, strides=pas, padding="same", activation="relu",
                     kernel_regularizer=reg)(x)
        x = L.MaxPooling1D(2)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(x)
    sortie = L.Dense(1, activation="sigmoid")(x)
    modele = tf.keras.Model(entree, sortie)
    modele.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy")
    return modele


class Patient:
    """Signaux filtrés d'un patient en mémoire (float16 : ~33 Mo par heure)."""

    def __init__(self, patient):
        self.noms, self.signaux, self.crises = [], [], []
        for chemin in lister_enregistrements(patient):
            eeg, _, crises = charger(chemin)
            self.noms.append(chemin.stem)
            self.signaux.append(filtrer(eeg).astype(np.float16))
            self.crises.append(crises)
        self.pli = plis(self.crises)

    def normalisation(self, fichiers):
        """Écart-type par canal (médiane sur les enregistrements d'entraînement)."""
        ecarts = [self.signaux[i][:, ::8].astype(np.float32).std(axis=1) for i in fichiers]
        return np.median(ecarts, axis=0).astype(np.float32)


def normaliser(f, ecart):
    """f : (..., canaux, n) -> (..., n, canaux) normalisé et écrêté, prêt pour le réseau."""
    x = np.clip(f / ecart[:, None], -ECRETAGE, ECRETAGE)
    return np.swapaxes(x, -1, -2).astype(np.float32)


class Lots(tf.keras.utils.Sequence):
    def __init__(self, pat, fichiers, ecart, graine=0):
        super().__init__()
        self.pat, self.ecart = pat, ecart
        self.rng = np.random.default_rng(graine)
        # positions de départ admissibles (en échantillons) pour chaque classe
        self.pos, self.neg, self.poids_neg = [], [], []
        for i in fichiers:
            n = self.pat.signaux[i].shape[1] - N_FENETRE
            hors_crise = np.ones(n, bool)
            for d, f in self.pat.crises[i]:
                # fenêtres recouvertes au moins à moitié par la crise
                a = int((d - DUREE_FENETRE * (1 - SEUIL_ICTAL)) * FS)
                b = int((f - DUREE_FENETRE * SEUIL_ICTAL) * FS)
                if b > a:
                    self.pos.append((i, max(a, 0), min(b, n)))
                hors_crise[max(int((d - DUREE_FENETRE) * FS), 0):int(f * FS)] = False
            self.neg.append((i, np.flatnonzero(hors_crise[::16]) * 16))  # tous les 1/8 s
            self.poids_neg.append(hors_crise.sum())
        self.poids_neg = np.array(self.poids_neg, float) / np.sum(self.poids_neg)
        self.poids_pos = np.array([b - a for _, a, b in self.pos], float)
        self.poids_pos /= self.poids_pos.sum()

    def __len__(self):
        return PAS_PAR_EPOQUE

    def _fenetre(self, i, debut):
        return self.pat.signaux[i][:, debut:debut + N_FENETRE].astype(np.float32)

    def __getitem__(self, _):
        moitie = TAILLE_LOT // 2
        x, y = [], []
        for k in self.rng.choice(len(self.pos), moitie, p=self.poids_pos):
            i, a, b = self.pos[k]
            x.append(self._fenetre(i, self.rng.integers(a, b)))
            y.append(1)
        for k in self.rng.choice(len(self.neg), moitie, p=self.poids_neg):
            i, departs = self.neg[k]
            x.append(self._fenetre(i, self.rng.choice(departs)))
            y.append(0)
        x = np.stack(x) * self.rng.uniform(0.8, 1.2, (len(x), 1, 1)).astype(np.float32)
        return normaliser(x, self.ecart), np.array(y, np.float32)


def predire(modele, signal, ecart, taille=1024):
    f = fenetres(signal)
    return np.concatenate([
        modele.predict_on_batch(normaliser(f[i:i + taille].astype(np.float32), ecart))[:, 0]
        for i in range(0, len(f), taille)]) if len(f) else np.zeros(0)


def main(patients):
    DOSSIER_PROBAS.mkdir(parents=True, exist_ok=True)
    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(0)
    for p in patients:
        if (DOSSIER_PROBAS / f"cnn_{p}.npz").exists():
            print(f"{p} : déjà fait", flush=True)
            continue
        t0 = time.time()
        pat = Patient(p)
        probas = [None] * len(pat.noms)
        for k in np.unique(pat.pli):
            app = np.flatnonzero(pat.pli != k)
            ecart = pat.normalisation(app)
            modele = construire_modele()
            lots = Lots(pat, app, ecart, graine=int(k))
            modele.fit(lots, epochs=EPOQUES, verbose=0)
            # contrôle : le modèle doit séparer ses propres données d'entraînement en inférence
            x, y = zip(*(lots[j] for j in range(8)))
            p_app = modele.predict_on_batch(np.concatenate(x))[:, 0]
            y = np.concatenate(y)
            controle = (float(p_app[y == 0].mean()), float(p_app[y == 1].mean()))
            for i in np.flatnonzero(pat.pli == k):
                probas[i] = predire(modele, pat.signaux[i], ecart)
            nom = f"cnn_{p}_pli{k}"
            modele.save(DOSSIER_MODELES / f"{nom}.keras")
            (DOSSIER_MODELES / f"{nom}_norm.json").write_text(json.dumps({
                "ecart_uv": ecart.tolist(), "ecretage": ECRETAGE,
                "test": [pat.noms[i] for i in np.flatnonzero(pat.pli == k)]}, indent=1))
            print(f"  {p} pli {k} : {time.time() - t0:.0f} s, proba moyenne sur l'entraînement "
                  f"hors crise {controle[0]:.3f} / crise {controle[1]:.3f}", flush=True)
            del modele
            tf.keras.backend.clear_session()
        np.savez(DOSSIER_PROBAS / f"cnn_{p}.npz", proba=np.concatenate(probas).astype(np.float32),
                 pli=pat.pli)
        print(f"{p} : {time.time() - t0:.0f} s", flush=True)
        del pat


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("patients", nargs="*", default=PATIENTS)
    main(a.parse_args().patients)

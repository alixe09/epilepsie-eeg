"""Réaction des deux méthodes à des défauts de capteur simulés (analyse des risques R4).

Part de 200 vraies fenêtres hors crise de chb01 (pli 0, jamais vues à
l'entraînement), leur applique un défaut, et mesure la part de fenêtres que
chaque modèle classerait « crise » (probabilité ≥ 0,5), sans puis avec le
contrôle de qualité du signal.

Usage : python src/robustesse.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from caracteristiques import calculer, charger_patient
from classique import entrainer, exemples_entrainement
from cnn import DOSSIER_MODELES, normaliser
from decoupage import plis
from donnees import RACINE, charger, lister_enregistrements
from pretraitement import FS, N_FENETRE, filtrer, fenetres, fraction_ictale
from qualite_signal import fenetres_exploitables


def defauts(f, rng):
    """Scénarios : nom -> fenêtres modifiées (µV, déjà filtrées comme en entrée du modèle)."""
    t = np.arange(N_FENETRE) / FS
    six = rng.choice(18, 6, replace=False)
    sat = f.copy()
    sat[:, six] = np.sign(rng.normal(size=(len(f), 6, N_FENETRE))) * 3276.7
    deux = f.copy()
    deux[:, :2] = 0
    mouvement = f + (800 * np.sin(2 * np.pi * 1.5 * t)).astype(np.float32)  # balancement, sous la saturation
    pointes = f.copy()
    pointes[:, :, ::FS] += 1500  # « pops » d'électrode, 1 par seconde
    return {
        "aucun défaut (référence)": f,
        "appareil déconnecté (tout plat)": np.zeros_like(f),
        "2 électrodes décollées": deux,
        "6 dérivations saturées": sat,
        "artefact de mouvement 800 µV, 1,5 Hz": mouvement,
        "pointes d'électrode 1 500 µV": pointes,
    }


def main():
    rng = np.random.default_rng(0)
    d = charger_patient("chb01")
    pli = plis(d["crises"])
    app = (pli[d["fichier"]] != 0)
    garder, y = exemples_entrainement(d["frac"])
    hgb = entrainer(d["X"][app & garder], y[app & garder])
    cnn = tf.keras.models.load_model(DOSSIER_MODELES / "cnn_chb01_pli0.keras")
    ecart = np.array(json.loads((DOSSIER_MODELES / "cnn_chb01_pli0_norm.json").read_text())["ecart_uv"],
                     np.float32)

    # 200 fenêtres de test hors crise, tirées dans les enregistrements du pli 0
    chemins = [c for i, c in enumerate(lister_enregistrements("chb01")) if pli[i] == 0]
    f = []
    for c in chemins[:4]:
        eeg, _, cr = charger(c)
        w = fenetres(filtrer(eeg))
        hors = np.flatnonzero(fraction_ictale(len(w), cr) == 0)
        f.append(w[rng.choice(hors, 50, replace=False)])
    f = np.concatenate(f)

    bilan = {}
    for nom, x in defauts(f, rng).items():
        p_hgb = hgb.predict_proba(calculer(x))[:, 1]
        p_cnn = cnn.predict_on_batch(normaliser(x, ecart))[:, 0]
        ok = fenetres_exploitables(x)
        bilan[nom] = {
            "classique_crise": float((p_hgb >= 0.5).mean()),
            "cnn_crise": float((p_cnn >= 0.5).mean()),
            "bloquees_par_controle": float((~ok).mean()),
            "classique_crise_apres_controle": float(((p_hgb >= 0.5) & ok).mean()),
            "cnn_crise_apres_controle": float(((p_cnn >= 0.5) & ok).mean()),
        }
        b = bilan[nom]
        print(f"{nom:38} classique {b['classique_crise']:5.0%} → {b['classique_crise_apres_controle']:5.0%}"
              f" · CNN {b['cnn_crise']:5.0%} → {b['cnn_crise_apres_controle']:5.0%}"
              f" · bloquées {b['bloquees_par_controle']:5.0%}")
    (RACINE / "resultats" / "robustesse.json").write_text(json.dumps(bilan, indent=1, ensure_ascii=False),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()

"""Budget embarqué des deux méthodes : taille, calcul par décision, latence, accord int8.

Usage : python src/embarque.py   (après cnn.py et export_tflite.py)
"""
import json
import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from caracteristiques import calculer, charger_patient
from classique import entrainer, exemples_entrainement
from cnn import DOSSIER_MODELES
from donnees import PATIENTS, RACINE
from export_tflite import ModeleInt8
from pretraitement import N_FENETRE


def macs(modele):
    """Multiplications-accumulations par fenêtre (convolutions et couche dense)."""
    total = 0
    for couche in modele.layers:
        if isinstance(couche, tf.keras.layers.Conv1D):
            longueur, filtres = couche.output.shape[1:]
            total += longueur * couche.kernel_size[0] * couche.input.shape[-1] * filtres
        elif isinstance(couche, tf.keras.layers.Dense):
            total += int(np.prod(couche.kernel.shape))
    return int(total)


def chrono(f, n=200):
    f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n * 1000


def main():
    rng = np.random.default_rng(0)
    fenetre = rng.normal(0, 50, (1, 18, N_FENETRE)).astype(np.float32)

    # CNN
    modele = tf.keras.models.load_model(DOSSIER_MODELES / "cnn_chb01_pli0.keras")
    tailles = [p.stat().st_size for p in DOSSIER_MODELES.glob("cnn_*_int8.tflite")]
    int8 = ModeleInt8((DOSSIER_MODELES / "cnn_chb01_pli0_int8.tflite").read_bytes(), taille_lot=1)
    x = np.swapaxes(fenetre / 50, 1, 2)
    accord = {}
    for p in PATIENTS:
        f32 = np.load(RACINE / "resultats" / "probas" / f"cnn_{p}.npz")["proba"]
        i8 = np.load(RACINE / "resultats" / "probas" / f"cnn_int8_{p}.npz")["proba"]
        accord[p] = {"ecart_moyen": float(np.abs(f32 - i8).mean()),
                     "accord_decision_0_5": float(((f32 >= 0.5) == (i8 >= 0.5)).mean())}
    cnn = {"parametres": modele.count_params(), "macs_par_decision": macs(modele),
           "taille_int8_ko": [min(tailles) / 1000, max(tailles) / 1000],
           "latence_int8_ms_pc": chrono(lambda: int8(x)), "accord_float_int8": accord}

    # classique : caractéristiques + gradient boosting
    d = charger_patient("chb01")
    garder, y = exemples_entrainement(d["frac"])
    hgb = entrainer(d["X"][garder], y[garder])
    feuilles = sum(arbre.nodes.shape[0] for iteration in hgb._predictors for arbre in iteration)
    classique = {
        "caracteristiques": d["X"].shape[1],
        "arbres": hgb.n_iter_,
        "noeuds": int(feuilles),
        "taille_modele_ko": len(pickle.dumps(hgb)) / 1000,
        "latence_caracteristiques_ms_pc": chrono(lambda: calculer(fenetre)),
        "latence_modele_ms_pc": chrono(lambda: hgb.predict_proba(calculer(fenetre))),
    }
    bilan = {"cnn": cnn, "classique": classique}
    (RACINE / "resultats" / "embarque.json").write_text(json.dumps(bilan, indent=1), encoding="utf-8")
    print(json.dumps(bilan, indent=1))


if __name__ == "__main__":
    main()

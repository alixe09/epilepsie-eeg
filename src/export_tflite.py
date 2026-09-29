"""Export des CNN en TensorFlow Lite int8 (poids ET activations), comme sur microcontrôleur.

Pour chaque patient et chaque pli : quantification calibrée sur des fenêtres
d'entraînement, puis inférence int8 sur les enregistrements de test du pli.
Les probabilités int8 sont sauvegardées sous la méthode « cnn_int8 » : les
métriques cliniques sont recalculées avec le modèle qui serait embarqué.

Usage : python src/export_tflite.py [chb01 ...]
"""
import argparse
import json
import os
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from cnn import DOSSIER_MODELES, DOSSIER_PROBAS, Lots, Patient, normaliser
from donnees import PATIENTS
from pretraitement import fenetres


def convertir(modele, lots, n_lots=20):
    def representatif():
        for i in range(n_lots):
            x, _ = lots[i]
            for fenetre in x[::4]:  # les lots sont ordonnés (crises puis reste) : prendre les deux
                yield [fenetre[None]]

    # Quantifier la sigmoïde écraserait les probabilités élevées (pas de 1/256, maximum
    # 0,996) alors que les seuils utiles sont vers 0,99+ : on quantifie le logit et la
    # sigmoïde est appliquée après (une opération par décision).
    sortie = modele.layers[-1]
    sortie.activation = tf.keras.activations.linear
    # from_keras_model échoue avec Keras 3 / TF 2.16 (BatchNormalization) : passage par un SavedModel
    with tempfile.TemporaryDirectory() as dossier:
        modele.export(dossier, verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(dossier)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representatif
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        return conv.convert()


class ModeleInt8:
    """Inférence TFLite int8 par lots (entrée et logit quantifiés), renvoie une probabilité."""

    def __init__(self, contenu, taille_lot=256):
        self.interp = tf.lite.Interpreter(model_content=contenu, num_threads=2)
        self.entree = self.interp.get_input_details()[0]
        self.sortie = self.interp.get_output_details()[0]
        self.taille_lot = taille_lot
        self.interp.resize_tensor_input(self.entree["index"], [taille_lot, *self.entree["shape"][1:]])
        self.interp.allocate_tensors()

    def __call__(self, x):
        (s_in, z_in), (s_out, z_out) = self.entree["quantization"], self.sortie["quantization"]
        sorties = []
        for i in range(0, len(x), self.taille_lot):
            lot = x[i:i + self.taille_lot]
            n = len(lot)
            if n < self.taille_lot:
                lot = np.concatenate([lot, np.zeros((self.taille_lot - n, *lot.shape[1:]), lot.dtype)])
            q = np.clip(np.round(lot / s_in + z_in), -128, 127).astype(np.int8)
            self.interp.set_tensor(self.entree["index"], q)
            self.interp.invoke()
            y = self.interp.get_tensor(self.sortie["index"])[:n, 0]
            logit = (y.astype(np.float32) - z_out) * s_out
            sorties.append(1 / (1 + np.exp(-logit)))
        return np.concatenate(sorties) if sorties else np.zeros(0, np.float32)


def main(patients):
    bilan = {}
    for p in patients:
        t0 = time.time()
        pat = Patient(p)
        probas = [None] * len(pat.noms)
        for k in np.unique(pat.pli):
            nom = f"cnn_{p}_pli{k}"
            modele = tf.keras.models.load_model(DOSSIER_MODELES / f"{nom}.keras")
            ecart = np.array(json.loads((DOSSIER_MODELES / f"{nom}_norm.json").read_text())["ecart_uv"],
                             np.float32)
            app = np.flatnonzero(pat.pli != k)
            contenu = convertir(modele, Lots(pat, app, ecart, graine=100 + int(k)))
            (DOSSIER_MODELES / f"{nom}_int8.tflite").write_bytes(contenu)
            int8 = ModeleInt8(contenu)
            for i in np.flatnonzero(pat.pli == k):
                f = fenetres(pat.signaux[i])
                probas[i] = np.concatenate([int8(normaliser(f[j:j + 1024].astype(np.float32), ecart))
                                            for j in range(0, len(f), 1024)])
            bilan[nom] = len(contenu)
        np.savez(DOSSIER_PROBAS / f"cnn_int8_{p}.npz", proba=np.concatenate(probas), pli=pat.pli)
        print(f"{p} : {time.time() - t0:.0f} s", flush=True)
        del pat
    print("tailles (octets) :", bilan)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("patients", nargs="*", default=PATIENTS)
    main(a.parse_args().patients)

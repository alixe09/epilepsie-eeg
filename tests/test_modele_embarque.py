"""Modèle int8 embarqué (EX-05, EX-06)."""
import json
from pathlib import Path

import numpy as np
import pytest

MODELES = Path(__file__).resolve().parents[1] / "resultats" / "modeles"
TFLITE = sorted(MODELES.glob("cnn_*_int8.tflite"))
pytestmark = pytest.mark.skipif(not TFLITE, reason="modèles int8 non générés (src/export_tflite.py)")


def test_taille_compatible_microcontroleur():
    assert max(p.stat().st_size for p in TFLITE) <= 100_000


def test_int8_proche_du_float_sur_des_fenetres_reelles():
    """Même décision float / int8 sur l'extrait de démo (enregistrement jamais vu par ce modèle)."""
    from export_tflite import ModeleInt8
    import tensorflow as tf
    from cnn import normaliser
    from pretraitement import fenetres

    demo = Path(__file__).resolve().parents[1] / "app" / "demo_data" / "chb01_03.npz"
    if not demo.exists():
        pytest.skip("données de démo absentes")
    with np.load(demo) as z:
        eeg = z["eeg"].astype(np.float32) * float(z["echelle_uv"])
    # le modèle de test de chb01_03 est celui dont la liste « test » contient cet enregistrement
    for norm in MODELES.glob("cnn_chb01_pli*_norm.json"):
        info = json.loads(norm.read_text())
        if "chb01_03" in info["test"]:
            break
    nom = norm.name.replace("_norm.json", "")
    x = normaliser(fenetres(eeg).astype(np.float32), np.array(info["ecart_uv"], np.float32))
    p_int8 = ModeleInt8((MODELES / f"{nom}_int8.tflite").read_bytes())(x)
    keras = MODELES / f"{nom}.keras"
    if not keras.exists():
        pytest.skip("modèle float absent")
    p_float = tf.keras.models.load_model(keras).predict_on_batch(x)[:, 0]
    assert np.abs(p_int8 - p_float).mean() < 0.02
    assert ((p_int8 >= 0.5) == (p_float >= 0.5)).mean() >= 0.99

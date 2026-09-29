"""Contrôle de qualité du signal (EX-07) : un défaut capteur ne doit pas produire d'alarme de crise."""
import numpy as np

from pretraitement import FS, N_FENETRE
from qualite_signal import appliquer, canaux_defectueux, fenetres_exploitables


def eeg_normal(n=10, seed=0):
    return np.random.default_rng(seed).normal(0, 30, (n, 18, N_FENETRE)).astype(np.float32)


def test_eeg_normal_exploitable():
    assert fenetres_exploitables(eeg_normal()).all()


def test_appareil_deconnecte_bloque():
    assert not fenetres_exploitables(np.zeros((3, 18, N_FENETRE), np.float32)).any()


def test_saturation_bloquee():
    f = eeg_normal()
    f[:, :6] = 3276.7
    assert not fenetres_exploitables(f).any()


def test_quelques_electrodes_decollees_toleres():
    """2 dérivations muettes sur 18 : on continue d'analyser, le défaut est visible par canal."""
    f = eeg_normal()
    f[:, :2] = 0
    assert fenetres_exploitables(f).all()
    assert canaux_defectueux(f)[:, :2].all() and not canaux_defectueux(f)[:, 2:].any()


def test_crise_de_forte_amplitude_non_bloquee():
    """Pointes-ondes de 2 400 µV (maximum mesuré pendant une crise dans CHB-MIT) : analysées."""
    t = np.arange(N_FENETRE) / FS
    f = np.broadcast_to(2400 * np.sin(2 * np.pi * 3 * t), (1, 18, N_FENETRE)).astype(np.float32)
    assert fenetres_exploitables(f).all()


def test_probabilite_forcee_a_zero_et_defaut_signale():
    proba, defaut = appliquer(np.array([0.99, 0.99]), np.array([True, False]))
    np.testing.assert_array_equal(proba, [0.99, 0.0])
    np.testing.assert_array_equal(defaut, [False, True])

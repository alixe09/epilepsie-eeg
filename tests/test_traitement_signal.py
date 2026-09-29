"""Lecture EDF, filtrage causal, fenêtrage et étiquettes."""
import numpy as np
import pytest

from edf import lire_edf
from pretraitement import FS, N_FENETRE, filtrer, fenetres, fraction_ictale, instants_decision


def ecrire_edf(chemin, signaux_numeriques, fs, phys=(-3200.0, 3200.0), dig=(-32768, 32767), unite="uV"):
    """Écrit un EDF minimal (1 enregistrement de données par seconde)."""
    ns, n = signaux_numeriques.shape
    n_rec = n // fs

    def champ(v, largeur):
        return str(v).ljust(largeur)[:largeur].encode("latin-1")

    entete = b"".join([champ(0, 8), champ("X", 80), champ("X", 80), champ("01.01.01", 8),
                       champ("00.00.00", 8), champ(256 * (ns + 1), 8), champ("", 44),
                       champ(n_rec, 8), champ(1, 8), champ(ns, 4)])
    for largeur, v in [(16, None), (80, ""), (8, unite), (8, phys[0]), (8, phys[1]),
                       (8, dig[0]), (8, dig[1]), (80, ""), (8, fs), (32, "")]:
        entete += b"".join(champ(f"C{i}" if v is None else v, largeur) for i in range(ns))
    donnees = signaux_numeriques[:, :n_rec * fs].reshape(ns, n_rec, fs).transpose(1, 0, 2)
    chemin.write_bytes(entete + donnees.astype("<i2").tobytes())


def test_lecture_edf_conversion_physique(tmp_path):
    rng = np.random.default_rng(0)
    num = rng.integers(-32768, 32767, (3, 256 * 5), dtype=np.int16)
    ecrire_edf(tmp_path / "a.edf", num, 256)
    x, fs, noms = lire_edf(tmp_path / "a.edf")
    attendu = (num + 32768) * 6400 / 65535 - 3200
    assert fs == 256 and noms == ["C0", "C1", "C2"]
    np.testing.assert_allclose(x, attendu, atol=1e-9)


def test_lecture_edf_millivolts(tmp_path):
    num = np.full((1, 256), 1000, dtype=np.int16)
    ecrire_edf(tmp_path / "b.edf", num, 256, phys=(-32.768, 32.767), dig=(-32768, 32767), unite="mV")
    x, _, _ = lire_edf(tmp_path / "b.edf")
    np.testing.assert_allclose(x, 1000.0, atol=1e-6)  # 1 mV = 1000 µV


def test_filtre_causal_ne_voit_pas_le_futur():
    """Modifier la fin du signal ne change pas le début filtré (utilisable en temps réel)."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, FS * 20)).astype(np.float32)
    b = a.copy()
    b[:, FS * 10:] += 500
    np.testing.assert_array_equal(filtrer(a)[:, :FS * 10], filtrer(b)[:, :FS * 10])


def test_filtre_passe_bande():
    t = np.arange(FS * 30) / FS
    for f, garde in [(0.1, False), (10, True), (60, False)]:
        y = filtrer(np.sin(2 * np.pi * f * t)[None])[0, FS * 10:]
        assert (y.std() > 0.5) == garde, f


def test_fenetres_et_instants():
    x = np.arange(FS * 10, dtype=np.float32)[None].repeat(2, 0)
    f = fenetres(x)
    assert f.shape == (7, 2, N_FENETRE)  # 10 s, fenêtres de 4 s au pas de 1 s
    assert f[3, 0, 0] == 3 * FS and f[3, 0, -1] == 3 * FS + N_FENETRE - 1
    np.testing.assert_array_equal(instants_decision(3), [4, 5, 6])
    assert fenetres(x[:, :100]).shape[0] == 0


def test_fraction_ictale():
    frac = fraction_ictale(10, np.array([[5.0, 7.0]]))
    # fenêtre k = [k, k+4) ; crise [5, 7)
    np.testing.assert_allclose(frac[[0, 1, 2, 3, 4, 5, 6, 7]], [0, 0, 0.25, 0.5, 0.5, 0.5, 0.25, 0])

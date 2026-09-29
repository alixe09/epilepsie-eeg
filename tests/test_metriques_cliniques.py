"""Post-traitement en alarmes et métriques par événement."""
import numpy as np

from decoupage import plis
from evaluation import alarmes, evaluer, lisser


def test_lissage_causal():
    p = np.array([0, 0, 1, 1, 1, 0])
    np.testing.assert_allclose(lisser(p, 2), [0, 0, 0.5, 1, 1, 0.5])
    np.testing.assert_allclose(lisser(p, 1), p)


def test_periode_refractaire():
    t = np.arange(200.0)
    p = np.zeros(200)
    p[10:15] = 1  # une alarme
    p[40:45] = 1  # trop proche : ignorée
    p[100:101] = 1  # > 60 s après : nouvelle alarme
    np.testing.assert_array_equal(alarmes(p, t, 0.5), [10, 100])


def test_detection_delai_et_fausses_alarmes():
    crises = [np.array([[100.0, 160.0]]), np.zeros((0, 2))]
    als = [np.array([50.0, 80.0, 112.0, 215.0, 300.0]), np.array([10.0])]
    m = evaluer(als, crises, [3600.0, 3600.0])
    # 80 est dans la tolérance avant (100 − 30 s) : détection en avance de 20 s
    assert m["detectees"] == 1 and m["delais"] == [-20.0]
    # 112 et 215 (≤ fin + 60 s) sont tolérées ; 50, 300 et 10 sont fausses
    assert m["fausses_alarmes"] == 3
    assert np.isclose(m["fa_par_24h"], 3 / 2 * 24)


def test_crise_manquee():
    m = evaluer([np.array([])], [np.array([[100.0, 150.0]])], [3600.0])
    assert m["sensibilite"] == 0 and np.isnan(m["delai_median"]) and m["fausses_alarmes"] == 0


def test_crise_breve_alarme_juste_apres():
    """Crise de 8 s : une alarme 20 s après la fin compte (tolérance SzCORE de 60 s)."""
    m = evaluer([np.array([128.0, 200.0])], [np.array([[100.0, 108.0]])], [3600.0])
    assert m["detectees"] == 1 and m["delais"] == [28.0] and m["fausses_alarmes"] == 1


def test_plis_chaque_pli_teste_une_crise():
    crises = [np.zeros((0, 2))] * 10
    crises = crises[:3] + [np.ones((1, 2))] + crises[3:6] + [np.ones((2, 2))] * 4
    pli = plis(crises, 4)
    for k in range(4):
        assert any(len(crises[i]) for i in np.flatnonzero(pli == k))
    assert set(pli) == {0, 1, 2, 3}

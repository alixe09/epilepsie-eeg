"""Choix du post-traitement et tableau des résultats cliniques.

Le seuil d'alarme et le lissage de chaque patient sont choisis sur les
5 AUTRES patients (validation croisée par patient sur le post-traitement) :
aucun réglage n'utilise les crises du patient évalué. Règle, fixée à l'avance :
maximiser la sensibilité moyenne sous la contrainte d'au plus CIBLE_FA fausses
alarmes par 24 h en moyenne (à défaut, minimiser les fausses alarmes).

Usage : python src/resumer.py [classique cnn classique_temporaux ...]
"""
import json
import sys

import numpy as np

from caracteristiques import charger_patient
from donnees import PATIENTS, RACINE
from evaluation import evaluer_probas

DOSSIER = RACINE / "resultats"
CIBLE_FA = 1.0  # fausse alarme par 24 h
SEUILS = np.round(np.concatenate([np.arange(0.1, 0.9, 0.05), np.arange(0.9, 0.99, 0.01),
                                  [0.99, 0.995, 0.998, 0.999]]), 3)
LISSAGES = [1, 3, 5, 10]


def grille(methode, patients):
    """metriques[patient][(lissage, seuil)] pour toute la grille."""
    res = {}
    for p in patients:
        d = charger_patient(p)
        proba = np.load(DOSSIER / "probas" / f"{methode}_{p}.npz")["proba"]
        assert len(proba) == len(d["t"]), (methode, p)
        res[p] = {(n, s): evaluer_probas(proba, d["fichier"], d["t"], d["crises"], d["duree"], s, n)
                  for n in LISSAGES for s in SEUILS}
    return res


def choisir(res, autres):
    def score(cfg):
        fa = np.mean([res[q][cfg]["fa_par_24h"] for q in autres])
        se = np.mean([res[q][cfg]["sensibilite"] for q in autres])
        # contrainte tenue : meilleure sensibilité ; sinon : le moins de fausses alarmes
        return (True, se, -fa) if fa <= CIBLE_FA else (False, -fa, se)
    return max(res[autres[0]], key=score)


def resumer(methode, patients=PATIENTS):
    res = grille(methode, patients)
    lignes = []
    for p in patients:
        cfg = choisir(res, [q for q in patients if q != p])
        m = res[p][cfg]
        # réglage individuel respectant la contrainte chez ce patient, choisi sur ses propres
        # résultats : borne optimiste de ce que donnerait une calibration par patient
        ideal = res[p][choisir(res, [p])]
        lignes.append({"patient": p, "lissage": cfg[0], "seuil": cfg[1],
                       "ideal": {k: ideal[k] for k in ["detectees", "fa_par_24h", "delai_median"]},
                       **{k: m[k] for k in ["crises", "detectees", "sensibilite", "fausses_alarmes",
                                            "heures", "fa_par_24h", "delai_median", "delais"]}})
    delais = [x for l in lignes for x in l["delais"]]
    total = {
        "crises": sum(l["crises"] for l in lignes),
        "detectees": sum(l["detectees"] for l in lignes),
        "sensibilite_moyenne": float(np.mean([l["sensibilite"] for l in lignes])),
        "fa_par_24h_moyenne": float(np.mean([l["fa_par_24h"] for l in lignes])),
        "fa_par_24h_globale": sum(l["fausses_alarmes"] for l in lignes)
                              / sum(l["heures"] for l in lignes) * 24,
        "delai_median": float(np.median(delais)) if delais else None,
        "heures": sum(l["heures"] for l in lignes),
        "ideal_detectees": sum(l["ideal"]["detectees"] for l in lignes),
        "ideal_fa_par_24h_moyenne": float(np.mean([l["ideal"]["fa_par_24h"] for l in lignes])),
    }
    # courbe globale sensibilité / fausses alarmes pour chaque lissage
    courbes = {n: [(float(np.mean([res[p][(n, s)]["fa_par_24h"] for p in patients])),
                    float(np.mean([res[p][(n, s)]["sensibilite"] for p in patients]))) for s in SEUILS]
               for n in LISSAGES}
    return {"methode": methode, "patients": lignes, "total": total,
            "courbes": {str(n): c for n, c in courbes.items()}, "seuils": SEUILS.tolist()}


def afficher(r):
    print(f"\n=== {r['methode']} ===")
    print(f"{'patient':8} {'crises':>8} {'FA/24h':>7} {'délai méd.':>10}  réglage (choisi sur les autres)")
    for l in r["patients"]:
        print(f"{l['patient']:8} {l['detectees']:>3} / {l['crises']:<3} {l['fa_par_24h']:7.2f} "
              f"{l['delai_median']:9.1f}s  seuil {l['seuil']}, lissage {l['lissage']}")
    t = r["total"]
    print(f"{'total':8} {t['detectees']:>3} / {t['crises']:<3} {t['fa_par_24h_globale']:7.2f} "
          f"{t['delai_median']:9.1f}s  ({t['heures']:.0f} h, sensibilité moyenne "
          f"{t['sensibilite_moyenne']:.0%})")
    print(f"réglage individuel, ≤ {CIBLE_FA:g} FA/24h chez chacun (optimiste) : {t['ideal_detectees']} / {t['crises']}, "
          f"{t['ideal_fa_par_24h_moyenne']:.2f} FA/24h en moyenne")


if __name__ == "__main__":
    for methode in sys.argv[1:] or ["classique", "cnn"]:
        r = resumer(methode)
        afficher(r)
        (DOSSIER / f"resume_{methode}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")

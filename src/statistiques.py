"""Incertitude des résultats : 6 patients et 50 crises, c'est peu.

- Sensibilité : intervalle de confiance à 95 % par bootstrap **sur les patients**
  (les crises d'un même enfant se ressemblent : les rééchantillonner une à une
  donnerait un intervalle trop étroit).
- Fausses alarmes par 24 h : intervalle exact de Poisson sur le nombre total,
  et bootstrap sur les patients (qui capte l'effet d'un patient comme chb24).
- Classique contre CNN : test de McNemar exact sur les mêmes crises (seules
  comptent les crises détectées par une méthode et pas par l'autre). Il suppose
  les crises indépendantes : on le complète par un test des signes **par patient**
  (combien de patients chaque méthode détecte-t-elle mieux ?), plus prudent.

Chaque crise est évaluée avec le réglage d'usine de son patient (resume_*.json).

Usage : python src/statistiques.py   (après resumer.py)
"""
import json

import numpy as np
from scipy.stats import binomtest, chi2

from caracteristiques import charger_patient
from donnees import PATIENTS, RACINE
from evaluation import evaluer_probas

RES = RACINE / "resultats"
SCENARIOS = {
    "chronologique": ("classique_chrono", "cnn_chrono"),
    "validation_croisee": ("classique", "cnn_int8"),
}
N_BOOTSTRAP = 10_000


def par_patient(methode):
    """Pour chaque patient : détection de chaque crise, fausses alarmes, heures testées."""
    reglages = {l["patient"]: l for l in json.loads(
        (RES / f"resume_{methode}.json").read_text(encoding="utf-8"))["patients"]}
    sortie = {}
    for p in PATIENTS:
        d = charger_patient(p)
        proba = np.load(RES / "probas" / f"{methode}_{p}.npz")["proba"]
        testes = [i for i in range(len(d["duree"])) if not np.isnan(proba[d["fichier"] == i]).any()]
        r = reglages[p]
        m = evaluer_probas(proba, d["fichier"], d["t"], d["crises"], d["duree"], r["seuil"], r["lissage"], testes)
        assert m["detectees"] == r["detectees"], (methode, p)  # cohérent avec resumer.py
        sortie[p] = {"detection": m["detection"], "fa": m["fausses_alarmes"], "heures": m["heures"]}
    return sortie


def ic_poisson(k, alpha=0.05):
    bas = chi2.ppf(alpha / 2, 2 * k) / 2 if k else 0.0
    haut = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return bas, haut


def bootstrap(res, rng):
    patients = list(res)
    se, fa = [], []
    for _ in range(N_BOOTSTRAP):
        tirage = rng.choice(patients, len(patients))
        det = [x for p in tirage for x in res[p]["detection"]]
        se.append(np.mean(det) if det else np.nan)
        fa.append(sum(res[p]["fa"] for p in tirage) / sum(res[p]["heures"] for p in tirage) * 24)
    return np.nanpercentile(se, [2.5, 97.5]), np.percentile(fa, [2.5, 97.5])


def resumer(methode, rng):
    res = par_patient(methode)
    det = [x for p in PATIENTS for x in res[p]["detection"]]
    k_fa, heures = sum(res[p]["fa"] for p in PATIENTS), sum(res[p]["heures"] for p in PATIENTS)
    ic_se, ic_fa_boot = bootstrap(res, rng)
    fa_bas, fa_haut = ic_poisson(k_fa)
    return res, {
        "crises": len(det), "detectees": int(sum(det)), "sensibilite": float(np.mean(det)),
        "ic95_sensibilite_bootstrap_patients": [float(x) for x in ic_se],
        "fausses_alarmes": k_fa, "heures": heures, "fa_par_24h": k_fa / heures * 24,
        "ic95_fa_poisson": [fa_bas / heures * 24, fa_haut / heures * 24],
        "ic95_fa_bootstrap_patients": [float(x) for x in ic_fa_boot],
    }


def main():
    rng = np.random.default_rng(0)
    bilan = {}
    for scenario, (m_classique, m_cnn) in SCENARIOS.items():
        res_c, s_c = resumer(m_classique, rng)
        res_n, s_n = resumer(m_cnn, rng)
        a = np.array([x for p in PATIENTS for x in res_c[p]["detection"]])
        b = np.array([x for p in PATIENTS for x in res_n[p]["detection"]])
        seul_c, seul_n = int((a & ~b).sum()), int((~a & b).sum())
        test = binomtest(seul_c, seul_c + seul_n, 0.5) if seul_c + seul_n else None
        ecarts = {p: sum(res_c[p]["detection"]) - sum(res_n[p]["detection"]) for p in PATIENTS}
        mieux_c = sum(e > 0 for e in ecarts.values())
        mieux_n = sum(e < 0 for e in ecarts.values())
        signes = binomtest(mieux_c, mieux_c + mieux_n, 0.5) if mieux_c + mieux_n else None
        bilan[scenario] = {
            m_classique: s_c, m_cnn: s_n,
            "mcnemar": {"seulement_classique": seul_c, "seulement_cnn": seul_n,
                        "p_valeur": test.pvalue if test else 1.0},
            "signes_par_patient": {"classique_mieux": mieux_c, "cnn_mieux": mieux_n,
                                   "egalite": len(PATIENTS) - mieux_c - mieux_n,
                                   "p_valeur": signes.pvalue if signes else 1.0},
        }
        print(f"\n=== {scenario} ===")
        for m, s in [(m_classique, s_c), (m_cnn, s_n)]:
            print(f"{m:18} {s['detectees']}/{s['crises']} = {s['sensibilite']:.0%} "
                  f"[IC95 {s['ic95_sensibilite_bootstrap_patients'][0]:.0%}–"
                  f"{s['ic95_sensibilite_bootstrap_patients'][1]:.0%}] · "
                  f"{s['fa_par_24h']:.2f} FA/24h [Poisson {s['ic95_fa_poisson'][0]:.2f}–"
                  f"{s['ic95_fa_poisson'][1]:.2f} ; patients {s['ic95_fa_bootstrap_patients'][0]:.2f}–"
                  f"{s['ic95_fa_bootstrap_patients'][1]:.2f}]")
        mc = bilan[scenario]["mcnemar"]
        print(f"McNemar : {mc['seulement_classique']} crises vues seulement par la méthode classique, "
              f"{mc['seulement_cnn']} seulement par le CNN, p = {mc['p_valeur']:.3f}")
        sg = bilan[scenario]["signes_par_patient"]
        print(f"Par patient : classique mieux chez {sg['classique_mieux']}, CNN mieux chez "
              f"{sg['cnn_mieux']}, égalité chez {sg['egalite']}, p = {sg['p_valeur']:.2f}")
    (RES / "statistiques.json").write_text(json.dumps(bilan, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Post-traitement des probabilités en alarmes, et métriques orientées patient.

Une probabilité par seconde et par enregistrement devient une suite d'alarmes :
  1. lissage causal (moyenne des n dernières décisions) ;
  2. alarme quand la moyenne franchit le seuil ;
  3. période réfractaire : pas de nouvelle alarme dans les 60 s qui suivent.

Métriques (par événement, comme en clinique, pas par fenêtre), avec les
tolérances du cadre SzCORE (Dan et al., Epilepsia 2024) :
  - crise détectée si une alarme tombe dans [début − 30 s, fin + 60 s] ;
  - délai = première alarme de cet intervalle − début annoté ;
  - fausse alarme = alarme hors de tout intervalle [début − 30 s, fin + 60 s] ;
  - fausses alarmes par 24 h = nombre / durée totale enregistrée.
"""
import numpy as np

REFRACTAIRE = 60.0  # s
TOLERANCE_AVANT = 30.0  # s : l'annotation du début est approximative
TOLERANCE_APRES = 60.0  # s : alarme juste après une crise brève encore utile


def lisser(proba, n):
    """Moyenne glissante causale sur les n dernières décisions."""
    if n <= 1:
        return np.asarray(proba, float)
    c = np.cumsum(np.insert(np.asarray(proba, float), 0, 0.0))
    idx = np.arange(1, len(proba) + 1)
    debut = np.maximum(idx - n, 0)
    return (c[idx] - c[debut]) / (idx - debut)


def alarmes(proba, t, seuil, n_lissage=1, refractaire=REFRACTAIRE):
    """Instants d'alarme (s) pour un enregistrement."""
    candidats = np.asarray(t)[lisser(proba, n_lissage) >= seuil]
    sorties, i = [], 0
    while i < len(candidats):
        sorties.append(candidats[i])
        i = np.searchsorted(candidats, candidats[i] + refractaire)  # saute la période réfractaire
    return np.array(sorties)


def evaluer(alarmes_par_fichier, crises_par_fichier, durees):
    """Métriques sur un ensemble d'enregistrements (listes alignées)."""
    delais, n_crises, n_detectees, n_fa = [], 0, 0, 0
    for al, crises in zip(alarmes_par_fichier, crises_par_fichier):
        al = np.asarray(al)
        tolere = np.zeros(len(al), bool)
        for debut, fin in crises:
            n_crises += 1
            dedans = (al >= debut - TOLERANCE_AVANT) & (al <= fin + TOLERANCE_APRES)
            if dedans.any():
                n_detectees += 1
                delais.append(al[dedans][0] - debut)
            tolere |= dedans
        n_fa += int((~tolere).sum())
    heures = float(np.sum(durees)) / 3600
    return {
        "crises": n_crises,
        "detectees": n_detectees,
        "sensibilite": n_detectees / n_crises if n_crises else np.nan,
        "fausses_alarmes": n_fa,
        "heures": heures,
        "fa_par_24h": n_fa / heures * 24 if heures else np.nan,
        "delais": delais,
        "delai_median": float(np.median(delais)) if delais else np.nan,
    }


def evaluer_probas(proba, fichier, t, crises, durees, seuil, n_lissage, fichiers=None):
    """Raccourci : probabilités concaténées (avec index d'enregistrement) -> métriques,
    sur les enregistrements `fichiers` (tous par défaut)."""
    fichiers = range(len(durees)) if fichiers is None else fichiers
    als = [alarmes(proba[fichier == i], t[fichier == i], seuil, n_lissage) for i in fichiers]
    return evaluer(als, [crises[i] for i in fichiers], [durees[i] for i in fichiers])

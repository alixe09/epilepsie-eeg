"""Figures du README (docs/).

Usage : python src/figures.py   (après resumer.py et make_demo_data.py)
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from donnees import RACINE
from evaluation import alarmes, lisser

DOCS = RACINE / "docs"
COULEURS = {"classique": "#2a78d6", "cnn_int8": "#eb6834", "classique_temporaux": "#1baf7a"}
NOMS = {"classique": "Classique, 18 canaux", "cnn_int8": "CNN int8, 18 canaux",
        "classique_temporaux": "Classique, 4 canaux temporaux"}
ENCRE, ENCRE_2 = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": ENCRE_2, "axes.labelcolor": ENCRE,
                     "xtick.color": ENCRE_2, "ytick.color": ENCRE_2, "axes.spines.top": False,
                     "axes.spines.right": False})


def front_pareto(points):
    """Meilleure sensibilité atteignable pour chaque niveau de fausses alarmes."""
    pts = sorted(points)
    front, meilleur = [], -1
    for fa, se in pts:
        if se > meilleur:
            front.append((fa, se))
            meilleur = se
    return np.array(front)


def courbe_compromis():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in COULEURS:
        chemin = RACINE / "resultats" / f"resume_{m}.json"
        if not chemin.exists():
            continue
        r = json.loads(chemin.read_text(encoding="utf-8"))
        f = front_pareto([tuple(p) for c in r["courbes"].values() for p in c])
        ax.step(f[:, 0], f[:, 1] * 100, where="post", color=COULEURS[m], lw=2, label=NOMS[m])
        t = r["total"]
        ax.plot(t["fa_par_24h_moyenne"], t["sensibilite_moyenne"] * 100, "o", ms=8,
                color=COULEURS[m], mec="white", mew=2)
    # scénario chronologique (réglage d'usine) : losanges évidés
    for m, chrono in [("classique", "classique_chrono"), ("cnn_int8", "cnn_chrono")]:
        chemin = RACINE / "resultats" / f"resume_{chrono}.json"
        if chemin.exists():
            t = json.loads(chemin.read_text(encoding="utf-8"))["total"]
            ax.plot(t["fa_par_24h_moyenne"], t["sensibilite_moyenne"] * 100, "D", ms=8,
                    mfc="white", mec=COULEURS[m], mew=2)
    ax.plot([], [], "D", ms=7, mfc="white", mec=ENCRE_2, mew=1.5, label="scénario chronologique")
    ax.axvline(1, color=ENCRE_2, lw=0.8, ls=":")
    ax.text(1.05, 8, "cible :\n1 fausse alarme / 24 h", color=ENCRE_2, fontsize=8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 102)
    ax.set_xlabel("fausses alarmes par 24 h (moyenne des 6 patients)")
    ax.set_ylabel("crises détectées (%, moyenne)")
    ax.set_title("Compromis détection / fausses alarmes\n"
                 "courbe : tous les réglages (optimiste) · point : réglage choisi sur les autres patients\n"
                 "rond : validation croisée · losange : scénario chronologique",
                 fontsize=9, loc="left", color=ENCRE)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(DOCS / "compromis_detection_fa.png", dpi=150)


def exemple(nom="chb01_03", canaux=("F7-T7", "T7-P7", "F8-T8", "T8-P8", "FZ-CZ")):
    with np.load(RACINE / "app" / "demo_data" / f"{nom}.npz") as z:
        d = {k: z[k] for k in z.files}
    fs, eeg = int(d["fs"]), d["eeg"].astype(np.float32) * float(d["echelle_uv"])
    reglages = json.loads(str(d["reglages"]))
    c0, c1 = d["crises"][0]
    a, b = c0 - 60, c1 + 60
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    t = np.arange(eeg.shape[1]) / fs
    sel = (t >= a) & (t < b)
    noms = [str(c) for c in d["canaux"]]
    for k, c in enumerate(canaux):
        ax1.plot(t[sel], eeg[noms.index(c), sel] - k * 300, lw=0.5, color="#1f3b57")
    ax1.set_yticks(-np.arange(len(canaux)) * 300, canaux)
    ax1.spines["left"].set_visible(False)
    for ax in (ax1, ax2):
        ax.axvspan(c0, c1, color="#e34948", alpha=0.12, lw=0)
    ax1.set_title(f"{nom} : crise annotée de {c1 - c0:.0f} s (bande rouge), 5 des 18 dérivations",
                  fontsize=9, loc="left")
    tt = d["t"]
    for rang, m in enumerate(["classique", "cnn_int8"]):
        r = reglages[m]
        p = d[f"proba_{m}"].astype(np.float32)
        s = (tt >= a) & (tt < b)
        ax2.plot(tt[s], lisser(p, r["lissage"])[s], color=COULEURS[m], lw=1.6,
                 label=f"{NOMS[m]} (seuil {r['seuil']:g})")
        for al in alarmes(p, tt, r["seuil"], r["lissage"]):
            if a <= al < b:
                ax2.axvline(al, color=COULEURS[m], lw=1, ls="--")
                ax2.annotate(f"alarme {NOMS[m].split(',')[0].lower()} +{al - c0:.0f} s", (al, 0.55 - 0.2 * rang),
                             color=COULEURS[m], fontsize=8, xytext=(4, 0), textcoords="offset points")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("probabilité lissée")
    ax2.set_xlabel("temps dans l'extrait (s)")
    ax2.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(DOCS / "exemple_detection.png", dpi=150)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    courbe_compromis()
    exemple()
    print("figures écrites dans", DOCS)

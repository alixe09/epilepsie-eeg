"""Démo : un dispositif portable surveille l'EEG et déclenche une alarme en cas de crise.

Rejoue des extraits d'enregistrements CHB-MIT jamais vus par les modèles
(probabilités hors-pli), avec le réglage d'alarme choisi sur les autres patients.

Lancer : streamlit run app/streamlit_app.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent / "src"))
from evaluation import TOLERANCE_APRES, TOLERANCE_AVANT, alarmes, lisser  # noqa: E402

EXEMPLES = {
    "chb01_03": "Patiente de 11 ans · crise de 40 s (cas typique)",
    "chb05_13": "Patiente de 7 ans · crise de près de 2 min",
    "chb24_04": "Âge et sexe non renseignés · 3 crises en 12 min",
    "chb16_17": "Patiente de 7 ans · crises très brèves (6 et 8 s) : le cas difficile",
}
METHODES = {"cnn_int8": "Réseau de neurones (CNN int8, 16 k paramètres)",
            "classique": "Méthode classique (énergie par bande + gradient boosting)"}
FENETRE_AFFICHEE = 20  # s d'EEG visibles
PAS_LECTURE = 2  # s d'enregistrement par image

st.set_page_config(page_title="Détection de crises EEG", page_icon="🧠", layout="wide")


@st.cache_data
def charger(nom):
    with np.load(APP / "demo_data" / f"{nom}.npz") as z:
        d = {k: z[k] for k in z.files}
    d["eeg"] = d["eeg"].astype(np.float32) * float(d["echelle_uv"])
    d["reglages"] = json.loads(str(d["reglages"]))
    return d


def statut(t_alarmes, crises, maintenant):
    """Texte d'état du dispositif à l'instant courant."""
    recentes = t_alarmes[(t_alarmes <= maintenant) & (t_alarmes > maintenant - 60)]
    if len(recentes):
        a = recentes[-1]
        vraie = any(d - TOLERANCE_AVANT <= a <= f + TOLERANCE_APRES for d, f in crises)
        return ("error", f"🔔 **ALARME** déclenchée à {a:.0f} s — "
                         + ("crise confirmée par l'annotation" if vraie else "**fausse alarme**"))
    en_cours = [c for c in crises if c[0] <= maintenant <= c[1]]
    if en_cours:
        return "warning", f"⚠️ Crise en cours depuis {maintenant - en_cours[0][0]:.0f} s (annotation neurologue) — pas encore d'alarme"
    return "success", "✅ Surveillance — aucune alarme"


def trace_eeg(d, maintenant, t_alarmes):
    fs = int(d["fs"])
    fin = max(maintenant, FENETRE_AFFICHEE)
    debut = fin - FENETRE_AFFICHEE
    seg = d["eeg"][:, int(debut * fs):int(fin * fs)]
    t = debut + np.arange(seg.shape[1]) / fs
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ecart = 250  # µV entre deux traces
    for i, nom in enumerate(d["canaux"]):
        ax.plot(t, seg[i] - i * ecart, lw=0.6, color="#1f3b57")
    ax.set_yticks(-np.arange(len(d["canaux"])) * ecart, [str(c) for c in d["canaux"]], fontsize=7)
    for c0, c1 in d["crises"]:
        if c1 > debut and c0 < fin:
            ax.axvspan(max(c0, debut), min(c1, fin), color="#e4572e", alpha=0.15, lw=0)
    for a in t_alarmes[(t_alarmes >= debut) & (t_alarmes <= fin)]:
        ax.axvline(a, color="#e4572e", lw=2)
    ax.set_xlim(debut, fin)
    ax.set_ylim(-len(d["canaux"]) * ecart, ecart)
    ax.set_xlabel("temps dans l'extrait (s)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"EEG des {FENETRE_AFFICHEE} dernières secondes (18 dérivations, filtré 0,5–40 Hz)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def trace_proba(d, proba, seuil, lissage, t_alarmes, maintenant):
    t = d["t"]
    vu = t <= maintenant
    fig, ax = plt.subplots(figsize=(11, 2.4))
    for c0, c1 in d["crises"]:
        ax.axvspan(c0, c1, color="#e4572e", alpha=0.15, lw=0, label="crise annotée")
    ax.plot(t[vu], proba[vu], lw=0.6, color="#9aa7b4", label="probabilité par seconde")
    ax.plot(t[vu], lisser(proba, lissage)[vu], lw=1.6, color="#1f3b57",
            label=f"moyenne des {lissage} dernières s")
    ax.axhline(seuil, ls="--", color="#e4572e", lw=1, label=f"seuil d'alarme ({seuil:g})")
    for a in t_alarmes[t_alarmes <= maintenant]:
        ax.axvline(a, color="#e4572e", lw=2)
    ax.axvline(maintenant, color="black", lw=0.8)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("temps dans l'extrait (s)")
    ax.spines[["top", "right"]].set_visible(False)
    poignees, etiquettes = ax.get_legend_handles_labels()
    uniques = dict(zip(etiquettes, poignees))
    ax.legend(uniques.values(), uniques.keys(), fontsize=7, loc="upper left", ncol=4, frameon=False)
    fig.tight_layout()
    return fig


st.title("🧠 Détection de crises d'épilepsie sur EEG portable")
st.caption("⚠️ Projet pédagogique — ce n'est **pas** un dispositif médical. "
           "Données : CHB-MIT Scalp EEG Database (PhysioNet, ODC-By), EEG pédiatrique.")

# lien direct : ?enregistrement=chb16_17&methode=classique&t=1700
parametres = st.query_params
nom_url = parametres.get("enregistrement")
methode_url = parametres.get("methode")

with st.sidebar:
    nom = st.selectbox("Enregistrement", list(EXEMPLES), format_func=lambda n: f"{n} — {EXEMPLES[n]}",
                       index=list(EXEMPLES).index(nom_url) if nom_url in EXEMPLES else 0)
    methode = st.radio("Algorithme", list(METHODES), format_func=METHODES.get,
                       index=list(METHODES).index(methode_url) if methode_url in METHODES else 0)
    d = charger(nom)
    r = d["reglages"][methode]
    st.markdown(f"**Réglage d'usine** (choisi sur les 5 autres patients) : seuil {r['seuil']:g}, "
                f"moyenne sur {r['lissage']} s, pas de nouvelle alarme pendant 60 s.")
    if st.toggle("Modifier le réglage"):
        r = {"seuil": st.slider("Seuil", 0.1, 0.999, float(r["seuil"]), 0.005),
             "lissage": st.select_slider("Moyenne sur (s)", [1, 3, 5, 10], r["lissage"])}
    st.divider()
    st.markdown(
        "**Comment lire la démo**\n\n"
        "- Le dispositif calcule chaque seconde une probabilité de crise sur les 4 dernières secondes d'EEG.\n"
        "- Il déclenche une alarme quand la moyenne de ces probabilités dépasse le seuil.\n"
        "- Les bandes rouges sont les crises annotées par les neurologues ; "
        "les traits rouges sont les alarmes.\n"
        "- Les modèles n'ont **jamais vu** cet enregistrement (validation croisée).")

proba = d[f"proba_{methode}"].astype(np.float32)
t_alarmes = alarmes(proba, d["t"], r["seuil"], r["lissage"])
duree = float(d["t"][-1])

if "t" not in st.session_state or st.session_state.get("nom") != nom:
    t0 = float(max(FENETRE_AFFICHEE, d["crises"][0][0] - 40)) if len(d["crises"]) else 60.0
    if "t" in parametres and nom == nom_url and "nom" not in st.session_state:
        try:
            t0 = float(np.clip(float(parametres["t"]), FENETRE_AFFICHEE, duree))
        except ValueError:
            pass
    st.session_state.update(t=t0, nom=nom, lecture=False)

if st.session_state.lecture:  # avance avant de créer le curseur (Streamlit l'exige)
    st.session_state.t = min(st.session_state.t + PAS_LECTURE, duree)
    st.session_state.lecture = st.session_state.t < duree

c1, c2 = st.columns([1, 6])
with c1:
    if st.button("⏸ Pause" if st.session_state.lecture else "▶ Lecture", use_container_width=True):
        st.session_state.lecture = not st.session_state.lecture
with c2:
    st.slider("Instant (s)", float(FENETRE_AFFICHEE), duree, key="t", step=1.0, label_visibility="collapsed")
maintenant = st.session_state.t

niveau, texte = statut(t_alarmes, d["crises"], maintenant)
getattr(st, niveau)(texte)
st.pyplot(trace_eeg(d, maintenant, t_alarmes), clear_figure=True)
st.pyplot(trace_proba(d, proba, r["seuil"], r["lissage"], t_alarmes, maintenant), clear_figure=True)

# bilan de l'extrait complet
detectees, delais = 0, []
for c0, c1_ in d["crises"]:
    ok = t_alarmes[(t_alarmes >= c0 - TOLERANCE_AVANT) & (t_alarmes <= c1_ + TOLERANCE_APRES)]
    if len(ok):
        detectees += 1
        delais.append(ok[0] - c0)
fausses = sum(not any(c0 - TOLERANCE_AVANT <= a <= c1_ + TOLERANCE_APRES for c0, c1_ in d["crises"])
              for a in t_alarmes)
m1, m2, m3 = st.columns(3)
m1.metric("Crises détectées (extrait entier)", f"{detectees} / {len(d['crises'])}")
m2.metric("Délai après le début", " · ".join(f"{x:+.0f} s" for x in delais) or "—")
m3.metric("Fausses alarmes", fausses)

with st.expander("Résultats sur les 166 h d'enregistrement (6 patients, 50 crises)"):
    chemin = APP / "resultats.md"
    st.markdown(chemin.read_text(encoding="utf-8") if chemin.exists() else "—")

if st.session_state.lecture:
    time.sleep(0.25)
    st.rerun()

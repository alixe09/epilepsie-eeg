"""Relance les tests et régénère la matrice de traçabilité exigences → vérification → résultat.

Usage : python src/verification_report.py
"""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

from donnees import RACINE

RES = RACINE / "resultats"
SORTIE = RACINE / "docs" / "dispositif-medical" / "03-matrice-tracabilite.md"
# scénario principal (chronologique, comme l'usage prévu) puis validation croisée
SCENARIOS = {
    "Chronologique (calibration sur 3 crises, puis surveillance)":
        {"classique_chrono": "classique", "cnn_chrono": "CNN"},
    "Validation croisée (50 crises)": {"classique": "classique", "cnn_int8": "CNN int8"},
}
METHODES = {m: n for sc in SCENARIOS.values() for m, n in sc.items()}


def tests():
    xml = RES / "pytest.xml"
    subprocess.run([sys.executable, "-m", "pytest", str(RACINE / "tests"), "-q", f"--junitxml={xml}"],
                   cwd=RACINE, capture_output=True)
    etat = {}
    for cas in ET.parse(xml).getroot().iter("testcase"):
        nom = f"{cas.get('classname').split('.')[-1]}::{cas.get('name')}"
        etat[nom] = "échec" if cas.find("failure") is not None or cas.find("error") is not None else (
            "ignoré" if cas.find("skipped") is not None else "ok")
    xml.unlink()
    return etat


def statut(ok):
    return "✅ conforme" if ok else "❌ non conforme"


def main():
    etat = tests()
    r = {m: json.loads((RES / f"resume_{m}.json").read_text(encoding="utf-8")) for m in METHODES}
    emb = json.loads((RES / "embarque.json").read_text(encoding="utf-8"))
    rob = json.loads((RES / "robustesse.json").read_text(encoding="utf-8"))

    def par_patient(cle, fmt, cond):
        cellules, conforme = [], True
        for scenario, methodes in SCENARIOS.items():
            cellules.append(f"*{scenario.split(' (')[0]}*")
            for m, nom in methodes.items():
                ko = [l["patient"] for l in r[m]["patients"] if not cond(l[cle])]
                conforme &= not ko
                cellules.append(f"{nom} : {fmt(r[m])}" + (f" — **hors cible : {', '.join(ko)}**" if ko else ""))
        return "<br>".join(cellules), conforme

    lignes = []
    txt, ok = par_patient(
        "sensibilite", lambda x: f"{x['total']['detectees']}/{x['total']['crises']} crises",
        lambda v: v >= 0.9)
    lignes.append(("EX-01", "Sensibilité ≥ 90 % par patient", "Scénario chronologique (modèle entraîné "
                   "sur le passé seulement) et validation croisée par enregistrement ; réglage choisi "
                   "sur les autres patients", txt, statut(ok)))
    txt, ok = par_patient(
        "fa_par_24h", lambda x: f"{x['total']['fa_par_24h_globale']:.2f} FA/24 h au total",
        lambda v: v <= 1.0)
    lignes.append(("EX-02", "≤ 1 fausse alarme / 24 h par patient", "Idem", txt, statut(ok)))
    txt, ok = par_patient(
        "delai_median", lambda x: f"médiane {x['total']['delai_median']:.0f} s",
        lambda v: v != v or v <= 30)  # NaN (aucune détection) : couvert par EX-01
    lignes.append(("EX-03", "Délai médian ≤ 30 s", "Idem (crises détectées)", txt, statut(ok)))
    t = [k for k in etat if "causal" in k or "fenetres_et" in k]
    lignes.append(("EX-04", "Traitement causal", "Tests : " + ", ".join(f"`{k}`" for k in t),
                   ", ".join(etat[k] for k in t), statut(all(etat[k] == "ok" for k in t))))
    c = emb["cnn"]
    lignes.append(("EX-05", "Modèle ≤ 100 Ko, calcul compatible microcontrôleur",
                   "Mesure des fichiers .tflite, décompte des multiplications ; test `test_taille_compatible_microcontroleur`",
                   f"{c['taille_int8_ko'][1]:.0f} Ko, {c['macs_par_decision'] / 1e6:.2f} M MAC par décision (1/s)",
                   statut(c["taille_int8_ko"][1] <= 100)))
    a = r["cnn_int8"]["total"]
    f = json.loads((RES / "resume_cnn.json").read_text(encoding="utf-8"))["total"]
    accord = min(v["accord_decision_0_5"] for v in c["accord_float_int8"].values())
    lignes.append(("EX-06", "Quantification int8 sans perte clinique",
                   "Métriques cliniques float vs int8 sur les 166 h ; test `test_int8_proche_du_float…`",
                   f"float {f['detectees']}/{f['crises']}, {f['fa_par_24h_globale']:.2f} FA/24 h · "
                   f"int8 {a['detectees']}/{a['crises']}, {a['fa_par_24h_globale']:.2f} FA/24 h · "
                   f"accord des décisions ≥ {accord:.1%}",
                   statut(a["detectees"] >= f["detectees"] and a["fa_par_24h_globale"] <= f["fa_par_24h_globale"] + 0.1)))
    t = [k for k in etat if k.startswith("test_securite")]
    mvt = rob["artefact de mouvement 800 µV, 1,5 Hz"]
    lignes.append(("EX-07", "Défaut capteur → pas de fausse alarme silencieuse",
                   f"{len(t)} tests `test_securite` ; défauts simulés sur EEG réel (`src/robustesse.py`)",
                   f"tests : {sum(etat[k] == 'ok' for k in t)}/{len(t)} ok ; déconnexion et saturation bloquées ; "
                   f"**artefact de mouvement non détecté : {mvt['cnn_crise_apres_controle']:.0%} des fenêtres "
                   f"classées crise (CNN), {mvt['classique_crise_apres_controle']:.0%} (classique)**",
                   "⚠️ partiel"))

    md = ["# 3. Matrice de traçabilité", "",
          "> **Générée automatiquement** par `python src/verification_report.py` (relance les tests "
          "et relit les résultats dans `resultats/`). Ne pas modifier à la main.", "",
          "| ID | Exigence | Méthode de vérification | Résultat | Statut |", "|---|---|---|---|---|"]
    md += [f"| {' | '.join(l)} |" for l in lignes]
    n_ok = sum(v == "ok" for v in etat.values())
    md += ["", f"Tests automatisés : **{n_ok} réussis**, "
               f"{sum(v == 'échec' for v in etat.values())} en échec, "
               f"{sum(v == 'ignoré' for v in etat.values())} ignorés (sur {len(etat)}).", ""]
    SORTIE.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

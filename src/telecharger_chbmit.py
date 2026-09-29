"""Téléchargement léger de CHB-MIT (PhysioNet, licence ODC-By).

Pour chaque enregistrement : télécharge l'EDF brut (~40 Mo), garde les 18
dérivations bipolaires standard, rééchantillonne 256 -> 128 Hz (filtre
anti-repliement inclus), stocke en int16 (0,1 µV) dans un .npz compressé,
puis supprime l'EDF. Relançable : les fichiers déjà convertis sont sautés.

Usage : python src/telecharger_chbmit.py [chb01 chb05 ...]
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from scipy.signal import resample_poly

from edf import lire_edf

# Miroir S3 officiel de PhysioNet : ~13x plus rapide que physionet.org/files depuis ce poste
BASE_URL = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
PATIENTS = ["chb01", "chb05", "chb08", "chb16", "chb23", "chb24"]
CANAUX = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
FS_CIBLE = 128
ECHELLE_UV = 0.1  # 1 unité int16 = 0,1 µV -> plage ±3276 µV
N_WORKERS = 3

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_TMP = RACINE / "data" / "tmp"
DOSSIER_SORTIE = RACINE / "data" / "chbmit_128hz"
JOURNAL = RACINE / "logs" / "telechargement.log"

session = requests.Session()


def log(msg):
    ligne = f"{time.strftime('%H:%M:%S')} {msg}"
    print(ligne, flush=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(ligne + "\n")


def get(url, **kw):
    for essai in range(5):
        try:
            r = session.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if essai == 4:
                raise
            log(f"  nouvel essai ({e}) : {url}")
            time.sleep(5 * (essai + 1))


def lire_resume(patient):
    """Renvoie {nom_fichier: [(debut_s, fin_s), ...]} depuis chbXX-summary.txt."""
    texte = get(f"{BASE_URL}/{patient}/{patient}-summary.txt").text
    crises, courant = {}, None
    debuts = []
    for ligne in texte.splitlines():
        if m := re.match(r"File Name:\s*(\S+\.edf)", ligne):
            courant, debuts = m.group(1), []
            crises[courant] = []
        elif courant and (m := re.search(r"Seizure.*Start Time:\s*(\d+)", ligne)):
            debuts.append(int(m.group(1)))
        elif courant and (m := re.search(r"Seizure.*End Time:\s*(\d+)", ligne)):
            crises[courant].append((debuts.pop(0), int(m.group(1))))
    return crises


def telecharger(url, dest):
    with get(url, stream=True) as r, open(dest, "wb") as f:
        for bloc in r.iter_content(1 << 20):
            f.write(bloc)


def choisir_canaux(noms):
    """Associe chaque canal cible à son index (T8-P8 est en double : on prend le premier)."""
    index = []
    for cible in CANAUX:
        trouve = [i for i, n in enumerate(noms) if n.upper() == cible]
        if not trouve:
            return None
        index.append(trouve[0])
    return index


def traiter(patient, fichier, crises):
    sortie = DOSSIER_SORTIE / patient / fichier.replace(".edf", ".npz")
    if sortie.exists():
        return "déjà fait"
    edf = DOSSIER_TMP / fichier
    try:
        telecharger(f"{BASE_URL}/{patient}/{fichier}", edf)
        x, fs, noms = lire_edf(edf)
        idx = choisir_canaux(noms)
        if idx is None:
            return f"IGNORÉ (canaux manquants : {noms})"
        fs = int(round(fs))
        x = resample_poly(x[idx], FS_CIBLE, fs, axis=1)  # déjà en µV
        x = np.clip(np.round(x / ECHELLE_UV), -32768, 32767).astype(np.int16)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        tmp = sortie.with_suffix(".part.npz")
        np.savez_compressed(
            tmp, eeg=x, fs=FS_CIBLE, echelle_uv=ECHELLE_UV,
            canaux=np.array(CANAUX), crises=np.array(crises, dtype=np.int32).reshape(-1, 2),
        )
        tmp.replace(sortie)
        return f"{x.shape[1] / FS_CIBLE / 3600:.2f} h, {len(crises)} crise(s), {sortie.stat().st_size / 1e6:.1f} Mo"
    finally:
        edf.unlink(missing_ok=True)


def main(patients):
    for d in (DOSSIER_TMP, DOSSIER_SORTIE, JOURNAL.parent):
        d.mkdir(parents=True, exist_ok=True)
    for vieux in DOSSIER_TMP.glob("*.edf"):  # restes d'une exécution interrompue
        vieux.unlink()

    records = get(f"{BASE_URL}/RECORDS").text.split()  # "chb01/chb01_01.edf", ...
    for patient in patients:
        crises = lire_resume(patient)
        fichiers = [r.split("/")[-1] for r in records if r.startswith(patient + "/")]
        (DOSSIER_SORTIE / patient).mkdir(exist_ok=True)
        (DOSSIER_SORTIE / patient / "crises.json").write_text(
            json.dumps(crises, indent=1), encoding="utf-8")
        log(f"== {patient} : {len(fichiers)} fichiers, "
            f"{sum(len(v) for v in crises.values())} crises")
        with ThreadPoolExecutor(N_WORKERS) as pool:
            taches = {pool.submit(traiter, patient, f, crises.get(f, [])): f for f in fichiers}
            for t in as_completed(taches):
                try:
                    log(f"  {taches[t]} : {t.result()}")
                except Exception as e:
                    log(f"  {taches[t]} : ERREUR {e!r}")
    log("Terminé.")


if __name__ == "__main__":
    main(sys.argv[1:] or PATIENTS)

# 3. Matrice de traçabilité

> **Générée automatiquement** par `python src/verification_report.py` (relance les tests et relit les résultats dans `resultats/`). Ne pas modifier à la main.

| ID | Exigence | Méthode de vérification | Résultat | Statut |
|---|---|---|---|---|
| EX-01 | Sensibilité ≥ 90 % par patient | Scénario chronologique (modèle entraîné sur le passé seulement) et validation croisée par enregistrement ; réglage choisi sur les autres patients | *Chronologique*<br>classique : 22/31 crises — **hors cible : chb16, chb23**<br>CNN : 16/31 crises — **hors cible : chb16, chb23, chb24**<br>*Validation croisée*<br>classique : 35/50 crises — **hors cible : chb01, chb16, chb24**<br>CNN int8 : 28/50 crises — **hors cible : chb16, chb23, chb24** | ❌ non conforme |
| EX-02 | ≤ 1 fausse alarme / 24 h par patient | Idem | *Chronologique*<br>classique : 5.85 FA/24 h au total — **hors cible : chb24**<br>CNN : 2.47 FA/24 h au total — **hors cible : chb08, chb24**<br>*Validation croisée*<br>classique : 1.01 FA/24 h au total — **hors cible : chb24**<br>CNN int8 : 1.01 FA/24 h au total — **hors cible : chb08, chb23, chb24** | ❌ non conforme |
| EX-03 | Délai médian ≤ 30 s | Idem (crises détectées) | *Chronologique*<br>classique : médiane 13 s<br>CNN : médiane 10 s<br>*Validation croisée*<br>classique : médiane 13 s<br>CNN int8 : médiane 12 s | ✅ conforme |
| EX-04 | Traitement causal | Tests : `test_metriques_cliniques::test_lissage_causal`, `test_traitement_signal::test_filtre_causal_ne_voit_pas_le_futur`, `test_traitement_signal::test_fenetres_et_instants` | ok, ok, ok | ✅ conforme |
| EX-05 | Modèle ≤ 100 Ko, calcul compatible microcontrôleur | Mesure des fichiers .tflite, décompte des multiplications ; test `test_taille_compatible_microcontroleur` | 29 Ko, 1.37 M MAC par décision (1/s) | ✅ conforme |
| EX-06 | Quantification int8 sans perte clinique | Métriques cliniques float vs int8 sur les 166 h ; test `test_int8_proche_du_float…` | float 27/50, 1.01 FA/24 h · int8 28/50, 1.01 FA/24 h · accord des décisions ≥ 99.8% | ✅ conforme |
| EX-07 | Défaut capteur → pas de fausse alarme silencieuse | 6 tests `test_securite` ; défauts simulés sur EEG réel (`src/robustesse.py`) | tests : 6/6 ok ; déconnexion et saturation bloquées ; **artefact de mouvement non détecté : 100% des fenêtres classées crise (CNN), 71% (classique)** | ⚠️ partiel |

Tests automatisés : **23 réussis**, 0 en échec, 0 ignorés (sur 23).

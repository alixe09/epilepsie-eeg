# 2. Analyse des risques (démarche inspirée d'ISO 14971)

> Exercice pédagogique, limité au **logiciel de détection**. Les fréquences sont
> estimées à partir des mesures du projet (166 h d'EEG hospitalier CHB-MIT,
> 6 enfants, 50 crises), pas d'un usage réel à domicile.

## Échelles

**Gravité** du dommage :
1 = négligeable (gêne) · 2 = mineure (réveil inutile, anxiété, perte de confiance) ·
3 = sérieuse (blessure pendant une crise non surveillée, crise prolongée non traitée) ·
4 = critique (état de mal épileptique non pris en charge, mort subite inattendue — SUDEP).

**Fréquence** estimée : fréquente (plus d'une fois par jour) · occasionnelle (plus
d'une fois par mois) · rare (moins) · non estimée.

**Acceptabilité** : ❌ inacceptable · ⚠️ à réduire autant que possible · ✅ acceptable.

## Tableau des risques

| ID | Danger → situation dangereuse → dommage | Gravité | Avant mesures | Mesures de réduction (vérification) | Après mesures | Risque résiduel |
|---|---|---|---|---|---|---|
| **R1** | Crise non détectée → l'aidant n'est pas prévenu → blessure, crise prolongée non traitée | 3–4 | — | Modèle par patient ; lissage et seuil réglés sur d'autres patients (EX-01) ; journal relu par le neurologue | Validation croisée : **classique 15 crises manquées sur 50 (30 %)**, **CNN int8 22 sur 50 (44 %)**. Scénario chronologique (usage prévu) : **classique 9 sur 31 (29 %)**, **CNN 15 sur 31 (48 %)**. Toujours toutes les crises de chb16 ; aussi chb23 en chronologique | ❌ |
| **R2** | **Crise brève (< 15 s) jamais détectée** : le lissage sur 10 s, choisi sur des patients aux crises longues, l'empêche de franchir le seuil (constaté) | 3 | chb16 : 0 / 10 | Proposé : réglage du lissage par patient lors de la calibration ; avec un réglage individuel, la méthode classique en détecte 3 / 10 sans fausse alarme | Non corrigé | ❌ pour ce type de patient |
| **R3** | Fausse alarme → réveil de la famille ; répétée → fatigue d'alarme, appareil désactivé → retour à R1 | 2 | Seuil bas : plus de 300 FA / 24 h (CNN, chb08) | M1 lissage + seuil ; M2 période réfractaire de 60 s ; réglage sous contrainte ≤ 1 FA / 24 h (EX-02) | Validation croisée : **1,0 FA / 24 h au total**, mais 3,4 à 6,8 chez chb24. **Scénario chronologique : 2,5 (CNN) à 5,9 (classique) au total, 13 à 34 chez chb24** : les modèles appris sur le début de l'enregistrement dérivent. **Proposé** : recalibration régulière intégrant les fausses alarmes signalées par l'aidant | ⚠️ (❌ chb24) |
| **R4** | **Artefact de mouvement pris pour une crise** : un balancement de 800 µV à 1,5 Hz ajouté à un EEG réel est classé « crise » dans 100 % des fenêtres par le CNN et 71 % par la méthode classique (constaté, `src/robustesse.py`) | 2 (FA) | Chaque mouvement rythmique ample | Le contrôle de qualité (R5) ne le détecte pas (amplitude plausible). **Proposé** : accéléromètre dans le boîtier (comme les détecteurs portés du commerce) ; apprentissage avec des artefacts | Non corrigé | ❌ en ambulatoire (CHB-MIT : patients alités) |
| **R5** | Électrode décollée, appareil déconnecté, saturation → décision sur un signal faux | 3 | Non mesuré | M3 contrôle de qualité (`src/qualite_signal.py`) : > 4 dérivations plates ou saturées → pas de décision « crise », défaut signalé à l'aidant (EX-07) | Déconnexion et saturation bloquées (6 tests). Sur les 166 h réelles : ≤ 0,016 % de fenêtres bloquées, **aucune fenêtre de crise** | ✅ défauts testés · ⚠️ 1–4 électrodes décollées : analysé quand même, non évalué |
| **R6** | Seuil de saturation trop bas → une vraie crise de forte amplitude est bloquée | 3 | Seuil initial à 1 500 µV : des crises atteignent 2 400 µV (chb24, mesuré) | Seuil relevé à 3 000 µV, proche de la limite de mesure ; test `test_crise_de_forte_amplitude_non_bloquee` | Marge de 25 % seulement | ⚠️ |
| **R7** | Portage embarqué incorrect (quantification int8) | 3 | **Constaté deux fois** : (a) sortie sigmoïde quantifiée plafonnée à 0,996, les seuils utiles (0,999 à 0,999999) devenaient inatteignables ; (b) calibration sur des fenêtres de crise seulement : toute probabilité basse écrasée à 0,5 | Logit quantifié, sigmoïde après ; calibration sur fenêtres mélangées ; métriques cliniques recalculées avec le modèle int8 (EX-06) | Accord des décisions ≥ 99,8 % ; int8 28 / 50, float 27 / 50, même taux de fausses alarmes | ✅ · ⚠️ seuils extrêmes sensibles à toute dérive de calibration |
| **R8** | Modèle dégénéré après entraînement → alarme permanente ou jamais d'alarme | 3 | **Constaté** : 1 modèle sur 4 (chb01) prédisait « crise » en permanence (BatchNormalization + lots équilibrés) | Architecture sans BatchNormalization ; contrôle automatique après chaque entraînement (probabilité moyenne sur l'entraînement, journalisée) | 23 modèles sur 23 conformes | ✅ · **proposé** : bloquer la mise en service si le contrôle échoue |
| **R9** | Patient non calibré ou trop peu de crises pour calibrer | 3 | Non évalué (tous les modèles sont par patient) | Usage prévu : calibration sur ≥ 3 crises | Non vérifié | ⚠️ hors périmètre évalué |
| **R10** | Données EEG = données de santé → atteinte à la vie privée | — | — | Données publiques pseudonymisées ; la démo n'enregistre rien | — | ✅ |

## Ce que l'analyse a changé dans le projet

Trois défauts ont été **trouvés par la démarche** et corrigés avant les résultats
finaux : le modèle dégénéré (R8), les deux erreurs de quantification (R7) et le
seuil de saturation trop bas (R6). Un quatrième problème concernait l'évaluation
elle-même : la grille de seuils, arrêtée à 0,999, rendait la contrainte de fausses
alarmes inatteignable et masquait l'écart entre le CNN (28 / 50 une fois corrigé) et la
méthode classique (35 / 50). Aucun n'aurait été visible dans une simple
précision par fenêtre : il a fallu regarder les fausses alarmes par
enregistrement, les probabilités int8 elles-mêmes et l'amplitude des vraies crises.

## Conclusion bénéfice / risque (exercice)

- Les défauts **techniques** (capteur débranché, saturation, portage int8,
  modèle dégénéré, causalité) sont maîtrisés et vérifiés par des tests.
- Les risques liés à la **performance de détection** restent inacceptables pour
  une mise sur le marché : 29 à 48 % de crises manquées selon la méthode et le
  scénario, un patient entier non détecté (crises brèves), un patient à 13–34 fausses
  alarmes par jour dans le scénario chronologique (dérive).
- Le risque **artefact de mouvement** (R4) n'est pas évaluable sur CHB-MIT
  (patients hospitalisés) et serait probablement dominant à domicile.
- Pistes, non réalisées : lissage réglé par patient à la calibration (R2),
  accéléromètre (R4), évaluation sur des données ambulatoires portées (par
  exemple des bases avec EEG derrière l'oreille).

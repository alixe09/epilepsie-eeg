**Scénario dispositif (chronologique)** : calibration sur les 3 premières crises, puis
surveillance avec un modèle entraîné sur le passé seulement. Classique **22 / 31** crises,
5,9 fausses alarmes / 24 h ; CNN 16 / 31, 2,5 / 24 h (107 h testées, surtout chb24).

**Validation croisée** (probabilités utilisées par cette démo) : chaque heure testée par un modèle qui ne l'a pas vue,
réglage d'alarme choisi sur les 5 autres patients. Tolérances SzCORE : alarme entre 30 s avant
le début et 60 s après la fin de la crise.

| Patient | Classique : crises | FA / 24 h | CNN int8 : crises | FA / 24 h |
|---|---|---|---|---|
| chb01 (40,6 h) | 6 / 7 | 0,0 | **7 / 7** | 0,0 |
| chb05 (39,0 h) | 5 / 5 | 0,0 | 5 / 5 | 0,0 |
| chb08 (20,0 h) | 5 / 5 | **0,0** | 5 / 5 | 2,4 |
| chb16 (19,0 h) | 0 / 10 | 0,0 | 0 / 10 | 0,0 |
| chb23 (26,6 h) | **7 / 7** | **0,9** | 4 / 7 | 1,8 |
| chb24 (21,3 h) | **12 / 16** | 6,8 | 7 / 16 | **3,4** |
| **Total (166 h)** | **35 / 50** | **1,0** | 28 / 50 | **1,0** |

Délai médian : 13 s (classique), 12 s (CNN). Les crises de chb16 durent 6 à 14 s :
trop brèves pour le lissage sur 10 s choisi sur les autres patients.

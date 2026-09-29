# Détection de crises d'épilepsie sur EEG pour un dispositif portable

Détecter automatiquement une crise d'épilepsie sur l'EEG et **alerter un aidant**
en quelques secondes, avec les contraintes d'un appareil porté : décision en
temps réel sans regarder le futur, modèle de quelques dizaines de Ko, et surtout
**très peu de fausses alarmes** (sinon la famille éteint l'appareil).

Quatrième volet d'un portfolio en IA appliquée à la santé, après
[stroke-risk-predictor](https://github.com/alixe09/stroke-risk-predictor) (données tabulaires),
[radio-thoracique](https://github.com/alixe09/radio-thoracique) (images, CNN, Grad-CAM) et
[emg-prothese](https://github.com/alixe09/emg-prothese) (EMG, CNN int8 embarqué, dossier
« dispositif médical ») : ici, **détection d'événements rares** dans un signal continu,
avec la même démarche (pipeline complet, évaluation honnête, métriques orientées patient,
démo Streamlit, dossier dispositif médical).

🔗 **Démo en ligne** : [Ouvrir l'application](https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/)
· crise détectée : [chb01, crise de 40 s](https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/?enregistrement=chb01_03&t=320)
· échec : [chb16, crise de 6 s manquée](https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/?enregistrement=chb16_17&methode=classique&t=205)

⚠️ **Disclaimer** : projet pédagogique / recherche de stage. Ce n'est **pas** un
dispositif médical.

![Crise de 40 s chez chb01 : alarme 11 s après le début pour le CNN, 16 s pour la méthode classique](docs/exemple_detection.png)

## En bref

- **166 h d'EEG, 6 enfants, 50 crises** (CHB-MIT). Le réglage d'alarme de chaque patient
  est toujours choisi **sur les 5 autres**.
- **Scénario « dispositif », dans l'ordre du temps** (calibration sur les 3 premières
  crises, puis surveillance avec un modèle qui ne connaît que le passé) : **méthode
  classique 22 / 31 crises**, CNN 16 / 31. Les fausses alarmes (5,9 et 2,5 par 24 h)
  viennent presque toutes d'un seul patient (chb24).
- **Validation croisée** (toutes les crises, modèles entraînés aussi sur des heures
  postérieures au test) : méthode classique 35 / 50 à 1,0 fausse alarme par 24 h, CNN
  28 / 50. Délai médian 10 à 13 s selon la méthode et le scénario.
- **La méthode classique détecte plus de crises dans les deux scénarios** ; le CNN
  détecte un peu plus vite. **Mais l'écart n'est pas établi** : il tient à 2 patients sur
  6, et les intervalles de confiance sont très larges (voir « Incertitude »).
- **Le CNN tient dans 29 Ko en int8**, sans perte clinique par rapport au float.
- **Un résultat corrigé en route** : avec une grille de seuils arrêtée à 0,999, le CNN
  semblait à égalité (35 / 50). C'était un effet de bord de la grille (détails plus bas).
- **Échec assumé** : les 10 crises de chb16, très brèves (6–14 s), ne sont jamais détectées.
- **Le dossier dispositif médical a trouvé 3 défauts**, corrigés avant les résultats
  finaux, et 2 risques qui restent ouverts.

## Dataset

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet,
licence ODC-By 1.0) : EEG de scalp d'enfants épileptiques hospitalisés à Boston, 23
dérivations à 256 Hz, crises annotées par des neurologues.

> Shoeb, A. H. (2009). *Application of machine learning to epileptic seizure onset
> detection and treatment.* PhD Thesis, MIT.
> Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*
> 101(23), e215–e220.

**Sous-ensemble** (contrainte : ~9 Go de disque libre, 6 Go de RAM, pas de GPU) :
6 patients choisis pour leur faible volume et leur nombre de crises.

| Patient | Sexe, âge | Durée | Crises | Durée des crises |
|---|---|---|---|---|
| chb01 | F, 11 ans | 40,6 h | 7 | 27–101 s |
| chb05 | F, 7 ans | 39,0 h | 5 | 96–120 s |
| chb08 | M, 3,5 ans | 20,0 h | 5 | 134–264 s |
| chb16 | F, 7 ans | 19,0 h | 10 | **6–14 s** |
| chb23 | F, 6 ans | 26,6 h | 7 | 20–113 s |
| chb24 | non renseigné | 21,3 h | 16 | 16–70 s |

**Stockage compact** ([`src/telecharger_chbmit.py`](src/telecharger_chbmit.py)) : chaque
EDF (42 Mo) est téléchargé, réduit aux 18 dérivations bipolaires communes,
rééchantillonné à 128 Hz, stocké en int16 compressé, puis supprimé. 7,2 Go d'EDF
deviennent **2,0 Go**. Lecteur EDF écrit en numpy ([`src/edf.py`](src/edf.py), vérifié
identique à MNE à 1e-13 µV près) pour éviter une dépendance lourde.

## Méthode

**Prétraitement commun, causal** ([`src/pretraitement.py`](src/pretraitement.py)) :
passe-bande 0,5–40 Hz (Butterworth, filtrage avant uniquement), fenêtres de 4 s,
**une décision par seconde**.

| | Méthode classique | Réseau de neurones |
|---|---|---|
| Entrée | 8 caractéristiques × 18 dérivations : énergie dans 5 bandes (δ θ α β γ), longueur de ligne, paramètres de Hjorth | EEG filtré brut, 4 s × 18 dérivations, normalisé par canal |
| Modèle | Gradient boosting (scikit-learn), classes pondérées | CNN 1D, 4 convolutions, 16 000 paramètres, exporté en **int8** |
| Déséquilibre (crises = 0,3 % du temps) | Pondération des classes | Lots tirés à moitié dans les crises, avec décalage temporel aléatoire |

**Évaluation** ([`src/decoupage.py`](src/decoupage.py), [`src/evaluation.py`](src/evaluation.py)) :

- **Par patient**, validation croisée à 4 plis **par enregistrement d'1 h** : aucune
  fenêtre de test ne chevauche l'entraînement, chaque pli contient au moins une crise,
  chaque heure est testée une fois.
- **Des probabilités aux alarmes** : moyenne des *n* dernières décisions, alarme au
  franchissement du seuil, puis 60 s sans nouvelle alarme.
- **Deux scénarios de découpage** :
  - **chronologique (principal, comme l'usage prévu)** : les enregistrements jusqu'à la
    3ᵉ crise servent à calibrer et ne sont jamais testés ; chaque enregistrement suivant
    est testé par un modèle entraîné **uniquement sur ce qui précède**, ré-entraîné après
    chaque nouvelle crise. 31 crises et 107 h testées ;
  - **validation croisée** : les plis mélangent passé et futur, mais toutes les crises
    (50) et toutes les heures (166 h) sont testées.
- **Réglage d'usine** : le seuil et *n* d'un patient sont choisis sur les 5 autres, avec
  une règle fixée à l'avance : maximiser la sensibilité sous la contrainte de ≤ 1 fausse
  alarme par 24 h en moyenne.
- **Métriques par événement**, tolérances du cadre SzCORE (Dan et al., *Epilepsia*
  2024) : une crise est détectée si une alarme tombe entre 30 s avant son début et 60 s
  après sa fin ; toute autre alarme est fausse.

## Résultats

### Scénario dispositif (chronologique) — résultat principal

Calibration sur les enregistrements jusqu'à la 3ᵉ crise, puis surveillance ; le modèle
est ré-entraîné après chaque nouvelle crise avec tout le passé
([`src/decoupage.py`](src/decoupage.py), `--decoupage chrono`). CNN évalué en float
(l'int8 donne les mêmes décisions à ≥ 99,8 %).

| Patient (heures testées) | Classique : crises | FA / 24 h | CNN : crises | FA / 24 h |
|---|---|---|---|---|
| chb01 (25,6 h) | 4 / 4 | 0,0 | 4 / 4 | 0,0 |
| chb05 (23,0 h) | 2 / 2 | 0,0 | 2 / 2 | 0,0 |
| chb08 (14,0 h) | 2 / 2 | **0,0** | 2 / 2 | 1,7 |
| chb16 (5,0 h) | 0 / 7 | 0,0 | 0 / 7 | 0,0 |
| chb23 (20,9 h) | **2 / 4** | 0,0 | 0 / 4 | 0,0 |
| chb24 (18,3 h) | **12 / 12** | 34,1 | 8 / 12 | **13,1** |
| **Total, 107 h** | **22 / 31** | 5,9 | 16 / 31 | **2,5** |

Délai médian : 13 s (classique), 10 s (CNN).

Lecture :

- **Trois patients restent au niveau de la validation croisée** (chb01, chb05, chb08 :
  toutes les crises, 0 à 1,7 fausse alarme par jour). Apprendre seulement sur le passé
  ne les pénalise pas.
- **chb24 révèle un vrai problème de dérive** : en validation croisée, 3,4 à 6,8 fausses
  alarmes par jour ; ici, 13 à 34. Les modèles appris sur les premières heures se
  trompent davantage sur les heures suivantes. Pour un dispositif, cela plaide pour une
  recalibration régulière, y compris sur les fausses alarmes signalées par les aidants.
- **chb23 perd des crises** (2 / 4 et 0 / 4) : ses 3 crises de calibration ne suffisent
  pas à reconnaître les 4 suivantes, toutes dans le même enregistrement de 4 h.
- **La hiérarchie des méthodes ne change pas** : la méthode classique détecte plus de
  crises (22 contre 16) ; le CNN fait moins de fausses alarmes chez chb24, mais en
  manque 4.
- **Seuls 31 crises et 107 h sont testées** : ce scénario est plus réaliste mais encore
  moins précis statistiquement.

### Validation croisée (toutes les crises)

| Patient | Classique : crises | FA / 24 h | Délai méd. | CNN int8 : crises | FA / 24 h | Délai méd. |
|---|---|---|---|---|---|---|
| chb01 | 6 / 7 | 0,0 | 15 s | **7 / 7** | 0,0 | 9 s |
| chb05 | 5 / 5 | 0,0 | 17 s | 5 / 5 | 0,0 | 12 s |
| chb08 | 5 / 5 | **0,0** | 14 s | 5 / 5 | 2,4 | 20 s |
| chb16 | 0 / 10 | 0,0 | — | 0 / 10 | 0,0 | — |
| chb23 | **7 / 7** | **0,9** | 24 s | 4 / 7 | 1,8 | 25 s |
| chb24 | **12 / 16** | 6,8 | 7 s | 7 / 16 | **3,4** | 12 s |
| **Total, 166 h** | **35 / 50** | **1,0** | 13 s | 28 / 50 | **1,0** | **12 s** |

(CNN en float : 27 / 50, 1,0 FA / 24 h.)

![Compromis entre crises détectées et fausses alarmes](docs/compromis_detection_fa.png)

Lecture :

- **La méthode classique fait mieux avec le réglage d'usine** (35 contre 28 crises, au
  même taux de fausses alarmes). Sur la courbe optimiste (meilleur réglage global), le
  CNN est derrière sous 1 fausse alarme / 24 h (71 % contre 80 % des crises en moyenne)
  et devant au-delà de 2 (92 % contre 85 % à 5 FA / 24 h) : il gagnerait si l'on
  tolérait plus de fausses alarmes. Une partie de l'écart vient aussi du transfert du
  réglage d'un patient à l'autre : le CNN est très confiant (probabilités de 0,99999 et
  plus), et son seuil utile, entre 0,9995 et 0,999999, varie d'un patient à l'autre ; celui
  de la méthode classique (0,94) se transpose mieux. L'écart tient à chb23 et chb24 et
  n'est pas statistiquement établi (voir « Incertitude ») ; il va dans le même sens
  qu'emg-prothese : à données limitées, la méthode classique est au moins aussi bonne, et
  plus facile à expliquer.
- **Le CNN garde deux atouts** : il détecte plus vite (délai médian 12 s contre 13 s,
  5 à 6 s de moins chez chb01 et chb05) et fait moins de fausses alarmes chez chb24.
- **Avec la méthode classique, 4 patients sur 6 sont proches d'un usage réel** (toutes
  ou presque toutes les crises, ≤ 1 fausse alarme par jour). **chb24 en reste loin** : 3
  à 7 fausses alarmes par jour selon la méthode, c'est trop pour une famille.
- **chb16 est entièrement manqué, et c'est instructif** : ses crises durent 6 à 14 s. Le
  réglage choisi sur les autres patients (moyenne sur 10 s, seuil haut) empêche une crise
  aussi brève de franchir le seuil. Le modèle, lui, les voit souvent (méthode classique :
  probabilité ≥ 0,87 sur au moins une fenêtre pendant 6 crises sur 10). Un réglage individuel en récupère 3 sur 10 sans fausse
  alarme : la durée typique des crises d'un patient est un paramètre de calibration.
- **Délai médian de 12–13 s** : l'alarme arrive pendant la crise, pas avant ; ce
  système détecte, il ne prédit pas.
- **Correction en cours de projet** : la grille de seuils s'arrêtait d'abord à 0,999.
  Pour 5 patients sur 6, aucun réglage ne tenait alors ≤ 1 fausse alarme / 24 h sur les
  autres patients, et la règle se rabattait sur « le moins de fausses alarmes », soit le
  bord de la grille (0,999), qui s'est trouvé bien marcher : **35 / 50 pour le CNN,
  égalité apparente**. Grille étendue jusqu'à 1 − 10⁻⁶, la contrainte devient atteignable,
  la règle fonctionne comme prévu… et le CNN tombe à 28 / 50. Le chiffre retenu est
  celui de la grille étendue : revenir en arrière parce qu'il est moins flatteur serait
  choisir le résultat après l'avoir vu.
- **Montage réduit « portable »** (4 dérivations temporales, proches d'électrodes autour
  de l'oreille, méthode classique) : 36 / 50 crises mais 2,3 fausses alarmes par 24 h
  (point vert sous la courbe). Réduire le nombre d'électrodes coûte surtout en fausses
  alarmes.

Détails par patient et courbes complètes : `resultats/resume_*.json`, régénérés par
`python src/resumer.py classique cnn cnn_int8 classique_temporaux classique_chrono cnn_chrono`.

### Incertitude

6 patients, c'est peu : les chiffres ci-dessus doivent se lire avec leurs intervalles de
confiance ([`src/statistiques.py`](src/statistiques.py), `resultats/statistiques.json`).

| | Crises détectées [IC 95 %] | Fausses alarmes / 24 h [IC 95 % Poisson] |
|---|---|---|
| Chronologique, classique | 22 / 31 = 71 % [26–100 %] | 5,9 [3,8–8,6] |
| Chronologique, CNN | 16 / 31 = 52 % [15–85 %] | 2,5 [1,2–4,4] |
| Validation croisée, classique | 35 / 50 = 70 % [36–97 %] | 1,0 [0,4–2,1] |
| Validation croisée, CNN int8 | 28 / 50 = 56 % [30–92 %] | 1,0 [0,4–2,1] |

- **Intervalles de la sensibilité par bootstrap sur les patients**, pas sur les crises :
  les crises d'un même enfant se ressemblent (chb16 : 10 crises, toutes manquées). Un
  intervalle calculé crise par crise serait deux à trois fois trop étroit.
- **Les fausses alarmes dépendent surtout d'un patient** : l'intervalle de Poisson
  ci-dessus suppose des alarmes indépendantes ; en rééchantillonnant les patients, il va
  de 0 à 19 par 24 h (classique, chronologique), à cause de chb24.
- **Classique contre CNN, sur les mêmes crises** : en scénario chronologique, 6 crises ne
  sont vues que par la méthode classique et aucune que par le CNN (McNemar exact,
  p = 0,03) ; en validation croisée, 12 contre 5 (p = 0,14). **Mais toutes ces crises
  discordantes viennent de chb23 et chb24.** Compté par patient, la méthode classique
  fait mieux chez 2 patients, le CNN chez 0 ou 1, égalité chez les autres (test des
  signes, p = 0,5 et 1) : on ne peut pas conclure qu'une méthode est meilleure.
- **Pour trancher, il faudrait plus de patients**, pas plus de crises chez les mêmes :
  les 18 autres patients de CHB-MIT (148 crises) n'ont pas été téléchargés faute de
  place sur le disque (le plus rentable : chb12, 40 crises pour 1,2 Go d'EDF).

### Embarqué

| | Classique | CNN int8 |
|---|---|---|
| Taille du modèle | 66 arbres, ~1 900 nœuds | **29 Ko** (TensorFlow Lite int8, poids et activations) |
| Calcul par décision (1 / s) | 18 FFT de 512 points + parcours des arbres | **1,4 M multiplications-accumulations** |
| Temps de calcul sur PC | 0,6 ms (caractéristiques) | 0,23 ms |
| Perte due à l'int8 | — | aucune : 28 / 50 (float : 27 / 50) à 1,0 FA / 24 h, décisions identiques à ≥ 99,8 % |

1,4 M opérations par seconde, c'est une petite fraction de ce que calcule un
microcontrôleur Cortex-M4 ; la latence sur cible n'a pas été mesurée (pas de carte).
Pièges rencontrés et corrigés ([`src/export_tflite.py`](src/export_tflite.py)) : la
sigmoïde quantifiée plafonnait à 0,996 alors que les seuils utiles dépassent 0,999 (on
quantifie désormais le logit, et les seuils jusqu'à 0,999999 restent atteignables) ; la calibration de la quantification ne voyait que des fenêtres de
crise, et toutes les probabilités basses tombaient à 0,5.

### Limites

- 6 patients sur les 24 de CHB-MIT : les intervalles de confiance (voir « Incertitude »)
  vont de 26 à 100 % de crises détectées, et aucune différence entre méthodes n'est
  établie au niveau des patients. Un seul entraînement par configuration (la variabilité
  due à l'initialisation du CNN n'est pas mesurée).
- EEG **hospitalier** de 18 dérivations, patients alités, souvent en sevrage de
  traitement : un appareil porté aurait moins d'électrodes et beaucoup plus d'artefacts
  de mouvement. Un artefact simulé de 800 µV à 1,5 Hz est pris pour une crise par les
  deux méthodes ([`src/robustesse.py`](src/robustesse.py)).
- Modèles spécifiques au patient : il faut avoir enregistré des crises pour calibrer.
  Le cas « nouveau patient sans calibration » n'est pas évalué.
- Le CNN dépend de seuils extrêmes (0,9995 à 0,999999) : une probabilité aussi saturée
  est fragile (calibration, dérive du signal). Un seuil sur le logit ou une calibration
  des probabilités par patient seraient à essayer.
- Le scénario chronologique ne teste que 31 crises (107 h), dont 7 crises en 5 h pour
  chb16. La validation croisée teste tout, mais laisse les modèles apprendre sur des
  heures postérieures au test (comme la plupart des travaux sur CHB-MIT).
- chb24 n'a pas d'horodatage dans CHB-MIT : l'ordre chronologique de ses enregistrements
  est supposé suivre leur numérotation.

## Dossier « dispositif médical » et tests

Exercice appliquant la démarche d'un fabricant, dans
[`docs/dispositif-medical/`](docs/dispositif-medical/README.md) : usage prévu et statut
réglementaire (MDR, IEC 62304), 7 exigences fixées **avant** les résultats, **analyse des
risques** inspirée d'ISO 14971 et **matrice de traçabilité** générée automatiquement.

La démarche a trouvé **trois défauts, corrigés avant les résultats finaux** :

1. **Un modèle sur quatre prédisait « crise » en permanence** (360 fausses alarmes par
   jour chez chb01), à cause de la BatchNormalization, dont les statistiques étaient
   apprises sur des lots équilibrés 50/50, très loin du flux réel. Architecture corrigée
   et contrôle automatique après chaque entraînement.
2. **Deux erreurs de quantification int8** (ci-dessus).
3. **Un seuil de saturation qui aurait bloqué de vraies crises** : fixé d'abord à
   1 500 µV, il a été relevé à 3 000 µV après avoir mesuré des crises montant à
   2 400 µV. Le contrôle de qualité du signal ([`src/qualite_signal.py`](src/qualite_signal.py))
   bloque l'appareil débranché et les dérivations saturées, touche 0,016 % des fenêtres
   réelles au plus et aucune fenêtre de crise.

```bash
python -m pytest tests            # 24 tests : signal, métriques cliniques, sécurité, modèle int8, démo
python src/verification_report.py # relance les tests et régénère la matrice de traçabilité
```

## Démo

Application Streamlit qui rejoue 4 extraits d'enregistrements comme le ferait
l'appareil : l'EEG défile, la probabilité de crise monte, l'alarme se déclenche (ou
pas). Les probabilités viennent des modèles qui n'ont **jamais vu** l'enregistrement,
avec le réglage d'usine de chaque patient ; on peut aussi modifier le réglage pour voir
le compromis détection / fausses alarmes. Le cas chb16 (crises de 6 et 8 s) montre
l'échec.

![Démo : crise de chb01 détectée par le CNN int8, alarme 11 s après le début, aucune fausse alarme](docs/demo-detection.png)

*chb01_03, CNN int8 ([ouvrir ce cas](https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/?enregistrement=chb01_03&t=320)) : pendant la crise
(bande rouge), la probabilité lissée atteint le seuil de 0,999999 et l'alarme part 11 s après le
début annoté.*

![Démo : crise de 6 s chez chb16, manquée par la méthode classique](docs/demo-crise-breve.png)

*chb16_17, méthode classique ([ouvrir ce cas](https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/?enregistrement=chb16_17&methode=classique&t=205)) :
la crise de 6 s est bien visible sur l'EEG, mais la moyenne des 10 dernières secondes plafonne
vers 0,3, loin du seuil de 0,94. Le dispositif reste en « surveillance » : c'est la limite
décrite dans les résultats (risque R2 du dossier dispositif médical).*

```bash
streamlit run app/streamlit_app.py
```

En ligne : [https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/](https://alixe09-epilepsie-eeg-appstreamlit-app-cfse0q.streamlit.app/). Lien direct vers un cas précis :
`?enregistrement=chb16_17&methode=classique&t=205` (enregistrement, `cnn_int8` ou
`classique`, instant en secondes dans l'extrait).

## Reproduire

Python 3.12, dépendances dans `requirements.txt` (~1 h 30 de calcul sur un portable
sans GPU, hors téléchargement).

```bash
python src/telecharger_chbmit.py          # ~7 Go téléchargés, 2 Go conservés
python src/caracteristiques.py            # cache des caractéristiques
python src/classique.py
python src/classique.py --canaux temporaux
python src/cnn.py                         # 23 modèles (un par patient et par pli)
python src/export_tflite.py               # int8 + probabilités int8
python src/classique.py --decoupage chrono     # scénario chronologique
python src/cnn.py --decoupage chrono
python src/resumer.py classique classique_temporaux cnn cnn_int8 classique_chrono cnn_chrono
python src/statistiques.py                # intervalles de confiance, tests appariés
python src/robustesse.py
python src/embarque.py
python src/make_demo_data.py
python src/figures.py
python src/verification_report.py
```

## Structure

```
src/
  telecharger_chbmit.py  edf.py  donnees.py     données
  pretraitement.py  qualite_signal.py            filtrage causal, fenêtres, contrôle de qualité
  caracteristiques.py  classique.py              méthode classique
  cnn.py  export_tflite.py  embarque.py          réseau de neurones, int8, budget embarqué
  decoupage.py  evaluation.py  resumer.py        validation croisée, alarmes, métriques cliniques
  statistiques.py                                intervalles de confiance, tests appariés
  robustesse.py  verification_report.py          analyse des risques, traçabilité
  make_demo_data.py  figures.py
tests/                                           24 tests pytest
app/                                             démo Streamlit (4 extraits, 8 Mo)
resultats/                                       résumés JSON, modèles int8 (.tflite)
docs/                                            figures, dossier dispositif médical
```

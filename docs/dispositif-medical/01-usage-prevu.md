# 1. Usage prévu et statut réglementaire

> Exercice pédagogique : ce projet n'est **pas** un dispositif médical.

## Usage prévu (proposé)

| | |
|---|---|
| **Fonction** | Analyser en continu l'EEG de scalp et **alerter un aidant** (parent, soignant) lorsqu'une crise d'épilepsie est probablement en cours ; tenir un **journal des crises** horodaté pour le neurologue. |
| **Population** | Enfants et adolescents épileptiques dont les crises ont une traduction EEG sur scalp, **après calibration** sur au moins 3 crises enregistrées du patient (modèles spécifiques au patient). |
| **Utilisateurs** | Aidants (réception de l'alarme) ; neurologue (calibration, relecture du journal). |
| **Environnement** | Domicile, nuit et jour, patient au repos ou en activité calme. |
| **Nature de l'information** | Aide à la surveillance. **Ne pose pas de diagnostic**, ne décide d'aucun traitement, ne remplace ni la surveillance humaine ni l'EEG hospitalier. |

**Hors périmètre** : diagnostic d'épilepsie, prédiction des crises avant leur
début, patients non calibrés, crises sans traduction EEG de scalp, usage en
réanimation ou pour déclencher automatiquement un traitement.

## Pourquoi ces choix

- **Alerter plutôt que diagnostiquer** : une détection manquée n'est pas pire
  qu'aujourd'hui (pas de surveillance) ; une fausse alarme réveille la famille.
  Les deux erreurs ont un coût, d'où les deux métriques suivies ensemble :
  crises détectées et fausses alarmes par 24 h.
- **Calibration par patient** : la forme EEG d'une crise varie énormément d'un
  enfant à l'autre (voir les résultats). C'est l'approche de référence sur
  CHB-MIT (Shoeb, 2009) et une contrainte d'usage explicite.

## Exigences de performance (fixées avant les résultats)

| ID | Exigence | Justification |
|---|---|---|
| EX-01 | Sensibilité ≥ 90 % des crises, par patient | Une alarme qui manque une crise sur dix reste utile ; en dessous, la confiance des aidants s'effondre |
| EX-02 | ≤ 1 fausse alarme par 24 h, par patient | Au-delà, fatigue d'alarme : les familles désactivent l'appareil |
| EX-03 | Délai médian de détection ≤ 30 s après le début de la crise | Laisser le temps d'intervenir (mise en sécurité, médicament de secours au-delà de 5 min) |
| EX-04 | Traitement causal : aucune décision n'utilise d'EEG futur | Fonctionnement en temps réel |
| EX-05 | Modèle embarquable : ≤ 100 Ko, calcul compatible microcontrôleur | Dispositif portable sur batterie |
| EX-06 | Quantification int8 sans perte clinique | Portage sur la cible |
| EX-07 | Signal absent ou saturé → pas de fausse alarme silencieuse ; défaut signalé | Sécurité en cas de panne capteur |

## Statut réglementaire (analyse d'exercice)

- **UE, règlement (UE) 2017/745 (MDR)** : un logiciel destiné à surveiller un
  processus physiologique est un dispositif médical. Règle 11 : un logiciel qui
  fournit des informations utilisées pour prendre des décisions à des fins
  thérapeutiques ou diagnostiques est au minimum en classe IIa. La surveillance
  de crises chez l'enfant, où une défaillance peut contribuer à un dommage
  sérieux, justifierait une discussion vers la **classe IIb**.
- **IEC 62304 (cycle de vie du logiciel)** : une crise non détectée peut
  contribuer à une blessure sérieuse (chute, crise prolongée non traitée) →
  **classe C** proposée, ramenée à **B** si l'architecture garantit que le
  dispositif n'est pas le seul moyen de surveillance (usage prévu : aide).
- **Autres normes à considérer** : ISO 14971 (gestion des risques), IEC 60601-1
  et IEC 60601-2-26 (sécurité des électroencéphalographes), IEC 60601-1-8
  (systèmes d'alarme), IEC 62366-1 (aptitude à l'utilisation), RGPD (données de
  santé).

## Ce qui manquerait pour un vrai dossier

- Données : CHB-MIT est une base hospitalière (EEG 23 canaux, enfants
  hospitalisés pour bilan, souvent avec sevrage médicamenteux) ; un portable
  aurait moins d'électrodes, plus d'artefacts de mouvement et d'autres patients.
- Investigation clinique prospective à domicile, sur la population visée.
- Matériel : électrodes, qualité du contact, autonomie, transmission de l'alarme.
- Système qualité (ISO 13485), documentation IEC 62304 complète, cybersécurité.

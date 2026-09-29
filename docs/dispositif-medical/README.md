# Dossier « dispositif médical » (exercice)

> ⚠️ Exercice pédagogique réalisé dans le cadre d'un projet personnel. Ce projet
> n'est **pas** un dispositif médical ; ce dossier n'a aucune valeur réglementaire.
> Il applique, de façon simplifiée, la démarche d'un fabricant : définir l'usage
> prévu, fixer des exigences **avant** de connaître les résultats, analyser les
> risques et prouver ce qui est vérifié.

| Document | Contenu |
|---|---|
| [1. Usage prévu et statut réglementaire](01-usage-prevu.md) | Alerter un aidant, pas diagnostiquer ; 7 exigences ; MDR (classe IIa–IIb), IEC 62304 (classe B/C), normes applicables |
| [2. Analyse des risques](02-analyse-risques.md) | 10 risques (démarche ISO 14971), dont 3 défauts trouvés et corrigés pendant le projet |
| [3. Matrice de traçabilité](03-matrice-tracabilite.md) | 7 exigences → méthode de vérification → résultat → statut. **Générée automatiquement** |

## En bref

- **Conforme** : délai médian (12–13 s), traitement causal, modèle embarquable
  (29 Ko, 1,4 M multiplications par seconde), quantification int8 sans perte.
- **Non conforme** : sensibilité et fausses alarmes par patient, dans les deux scénarios.
  Scénario chronologique (usage prévu) : 22 / 31 crises en classique, 16 / 31 pour le
  CNN, chb24 à 13–34 fausses alarmes / 24 h. Validation croisée : 35 / 50 et 28 / 50.
- **Défauts trouvés par la démarche et corrigés** : un modèle sur quatre qui
  prédisait « crise » en permanence, deux erreurs de quantification int8, un
  seuil de saturation qui aurait bloqué de vraies crises.
- **Risques ouverts** : crises brèves jamais détectées ; artefact de mouvement
  pris pour une crise (non détectable par le contrôle de qualité).

## Régénérer

```bash
python -m pytest tests            # 23 tests : signal, métriques cliniques, sécurité, modèle embarqué, démo
python src/verification_report.py # relance les tests et régénère la matrice
```

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

- **Conforme** : délai médian (13–16 s), traitement causal, modèle embarquable
  (29 Ko, 1,4 M multiplications par seconde), quantification int8 sans perte.
- **Non conforme** : sensibilité (35 / 50 crises) et fausses alarmes par patient
  (chb24 à 4,5–6,8 / 24 h).
- **Défauts trouvés par la démarche et corrigés** : un modèle sur quatre qui
  prédisait « crise » en permanence, deux erreurs de quantification int8, un
  seuil de saturation qui aurait bloqué de vraies crises.
- **Risques ouverts** : crises brèves jamais détectées ; artefact de mouvement
  pris pour une crise (non détectable par le contrôle de qualité).

## Régénérer

```bash
python -m pytest tests            # 20 tests : signal, métriques cliniques, sécurité, modèle embarqué
python src/verification_report.py # relance les tests et régénère la matrice
```

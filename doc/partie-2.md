# Partie 2 – Preprocessing, Pipeline MLOps et Données synthétiques

---

## Feature engineering - enrichissement des variables

### Objectifs du preprocessing
- **Problème** : Données brutes avec seulement 31 variables, pas optimisées pour ML
- **Solution** : Pipeline de transformation automatisé avec feature engineering  
- **Résultat** : +8 nouvelles variables métier, données prêtes pour modélisation

### Transformations réalisées

#### Variables temporelles créées
- **Hour** : Heure de la transaction (0-23) extraite de `Time`
- **Day** : Jour depuis le début (0, 1, 2...) pour identifier les patterns hebdomadaires
- **Is_Night** : Variable binaire pour transactions nocturnes (22h-6h)  
- **Is_Weekend** : Variable binaire pour transactions weekend

#### Variables de montant enrichies
- **Amount_log** : Transformation logarithmique `log(Amount + 1)` pour normaliser la distribution
- **Amount_Category** : Catégorisation en 4 groupes (Micro/Small/Medium/Large)
  - Micro : 0-10€
  - Small : 10-50€  
  - Medium : 50-200€
  - Large : >200€

#### Variables PCA enrichies
- **PCA_Magnitude** : Norme euclidienne des 28 composantes PCA `√(V1² + V2² + ... + V28²)`
- **PCA_Extreme_Count** : Nombre de variables PCA avec valeurs extrêmes (>3σ)

### Insights du feature engineering

#### Patterns temporels identifiés
- **Taux de fraude nuit** : 0.180% (légèrement supérieur)
- **Taux de fraude jour** : 0.172% (niveau de base)
- **Conclusion** : Faible différence mais exploitable par ML

#### Distribution par catégorie de montant
- **Micro (0-10€)** : 0.213% de fraude, 21,934 transactions
- **Small (10-50€)** : 0.169% de fraude, 118,063 transactions  
- **Medium (50-200€)** : 0.151% de fraude, 98,824 transactions
- **Large (>200€)** : 0.172% de fraude, 45,986 transactions

#### Variables PCA extrêmes
- **Transactions normales** : 0.30 valeurs extrêmes en moyenne
- **Transactions frauduleuses** : 4.11 valeurs extrêmes en moyenne
- **Ratio** : 13.7x plus de valeurs extrêmes dans les fraudes
- **Impact** : Fort pouvoir discriminant pour la détection

### Résultats du preprocessing
**Avant** : 31 variables brutes, distributions non optimisées  
**Après** : 39 variables enrichies, normalisées, prêtes pour ML  
**Validation** : 100% de complétude, 0 valeurs manquantes

---

## Pipeline MLOps avec Airflow

### Architecture du pipeline automatisé

#### Configuration Airflow
```bash
# Démarrage du pipeline
cd /chemin/vers/projet
./scripts/start-airflow.sh

# Interface web
URL: http://localhost:8080
Username: admin
Password: admin
```

#### Structure du DAG `fraud_detection_pipeline`
```
check_data
     ↓
preprocess_data
     ↓
synthetic_generation/
  ├── generate_smote
  └── generate_sdv
     ↓
select_best
     ↓
notify
```

### Vues de monitoring disponibles

#### **Graph view**
- Vue graphique complète du pipeline
- Dépendances entre les tâches visualisées
- Groupes parallèles pour optimisation
- Status coloré par tâche (Vert=réussi, Rouge=échec, Bleu=en cours)

#### **Tree view**
- Historique des exécutions dans le temps
- Timeline détaillée de chaque run
- Accès direct aux logs par tâche

#### **Gantt chart**
- Durée d'exécution de chaque tâche
- Identification des goulots d'étranglement
- Optimisation des performances du pipeline

### Fonctionnalités opérationnelles

#### Exécution et monitoring
```bash
# Trigger manuel du pipeline
airflow dags trigger fraud_detection_pipeline

# Consultation des logs
airflow tasks logs fraud_detection_pipeline check_raw_data 2024-01-01

# Arrêt propre
pkill -f airflow
```

#### Gestion des erreurs
- **Retry automatique** : 3 tentatives par tâche en cas d'échec
- **Alertes** : Notifications Slack/email en cas de problème
- **Rollback** : Retour version précédente si validation échoue
- **Monitoring** : Dashboards temps réel des KPIs

---

## Génération de données synthétiques

### Problématique du déséquilibre extrême

**Défi initial** : 0.17% fraudes vs 99.83% normales (ratio 1:578)  
**Impact critique** : Les modèles ML prédisent systématiquement "normal"  
**Solution adoptée** : Génération de fraudes synthétiques réalistes

### Méthodes testées et comparées

#### SMOTE (Synthetic Minority Oversampling Technique)
- **Principe** : Interpolation entre fraudes existantes dans l'espace des features
- **Algorithme** : Génère des exemples sur les segments reliant les k plus proches voisins
- **Résultat** : 28,431 nouvelles fraudes synthétiques
- **Score qualité** : 0.92/1.0
- **Temps d'exécution** : 2 secondes
- **Status** : **Méthode retenue**

#### SDV GaussianCopula (Statistical Data Vault)
- **Principe** : Modélisation statistique complète des distributions et corrélations
- **Algorithme** : Apprentissage des patterns statistiques puis génération probabiliste
- **Résultat** : 2,000 fraudes synthétiques haute qualité
- **Score qualité** : 0.95/1.0 (meilleur score)
- **Temps d'exécution** : 15 secondes
- **Status** : 🔄 **Méthode de backup** (plus lente mais plus précise)

#### BorderlineSMOTE (Borderline Cases Focus)
- **Principe** : Focus sur les fraudes proches de la frontière de décision
- **Algorithme** : Identification des cas limites puis sur-échantillonnage ciblé
- **Résultat** : 28,431 fraudes synthétiques
- **Score qualité** : 0.89/1.0
- **Temps d'exécution** : 3 secondes
- **Status** : **Alternative** (bon compromis)

### Résultats de l'équilibrage

#### Impact quantitatif de SMOTE
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **Transactions normales** | 284,315 | 284,315 | - |
| **Transactions frauduleuses** | 492 | 28,923 | +58x |
| **Ratio de déséquilibre** | 1:578 | 1:10 | **57x plus équilibré** |
| **Dataset total** | 284,807 | 313,238 | +10% |

#### Validation statistique
- **Cohérence des distributions** : Les fraudes synthétiques respectent les patterns originaux
- **Préservation des corrélations** : Les relations entre variables V1-V28 sont maintenues
- **Diversité générative** : Pas de duplication, chaque échantillon est unique
- **Qualité globale** : Score de validation 0.92/1.0

### Justification du choix SMOTE
1. **Équilibre qualité/performance** : Score 0.92 avec exécution rapide (2s)
2. **Robustesse** : Algorithme stable et reproductible
3. **Intégration** : Compatible avec tous les frameworks ML
4. **Scalabilité** : Fonctionne sur datasets volumineux
5. **Maintenance** : Pipeline simple sans dépendances complexes

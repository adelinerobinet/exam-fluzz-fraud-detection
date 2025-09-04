# Partie 1 – Exploration et cycle de vie des données

---

## 1 - Fiche descriptive des données - Détection de fraude

### Vue d'ensemble rapide
- **Dataset** : Credit Card Fraud Detection (Kaggle/ULB)  
- **Volume** : 284 807 transactions sur 2 jours  
- **Défi principal** : Déséquilibre extrême (0.17% de fraudes)

### Caractéristiques clés
- **39 variables** (31 originales + 8 enrichies) :  
  - 28 anonymisées (PCA)  
  - `Time` (temps en secondes depuis 1ère transaction)  
  - `Amount` (montant en €)  
  - `Class` (0=Normal, 1=Fraude)  
  - **8 nouvelles variables créées** :
    - Temporelles : `Hour`, `Day`, `Is_Night`, `Is_Weekend`
    - Montants : `Amount_log`, `Amount_Category`
    - PCA : `PCA_Magnitude`, `PCA_Extreme_Count`
- **492 fraudes** vs **284 315 normales** → Ratio ≈ **1:578**  
- Données collectées en **Europe (septembre 2013)**, anonymisées pour confidentialité  
- **Qualité** : 0 valeurs manquantes ✅

---

### Déséquilibre des classes
- **Normales** : 99.83%  
- **Fraudes** : 0.17%  
- **Impact** : les modèles tendent à prédire "normal" par défaut  
- **Solution** : techniques de rééquilibrage (SMOTE, sous-échantillonnage)

---

### Analyses exploratoires

#### Montants
- Transactions normales → Moyenne ≈ **88€**, médiane ≈ **23€**  
- Transactions frauduleuses → Moyenne ≈ **122€**, médiane ≈ **9€**  
- **Insight** : Les fraudes concernent souvent de **petits montants (0-50€)** mais quelques cas isolés à très gros montants augmentent la moyenne.

#### Temporalité
- Durée couverte : ~**2 jours**  
- Transactions normales → pic d’activité en journée (**8h-20h**)  
- Fraudes → présentes jour et nuit, mais en nombre plus faible la nuit  
- **Insight** : l’heure de la transaction (`Hour`) est une feature potentiellement utile.

#### Features PCA
- Les 5 plus corrélées avec la cible `Class` : **V14, V17, V12, V10, V16**  
- Distribution très différente entre fraudes et normales → **fort pouvoir discriminant**

---

### Métriques de performance à retenir
- ❌ **Accuracy** → trompeuse (modèle biaisé vers classe majoritaire)  
- ✅ **Recall** → détecter un maximum de fraudes (priorité métier)  
- ✅ **Precision** → limiter les faux positifs  
- ✅ **F1-Score** → équilibre Recall / Precision  
- ✅ **AUC-ROC** → évaluer la séparation globale  

---

## 2 - Cycle de vie des données - Détection de fraude

### Pipeline MLOps simplifié
```
📥 Ingestion → 🗃️ Stockage → ⚙️ Preprocessing → 🤖 Modélisation → 🚀 Déploiement → 📊 Monitoring
```

---

### 1. Ingestion des données
- **Temps réel** : Kafka (flux de transactions live)  
- **Batch** : CSV historiques via Airflow  
- **Externes** : listes noires, géolocalisation  

---

### 2. Stockage data lake (3 couches)
- **Bronze** → données brutes (S3/Parquet)  
- **Silver** → données nettoyées (Delta Lake)  
- **Gold** → données prêtes pour ML (PostgreSQL)  

---

### 3. Preprocessing automatisé
1. Validation des schémas et cohérence  
2. Nettoyage doublons, valeurs manquantes  
3. Feature engineering (8 nouvelles variables) :
   - Temporelles : `Hour`, `Day`, `Is_Night`, `Is_Weekend`
   - Montants : `Amount_log`, `Amount_Category`
   - PCA : `PCA_Magnitude`, `PCA_Extreme_Count`
4. Rééquilibrage → SMOTE / données synthétiques  

---

### 4. Pipeline Airflow automatisé
**DAG quotidien (2h00 UTC)** :  
```
Ingestion → Preprocessing → Synthétique → Entraînement → Validation → Déploiement
```
- Retry automatique  
- Alertes en cas d’échec  
- Monitoring en continu  

---

### 5. Déploiement et production
- **Architecture** : Kubernetes + FastAPI + autoscaling  
- **API prédiction** : `/api/predict` (latence <100ms)  
- **CI/CD** : tests → build → staging → production  

---

### 6. Monitoring temps réel
- **KPIs métier** :  
  - Taux de détection fraude > 85%  
  - Faux positifs < 5%  
  - Latence API < 100ms  
- **Outils** : Grafana dashboards + Alertes Slack/PagerDuty  

---

### 7. Amélioration continue
- Réentraînement déclenché si :  
  - Baisse de performance (AUC < seuil)  
  - Data drift détecté  
  - Feedback métier (enquêtes manuelles)  

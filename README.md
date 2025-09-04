# 🏦 Système de détection de fraude bancaire - Néobanque Fluzz

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.7+-red.svg)](https://airflow.apache.org)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

Système complet de machine learning pour la détection automatique de fraudes bancaires, développé pour la néobanque Fluzz. Le projet implémente un pipeline MLOps robuste allant de l'exploration des données jusqu'au déploiement en production.

## Table des matières

- [ Objectifs du Projet](#-objectifs-du-projet)
- [️ Architecture](#️-architecture)
- [ Installation](#-installation)
- [ Structure du Projet](#-structure-du-projet)
- [ Pipeline ML](#-pipeline-ml)
- [ Documentation](#-documentation)
- [ Monitoring](#-monitoring)
- [ Déploiement](#-déploiement)

## Objectifs du projet

### Contexte business
La néobanque Fluzz fait face à une augmentation des transactions frauduleuses qui impactent directement la rentabilité et la satisfaction client. Ce projet vise à :

- **Détecter automatiquement** les tentatives de fraude en temps réel
- **Réduire les faux positifs** pour améliorer l'expérience client
- **Automatiser** le processus de détection pour réduire les coûts opérationnels
- **Respecter** les réglementations bancaires (RGPD, PCI DSS)

### Objectifs techniques
- Taux de détection > 75%
- Taux de faux positifs < 5%
- Latence de prédiction < 100ms
- Pipeline automatisé et reproductible
- Solution scalable et maintenable

## Architecture

### Stack technologique

| Composant | Technologie | Version | Usage |
|-----------|-------------|---------|-------|
| **Language** | Python | 3.9+ | Développement principal |
| **Orchestration** | Apache Airflow | 2.7+ | Pipeline ML automatisé |
| **ML Libraries** | scikit-learn, imbalanced-learn | Latest | Modélisation |
| **Données Synthétiques** | SDV | Latest | Augmentation de données |
| **API** | FastAPI | Latest | Service de prédiction |
| **Containerisation** | Docker | Latest | Déploiement |
| **Monitoring** | Grafana + Prometheus | Latest | Observabilité |
| **Data Storage** | Fichiers locaux (CSV, PKL) | - | Persistance |

## Installation

### Prérequis Système
```bash
# Python 3.9+ requis
python --version

# Docker (optionnel pour déploiement)
docker --version

# Git pour le versioning
git --version
```

### Installation locale

1. **Cloner le repository**
```bash
# Remplacer par l'URL de votre repository
git clone <URL_DU_REPOSITORY>
cd exam-fluzz-fraud-detection/adeline
```

2. **Créer l'environnement virtuel**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

3. **Installer les dépendances**
```bash
# Dépendances principales
pip install -r requirements.txt

# Dépendances pour notebooks (optionnel)
pip install -r requirements-notebooks.txt
```

4. **Configurer l'environnement**
```bash
# Créer la structure de répertoires nécessaire
python config/config.py

# Vérifier que tout est bien installé
python -c "import pandas, numpy, sklearn; print('Installation validée')"
```

### Installation avec Docker

```bash
# Aller dans le répertoire api
cd api/

# Construction de l'image
docker build -t fraud-detection-api:latest .

# Lancement de l'API
docker-compose up -d
```

## Structure du projet

```
fraud-detection/
├── 📁 .github/               # Workflows GitHub Actions
├── 📁 airflow/               # Configuration Airflow locale  
│   ├── airflow.cfg           # Configuration Airflow
│   ├── airflow.db           # Base de données Airflow
│   └── logs/                # Logs d'exécution
├── 📁 api/                   # API de prédiction
│   ├── Dockerfile           # Image Docker API
│   ├── docker-compose.yml   # Orchestration Docker
│   ├── app/                 # Code FastAPI
│   ├── k8s/                 # Manifests Kubernetes
│   └── monitoring/          # Config Grafana/Prometheus
├── 📁 config/               # Configuration centralisée
│   ├── config.py           # Paramètres globaux
│   └── logging.conf        # Configuration logs
├── 📁 data/                # Données du projet
│   ├── raw/                # Données brutes
│   ├── processed/          # Données nettoyées
│   └── synthetic/          # Données synthétiques
├── 📁 doc/                 # Documentation complète
│   ├── partie-1.md         # Exploration et cycle de vie des données
│   ├── partie-2.md         # Preprocessing et pipeline Airflow
│   ├── partie-3.md         # Enjeux sociétaux et éthiques
│   ├── partie-4.md         # Modélisation et optimisation
│   ├── partie-5.md         # Mesure de performance et suivi
│   ├── partie-6.md         # Sécurité et menaces
│   └── partie-7.md         # Industrialisation et déploiement
├── 📁 models/              # Modèles ML
│   └── generators/         # Générateurs synthétiques
├── 📁 notebook/            # Notebooks Jupyter
│   ├── partie-1.ipynb     # Exploration données
│   ├── partie-2.ipynb     # Preprocessing
│   ├── partie-3.ipynb     # Enjeux éthiques
│   ├── partie-4.ipynb     # Modélisation
│   ├── partie-5.ipynb     # Monitoring
│   ├── partie-6.ipynb     # Sécurité
│   └── partie-7.ipynb     # Industrialisation
├── 📁 oral/                # Préparation soutenance
├── 📁 pipelines/           # Pipelines Airflow
│   └── airflow/
│       └── fraud_pipeline.py
├── 📁 reports/             # Rapports générés
├── 📁 scripts/             # Scripts utilitaires
│   └── start-airflow.sh    # Démarrage Airflow
├── 📁 src/                 # Code source
│   ├── __init__.py
│   ├── utils.py           # Fonctions utilitaires
│   ├── data_processing.py # Traitement données
│   ├── model_training.py  # Entraînement modèles
│   └── synthetic_data.py  # Génération synthétique
├── requirements.txt        # Dépendances production
├── requirements-notebooks.txt # Dépendances notebooks
└── README.md              # Ce fichier
```

## Pipeline ML

### Vue d'Ensemble du Pipeline

Le pipeline est orchestré par Apache Airflow et se compose de plusieurs étapes :

1. **Ingestion de Données** (`check_data`)
2. **Preprocessing** (`preprocess_data`) 
3. **Génération Synthétique** (`synthetic_generation`)
4. **Sélection Meilleur Dataset** (`select_best`)
5. **Notifications** (`notify`)

### Lancement du pipeline

#### Via Airflow
```bash
# Démarrer Airflow
./scripts/start-airflow.sh

# Accéder à l'interface web
# http://localhost:8080 (admin/admin)

# Le DAG 'fraud_detection_pipeline' sera automatiquement détecté
```

#### Via Notebooks
```bash
# Lancer Jupyter
jupyter notebook

# Exécuter les notebooks dans l'ordre :
# 1. notebook/partie-1.ipynb - Exploration données
# 2. notebook/partie-2.ipynb - Preprocessing  
# 3. notebook/partie-3.ipynb - Enjeux éthiques
# 4. notebook/partie-4.ipynb - Modélisation
# 5. notebook/partie-5.ipynb - Monitoring
# 6. notebook/partie-6.ipynb - Sécurité
# 7. notebook/partie-7.ipynb - Industrialisation
```

### Configuration du pipeline

Les paramètres du pipeline sont centralisés dans `config/config.py` :

```python
# Exemple de configuration
SYNTHETIC_DATA_CONFIG = {
    "smote": {
        "sampling_strategy": 0.1,
        "k_neighbors": 5
    },
    "sdv": {
        "num_samples": 2000,
        "synthesizer": "GaussianCopulaSynthesizer"
    }
}
```

## Documentation

### Documentation Technique

La documentation est organisée en 7 parties correspondant aux exigences du projet :

| Partie | Document | Description |
|--------|----------|-------------|
| **1** | [Exploration et Cycle de Vie](doc/partie-1.md) | Fiche descriptive des données et architecture MLOps |
| **2** | [Preprocessing et Pipeline](doc/partie-2.md) | Feature engineering, Airflow et données synthétiques |
| **3** | [Enjeux Éthiques](doc/partie-3.md) | Identification des biais et charte éthique RGPD |
| **4** | [Modélisation](doc/partie-4.md) | Comparaison de modèles et optimisation |
| **5** | [Monitoring](doc/partie-5.md) | Dashboard temps réel et détection de drift |
| **6** | [Sécurité](doc/partie-6.md) | Détection d'anomalies et analyse des logs |
| **7** | [Industrialisation](doc/partie-7.md) | Containerisation Docker et Kubernetes |
| **Sujet** | [Sujet d'Examen](doc/sujet.md) | Énoncé complet du projet |

### API Documentation

Une fois l'API déployée, la documentation interactive est accessible sur :
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

## Monitoring

### Métriques business
- **Taux de détection de fraude** : > 85%
- **Taux de faux positifs** : < 5%
- **Latence de prédiction** : < 100ms
- **Disponibilité du service** : > 99.9%

### Métriques techniques
- **Dérive des données** (Data Drift)
- **Dérive du modèle** (Model Drift)
- **Performance des prédictions**
- **Utilisation des ressources**

### Dashboards

Les dashboards Grafana incluent :
- Vue temps réel des prédictions
- Métriques de performance historiques  
- Alertes et notifications
- Analyse des tendances

## Déploiement

### Déploiement Local (Développement)
```bash
# API FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Pipeline Airflow
./scripts/start-airflow.sh
```

### Déploiement Kubernetes (Conception théorique - Partie 7)

**Note**: Les manifestes Kubernetes sont fournis pour répondre aux exigences de la Partie 7 du projet, mais n'ont pas été déployés sur un cluster réel.

```bash
# Aller dans le répertoire api
cd api/

# Commandes de déploiement préparées
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml 
kubectl apply -f k8s/hpa.yaml

# Vérifications théoriques
kubectl get pods -n fluzz-banking
kubectl get services -n fluzz-banking
kubectl get hpa -n fluzz-banking
```

### Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `ENVIRONMENT` | Environnement | `development` |
| `LOG_LEVEL` | Niveau de logging | `INFO` |

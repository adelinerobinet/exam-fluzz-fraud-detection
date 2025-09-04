#!/bin/bash

#==============================================================================
# Script de Démarrage d'Apache Airflow - Projet Détection de Fraude
#==============================================================================
#
# Description:
#   Script automatisé pour démarrer Apache Airflow en local avec configuration
#   optimisée pour le pipeline de détection de fraude bancaire.
#   
# Fonctionnalités:
#   - Initialisation automatique de la base de données Airflow
#   - Création de l'utilisateur admin avec identifiants admin/admin
#   - Configuration des chemins et variables d'environnement
#   - Démarrage coordonné du scheduler et webserver
#   - Gestion propre de l'arrêt avec Ctrl+C
#
# Prérequis:
#   - Apache Airflow 2.7+ installé (pip install apache-airflow==2.7.3)
#   - Python 3.8+ avec les dépendances du projet
#   - Permissions d'écriture dans le répertoire du projet
#
# Usage:
#   ./scripts/start_airflow.sh
#
# Interface Web:
#   http://localhost:8080 (admin/admin)
#
# Auteur: Équipe Data Science - Néobanque Fluzz
# Version: 2.0
# Date: Août 2025
#==============================================================================

echo "🚀 Démarrage d'Airflow pour visualisation du pipeline de détection de fraude"

# Configuration des variables d'environnement
export AIRFLOW_HOME=$(pwd)/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/pipelines/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8080

# Créer le dossier Airflow s'il n'existe pas
mkdir -p $AIRFLOW_HOME

# Si c'est la première fois, initialiser avec admin/admin
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "📦 Première installation - Configuration avec admin/admin..."
    
    # Initialiser la base de données
    airflow db init
    
    # Créer l'utilisateur admin/admin
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    
    echo "✅ Utilisateur admin/admin créé!"
fi

echo "🌐 Démarrage d'Airflow..."
echo "📱 Interface accessible sur: http://localhost:8080"
echo "🔑 Login: admin / Password: admin"
echo ""
echo "Pour arrêter: Ctrl+C"

# Démarrer les services Airflow séparément pour contrôler les utilisateurs
echo "🚀 Démarrage du scheduler en arrière-plan..."
airflow scheduler &
SCHEDULER_PID=$!

echo "🌐 Démarrage du webserver..."
airflow webserver --port 8080 &
WEBSERVER_PID=$!

# Fonction pour nettoyer les processus à l'arrêt
cleanup() {
    echo ""
    echo "🛑 Arrêt d'Airflow..."
    kill $SCHEDULER_PID 2>/dev/null
    kill $WEBSERVER_PID 2>/dev/null
    exit 0
}

# Capturer Ctrl+C pour nettoyer
trap cleanup SIGINT SIGTERM

# Attendre que les processus se terminent
wait
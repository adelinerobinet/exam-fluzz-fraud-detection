#!/bin/bash

# Script de démarrage du monitoring Grafana/Prometheus pour la partie 5

echo "🚀 Démarrage de l'environnement de monitoring - Partie 5"
echo "=================================================="

# Vérifier que Docker est démarré
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker n'est pas démarré. Veuillez démarrer Docker Desktop."
    exit 1
fi

echo "✅ Docker est actif"

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Fichier docker-compose.yml non trouvé."
    echo "💡 Veuillez vous placer dans le répertoire adeline/api/"
    exit 1
fi

echo "✅ Fichier docker-compose.yml trouvé"

# Arrêter les anciens conteneurs s'ils existent
echo "🧹 Nettoyage des anciens conteneurs..."
docker-compose down 2>/dev/null

# Démarrer tous les services nécessaires (API incluse pour les métriques)
echo "🔄 Démarrage des conteneurs..."
if docker-compose up -d; then
    echo "✅ Conteneurs démarrés avec succès"
else
    echo "❌ Erreur lors du démarrage des conteneurs"
    exit 1
fi

# Attendre que les services soient prêts
echo "⏳ Attente du démarrage des services (30 secondes)..."
sleep 30

# Vérifier le statut des services
echo "📊 Statut des conteneurs:"
docker-compose ps

# Vérifier que Grafana répond
echo "🔍 Vérification de Grafana..."
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Grafana est accessible"
else
    echo "⚠️  Grafana pas encore prêt, patientez quelques secondes..."
fi

# Vérifier que Prometheus répond
echo "🔍 Vérification de Prometheus..."
if curl -s http://localhost:9090 > /dev/null; then
    echo "✅ Prometheus est accessible"
else
    echo "⚠️  Prometheus pas encore prêt, patientez quelques secondes..."
fi

echo ""
echo "🎉 Environnement de monitoring démarré!"
echo "======================================"
echo ""
echo "🌐 Accès aux interfaces:"
echo "   📊 Grafana: http://localhost:3000"
echo "      👤 Username: admin"
echo "      🔑 Password: admin123"
echo "   📈 Prometheus: http://localhost:9090"
echo "   🔧 API Métriques: http://localhost:8000/metrics"
echo ""
echo "🎯 Instructions:"
echo "   1. 📂 Ouvrez et exécutez le notebook: partie-5.ipynb"
echo "   2. 🌐 Allez sur Grafana: http://localhost:3000"
echo "   3. 📊 Le dashboard '🏦 Dashboard Anti-Fraude Fluzz' sera disponible"
echo "   4. ⚡ Les 3 métriques se mettront à jour automatiquement:"
echo "      • Taux de détection des fraudes"
echo "      • Taux de faux positifs"
echo "      • Score de dérive des données"
echo ""
echo "📋 Métriques Prometheus disponibles:"
echo "   • fraud_detection_rate"
echo "   • fraud_false_positive_rate"
echo "   • fraud_data_drift_score"
echo ""
echo "⚠️  Pour arrêter l'environnement:"
echo "   docker-compose down"
echo ""
echo "🚀 Statut: Partie 5 prête pour démonstration!"
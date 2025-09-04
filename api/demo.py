#!/usr/bin/env python3
"""
Script de démonstration de l'API Fraud Detection
Utilise l'API déployée localement pour montrer les fonctionnalités
"""

import requests
import json
import time
import random

API_BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test des différents endpoints de l'API"""
    
    print("🚀 Démonstration de l'API Fraud Detection - Banque Fluzz")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. 🏥 Test Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   📊 Modèle chargé: {data['model_loaded']}")
            print(f"   🔧 Scaler chargé: {data['scaler_loaded']}")
            print(f"   📅 Version: {data['version']}")
        else:
            print(f"   ❌ Erreur: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connexion échouée: {e}")
        return False
    
    # Test 2: Info endpoint
    print("\n2. ℹ️  Test Informations API")
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   📱 API: {data['api']['name']}")
            print(f"   🤖 Modèle: {data['model']['type']}")
            print(f"   📊 Features: {data['model']['features_count']}")
        else:
            print(f"   ❌ Erreur: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Test 3: Prédiction normale (transaction légitime)
    print("\n3. 💳 Test Transaction Normale")
    normal_transaction = {
        "features": [0.1, 0.2, -0.1, 0.5, -0.3, 0.0, 0.1, -0.2, 0.3, -0.1,
                    0.0, 0.2, -0.4, 0.1, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2,
                    0.3, -0.1, 0.0, 0.4, -0.2, 0.1, 0.0, -0.3, 0.2, 50.0],
        "transaction_id": "demo_normal_001"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/predict", 
            json=normal_transaction,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   📝 Transaction ID: {data['transaction_id']}")
            print(f"   🚨 Fraude détectée: {data['is_fraud']}")
            print(f"   📈 Probabilité fraude: {data['fraud_probability']:.3f}")
            print(f"   🎯 Confiance: {data['confidence_score']:.3f}")
        else:
            print(f"   ❌ Erreur: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Test 4: Prédiction frauduleuse (transaction suspecte)
    print("\n4. 🚨 Test Transaction Suspecte")
    fraud_transaction = {
        "features": [5.0, -8.2, 12.3, -15.7, 20.1, -25.4, 30.2, -35.8, 40.1, -45.6,
                    50.2, -55.9, 60.4, -65.1, 70.8, -75.3, 80.2, -85.7, 90.1, -95.4,
                    100.2, -105.8, 110.3, -115.9, 120.4, -125.1, 130.8, -135.3, 140.2, 5000.0],
        "transaction_id": "demo_fraud_001"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/predict", 
            json=fraud_transaction,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   📝 Transaction ID: {data['transaction_id']}")
            print(f"   🚨 Fraude détectée: {data['is_fraud']}")
            print(f"   📈 Probabilité fraude: {data['fraud_probability']:.3f}")
            print(f"   🎯 Confiance: {data['confidence_score']:.3f}")
        else:
            print(f"   ❌ Erreur: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Test 5: Prédictions en lot
    print("\n5. 📦 Test Prédictions en Lot")
    batch_transactions = {
        "transactions": [
            {
                "features": [random.uniform(-2, 2) for _ in range(29)] + [random.uniform(10, 1000)],
                "transaction_id": f"batch_txn_{i:03d}"
            }
            for i in range(5)
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch_predict", 
            json=batch_transactions,
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Nombre de prédictions: {data['count']}")
            print("   📋 Résultats:")
            for pred in data['predictions']:
                fraud_status = "🚨 FRAUDE" if pred['is_fraud'] else "✅ OK"
                print(f"     - {pred['transaction_id']}: {fraud_status} ({pred['fraud_probability']:.3f})")
        else:
            print(f"   ❌ Erreur: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Test 6: Métriques Prometheus
    print("\n6. 📊 Test Métriques Prometheus")
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            fraud_metrics = [line for line in metrics.split('\n') if 'fraud_' in line and not line.startswith('#')]
            print("   📈 Métriques disponibles:")
            for metric in fraud_metrics[:5]:  # Afficher seulement les 5 premières
                if metric.strip():
                    print(f"     - {metric}")
        else:
            print(f"   ❌ Erreur: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Démonstration terminée!")
    print("🌐 Documentation interactive disponible sur: http://localhost:8000/docs")
    print("📊 Métriques Prometheus: http://localhost:8000/metrics")
    
    return True

def performance_test():
    """Test de performance simple"""
    print("\n🚀 Test de Performance")
    print("-" * 40)
    
    # Générer des transactions de test
    test_transactions = []
    for i in range(20):
        features = [random.uniform(-3, 3) for _ in range(29)] + [random.uniform(1, 10000)]
        test_transactions.append({
            "features": features,
            "transaction_id": f"perf_test_{i:03d}"
        })
    
    start_time = time.time()
    successful_predictions = 0
    
    for transaction in test_transactions:
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/predict", 
                json=transaction,
                timeout=5
            )
            if response.status_code == 200:
                successful_predictions += 1
        except Exception as e:
            print(f"   ⚠️  Erreur sur transaction {transaction['transaction_id']}: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   📊 Transactions testées: {len(test_transactions)}")
    print(f"   ✅ Prédictions réussies: {successful_predictions}")
    print(f"   ⏱️  Temps total: {total_time:.2f}s")
    print(f"   🚀 Débit: {successful_predictions/total_time:.1f} req/sec")
    print(f"   ⚡ Latence moyenne: {(total_time/successful_predictions)*1000:.1f}ms")

if __name__ == "__main__":
    print("Démarrez d'abord l'API avec:")
    print("  cd adeline/api")
    print("  uvicorn app.main:app --host 127.0.0.1 --port 8000")
    print("")
    
    input("Appuyez sur Entrée quand l'API est démarrée...")
    
    # Test principal
    if test_api_endpoints():
        print("\n🎯 Voulez-vous lancer le test de performance? (y/N)")
        choice = input().lower()
        if choice == 'y':
            performance_test()
    
    print("\n🏁 Démonstration complète terminée!")
    print("   L'API est prête pour la production avec Kubernetes!")
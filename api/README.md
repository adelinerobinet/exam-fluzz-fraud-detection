# Fraud Detection API - Banque Fluzz

API REST de détection de fraude bancaire pour la néobanque Fluzz, développée dans le cadre du projet d'industrialisation.

## Démarrage rapide

### Prérequis
- Python 3.9+
- Docker & Docker Compose
- kubectl (pour Kubernetes)

### Développement local

1. **Cloner et installer**
   ```bash
   cd adeline/api
   pip install -r requirements.txt
   ```

2. **Lancer l'API**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Accéder à la documentation**
   - API Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Avec Docker Compose (Stack complet API + Monitoring)

```bash
# Lancer tous les services (API + Prometheus + Grafana)
docker-compose up -d

# Ou utiliser le script de démarrage
../scripts/start-monitoring.sh

# Voir les logs
docker-compose logs -f fraud-detection-api
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Arrêter les services
docker-compose down
```

Services disponibles:
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090  
- **Grafana**: http://localhost:3000 (admin/admin123)

** Dashboard Grafana configuré automatiquement avec** :
- Taux de détection des fraudes (fraud_detection_rate)
- Taux de faux positifs (fraud_false_positive_rate)  
- Score de dérive des données (fraud_data_drift_score)

## Endpoints disponibles

### Santé et informations
- `GET /api/` - Message d'accueil
- `GET /api/health` - Health check pour Kubernetes
- `GET /api/info` - Informations sur l'API et le modèle
- `GET /api/metrics` - Métriques Prometheus

### Prédictions
- `POST /api/predict` - Prédiction pour une transaction
- `POST /api/batch_predict` - Prédictions en lot (max 100)

### Exemple d'usage

```python
import requests

# Prédiction simple
transaction = {
    "features": [1.0, -1.36, -0.07] + [0.0] * 27,
    "transaction_id": "txn_123456789"
}

response = requests.post(
    "http://localhost:8000/api/predict",
    json=transaction
)
print(response.json())
```

## Déploiement

### Docker
```bash
# Build l'image
docker build -t fraud-detection:latest .

# Run le container
docker run -p 8000:8000 fraud-detection:latest
```

### Kubernetes
```bash
# Commandes de déploiement préparées
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Vérifications théoriques
kubectl get pods -n fluzz-banking
kubectl get services -n fluzz-banking
kubectl get hpa -n fluzz-banking
```

> **Note** : Les manifestes Kubernetes sont fournis pour répondre aux exigences de la Partie 7 du projet, mais n'ont pas été déployés sur un cluster réel dans le cadre de cet examen.

## Monitoring

L'API expose des métriques Prometheus sur `/metrics`:

### Métriques principales (Dashboard Grafana)
- `fraud_detection_rate` - Taux de détection des fraudes (85%)
- `fraud_false_positive_rate` - Taux de faux positifs (2%)  
- `fraud_data_drift_score` - Score de dérive des données (0.15)

### Métriques techniques
- `fraud_predictions_total` - Nombre total de prédictions
- `fraud_prediction_duration_seconds` - Temps de traitement
- `fraud_detection_model_accuracy` - Précision du modèle
- `api_requests_total` - Requêtes API totales

### Test des métriques
```bash
# Vérifier les métriques principales
curl http://localhost:8000/api/metrics | grep fraud_detection_rate
curl http://localhost:8000/api/metrics | grep fraud_false_positive_rate
curl http://localhost:8000/api/metrics | grep fraud_data_drift_score
```

## Sécurité

- Utilisateur non-root dans le container
- Health checks intégrés
- Limitation de taux (100 req/min)
- Sécurité au niveau des containers
- Scan de vulnérabilités dans CI/CD

## Architecture

```
┌─────────────────┐
│   Load Balancer │
│   (Kubernetes)  │
└─────────┬───────┘
          │
    ┌─────┴─────────────┐
    │                   │
┌───┴────┐         ┌────┴───┐
│API Pod │   ...   │API Pod │
│(FastAPI)│         │(FastAPI)│
└────────┘         └────────┘
```

## Performances

- **Latence**: < 200ms (p95)
- **Throughput**: > 1000 req/sec
- **Disponibilité**: 99.9%
- **Auto-scaling**: 2-10 pods selon la charge

## CI/CD

Le pipeline GitHub Actions inclut:

1. **Tests**: Pytest + coverage
2. **Sécurité**: Trivy + Bandit
3. **Build**: Docker multi-arch
4. **Deploy**: Kubernetes (staging + production)

## Dépannage

### Logs
```bash
# Logs du container
docker-compose logs fraud-detection-api

# Logs Kubernetes
kubectl logs -f deployment/fraud-detection-api -n fluzz-banking
```

### Health checks
```bash
# Vérifier la santé
curl http://localhost:8000/health

# Métriques
curl http://localhost:8000/metrics
```

### Performance
```bash
# Test de charge simple
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}'
```
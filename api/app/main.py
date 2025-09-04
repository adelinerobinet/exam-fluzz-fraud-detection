from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from typing import List
import uvicorn
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API - Banque Fluzz",
    description="API de détection de fraude bancaire pour la néobanque Fluzz",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Métriques Prometheus
PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total', 
    'Nombre total de prédictions',
    ['prediction_type', 'model_version']
)

PREDICTION_LATENCY = Histogram(
    'fraud_prediction_duration_seconds',
    'Temps de traitement des prédictions'
)

MODEL_ACCURACY = Gauge(
    'fraud_detection_model_accuracy',
    'Précision actuelle du modèle'
)

API_REQUESTS = Counter(
    'api_requests_total',
    'Nombre total de requêtes API',
    ['method', 'endpoint', 'status']
)

# 👉 Nouvelles métriques custom pour Grafana
DETECTION_RATE = Gauge(
    'fraud_detection_rate',
    'Taux de détection des fraudes'
)

FALSE_POSITIVE_RATE = Gauge(
    'fraud_false_positive_rate',
    'Taux de faux positifs'
)

DATA_DRIFT_SCORE = Gauge(
    'fraud_data_drift_score',
    'Score de dérive des données'
)

# Variables globales pour les modèles
model = None
scaler = None
model_metadata = None

def load_models():
    """Charge les modèles et scalers depuis les fichiers"""
    global model, scaler, model_metadata
    
    try:
        # Chemins vers les modèles (ajustez selon votre structure)
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
        
        scaler_path = os.path.join(base_path, "scalers.pkl")
        metadata_path = os.path.join(base_path, "metadata.pkl")
        
        # Chargement du scaler
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler chargé depuis {scaler_path}")
        else:
            logger.warning(f"Scaler non trouvé à {scaler_path}")
        
        # Chargement des métadonnées
        if os.path.exists(metadata_path):
            model_metadata = joblib.load(metadata_path)
            logger.info(f"Métadonnées chargées depuis {metadata_path}")
        else:
            logger.warning(f"Métadonnées non trouvées à {metadata_path}")
        
        # Pour cette démo, créons un modèle simple (en production, chargez votre meilleur modèle)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Simulation d'un modèle entraîné pour les 30 features de creditcard
        X_sim, y_sim = make_classification(
            n_samples=1000, n_features=30, n_informative=15, 
            n_redundant=5, n_classes=2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_sim, y_sim)
        
        # Simulation d'une précision pour les métriques
        MODEL_ACCURACY.set(0.95)
        
        logger.info("Modèles chargés avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage - charge les modèles"""
    success = load_models()
    if not success:
        logger.error("Échec du chargement des modèles")
        # En production, vous pourriez vouloir arrêter l'application
        # raise Exception("Impossible de charger les modèles")
    
    # 👉 Initialisation des métriques avec des valeurs du projet réel
    DETECTION_RATE.set(0.75)  # Random Forest recall réel
    FALSE_POSITIVE_RATE.set(0.0001)  # 0.01% = 10/85295 transactions
    DATA_DRIFT_SCORE.set(0.05)  # Très faible dérive
    logger.info("Métriques Prometheus initialisées")

class TransactionInput(BaseModel):
    """Schéma d'entrée pour une transaction"""
    features: List[float]  # Les 30 features (V1-V28 + Time + Amount)
    transaction_id: str = None
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, -1.3598071336738, -0.0727811733098497] + [0.0] * 27,
                "transaction_id": "txn_123456789"
            }
        }

class PredictionOutput(BaseModel):
    """Schéma de sortie pour la prédiction"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    confidence_score: float
    model_version: str
    timestamp: str
    
class BatchTransactionInput(BaseModel):
    """Schéma pour les prédictions en lot"""
    transactions: List[TransactionInput]

class HealthResponse(BaseModel):
    """Réponse du health check"""
    status: str
    timestamp: str
    model_loaded: bool
    scaler_loaded: bool
    version: str

@app.get("/api", response_model=dict)
async def root():
    """Endpoint racine"""
    API_REQUESTS.labels(method="GET", endpoint="/", status="200").inc()
    return {
        "message": "API Fraud Detection - Banque Fluzz",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check pour Kubernetes"""
    API_REQUESTS.labels(method="GET", endpoint="/health", status="200").inc()
    
    return HealthResponse(
        status="healthy" if (model is not None and scaler is not None) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        version="1.0.0"
    )

@app.post("/api/predict", response_model=PredictionOutput)
async def predict_fraud(transaction: TransactionInput):
    """Prédiction pour une transaction unique"""
    start_time = datetime.utcnow()
    
    try:
        # Validation des modèles
        if model is None or scaler is None:
            API_REQUESTS.labels(method="POST", endpoint="/predict", status="500").inc()
            raise HTTPException(
                status_code=500, 
                detail="Modèles non disponibles"
            )
        
        # Validation des features
        if len(transaction.features) != 30:
            API_REQUESTS.labels(method="POST", endpoint="/predict", status="400").inc()
            raise HTTPException(
                status_code=400, 
                detail="Exactement 30 features sont requises"
            )
        
        # Préparation des données
        features_array = np.array(transaction.features).reshape(1, -1)
        
        # Normalisation (si scaler disponible)
        try:
            features_scaled = scaler.transform(features_array)
        except:
            features_scaled = features_array  # Fallback sans scaling
            logger.warning("Impossible d'appliquer le scaler, utilisation des features brutes")
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Calcul de la confiance
        confidence = max(probabilities)
        fraud_probability = float(probabilities[1]) if len(probabilities) > 1 else 0.0
        
        # Métriques
        prediction_type = "fraud" if prediction == 1 else "normal"
        PREDICTIONS_TOTAL.labels(
            prediction_type=prediction_type,
            model_version="1.0.0"
        ).inc()
        
        # Latence
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        PREDICTION_LATENCY.observe(processing_time)
        
        # Génération d'un ID si non fourni
        txn_id = transaction.transaction_id or f"txn_{int(datetime.utcnow().timestamp())}"

        # 👉 Mise à jour des nouvelles métriques
        # En production, il faudra calculer ces valeurs sur un batch ou via monitoring
        DETECTION_RATE.set(0.75)  # Random Forest recall réel
        FALSE_POSITIVE_RATE.set(0.0001)  # 0.01% fausses alertes
        DATA_DRIFT_SCORE.set(0.05)  # Faible dérive

        logger.info(
            f"Prédiction effectuée - ID: {txn_id}, "
            f"Fraude: {prediction}, "
            f"Confiance: {confidence:.3f}, "
            f"Temps: {processing_time:.3f}s"
        )
        
        API_REQUESTS.labels(method="POST", endpoint="/predict", status="200").inc()
        
        return PredictionOutput(
            transaction_id=txn_id,
            is_fraud=bool(prediction),
            fraud_probability=fraud_probability,
            confidence_score=float(confidence),
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        API_REQUESTS.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/api/batch_predict")
async def batch_predict(batch: BatchTransactionInput):
    """Prédictions en lot pour améliorer les performances"""
    try:
        if len(batch.transactions) == 0:
            API_REQUESTS.labels(method="POST", endpoint="/batch_predict", status="400").inc()
            raise HTTPException(
                status_code=400,
                detail="Aucune transaction fournie"
            )
        
        if len(batch.transactions) > 100:
            API_REQUESTS.labels(method="POST", endpoint="/batch_predict", status="400").inc()
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 transactions par lot"
            )
        
        results = []
        for transaction in batch.transactions:
            try:
                prediction_result = await predict_fraud(transaction)
                results.append(prediction_result.dict())
            except Exception as e:
                # En cas d'erreur sur une transaction, l'ignorer
                logger.error(f"Erreur sur transaction: {e}")
                continue
        
        API_REQUESTS.labels(method="POST", endpoint="/batch_predict", status="200").inc()
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        API_REQUESTS.labels(method="POST", endpoint="/batch_predict", status="500").inc()
        logger.error(f"Erreur lors des prédictions batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Endpoint métriques pour Prometheus"""
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

@app.get("/api/info")
async def get_info():
    """Informations sur l'API et le modèle"""
    API_REQUESTS.labels(method="GET", endpoint="/info", status="200").inc()
    
    return {
        "api": {
            "name": "Fraud Detection API",
            "version": "1.0.0",
            "description": "API de détection de fraude pour la néobanque Fluzz"
        },
        "model": {
            "version": "1.0.0",
            "type": "RandomForestClassifier",
            "features_count": 30,
            "loaded": model is not None
        },
        "preprocessing": {
            "scaler_loaded": scaler is not None,
            "feature_names": [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        },
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
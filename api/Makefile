# Makefile pour l'API Fraud Detection

.PHONY: help install test lint format build run clean docker-build docker-run k8s-deploy

# Variables
APP_NAME = fraud-detection
VERSION = 1.0.0
REGISTRY = ghcr.io/fluzz
IMAGE_NAME = $(REGISTRY)/$(APP_NAME):$(VERSION)

# Couleurs pour l'output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Affiche l'aide
	@echo "$(GREEN)Fraud Detection API - Commandes disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install les dépendances Python
	@echo "$(GREEN)Installation des dépendances...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 httpx

test: ## Lance les tests
	@echo "$(GREEN)Lancement des tests...$(NC)"
	pytest tests/ -v --cov=app --cov-report=term-missing

test-coverage: ## Lance les tests avec rapport de couverture HTML
	@echo "$(GREEN)Tests avec couverture HTML...$(NC)"
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
	@echo "$(YELLOW)Rapport disponible dans htmlcov/index.html$(NC)"

lint: ## Vérifie le code avec flake8
	@echo "$(GREEN)Vérification du code...$(NC)"
	flake8 app --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 app --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format: ## Formate le code avec black
	@echo "$(GREEN)Formatage du code...$(NC)"
	black app/ tests/

format-check: ## Vérifie le formatage sans modifier
	@echo "$(GREEN)Vérification du formatage...$(NC)"
	black --check --diff app/ tests/

security: ## Scan de sécurité avec bandit
	@echo "$(GREEN)Scan de sécurité...$(NC)"
	pip install bandit[toml]
	bandit -r app/ -f json -o bandit-report.json || true
	bandit -r app/ || true

run: ## Lance l'API en développement
	@echo "$(GREEN)Démarrage de l'API...$(NC)"
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Lance l'API en mode production
	@echo "$(GREEN)Démarrage en mode production...$(NC)"
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-build: ## Build l'image Docker
	@echo "$(GREEN)Build de l'image Docker...$(NC)"
	docker build -t $(APP_NAME):$(VERSION) .
	docker build -t $(APP_NAME):latest .

docker-run: ## Lance le container Docker
	@echo "$(GREEN)Lancement du container...$(NC)"
	docker run -p 8000:8000 --name $(APP_NAME) $(APP_NAME):$(VERSION)

docker-stop: ## Arrête et supprime le container
	@echo "$(GREEN)Arrêt du container...$(NC)"
	docker stop $(APP_NAME) || true
	docker rm $(APP_NAME) || true

docker-compose-up: ## Lance avec docker-compose
	@echo "$(GREEN)Lancement avec docker-compose...$(NC)"
	docker-compose up -d

docker-compose-down: ## Arrête docker-compose
	@echo "$(GREEN)Arrêt docker-compose...$(NC)"
	docker-compose down

docker-compose-logs: ## Affiche les logs docker-compose
	docker-compose logs -f fraud-detection-api

build-multi-arch: ## Build multi-architecture pour production
	@echo "$(GREEN)Build multi-architecture...$(NC)"
	docker buildx create --use --name multiarch || true
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE_NAME) --push .

k8s-deploy: ## Déploie sur Kubernetes
	@echo "$(GREEN)Déploiement sur Kubernetes...$(NC)"
	kubectl apply -f k8s/
	kubectl rollout status deployment/fraud-detection-api -n fluzz-banking

k8s-status: ## Affiche le statut Kubernetes
	@echo "$(GREEN)Statut Kubernetes...$(NC)"
	kubectl get pods -n fluzz-banking -l app=fraud-detection
	kubectl get services -n fluzz-banking -l app=fraud-detection
	kubectl get ingress -n fluzz-banking

k8s-logs: ## Affiche les logs Kubernetes
	kubectl logs -f deployment/fraud-detection-api -n fluzz-banking

k8s-delete: ## Supprime le déploiement Kubernetes
	@echo "$(RED)Suppression du déploiement...$(NC)"
	kubectl delete -f k8s/

health-check: ## Vérifie la santé de l'API
	@echo "$(GREEN)Health check...$(NC)"
	curl -f http://localhost:8000/api/health || echo "$(RED)API non accessible$(NC)"
	curl -f http://localhost:8000/ || echo "$(RED)API non accessible$(NC)"

load-test: ## Test de charge simple
	@echo "$(GREEN)Test de charge...$(NC)"
	for i in {1..10}; do \
		curl -X POST http://localhost:8000/api/predict \
		-H "Content-Type: application/json" \
		-d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}' & \
	done; \
	wait; \
	echo "$(GREEN)Test de charge terminé$(NC)"

metrics: ## Affiche les métriques Prometheus
	@echo "$(GREEN)Métriques Prometheus:$(NC)"
	curl -s http://localhost:8000/metrics | grep -E "^(fraud_|api_)" | head -20

clean: ## Nettoie les fichiers temporaires
	@echo "$(GREEN)Nettoyage...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf bandit-report.json
	docker system prune -f

all-checks: format-check lint security test ## Lance tous les checks de qualité
	@echo "$(GREEN)✅ Tous les checks de qualité passés!$(NC)"

dev-setup: install ## Setup environnement de développement
	@echo "$(GREEN)✅ Environnement de développement configuré!$(NC)"
	@echo "$(YELLOW)Commandes utiles:$(NC)"
	@echo "  make run          # Lance l'API"
	@echo "  make test         # Lance les tests" 
	@echo "  make docker-compose-up  # Lance avec monitoring"

# Commande par défaut
.DEFAULT_GOAL := help
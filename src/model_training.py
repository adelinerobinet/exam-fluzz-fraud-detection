"""
Module d'Entraînement des Modèles ML pour Détection de Fraude
=============================================================

Ce module contient toutes les fonctions pour l'entraînement, l'optimisation
et l'évaluation des modèles de machine learning pour la détection de fraude.

Fonctionnalités principales:
- Entraînement de multiples algorithmes ML
- Optimisation des hyperparamètres 
- Évaluation et comparaison des modèles
- Sauvegarde et chargement des modèles

Auteur: Équipe Data Science - Néobanque Fluzz  
Version: 1.0
Date: Août 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, precision_score, recall_score
)
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .utils import setup_logging, save_model_artifacts

# Configuration du logger
logger = setup_logging("INFO")


class FraudModelTrainer:
    """
    Classe principale pour l'entraînement des modèles de détection de fraude.
    
    Cette classe encapsule l'entraînement de multiples algorithmes,
    l'optimisation des hyperparamètres, et l'évaluation comparative.
    
    Attributes:
        models (Dict): Dictionnaire des modèles disponibles
        best_model: Meilleur modèle après comparaison
        results (Dict): Résultats d'évaluation de tous les modèles
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le trainer avec la configuration des modèles.
        
        Args:
            config (Dict, optional): Configuration des modèles et hyperparamètres.
                                   Si None, utilise la configuration par défaut.
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.best_model = None
        self.results = {}
        self.logger = setup_logging("INFO")
        
    def _get_default_config(self) -> Dict:
        """Configuration par défaut des modèles."""
        return {
            'models': {
                'logistic_regression': {
                    'class': LogisticRegression,
                    'params': {
                        'C': [0.01, 0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear'],
                        'max_iter': [1000]
                    },
                    'enabled': True
                },
                'random_forest': {
                    'class': RandomForestClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                        'class_weight': ['balanced']
                    },
                    'enabled': True
                },
                'gradient_boosting': {
                    'class': GradientBoostingClassifier,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]
                    },
                    'enabled': True
                }
            },
            'cv_folds': 5,
            'scoring': 'f1',
            'n_jobs': -1
        }
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Entraîne tous les modèles configurés et les compare.
        
        Args:
            X_train (pd.DataFrame): Features d'entraînement
            y_train (pd.Series): Target d'entraînement  
            X_val (pd.DataFrame, optional): Features de validation
            y_val (pd.Series, optional): Target de validation
            
        Returns:
            Dict[str, Any]: Résultats de comparaison des modèles
        """
        logger.info("Début de l'entraînement de tous les modèles")
        
        results = {}
        
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                logger.info(f"Modèle {model_name} désactivé, passage au suivant")
                continue
                
            logger.info(f"Entraînement du modèle: {model_name}")
            
            try:
                # Entraîner le modèle avec optimisation hyperparamètres
                best_model, cv_results = self._train_single_model(
                    model_name, model_config, X_train, y_train
                )
                
                # Évaluer sur validation si fournie
                val_scores = {}
                if X_val is not None and y_val is not None:
                    val_scores = self._evaluate_model(best_model, X_val, y_val)
                
                results[model_name] = {
                    'model': best_model,
                    'cv_scores': cv_results,
                    'validation_scores': val_scores,
                    'best_params': best_model.best_params_ if hasattr(best_model, 'best_params_') else {}
                }
                
                logger.info(f"✅ {model_name} terminé - F1: {cv_results.get('f1', 'N/A'):.3f}")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement de {model_name}: {str(e)}")
                continue
        
        self.results = results
        
        # Sélectionner le meilleur modèle
        self.best_model = self._select_best_model(results)
        
        logger.info(f"🏆 Meilleur modèle: {self.best_model['name']} (F1: {self.best_model['score']:.3f})")
        
        return results
    
    def _train_single_model(
        self,
        model_name: str,
        model_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[Any, Dict]:
        """Entraîne un seul modèle avec optimisation des hyperparamètres."""
        
        # Initialiser le modèle
        model_class = model_config['class']
        base_model = model_class(random_state=42)
        
        # Configuration de la validation croisée
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=42
        )
        
        # Grid Search pour optimisation hyperparamètres
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=model_config['params'],
            cv=cv,
            scoring=self.config['scoring'],
            n_jobs=self.config['n_jobs'],
            verbose=0
        )
        
        # Entraînement
        grid_search.fit(X_train, y_train)
        
        # Évaluation cross-validation du meilleur modèle
        cv_scores = cross_val_score(
            grid_search.best_estimator_,
            X_train,
            y_train,
            cv=cv,
            scoring=self.config['scoring']
        )
        
        cv_results = {
            'f1': cv_scores.mean(),
            'f1_std': cv_scores.std(),
            'scores_detail': cv_scores
        }
        
        return grid_search, cv_results
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Évalue un modèle sur un ensemble de test."""
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def _select_best_model(self, results: Dict) -> Dict:
        """Sélectionne le meilleur modèle basé sur le F1-score de validation croisée."""
        
        best_score = -1
        best_model_name = None
        
        for model_name, model_results in results.items():
            f1_score = model_results['cv_scores']['f1']
            
            if f1_score > best_score:
                best_score = f1_score
                best_model_name = model_name
        
        if best_model_name:
            return {
                'name': best_model_name,
                'model': results[best_model_name]['model'],
                'score': best_score
            }
        
        return None
    
    def generate_model_comparison_report(self) -> str:
        """Génère un rapport de comparaison des modèles."""
        
        if not self.results:
            return "Aucun modèle entraîné pour le moment."
        
        report_lines = [
            "# Rapport de Comparaison des Modèles ML",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Résultats Cross-Validation",
            "",
            "| Modèle | F1-Score | Std | Meilleur |",
            "|--------|----------|-----|----------|"
        ]
        
        for model_name, model_results in self.results.items():
            cv_results = model_results['cv_scores']
            f1_mean = cv_results['f1']
            f1_std = cv_results['f1_std']
            is_best = "🏆" if (self.best_model and self.best_model['name'] == model_name) else ""
            
            report_lines.append(f"| {model_name} | {f1_mean:.3f} | {f1_std:.3f} | {is_best} |")
        
        # Ajouter les paramètres du meilleur modèle
        if self.best_model:
            report_lines.extend([
                "",
                "## Meilleur Modèle - Paramètres Optimaux",
                "",
                f"**Modèle:** {self.best_model['name']}",
                f"**Score F1:** {self.best_model['score']:.3f}",
                "",
                "**Hyperparamètres:**"
            ])
            
            model_results = self.results[self.best_model['name']]
            best_params = model_results['best_params']
            
            for param, value in best_params.items():
                report_lines.append(f"- **{param}:** {value}")
        
        return "\n".join(report_lines)
    
    def save_best_model(self, output_dir: str, model_name: Optional[str] = None) -> Dict[str, str]:
        """
        Sauvegarde le meilleur modèle et ses métadonnées.
        
        Args:
            output_dir (str): Répertoire de destination
            model_name (str, optional): Nom personnalisé pour les fichiers
            
        Returns:
            Dict[str, str]: Chemins des fichiers sauvegardés
        """
        if not self.best_model:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        # Nom par défaut
        if not model_name:
            model_name = f"fraud_model_{self.best_model['name']}"
        
        # Préparer les métadonnées
        metadata = {
            'model_type': self.best_model['name'],
            'f1_score': self.best_model['score'],
            'training_date': datetime.now().isoformat(),
            'best_params': self.results[self.best_model['name']]['best_params'],
            'cv_results': self.results[self.best_model['name']]['cv_scores']
        }
        
        # Artefacts à sauvegarder
        artifacts = {
            'model': self.best_model['model'],
            'metadata': metadata
        }
        
        # Sauvegarder
        saved_paths = save_model_artifacts(artifacts, output_dir, model_name)
        
        logger.info(f"✅ Modèle sauvegardé dans {output_dir}")
        return saved_paths


def compare_models_performance(
    models_results: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare les performances de plusieurs modèles sur un ensemble de test.
    
    Args:
        models_results (Dict[str, Any]): Résultats des modèles entraînés
        X_test (pd.DataFrame): Features de test
        y_test (pd.Series): Target de test
        
    Returns:
        pd.DataFrame: Tableau comparatif des performances
    """
    comparison_data = []
    
    for model_name, model_info in models_results.items():
        model = model_info['model']
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        metrics = {
            'Model': model_name,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred), 
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df.sort_values('F1-Score', ascending=False)


def calculate_business_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.Series,
    fraud_cost: float = 100.0,
    investigation_cost: float = 10.0
) -> Dict[str, float]:
    """
    Calcule les métriques business pour l'évaluation du modèle.
    
    Args:
        y_true (pd.Series): Vraies labels
        y_pred (pd.Series): Prédictions
        y_pred_proba (pd.Series): Probabilités de fraude
        fraud_cost (float): Coût d'une fraude non détectée
        investigation_cost (float): Coût d'investigation d'une alerte
        
    Returns:
        Dict[str, float]: Métriques business
    """
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Coûts
    cost_missed_frauds = fn * fraud_cost  # Fraudes ratées
    cost_investigations = fp * investigation_cost  # Fausses alertes
    total_cost = cost_missed_frauds + cost_investigations
    
    # Bénéfices
    prevented_frauds = tp * fraud_cost
    net_benefit = prevented_frauds - total_cost
    
    # ROI
    investigation_total_cost = (tp + fp) * investigation_cost
    roi = (prevented_frauds - investigation_total_cost) / max(investigation_total_cost, 1)
    
    return {
        'total_cost': total_cost,
        'cost_missed_frauds': cost_missed_frauds,
        'cost_false_alerts': cost_investigations,
        'prevented_fraud_value': prevented_frauds,
        'net_benefit': net_benefit,
        'roi': roi,
        'cost_per_transaction': total_cost / len(y_true)
    }


if __name__ == "__main__":
    # Test du module avec données factices
    logger.info("Test du module model_training")
    
    from sklearn.datasets import make_classification
    
    # Créer des données de test déséquilibrées
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.995, 0.005],  # Très déséquilibré
        random_state=42
    )
    
    X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_train = pd.Series(y)
    
    # Tester le trainer
    trainer = FraudModelTrainer()
    
    try:
        # Entraîner les modèles (configuration réduite pour test rapide)
        test_config = {
            'models': {
                'logistic_regression': {
                    'class': LogisticRegression,
                    'params': {'C': [0.1, 1], 'max_iter': [1000]},
                    'enabled': True
                },
                'random_forest': {
                    'class': RandomForestClassifier,
                    'params': {'n_estimators': [50], 'max_depth': [5]},
                    'enabled': True
                }
            },
            'cv_folds': 3,
            'scoring': 'f1',
            'n_jobs': 1
        }
        
        trainer.config = test_config
        results = trainer.train_all_models(X_train, y_train)
        
        logger.info(f"✅ {len(results)} modèles entraînés")
        logger.info(f"✅ Meilleur modèle: {trainer.best_model['name'] if trainer.best_model else 'Aucun'}")
        
        # Générer rapport
        report = trainer.generate_model_comparison_report()
        logger.info("✅ Rapport généré")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {str(e)}")
        raise
    
    logger.info("🎉 Tests du module model_training terminés avec succès!")
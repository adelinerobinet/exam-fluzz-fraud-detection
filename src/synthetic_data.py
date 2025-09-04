"""
Module de Génération de Données Synthétiques pour Détection de Fraude
======================================================================

Ce module contient toutes les fonctions pour générer des données synthétiques
afin de rééquilibrer le dataset de fraude bancaire extrêmement déséquilibré.

Fonctionnalités principales:
- Génération SMOTE et variantes
- Génération SDV (Synthetic Data Vault)
- Évaluation de la qualité des données synthétiques  
- Sélection de la meilleure méthode

Auteur: Équipe Data Science - Néobanque Fluzz
Version: 1.0
Date: Août 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import conditionnel de SDV (peut ne pas être installé)
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

from .utils import setup_logging, calculate_class_balance_metrics

# Configuration du logger
logger = setup_logging("INFO")


class SyntheticDataGenerator:
    """
    Classe principale pour la génération de données synthétiques.
    
    Cette classe encapsule différentes méthodes de génération de données
    synthétiques et permet de comparer leur qualité pour choisir la meilleure.
    
    Attributes:
        methods (Dict): Dictionnaire des méthodes disponibles
        generated_data (Dict): Données générées par chaque méthode
        quality_scores (Dict): Scores de qualité pour chaque méthode
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le générateur avec la configuration.
        
        Args:
            config (Dict, optional): Configuration des méthodes de génération.
                                   Si None, utilise la configuration par défaut.
        """
        self.config = config or self._get_default_config()
        self.methods = {}
        self.generated_data = {}
        self.quality_scores = {}
        self.logger = setup_logging("INFO")
        
    def _get_default_config(self) -> Dict:
        """Configuration par défaut des méthodes de génération."""
        return {
            'target_samples': 2000,
            'methods': {
                'smote': {
                    'enabled': True,
                    'params': {
                        'sampling_strategy': 0.1,  # Ratio final minority/majority
                        'k_neighbors': 5,
                        'random_state': 42
                    }
                },
                'borderline_smote': {
                    'enabled': True,
                    'params': {
                        'sampling_strategy': 0.1,
                        'k_neighbors': 5,
                        'kind': 'borderline-1',
                        'random_state': 42
                    }
                },
                'sdv': {
                    'enabled': SDV_AVAILABLE,
                    'params': {
                        'num_samples': 2000,
                        'enforce_min_max_values': True,
                        'enforce_rounding': False
                    }
                }
            },
            'evaluation': {
                'test_size': 0.3,
                'discriminator_model': RandomForestClassifier,
                'quality_threshold': 0.8
            }
        }
    
    def generate_all_methods(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_column: str = 'Class'
    ) -> Dict[str, pd.DataFrame]:
        """
        Génère des données synthétiques avec toutes les méthodes configurées.
        
        Args:
            X (pd.DataFrame): Features originales
            y (pd.Series): Target originale
            target_column (str): Nom de la colonne target
            
        Returns:
            Dict[str, pd.DataFrame]: Données générées par chaque méthode
        """
        logger.info("Début de la génération de données synthétiques")
        
        results = {}
        
        # Analyser le déséquilibre initial
        balance_metrics = calculate_class_balance_metrics(y)
        logger.info(f"Déséquilibre initial: {balance_metrics['imbalance_ratio']:.1f}:1")
        
        for method_name, method_config in self.config['methods'].items():
            if not method_config.get('enabled', False):
                logger.info(f"Méthode {method_name} désactivée")
                continue
                
            logger.info(f"Génération avec {method_name}...")
            
            try:
                synthetic_df = self._generate_single_method(
                    method_name, method_config, X, y, target_column
                )
                
                if synthetic_df is not None:
                    results[method_name] = synthetic_df
                    
                    # Calculer le nouveau ratio
                    new_balance = calculate_class_balance_metrics(synthetic_df[target_column])
                    logger.info(f"✅ {method_name}: {len(synthetic_df)} échantillons, "
                              f"ratio {new_balance['imbalance_ratio']:.1f}:1")
                else:
                    logger.warning(f"⚠️ {method_name}: Génération échouée")
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors de la génération avec {method_name}: {str(e)}")
                continue
        
        self.generated_data = results
        return results
    
    def _generate_single_method(
        self,
        method_name: str,
        method_config: Dict,
        X: pd.DataFrame,
        y: pd.Series,
        target_column: str
    ) -> Optional[pd.DataFrame]:
        """Génère des données avec une méthode spécifique."""
        
        if method_name in ['smote', 'borderline_smote']:
            return self._generate_smote_variant(method_name, method_config, X, y, target_column)
        elif method_name == 'sdv':
            return self._generate_sdv(method_config, X, y, target_column)
        else:
            logger.error(f"Méthode inconnue: {method_name}")
            return None
    
    def _generate_smote_variant(
        self,
        method_name: str,
        method_config: Dict,
        X: pd.DataFrame,
        y: pd.Series,
        target_column: str
    ) -> pd.DataFrame:
        """Génère des données avec les variantes SMOTE."""
        
        # Sélectionner la classe SMOTE appropriée
        smote_classes = {
            'smote': SMOTE,
            'borderline_smote': BorderlineSMOTE,
            'svm_smote': SVMSMOTE
        }
        
        smote_class = smote_classes.get(method_name, SMOTE)
        
        # Créer et configurer SMOTE
        smote = smote_class(**method_config['params'])
        
        # Génération
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Créer le DataFrame résultat
        result_df = X_resampled.copy()
        result_df[target_column] = y_resampled
        
        return result_df
    
    def _generate_sdv(
        self,
        method_config: Dict,
        X: pd.DataFrame,
        y: pd.Series,
        target_column: str
    ) -> Optional[pd.DataFrame]:
        """Génère des données avec SDV (Synthetic Data Vault)."""
        
        if not SDV_AVAILABLE:
            logger.warning("SDV n'est pas disponible, installation requise")
            return None
        
        try:
            # Préparer les données complètes
            full_data = X.copy()
            full_data[target_column] = y
            
            # Ne garder que les fraudes pour l'entraînement SDV
            fraud_data = full_data[full_data[target_column] == 1].copy()
            
            if len(fraud_data) < 10:
                logger.warning("Pas assez de fraudes pour entraîner SDV")
                return None
            
            # Créer les métadonnées
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(fraud_data)
            
            # Configurer le synthesizer
            synthesizer = GaussianCopulaSynthesizer(
                metadata,
                enforce_min_max_values=method_config['params'].get('enforce_min_max_values', True),
                enforce_rounding=method_config['params'].get('enforce_rounding', False)
            )
            
            # Entraîner
            synthesizer.fit(fraud_data)
            
            # Générer des échantillons synthétiques
            num_samples = method_config['params'].get('num_samples', 1000)
            synthetic_frauds = synthesizer.sample(num_samples)
            
            # S'assurer que la colonne Class est bien définie
            synthetic_frauds[target_column] = 1
            
            # Combiner avec les données originales
            normal_data = full_data[full_data[target_column] == 0].copy()
            result_df = pd.concat([normal_data, synthetic_frauds], ignore_index=True)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Erreur SDV: {str(e)}")
            return None
    
    def evaluate_quality(
        self,
        original_data: pd.DataFrame,
        target_column: str = 'Class'
    ) -> Dict[str, float]:
        """
        Évalue la qualité des données synthétiques générées.
        
        Args:
            original_data (pd.DataFrame): Données originales pour comparaison
            target_column (str): Nom de la colonne target
            
        Returns:
            Dict[str, float]: Scores de qualité pour chaque méthode
        """
        logger.info("Évaluation de la qualité des données synthétiques")
        
        quality_scores = {}
        
        for method_name, synthetic_data in self.generated_data.items():
            try:
                score = self._evaluate_single_method(
                    method_name, synthetic_data, original_data, target_column
                )
                quality_scores[method_name] = score
                logger.info(f"✅ {method_name}: Score qualité {score:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Erreur évaluation {method_name}: {str(e)}")
                quality_scores[method_name] = 0.0
        
        self.quality_scores = quality_scores
        return quality_scores
    
    def _evaluate_single_method(
        self,
        method_name: str,
        synthetic_data: pd.DataFrame,
        original_data: pd.DataFrame,
        target_column: str
    ) -> float:
        """Évalue la qualité d'une méthode de génération spécifique."""
        
        # Préparer les données pour l'évaluation discriminateur
        X_orig = original_data.drop(columns=[target_column])
        X_synth = synthetic_data.drop(columns=[target_column])
        
        # Créer les labels (0=original, 1=synthétique)
        y_orig = np.zeros(len(X_orig))
        y_synth = np.ones(len(X_synth))
        
        # Combiner les données
        X_combined = pd.concat([X_orig, X_synth], ignore_index=True)
        y_combined = np.concatenate([y_orig, y_synth])
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined,
            test_size=self.config['evaluation']['test_size'],
            random_state=42,
            stratify=y_combined
        )
        
        # Normaliser les données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraîner discriminateur
        discriminator = RandomForestClassifier(n_estimators=100, random_state=42)
        discriminator.fit(X_train_scaled, y_train)
        
        # Prédire
        y_pred = discriminator.predict(X_test_scaled)
        
        # Score de qualité: 1 - accuracy du discriminateur
        # Si le discriminateur ne peut pas distinguer, la qualité est haute
        discriminator_accuracy = accuracy_score(y_test, y_pred)
        quality_score = 1 - discriminator_accuracy
        
        return quality_score
    
    def select_best_method(self) -> Tuple[str, pd.DataFrame]:
        """
        Sélectionne la meilleure méthode basée sur les scores de qualité.
        
        Returns:
            Tuple[str, pd.DataFrame]: Nom de la meilleure méthode et ses données
            
        Raises:
            ValueError: Si aucune méthode n'a été évaluée
        """
        if not self.quality_scores:
            raise ValueError("Aucune méthode n'a été évaluée")
        
        best_method = max(self.quality_scores, key=self.quality_scores.get)
        best_score = self.quality_scores[best_method]
        
        logger.info(f"🏆 Meilleure méthode: {best_method} (score: {best_score:.3f})")
        
        return best_method, self.generated_data[best_method]
    
    def generate_quality_report(self) -> str:
        """Génère un rapport de qualité des données synthétiques."""
        
        if not self.quality_scores:
            return "Aucune évaluation de qualité disponible."
        
        report_lines = [
            "# Rapport de Qualité - Données Synthétiques",
            "",
            "## Scores de Qualité par Méthode",
            "",
            "| Méthode | Score Qualité | Statut |",
            "|---------|---------------|--------|"
        ]
        
        threshold = self.config['evaluation']['quality_threshold']
        
        for method, score in sorted(self.quality_scores.items(), key=lambda x: x[1], reverse=True):
            status = "✅ Excellent" if score >= threshold else "⚠️ Acceptable" if score >= 0.5 else "❌ Faible"
            report_lines.append(f"| {method} | {score:.3f} | {status} |")
        
        # Recommandations
        report_lines.extend([
            "",
            "## Recommandations",
            ""
        ])
        
        best_method = max(self.quality_scores, key=self.quality_scores.get)
        best_score = self.quality_scores[best_method]
        
        if best_score >= threshold:
            report_lines.append(f"✅ **Recommandé**: Utiliser `{best_method}` (score: {best_score:.3f})")
        else:
            report_lines.append(f"⚠️ **Attention**: Meilleur score ({best_score:.3f}) sous le seuil ({threshold})")
            report_lines.append("- Considérer ajuster les paramètres de génération")
            report_lines.append("- Vérifier la qualité des données originales")
        
        return "\n".join(report_lines)


def create_balanced_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'smote',
    target_ratio: float = 0.1,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Crée un dataset équilibré avec la méthode spécifiée.
    
    Args:
        X (pd.DataFrame): Features originales
        y (pd.Series): Target originale
        method (str): Méthode à utiliser ('smote', 'borderline_smote', 'sdv')
        target_ratio (float): Ratio final minority/majority souhaité
        **kwargs: Paramètres additionnels pour la méthode
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features et target équilibrées
        
    Examples:
        >>> X_balanced, y_balanced = create_balanced_dataset(X, y, 'smote', 0.1)
    """
    config = {
        'methods': {
            method: {
                'enabled': True,
                'params': {
                    'sampling_strategy': target_ratio,
                    'random_state': 42,
                    **kwargs
                }
            }
        }
    }
    
    generator = SyntheticDataGenerator(config)
    
    # Créer un DataFrame temporaire pour la génération
    temp_df = X.copy()
    temp_df['Class'] = y
    
    results = generator.generate_all_methods(X, y, 'Class')
    
    if method in results:
        balanced_df = results[method]
        X_balanced = balanced_df.drop(columns=['Class'])
        y_balanced = balanced_df['Class']
        return X_balanced, y_balanced
    else:
        logger.error(f"Échec génération avec {method}")
        return X, y


if __name__ == "__main__":
    # Test du module avec données factices
    logger.info("Test du module synthetic_data")
    
    from sklearn.datasets import make_classification
    
    # Créer des données de test très déséquilibrées
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        weights=[0.99, 0.01],  # 1% de fraudes
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 11)])
    y_series = pd.Series(y)
    
    # Analyser le déséquilibre initial
    initial_balance = calculate_class_balance_metrics(y_series)
    logger.info(f"Déséquilibre initial: {initial_balance['imbalance_ratio']:.1f}:1")
    
    # Tester le générateur (sans SDV pour éviter les dépendances)
    test_config = {
        'methods': {
            'smote': {
                'enabled': True,
                'params': {'sampling_strategy': 0.2, 'k_neighbors': 3}
            },
            'borderline_smote': {
                'enabled': True,
                'params': {'sampling_strategy': 0.2, 'k_neighbors': 3}
            }
        }
    }
    
    generator = SyntheticDataGenerator(test_config)
    
    try:
        # Générer données synthétiques
        results = generator.generate_all_methods(X_df, y_series, 'Class')
        logger.info(f"✅ {len(results)} méthodes ont généré des données")
        
        # Évaluer qualité
        original_df = X_df.copy()
        original_df['Class'] = y_series
        
        quality_scores = generator.evaluate_quality(original_df, 'Class')
        logger.info("✅ Évaluation qualité terminée")
        
        # Sélectionner meilleure méthode
        if quality_scores:
            best_method, best_data = generator.select_best_method()
            logger.info(f"✅ Meilleure méthode sélectionnée: {best_method}")
        
        # Générer rapport
        report = generator.generate_quality_report()
        logger.info("✅ Rapport généré")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {str(e)}")
        raise
    
    logger.info("🎉 Tests du module synthetic_data terminés avec succès!")
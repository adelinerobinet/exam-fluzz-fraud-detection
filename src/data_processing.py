"""
Module de Traitement des Données pour Détection de Fraude
=========================================================

Ce module contient toutes les fonctions de preprocessing et feature engineering
pour préparer les données de transactions bancaires à l'entraînement ML.

Fonctionnalités principales:
- Nettoyage et validation des données
- Feature engineering temporel et métier
- Normalisation et mise à l'échelle
- Gestion du déséquilibre des classes

Auteur: Équipe Data Science - Néobanque Fluzz
Version: 1.0
Date: Août 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

from .utils import setup_logging, validate_dataset_quality, calculate_class_balance_metrics

# Configuration du logger
logger = setup_logging("INFO")

class DataPreprocessor:
    """
    Classe principale pour le preprocessing des données de transactions.
    
    Cette classe encapsule toutes les étapes de traitement des données,
    du nettoyage initial jusqu'à la préparation finale pour l'entraînement.
    
    Attributes:
        scalers (Dict): Dictionnaire des scalers ajustés
        feature_names (List[str]): Liste des noms de features finales
        is_fitted (bool): Indicateur si le preprocessor a été ajusté
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le preprocessor avec la configuration.
        
        Args:
            config (Dict, optional): Configuration de preprocessing.
                                   Si None, utilise la configuration par défaut.
        """
        self.config = config or self._get_default_config()
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        self.logger = setup_logging("INFO")
        
    def _get_default_config(self) -> Dict:
        """Configuration par défaut du preprocessing."""
        return {
            "temporal_features": True,
            "amount_features": True, 
            "pca_features": True,
            "scaling_method": "standard",
            "remove_outliers": True,
            "outlier_threshold": 3.0
        }
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Ajuste le preprocessor sur les données d'entraînement.
        
        Args:
            df (pd.DataFrame): Données d'entraînement
            
        Returns:
            DataPreprocessor: Self pour chaînage
            
        Raises:
            ValueError: Si les données ne sont pas valides
        """
        logger.info("Début de l'ajustement du preprocessor")
        
        # Validation des données
        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        validation = validate_dataset_quality(df, required_cols)
        
        if not validation['is_valid']:
            raise ValueError(f"Données invalides: {validation['message']}")
        
        # Étapes de preprocessing
        df_processed = df.copy()
        df_processed = self._clean_data(df_processed)
        df_processed = self._create_features(df_processed)
        df_processed = self._fit_scalers(df_processed)
        
        self.is_fitted = True
        self.feature_names = [col for col in df_processed.columns if col != 'Class']
        
        logger.info(f"Preprocessor ajusté avec {len(self.feature_names)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le preprocessing sur de nouvelles données.
        
        Args:
            df (pd.DataFrame): Données à transformer
            
        Returns:
            pd.DataFrame: Données transformées
            
        Raises:
            ValueError: Si le preprocessor n'a pas été ajusté
        """
        if not self.is_fitted:
            raise ValueError("Le preprocessor doit être ajusté avant transformation")
        
        logger.info("Transformation des données")
        
        df_processed = df.copy()
        df_processed = self._clean_data(df_processed)
        df_processed = self._create_features(df_processed)
        df_processed = self._apply_scaling(df_processed)
        
        # Assurer la cohérence des colonnes
        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0  # Valeur par défaut
        
        return df_processed[self.feature_names + (['Class'] if 'Class' in df_processed.columns else [])]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajuste et transforme les données en une seule étape.
        
        Args:
            df (pd.DataFrame): Données à ajuster et transformer
            
        Returns:
            pd.DataFrame: Données transformées
        """
        return self.fit(df).transform(df)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données (doublons, valeurs manquantes, outliers)."""
        logger.debug("Nettoyage des données")
        
        # Supprimer les doublons
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped_duplicates = initial_rows - len(df)
        
        if dropped_duplicates > 0:
            logger.info(f"Supprimés {dropped_duplicates} doublons")
        
        # Gérer les valeurs manquantes (très rares dans ce dataset)
        if df.isnull().any().any():
            logger.warning("Valeurs manquantes détectées, remplacement par médiane")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Supprimer les outliers extrêmes sur Amount
        if self.config.get("remove_outliers", True):
            threshold = self.config.get("outlier_threshold", 3.0)
            z_scores = np.abs((df['Amount'] - df['Amount'].mean()) / df['Amount'].std())
            outliers = len(df[z_scores > threshold])
            df = df[z_scores <= threshold]
            
            if outliers > 0:
                logger.info(f"Supprimés {outliers} outliers extrêmes")
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les nouvelles features d'ingénierie."""
        logger.debug("Création des features")
        
        df = df.copy()
        
        # Features temporelles
        if self.config.get("temporal_features", True):
            df = self._create_temporal_features(df)
        
        # Features de montant
        if self.config.get("amount_features", True):
            df = self._create_amount_features(df)
        
        # Features PCA enrichies
        if self.config.get("pca_features", True):
            df = self._create_pca_features(df)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features temporelles."""
        # Convertir Time en heures (Time est en secondes depuis début)
        df['Hour'] = (df['Time'] / 3600) % 24  # Heure du jour (0-23)
        df['Day'] = (df['Time'] / 86400).astype(int)  # Jour depuis début
        
        # Indicateurs binaires
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] < 6)).astype(int)
        df['Is_Weekend'] = (df['Day'] % 7 >= 5).astype(int)  # Simule weekend
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les features basées sur le montant."""
        # Log du montant (+ 1 pour éviter log(0))
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # Catégories de montant
        df['Amount_Category'] = pd.cut(
            df['Amount'], 
            bins=[0, 10, 50, 200, 1000, float('inf')],
            labels=['Micro', 'Small', 'Medium', 'Large', 'XLarge']
        ).cat.codes
        
        # Z-score du montant par jour
        daily_stats = df.groupby('Day')['Amount'].agg(['mean', 'std'])
        df = df.merge(daily_stats, on='Day', suffixes=('', '_day'))
        df['Amount_Daily_Zscore'] = (df['Amount'] - df['Amount_day']) / (df['Amount_day'].std() + 1e-8)
        
        # Nettoyer les colonnes temporaires
        df = df.drop(['Amount_day', 'Amount_day'], axis=1, errors='ignore')
        
        return df
    
    def _create_pca_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des features enrichies basées sur les composantes PCA."""
        pca_cols = [f'V{i}' for i in range(1, 29)]
        
        # Magnitude euclidienne des composantes PCA
        df['PCA_Magnitude'] = np.sqrt((df[pca_cols] ** 2).sum(axis=1))
        
        # Nombre de composantes "extrêmes" (> 2 écarts-type)
        pca_extreme = (np.abs(df[pca_cols]) > 2).sum(axis=1)
        df['PCA_Extreme_Count'] = pca_extreme
        
        return df
    
    def _fit_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajuste les scalers sur les données."""
        logger.debug("Ajustement des scalers")
        
        # Définir les groupes de colonnes pour différents types de scaling
        scaling_groups = {
            'standard': ['Time', 'Hour', 'Amount_Daily_Zscore', 'PCA_Magnitude'],
            'robust': ['Amount'],
            'minmax': ['Amount_log', 'Amount_Category', 'PCA_Extreme_Count']
        }
        
        scaler_classes = {
            'standard': StandardScaler,
            'robust': RobustScaler, 
            'minmax': MinMaxScaler
        }
        
        for scaler_type, columns in scaling_groups.items():
            # Filtrer les colonnes qui existent réellement
            existing_cols = [col for col in columns if col in df.columns]
            
            if existing_cols:
                scaler = scaler_classes[scaler_type]()
                scaler.fit(df[existing_cols])
                self.scalers[scaler_type] = {
                    'scaler': scaler,
                    'columns': existing_cols
                }
        
        return df
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les scalers ajustés."""
        df = df.copy()
        
        for scaler_type, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']
            
            # Vérifier que toutes les colonnes existent
            existing_cols = [col for col in columns if col in df.columns]
            if existing_cols:
                df[existing_cols] = scaler.transform(df[existing_cols])
        
        return df


def split_data(
    df: pd.DataFrame,
    target_column: str = 'Class',
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en ensembles train/validation/test stratifiés.
    
    Args:
        df (pd.DataFrame): Données complètes
        target_column (str): Nom de la colonne cible
        test_size (float): Proportion pour le test set
        val_size (float): Proportion pour le validation set (sur les données restantes)
        random_state (int): Seed pour reproductibilité
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test sets
        
    Examples:
        >>> train_df, val_df, test_df = split_data(df)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    """
    # Premier split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )
    
    # Deuxième split: train vs validation
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val[target_column]
    )
    
    logger.info(f"Données divisées - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test


def get_feature_importance_summary(
    feature_names: List[str],
    feature_importance: np.ndarray,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Crée un résumé des features les plus importantes.
    
    Args:
        feature_names (List[str]): Noms des features
        feature_importance (np.ndarray): Importances des features
        top_n (int): Nombre de top features à retourner
        
    Returns:
        pd.DataFrame: DataFrame avec features et importances triées
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    
    return importance_df.nlargest(top_n, 'importance').reset_index(drop=True)


if __name__ == "__main__":
    # Test du preprocessor avec données factices
    logger.info("Test du module data_processing")
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    
    test_data = {
        'Time': np.random.uniform(0, 172800, n_samples),  # 2 jours en secondes
        'Amount': np.random.lognormal(2, 1, n_samples),  # Distribution réaliste
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])  # Déséquilibré
    }
    
    # Ajouter les features V1-V28 (PCA fictives)
    for i in range(1, 29):
        test_data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    test_df = pd.DataFrame(test_data)
    
    # Tester le preprocessor
    preprocessor = DataPreprocessor()
    
    try:
        processed_df = preprocessor.fit_transform(test_df)
        logger.info(f"✅ Preprocessing réussi: {processed_df.shape}")
        logger.info(f"✅ Features créées: {len(preprocessor.feature_names)}")
        
        # Tester le split
        train_df, val_df, test_df = split_data(processed_df)
        logger.info("✅ Split des données réussi")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {str(e)}")
        raise
    
    logger.info("🎉 Tests du module data_processing terminés avec succès!")
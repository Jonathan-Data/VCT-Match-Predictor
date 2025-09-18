"""
Machine Learning Models for VCT 2025 Match Prediction

This module implements various ML algorithms for predicting match outcomes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
import joblib
import json
from dataclasses import dataclass

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

# TensorFlow/Keras for Neural Networks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Data class for storing model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cross_val_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None

class VCTMatchPredictor:
    """Match predictor using multiple ML algorithms."""
    
    def __init__(self, models_dir: Path = None):
        """Initialize the predictor."""
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.model_performances = {}
        self.feature_names = []
        self.is_trained = False
    
    def create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Create a Random Forest model with optimized parameters."""
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        return RandomForestClassifier(**default_params)
    
    def create_xgboost(self, **kwargs) -> Optional[xgb.XGBClassifier]:
        """Create an XGBoost model with optimized parameters."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping")
            return None
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        default_params.update(kwargs)
        
        return xgb.XGBClassifier(**default_params)
    
    def create_logistic_regression(self, **kwargs) -> LogisticRegression:
        """Create a Logistic Regression model."""
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'C': 1.0
        }
        default_params.update(kwargs)
        
        return LogisticRegression(**default_params)
    
    def create_svm(self, **kwargs) -> SVC:
        """Create an SVM model."""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        return SVC(**default_params)
    
    def create_neural_network(self, input_dim: int, **kwargs) -> Optional[Sequential]:
        """Create a neural network model."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network")
            return None
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def initialize_models(self, input_dim: int = None) -> None:
        """Initialize all available models."""
        logger.info("Initializing ML models...")
        
        # Traditional ML models
        self.models['random_forest'] = self.create_random_forest()
        self.models['logistic_regression'] = self.create_logistic_regression()
        self.models['svm'] = self.create_svm()
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = self.create_xgboost()
        
        # Neural Network if available and input dimension provided
        if TENSORFLOW_AVAILABLE and input_dim:
            self.models['neural_network'] = self.create_neural_network(input_dim)
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_sklearn_model(self, model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """Train a scikit-learn compatible model."""
        logger.info(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_') and len(model.coef_.shape) == 1:
            feature_importance = dict(zip(self.feature_names, abs(model.coef_)))
        
        performance = ModelPerformance(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cross_val_scores=cv_scores.tolist(),
            feature_importance=feature_importance
        )
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f}")
        return performance
    
    def train_neural_network(self, model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """Train the neural network model."""
        logger.info(f"Training {model_name}...")
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train the model
        history = model.fit(
            X_train.values, y_train.values,
            epochs=100,
            batch_size=32,
            validation_data=(X_test.values, y_test.values),
            callbacks=callbacks,
            verbose=0
        )
        
        # Make predictions
        y_pred_proba = model.predict(X_test.values).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Use validation accuracy as cross-validation proxy
        val_accuracies = history.history.get('val_accuracy', [accuracy] * 5)
        cv_scores = val_accuracies[-5:] if len(val_accuracies) >= 5 else [accuracy] * 5
        
        performance = ModelPerformance(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cross_val_scores=cv_scores
        )
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return performance
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelPerformance]:
        """Train all initialized models."""
        logger.info("Starting training for all models...")
        
        self.feature_names = list(X_train.columns)
        
        # Initialize models if not already done
        if not self.models:
            self.initialize_models(input_dim=X_train.shape[1])
        
        performances = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    performance = self.train_neural_network(model, model_name, X_train, y_train, X_test, y_test)
                else:
                    performance = self.train_sklearn_model(model, model_name, X_train, y_train, X_test, y_test)
                
                performances[model_name] = performance
                self.model_performances[model_name] = performance
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.is_trained = True
        logger.info(f"Training completed for {len(performances)} models")
        return performances
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> VotingClassifier:
        """Create an ensemble model combining the best performers."""
        logger.info("Creating ensemble model...")
        
        # Select models for ensemble (exclude neural network for now)
        ensemble_models = []
        for name, model in self.models.items():
            if name != 'neural_network' and hasattr(model, 'predict_proba'):
                ensemble_models.append((name, model))
        
        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble, need at least 2")
            return None
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',  # Use probability estimates
            n_jobs=-1
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        
        logger.info(f"Ensemble model created with {len(ensemble_models)} base models")
        return ensemble
    
    def predict_match(self, team1_features: Dict, team2_features: Dict, 
                     model_name: str = 'ensemble') -> Dict[str, Any]:
        """Predict the outcome of a match between two teams."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        # Create feature vector (this would need to match the training features)
        # This is a simplified version - in practice, you'd need to engineer the same features
        feature_vector = np.array([
            team1_features.get('rating', 0) - team2_features.get('rating', 0),
            team1_features.get('win_rate', 0.5) - team2_features.get('win_rate', 0.5),
            team1_features.get('round_win_rate', 0.5) - team2_features.get('round_win_rate', 0.5),
            team1_features.get('avg_combat_score', 200) - team2_features.get('avg_combat_score', 200),
            int(team1_features.get('region') == team2_features.get('region', 'unknown'))
        ]).reshape(1, -1)
        
        model = self.models[model_name]
        
        if model_name == 'neural_network' and TENSORFLOW_AVAILABLE:
            prediction_proba = model.predict(feature_vector)[0][0]
            prediction = int(prediction_proba > 0.5)
        else:
            prediction = model.predict(feature_vector)[0]
            prediction_proba = model.predict_proba(feature_vector)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        return {
            'prediction': prediction,  # 0 for team2 win, 1 for team1 win
            'probability': float(prediction_proba),
            'confidence': abs(prediction_proba - 0.5) * 2,  # 0-1 scale
            'model_used': model_name
        }
    
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get models ranked by performance."""
        if not self.model_performances:
            return []
        
        # Rank by F1 score (you could use other metrics)
        rankings = [(name, perf.f1) for name, perf in self.model_performances.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def save_models(self) -> None:
        """Save all trained models to disk."""
        logger.info("Saving models...")
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    model_path = self.models_dir / f"{model_name}.h5"
                    model.save(str(model_path))
                else:
                    model_path = self.models_dir / f"{model_name}.joblib"
                    joblib.dump(model, model_path)
                
                logger.info(f"Saved {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")
        
        # Save performance metrics
        if self.model_performances:
            performance_data = {}
            for name, perf in self.model_performances.items():
                performance_data[name] = {
                    'accuracy': perf.accuracy,
                    'precision': perf.precision,
                    'recall': perf.recall,
                    'f1': perf.f1,
                    'cross_val_scores': perf.cross_val_scores,
                    'feature_importance': perf.feature_importance
                }
            
            perf_path = self.models_dir / "model_performances.json"
            with open(perf_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            logger.info(f"Saved performance metrics to {perf_path}")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        logger.info("Loading models...")
        
        # Load performance metrics
        perf_path = self.models_dir / "model_performances.json"
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                performance_data = json.load(f)
            
            for name, data in performance_data.items():
                self.model_performances[name] = ModelPerformance(
                    model_name=name,
                    accuracy=data['accuracy'],
                    precision=data['precision'],
                    recall=data['recall'],
                    f1=data['f1'],
                    cross_val_scores=data['cross_val_scores'],
                    feature_importance=data.get('feature_importance')
                )
        
        # Load models
        model_files = {
            'random_forest': 'random_forest.joblib',
            'xgboost': 'xgboost.joblib',
            'logistic_regression': 'logistic_regression.joblib',
            'svm': 'svm.joblib',
            'ensemble': 'ensemble.joblib',
            'neural_network': 'neural_network.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    if model_name == 'neural_network' and TENSORFLOW_AVAILABLE:
                        self.models[model_name] = tf.keras.models.load_model(str(model_path))
                    else:
                        self.models[model_name] = joblib.load(model_path)
                    
                    logger.info(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
        
        if self.models:
            self.is_trained = True
            logger.info(f"Loaded {len(self.models)} models")
    
    def print_performance_summary(self) -> None:
        """Print a summary of model performances."""
        if not self.model_performances:
            print("No performance data available")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        rankings = self.get_model_rankings()
        
        for i, (model_name, f1_score) in enumerate(rankings, 1):
            perf = self.model_performances[model_name]
            print(f"\n{i}. {model_name.upper()}")
            print(f"   Accuracy:  {perf.accuracy:.4f}")
            print(f"   Precision: {perf.precision:.4f}")
            print(f"   Recall:    {perf.recall:.4f}")
            print(f"   F1 Score:  {perf.f1:.4f}")
            print(f"   CV Mean:   {np.mean(perf.cross_val_scores):.4f} (±{np.std(perf.cross_val_scores):.4f})")
        
        print("\n" + "="*60)

def main():\n    \"\"\"Main function to demonstrate model training.\"\"\"\n    # This is a demo function - in practice, you'd load your processed data\n    print(\"VCT Match Predictor - Model Training Demo\")\n    \n    # Create dummy data for demonstration\n    np.random.seed(42)\n    n_samples = 1000\n    n_features = 10\n    \n    X = pd.DataFrame(np.random.randn(n_samples, n_features), \n                     columns=[f'feature_{i}' for i in range(n_features)])\n    y = pd.Series(np.random.randint(0, 2, n_samples), name='winner')\n    \n    # Split data\n    from sklearn.model_selection import train_test_split\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Initialize predictor\n    predictor = VCTMatchPredictor()\n    \n    # Train models\n    performances = predictor.train_all_models(X_train, y_train, X_test, y_test)\n    \n    # Create ensemble\n    predictor.create_ensemble_model(X_train, y_train)\n    \n    # Print results\n    predictor.print_performance_summary()\n    \n    # Save models\n    predictor.save_models()\n    \n    print(\"\\nModel training completed!\")\n\nif __name__ == \"__main__\":\n    main()
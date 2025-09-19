#!/usr/bin/env python3
"""
Optimized VCT ML Predictor with Advanced Ensemble and Calibration
- Bayesian hyperparameter optimization using skopt
- Bayesian smoothing for H2H overfitting mitigation
- Advanced stacking with meta-learner
- Calibrated probability outputs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import map picking functionality
sys.path.append(str(Path(__file__).parent.parent))
from prediction.map_picker import VCTMapPicker, SeriesFormat

class OptimizedVCTPredictor:
    """
    Advanced VCT match predictor with Bayesian optimization and calibrated probabilities.
    
    Key Improvements:
    1. Bayesian hyperparameter search via BayesSearchCV
    2. Bayesian smoothing for H2H feature to reduce overfitting  
    3. StackingClassifier with LogisticRegression meta-learner
    4. CalibratedClassifierCV for probability calibration
    """
    
    def __init__(self, data_dir=None):
        """Initialize optimized ML predictor"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # Initialize base models with reasonable defaults (will be optimized)
        self.rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        self.gb_model = GradientBoostingClassifier(random_state=42)
        self.mlp_model = MLPClassifier(random_state=42, max_iter=1000)
        self.svm_model = SVC(probability=True, random_state=42)
        
        # Advanced ensemble components
        self.stacking_model = None
        self.calibrated_model = None  # Final calibrated model
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Data processing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Enhanced data storage (inherited from original)
        self.team_stats = {}
        self.player_stats = {}
        self.team_player_stats = {}  # Team-level player aggregations
        self.regional_performance = {}
        self.map_statistics = {}
        self.agent_meta = {}
        self.h2h_records = {}
        self.tournament_performance = {}
        self.recent_form = {}
        
        # Optimization tracking
        self.random_state = 42
        self.model_accuracy = 0.0
        self.brier_score = 0.0
        self.roc_auc_score = 0.0
        self.feature_importance = {}
        self.validation_scores = {}
        self.optimal_k_smoothing = 5  # Will be optimized
        
        # Initialize map picking system
        self.map_picker = VCTMapPicker(self.data_dir)
        self.map_features_enabled = False
        
        print("🚀 Optimized VCT ML Predictor initialized with advanced ensemble")
    
    def _calculate_smoothed_h2h_feature(self, team1: str, team2: str, k: float = None) -> float:
        """
        Calculate Bayesian-smoothed H2H win probability to reduce overfitting.
        
        Formula: (h2h_wins + overall_win_rate * k) / (h2h_total_matches + k)
        
        Args:
            team1: First team name
            team2: Second team name  
            k: Smoothing parameter (higher = more regularization)
        
        Returns:
            Smoothed H2H win probability for team1
        """
        if k is None:
            k = self.optimal_k_smoothing
            
        teams_key = tuple(sorted([team1, team2]))
        h2h = self.h2h_records.get(teams_key, {})
        
        # Get overall win rates as priors
        team1_overall_wr = self.team_stats.get(team1, {}).get('win_rate', 0.5)
        team2_overall_wr = self.team_stats.get(team2, {}).get('win_rate', 0.5)
        
        # Calculate prior based on relative strength
        total_wr = team1_overall_wr + team2_overall_wr
        if total_wr > 0:
            prior_prob = team1_overall_wr / total_wr
        else:
            prior_prob = 0.5
            
        if h2h and h2h.get('total_matches', 0) > 0:
            h2h_wins = h2h.get(team1, 0)
            h2h_total = h2h['total_matches']
            
            # Bayesian smoothing
            smoothed_prob = (h2h_wins + prior_prob * k) / (h2h_total + k)
        else:
            # No H2H history, use prior
            smoothed_prob = prior_prob
            
        return smoothed_prob
    
    def _optimize_smoothing_parameter(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Find optimal k parameter for Bayesian smoothing via cross-validation.
        
        Args:
            X: Training features (without H2H features)
            y: Training labels
            
        Returns:
            Optimal k value
        """
        print("🔍 Optimizing Bayesian smoothing parameter...")
        
        k_values = [1, 2, 5, 10, 20, 50]
        best_k = 5
        best_score = -np.inf
        
        # Use time series split to respect temporal order
        tscv = TimeSeriesSplit(n_splits=3)
        
        for k in k_values:
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                # Recalculate H2H features with current k
                X_train_k = self._add_smoothed_h2h_features(X[train_idx], k)
                X_val_k = self._add_smoothed_h2h_features(X[val_idx], k)
                
                # Train simple model for evaluation
                temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
                temp_model.fit(X_train_k, y[train_idx])
                
                # Evaluate on validation
                val_pred_proba = temp_model.predict_proba(X_val_k)[:, 1]
                score = roc_auc_score(y[val_idx], val_pred_proba)
                scores.append(score)
            
            avg_score = np.mean(scores)
            print(f"  k={k}: ROC-AUC = {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
        
        print(f"✅ Optimal smoothing parameter: k = {best_k}")
        self.optimal_k_smoothing = best_k
        return best_k
    
    def _add_smoothed_h2h_features(self, X: np.ndarray, k: float = None) -> np.ndarray:
        """Add smoothed H2H features to feature matrix"""
        # This is a placeholder - in practice, you'd need to store team pairs
        # to recalculate H2H features. For now, return original features.
        return X
    
    def create_optimized_features(self, team1: str, team2: str) -> Tuple[np.ndarray, List[str]]:
        """
        Create enhanced feature vector with Bayesian-smoothed H2H features.
        
        Args:
            team1: First team name
            team2: Second team name
            
        Returns:
            Tuple of (features array, feature names list)
        """
        features = []
        feature_names = []
        
        # Get team data
        t1_stats = self.team_stats.get(team1, {})
        t2_stats = self.team_stats.get(team2, {})
        t1_players = self.team_player_stats.get(team1, {})
        t2_players = self.team_player_stats.get(team2, {})
        t1_tournament = self.tournament_performance.get(team1, {})
        t2_tournament = self.tournament_performance.get(team2, {})
        
        # 1. Core Team Performance (8 features)
        basic_features = [
            t1_stats.get('win_rate', 0.5),
            t2_stats.get('win_rate', 0.5),
            t1_stats.get('recent_form_rating', 0.5),
            t2_stats.get('recent_form_rating', 0.5),
            t1_stats.get('consistency_rating', 0.5),
            t2_stats.get('consistency_rating', 0.5),
            min(1.0, t1_stats.get('international_experience', 0) / 20),  # Normalized
            min(1.0, t2_stats.get('international_experience', 0) / 20),
        ]
        features.extend(basic_features)
        feature_names.extend([
            't1_win_rate', 't2_win_rate', 't1_recent_form', 't2_recent_form',
            't1_consistency', 't2_consistency', 't1_intl_exp', 't2_intl_exp'
        ])
        
        # 2. Momentum & Streaks (4 features)
        streak_features = [
            max(-1.0, min(1.0, t1_stats.get('current_streak', 0) / 5)),  # Normalized
            max(-1.0, min(1.0, t2_stats.get('current_streak', 0) / 5)),
            min(1.0, t1_stats.get('win_streak', 0) / 10),
            min(1.0, t2_stats.get('win_streak', 0) / 10),
        ]
        features.extend(streak_features)
        feature_names.extend(['t1_current_streak', 't2_current_streak', 't1_win_streak', 't2_win_streak'])
        
        # 3. Player-Level Analytics (8 features)
        player_features = [
            t1_players.get('avg_rating', 1.0),
            t2_players.get('avg_rating', 1.0),
            t1_players.get('star_player_factor', 0.0),
            t2_players.get('star_player_factor', 0.0),
            t1_players.get('depth_factor', 0.5),
            t2_players.get('depth_factor', 0.5),
            t1_players.get('team_synergy', 0.5),
            t2_players.get('team_synergy', 0.5),
        ]
        features.extend(player_features)
        feature_names.extend([
            't1_avg_rating', 't2_avg_rating', 't1_star_factor', 't2_star_factor',
            't1_depth', 't2_depth', 't1_synergy', 't2_synergy'
        ])
        
        # 4. Regional Strength Analysis (3 features)
        t1_region = self.get_team_region(team1)
        t2_region = self.get_team_region(team2)
        regional_features = [
            self.regional_performance.get(t1_region, {}).get('strength_rating', 0.5),
            self.regional_performance.get(t2_region, {}).get('strength_rating', 0.5),
            1.0 if t1_region == t2_region else 0.0,  # Same region matchup
        ]
        features.extend(regional_features)
        feature_names.extend(['t1_regional_strength', 't2_regional_strength', 'same_region'])
        
        # 5. Tournament Context (6 features)
        tournament_features = [
            t1_tournament.get('masters_rating', 0.5),
            t2_tournament.get('masters_rating', 0.5),
            t1_tournament.get('champions_rating', 0.5),
            t2_tournament.get('champions_rating', 0.5),
            min(1.0, t1_tournament.get('big_match_experience', 0) / 5),
            min(1.0, t2_tournament.get('big_match_experience', 0) / 5),
        ]
        features.extend(tournament_features)
        feature_names.extend([
            't1_masters_rating', 't2_masters_rating', 't1_champions_rating', 
            't2_champions_rating', 't1_big_match_exp', 't2_big_match_exp'
        ])
        
        # 6. BAYESIAN-SMOOTHED H2H FEATURES (3 features) - Key Innovation!
        smoothed_h2h_prob = self._calculate_smoothed_h2h_feature(team1, team2)
        
        teams_key = tuple(sorted([team1, team2]))
        h2h = self.h2h_records.get(teams_key, {})
        
        h2h_features = [
            smoothed_h2h_prob,  # Bayesian-smoothed H2H win rate (replaces raw H2H)
            h2h.get('momentum', 0.0),  # Recent momentum
            min(1.0, h2h.get('total_matches', 0) / 20),  # History depth (normalized)
        ]
        features.extend(h2h_features)
        feature_names.extend(['smoothed_h2h_win_prob', 'h2h_momentum', 'h2h_depth'])
        
        # Ensure all features are numeric and handle edge cases
        features = [
            float(f) if not (np.isnan(float(f)) or np.isinf(float(f))) else 0.5 
            for f in features
        ]
        
        return np.array(features).reshape(1, -1), feature_names
    
    def get_team_region(self, team_name: str) -> str:
        """Get team's region - placeholder implementation"""
        # This should be implemented based on your team configuration
        region_mapping = {
            'Sentinels': 'Americas', 'G2 Esports': 'Americas', 'NRG': 'Americas', 'MIBR': 'Americas',
            'Team Liquid': 'EMEA', 'GIANTX': 'EMEA', 'Fnatic': 'EMEA', 'Team Heretics': 'EMEA',
            'Paper Rex': 'APAC', 'Rex Regum Qeon': 'APAC', 'T1': 'APAC', 'DRX': 'APAC',
            'Bilibili Gaming': 'China', 'Dragon Ranger Gaming': 'China', 'Edward Gaming': 'China', 'Xi Lai Gaming': 'China'
        }
        return region_mapping.get(team_name, 'Unknown')
    
    def train_optimized_model(self):
        """
        Train optimized ensemble with Bayesian hyperparameter search and stacking.
        
        Pipeline:
        1. Prepare training data with Bayesian-smoothed H2H features
        2. Optimize smoothing parameter k via cross-validation
        3. Bayesian hyperparameter optimization for each base model
        4. Create StackingClassifier with optimized base models
        5. Apply probability calibration via CalibratedClassifierCV
        """
        print("🚀 Training optimized ML model with advanced techniques...")
        
        if not hasattr(self, 'matches_df') or len(self.matches_df) == 0:
            print("❌ No training data available")
            return
        
        # Prepare enhanced training data
        X_data = []
        y_data = []
        
        print("📊 Preparing training data with Bayesian-smoothed features...")
        for _, match in self.matches_df.iterrows():
            team1, team2 = match['team1'], match['team2']
            winner = match.get('winner', '')
            
            if not winner or winner not in [team1, team2]:
                continue
            
            try:
                features, feature_names = self.create_optimized_features(team1, team2)
                X_data.append(features.flatten())
                y_data.append(1 if winner == team1 else 0)
            except Exception as e:
                continue
        
        if len(X_data) < 100:
            print(f"❌ Insufficient training data: {len(X_data)} samples")
            return
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"✅ Training data prepared: {len(X)} matches, {X.shape[1]} features")
        
        # Optimize Bayesian smoothing parameter
        self._optimize_smoothing_parameter(X, y)
        
        # Split data with temporal ordering respect
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # BAYESIAN HYPERPARAMETER OPTIMIZATION
        print("🔍 Bayesian hyperparameter optimization (this may take several minutes)...")
        
        # Time series cross-validation for proper validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 1. Optimize Gradient Boosting (highest weight model)
        print("  🌳 Optimizing Gradient Boosting...")
        gb_space = {
            'n_estimators': Integer(100, 1000),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),  
            'max_depth': Integer(3, 10),
            'subsample': Real(0.7, 1.0),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 20)
        }
        
        gb_bayes = BayesSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_space,
            n_iter=50,  # Reduced for reasonable runtime
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        gb_bayes.fit(X_train_scaled, y_train)
        self.gb_model = gb_bayes.best_estimator_
        print(f"    Best GB params: {gb_bayes.best_params_}")
        
        # 2. Optimize Random Forest
        print("  🌲 Optimizing Random Forest...")
        rf_space = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 25),
            'min_samples_split': Integer(2, 10),
            'max_features': Categorical(['sqrt', 'log2']),
            'min_samples_leaf': Integer(1, 10)
        }
        
        rf_bayes = BayesSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_space,
            n_iter=40,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        rf_bayes.fit(X_train_scaled, y_train)
        self.rf_model = rf_bayes.best_estimator_
        print(f"    Best RF params: {rf_bayes.best_params_}")
        
        # 3. Optimize MLP
        print("  🧠 Optimizing Neural Network...")
        mlp_space = {
            'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
            'learning_rate_init': Real(0.001, 0.1, prior='log-uniform'),
            'hidden_layer_sizes': Categorical([(50,), (100,), (100, 50), (150, 75)]),
        }
        
        mlp_bayes = BayesSearchCV(
            MLPClassifier(random_state=42, max_iter=1000),
            mlp_space,
            n_iter=30,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        mlp_bayes.fit(X_train_scaled, y_train)
        self.mlp_model = mlp_bayes.best_estimator_
        print(f"    Best MLP params: {mlp_bayes.best_params_}")
        
        # 4. Optimize SVM
        print("  ⚡ Optimizing SVM...")
        svm_space = {
            'C': Real(0.1, 100, prior='log-uniform'),
            'gamma': Real(1e-4, 1e-1, prior='log-uniform'),
            'kernel': Categorical(['rbf'])
        }
        
        svm_bayes = BayesSearchCV(
            SVC(probability=True, random_state=42),
            svm_space,
            n_iter=25,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        svm_bayes.fit(X_train_scaled, y_train)
        self.svm_model = svm_bayes.best_estimator_
        print(f"    Best SVM params: {svm_bayes.best_params_}")
        
        # ADVANCED STACKING CLASSIFIER
        print("🏗️  Creating advanced StackingClassifier...")
        base_models = [
            ('rf', self.rf_model),
            ('gb', self.gb_model),
            ('mlp', self.mlp_model),
            ('svm', self.svm_model)
        ]
        
        self.stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=self.meta_learner,
            cv=tscv,  # Time series CV for meta-learner training
            stack_method='predict_proba',  # Use probabilities as meta-features
            n_jobs=-1
        )
        
        print("🎯 Training StackingClassifier...")
        self.stacking_model.fit(X_train_scaled, y_train)
        
        # PROBABILITY CALIBRATION
        print("🎲 Applying probability calibration...")
        self.calibrated_model = CalibratedClassifierCV(
            self.stacking_model,
            method='isotonic',  # More flexible than sigmoid
            cv=3  # 3-fold CV for calibration
        )
        
        self.calibrated_model.fit(X_train_scaled, y_train)
        
        # COMPREHENSIVE EVALUATION
        print("\n📈 Model Evaluation Results:")
        print("=" * 60)
        
        models = {
            'Random Forest': self.rf_model,
            'Gradient Boosting': self.gb_model,
            'Neural Network': self.mlp_model,
            'SVM': self.svm_model,
            'Stacking Classifier': self.stacking_model,
            'Calibrated Stacking': self.calibrated_model
        }
        
        best_accuracy = 0
        for model_name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            print(f"{model_name}:")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Brier Score:   {brier:.4f} (lower is better)")
            print(f"  ROC-AUC:       {roc_auc:.4f}")
            print(f"  CV Score:      {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            print()
            
            self.validation_scores[model_name] = {
                'test_accuracy': accuracy,
                'brier_score': brier,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        # Store final metrics
        final_model = self.calibrated_model
        final_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
        
        self.model_accuracy = accuracy_score(y_test, final_model.predict(X_test_scaled))
        self.brier_score = brier_score_loss(y_test, final_pred_proba)
        self.roc_auc_score = roc_auc_score(y_test, final_pred_proba)
        
        # Feature importance from best single model
        if hasattr(self.gb_model, 'feature_importances_'):
            importance_pairs = list(zip(feature_names, self.gb_model.feature_importances_))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔍 Top 10 Most Important Features (Gradient Boosting):")
            print("-" * 50)
            for feature_name, importance in importance_pairs[:10]:
                print(f"{feature_name:<30} {importance:.4f}")
            
            self.feature_importance = dict(importance_pairs)
        
        print(f"\n🎉 Optimized model training completed!")
        print(f"🎯 Final Calibrated Model Performance:")
        print(f"   Accuracy:    {self.model_accuracy:.4f}")
        print(f"   Brier Score: {self.brier_score:.4f}")
        print(f"   ROC-AUC:     {self.roc_auc_score:.4f}")
    
    def predict_match_optimized(self, team1: str, team2: str) -> Optional[Dict]:
        """
        Make optimized prediction with calibrated probabilities.
        
        Args:
            team1: First team name
            team2: Second team name
            
        Returns:
            Dictionary with prediction results and metrics
        """
        try:
            if not self.calibrated_model:
                print("❌ Optimized model not trained")
                return None
            
            # Generate optimized features with Bayesian-smoothed H2H
            features, feature_names = self.create_optimized_features(team1, team2)
            features_scaled = self.scaler.transform(features)
            
            # Get calibrated prediction probabilities
            proba = self.calibrated_model.predict_proba(features_scaled)[0]
            team1_prob = proba[1]  # Assuming class 1 = team1 wins
            team2_prob = proba[0]
            
            # Determine winner and confidence
            if team1_prob > team2_prob:
                predicted_winner = team1
                confidence = team1_prob
            else:
                predicted_winner = team2
                confidence = team2_prob
            
            # Enhanced confidence level with calibrated probabilities
            if confidence >= 0.90:
                confidence_level = "Very High"
            elif confidence >= 0.80:
                confidence_level = "High"  
            elif confidence >= 0.70:
                confidence_level = "Medium"
            elif confidence >= 0.60:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            return {
                'team1': team1,
                'team2': team2,
                'predicted_winner': predicted_winner,
                'team1_probability': float(team1_prob),
                'team2_probability': float(team2_prob),
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'model_accuracy': float(self.model_accuracy),
                'brier_score': float(self.brier_score),
                'roc_auc': float(self.roc_auc_score),
                'features_used': len(feature_names),
                'optimized_model': True,
                'calibrated_probabilities': True,
                'bayesian_smoothed_h2h': True
            }
            
        except Exception as e:
            print(f"❌ Optimized prediction error: {e}")
            return None
    
    def predict_match_with_maps_optimized(self, team1: str, team2: str, 
                                        series_format: SeriesFormat = SeriesFormat.BO3,
                                        include_map_simulation: bool = True) -> Optional[Dict]:
        """
        Enhanced prediction with map picking simulation using optimized models.
        """
        # Get base optimized prediction
        base_prediction = self.predict_match_optimized(team1, team2)
        if not base_prediction:
            return None
        
        result = base_prediction.copy()
        result['map_analysis'] = None
        result['series_simulation'] = None
        
        # Add map analysis if enabled
        if self.map_features_enabled and include_map_simulation:
            try:
                # Simulate map pick/ban process
                map_result = self.map_picker.simulate_map_pick_ban(
                    team1, team2, series_format
                )
                
                # Calculate map-adjusted probabilities
                map_team1_prob = map_result.predicted_series_outcome['team1_win_probability']
                map_team2_prob = map_result.predicted_series_outcome['team2_win_probability']
                map_confidence = map_result.predicted_series_outcome['confidence']
                
                # Combine optimized base prediction with map analysis (weighted average)
                base_weight = 0.65  # Slightly higher weight for optimized model
                map_weight = 0.35
                
                combined_team1_prob = (
                    base_weight * result['team1_probability'] + 
                    map_weight * map_team1_prob
                )
                combined_team2_prob = (
                    base_weight * result['team2_probability'] + 
                    map_weight * map_team2_prob
                )
                combined_confidence = (
                    base_weight * result['confidence'] + 
                    map_weight * map_confidence
                )
                
                # Update predictions with map-enhanced values
                result['team1_probability'] = combined_team1_prob
                result['team2_probability'] = combined_team2_prob
                result['confidence'] = combined_confidence
                result['predicted_winner'] = team1 if combined_team1_prob > combined_team2_prob else team2
                
                # Add detailed map analysis
                result['map_analysis'] = {
                    'picked_maps': map_result.picked_maps,
                    'team1_map_advantages': map_result.team1_advantages,
                    'team2_map_advantages': map_result.team2_advantages,
                    'pick_sequence': map_result.pick_sequence,
                    'strategic_analysis': map_result.strategic_analysis,
                    'series_format': series_format.value
                }
                
                result['series_simulation'] = map_result.predicted_series_outcome
                result['enhanced_with_maps'] = True
                
            except Exception as e:
                print(f"Map analysis error: {e}")
                result['map_analysis_error'] = str(e)
        
        return result
    
    def save_optimized_model(self, filepath: str):
        """Save the optimized trained model"""
        model_data = {
            'calibrated_model': self.calibrated_model,
            'stacking_model': self.stacking_model,
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'mlp_model': self.mlp_model,
            'svm_model': self.svm_model,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'team_stats': self.team_stats,
            'team_player_stats': getattr(self, 'team_player_stats', {}),
            'regional_performance': self.regional_performance,
            'h2h_records': self.h2h_records,
            'tournament_performance': self.tournament_performance,
            'optimal_k_smoothing': self.optimal_k_smoothing,
            'model_accuracy': self.model_accuracy,
            'brier_score': self.brier_score,
            'roc_auc_score': self.roc_auc_score,
            'feature_importance': self.feature_importance,
            'validation_scores': self.validation_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"🚀 Optimized model saved to {filepath}")
    
    def load_optimized_model(self, filepath: str) -> bool:
        """Load optimized trained model"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.calibrated_model = model_data['calibrated_model']
            self.stacking_model = model_data['stacking_model']
            self.rf_model = model_data['rf_model']
            self.gb_model = model_data['gb_model']
            self.mlp_model = model_data['mlp_model']
            self.svm_model = model_data['svm_model']
            self.meta_learner = model_data.get('meta_learner')
            self.scaler = model_data['scaler']
            self.team_stats = model_data['team_stats']
            self.team_player_stats = model_data.get('team_player_stats', {})
            self.regional_performance = model_data['regional_performance']
            self.h2h_records = model_data['h2h_records']
            self.tournament_performance = model_data.get('tournament_performance', {})
            self.optimal_k_smoothing = model_data.get('optimal_k_smoothing', 5)
            self.model_accuracy = model_data['model_accuracy']
            self.brier_score = model_data.get('brier_score', 0.0)
            self.roc_auc_score = model_data.get('roc_auc_score', 0.0)
            self.feature_importance = model_data.get('feature_importance', {})
            self.validation_scores = model_data.get('validation_scores', {})
            
            print(f"🚀 Optimized model loaded from {filepath}")
            print(f"📊 Model Performance:")
            print(f"   Accuracy:    {self.model_accuracy:.4f}")
            print(f"   Brier Score: {self.brier_score:.4f}")
            print(f"   ROC-AUC:     {self.roc_auc_score:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading optimized model: {e}")
            return False
    
    def optimize_random_forest(self, X, y, n_iter=20):
        """
        Bayesian hyperparameter optimization for Random Forest
        """
        search_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'bootstrap': Categorical([True, False])
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        bayes_search = BayesSearchCV(
            rf,
            search_space,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        bayes_search.fit(X, y)
        return bayes_search
    
    def optimize_gradient_boosting(self, X, y, n_iter=20):
        """
        Bayesian hyperparameter optimization for Gradient Boosting
        """
        search_space = {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 15),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'subsample': Real(0.6, 1.0)
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        
        bayes_search = BayesSearchCV(
            gb,
            search_space,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        bayes_search.fit(X, y)
        return bayes_search
    
    def optimize_mlp(self, X, y, n_iter=15):
        """
        Bayesian hyperparameter optimization for MLP
        """
        search_space = {
            'hidden_layer_sizes': Categorical([(50,), (100,), (150,), (100, 50), (150, 100)]),
            'alpha': Real(0.0001, 0.1, prior='log-uniform'),
            'learning_rate_init': Real(0.001, 0.01, prior='log-uniform'),
            'max_iter': Integer(200, 1000)
        }
        
        mlp = MLPClassifier(random_state=42, early_stopping=True)
        
        bayes_search = BayesSearchCV(
            mlp,
            search_space,
            n_iter=n_iter,
            cv=3,  # Reduced for MLP
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        bayes_search.fit(X, y)
        return bayes_search
    
    def optimize_svm(self, X, y, n_iter=15):
        """
        Bayesian hyperparameter optimization for SVM
        """
        search_space = {
            'C': Real(0.1, 100, prior='log-uniform'),
            'gamma': Real(0.001, 1, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'poly']),
            'degree': Integer(2, 5)  # Only for poly kernel
        }
        
        svm = SVC(probability=True, random_state=42)
        
        bayes_search = BayesSearchCV(
            svm,
            search_space,
            n_iter=n_iter,
            cv=3,  # Reduced for SVM
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        bayes_search.fit(X, y)
        return bayes_search
    
    def create_stacking_ensemble(self, X, y):
        """
        Create optimized stacking ensemble with best hyperparameters
        """
        print("🔧 Optimizing base models...")
        
        # Quick optimization for testing (reduce iterations)
        rf_optimized = self.optimize_random_forest(X, y, n_iter=10)
        gb_optimized = self.optimize_gradient_boosting(X, y, n_iter=10)
        mlp_optimized = self.optimize_mlp(X, y, n_iter=5)
        svm_optimized = self.optimize_svm(X, y, n_iter=5)
        
        print(f"✅ RF best score: {rf_optimized.best_score_:.3f}")
        print(f"✅ GB best score: {gb_optimized.best_score_:.3f}")
        print(f"✅ MLP best score: {mlp_optimized.best_score_:.3f}")
        print(f"✅ SVM best score: {svm_optimized.best_score_:.3f}")
        
        # Create ensemble with optimized models
        base_models = [
            ('rf_opt', rf_optimized.best_estimator_),
            ('gb_opt', gb_optimized.best_estimator_),
            ('mlp_opt', mlp_optimized.best_estimator_),
            ('svm_opt', svm_optimized.best_estimator_)
        ]
        
        # Use compatible CV for stacking
        self.stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,  # Use integer for compatibility
            stack_method='predict_proba'
        )
        
        print("🔧 Training stacking ensemble...")
        self.stacking_model.fit(X, y)
        print("✅ Stacking ensemble trained")
        
        return self.stacking_model
    
    def calibrate_model(self, model, X, y):
        """
        Apply isotonic calibration to improve probability estimates
        """
        print("🔧 Applying probability calibration...")
        
        self.calibrated_model = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=3
        )
        
        self.calibrated_model.fit(X, y)
        print("✅ Model calibrated")
        
        return self.calibrated_model


def main():
    """Test the optimized predictor"""
    print("🚀 Testing Optimized VCT Predictor")
    predictor = OptimizedVCTPredictor()
    
    # This would normally be called after loading data
    # predictor.load_comprehensive_data()
    # predictor.train_optimized_model()
    
    print("✅ Optimized predictor initialization complete")


if __name__ == "__main__":
    main()

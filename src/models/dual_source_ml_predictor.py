
"""
Dual-Source Enhanced VCT ML Predictor
Integrates the optimized ML predictor with dual-source 58+ features from VLR.gg and RIB.gg
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
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


sys.path.append(str(Path(__file__).parent.parent))
from data_collection.dual_source_integrator import DualSourceIntegrator, UnifiedTeamStats
from preprocessing.enhanced_feature_engineering import EnhancedFeatureEngineer

class DualSourceMLPredictor:
    """
    Enhanced ML predictor utilizing dual-source data (VLR.gg + RIB.gg) with 58+ features.

    Key Enhancements:
    1. Integrates 58+ features from dual-source data
    2. Advanced Bayesian hyperparameter optimization
    3. Multi-source data validation and quality scoring
    4. Enhanced stacking ensemble with meta-learner
    5. Calibrated probability outputs with confidence intervals
    """

    def __init__(self, data_dir=None):
        """Initialize the dual-source enhanced ML predictor."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)


        self.dual_source_integrator = DualSourceIntegrator()
        self.feature_engineer = EnhancedFeatureEngineer()


        self.rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        self.gb_model = GradientBoostingClassifier(random_state=42)
        self.mlp_model = MLPClassifier(random_state=42, max_iter=1000)
        self.svm_model = SVC(probability=True, random_state=42)


        self.stacking_model = None
        self.calibrated_model = None
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)


        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()


        self.unified_team_data = {}
        self.training_features = None
        self.training_labels = None
        self.feature_names = []


        self.model_accuracy = 0.0
        self.brier_score = 0.0
        self.roc_auc_score = 0.0
        self.feature_importance = {}
        self.validation_scores = {}
        self.data_quality_score = 0.0

        print("🚀 Dual-Source Enhanced VCT ML Predictor initialized")
        print(f"📊 Expected features: {len(self.feature_engineer.get_feature_names())}")

    def collect_and_integrate_data(self) -> bool:
        """
        Collect data from both VLR.gg and RIB.gg sources and integrate them.

        Returns:
            bool: True if data collection was successful
        """
        try:
            print("🔄 Starting dual-source data collection...")


            self.unified_team_data = self.dual_source_integrator.collect_all_data()

            if not self.unified_team_data:
                print("❌ No data collected from sources")
                return False


            report = self.dual_source_integrator.generate_integration_report(self.unified_team_data)
            print(report)


            self.data_quality_score = np.mean([
                stats.data_confidence_score 
                for stats in self.unified_team_data.values()
            ])

            print(f"📈 Data collection successful: {len(self.unified_team_data)} teams")
            print(f"🔍 Overall data quality score: {self.data_quality_score:.3f}")

            return True

        except Exception as e:
            print(f"❌ Error in data collection: {e}")
            return False

    def prepare_training_data(self, match_data: List[Dict[str, Any]]) -> bool:
        """
        Prepare training data using enhanced 58+ features.

        Args:
            match_data: List of match dictionaries with team1, team2, winner

        Returns:
            bool: True if training data preparation was successful
        """
        try:
            print("🔄 Preparing enhanced training data with 58+ features...")

            X_data = []
            y_data = []
            valid_matches = 0

            for match in match_data:
                team1 = match.get('team1')
                team2 = match.get('team2')
                winner = match.get('winner')

                if not all([team1, team2, winner]) or winner not in [team1, team2]:
                    continue


                team1_stats = self.unified_team_data.get(team1.lower().replace(' ', '_'))
                team2_stats = self.unified_team_data.get(team2.lower().replace(' ', '_'))

                if not team1_stats or not team2_stats:
                    continue


                team1_dict = self._unified_stats_to_dict(team1_stats)
                team2_dict = self._unified_stats_to_dict(team2_stats)


                match_context = {
                    'tournament_importance': match.get('tournament_importance', 0.8),
                    'stage_importance': match.get('stage_importance', 0.7),
                    'tournament_region': match.get('tournament_region', 'International')
                }

                features = self.feature_engineer.create_enhanced_features(
                    team1_dict, team2_dict, match_context
                )

                X_data.append(list(features.values()))
                y_data.append(1 if winner == team1 else 0)
                valid_matches += 1

            if valid_matches < 50:
                print(f"❌ Insufficient training data: {valid_matches} valid matches")
                return False

            self.training_features = np.array(X_data)
            self.training_labels = np.array(y_data)
            self.feature_names = self.feature_engineer.get_feature_names()

            print(f"✅ Training data prepared:")
            print(f"   Matches: {valid_matches}")
            print(f"   Features: {self.training_features.shape[1]}")
            print(f"   Class balance: {np.mean(self.training_labels):.3f}")

            return True

        except Exception as e:
            print(f"❌ Error preparing training data: {e}")
            return False

    def _unified_stats_to_dict(self, unified_stats: UnifiedTeamStats) -> Dict[str, Any]:
        """Convert UnifiedTeamStats to dictionary for feature engineering."""
        return {
            'team_name': unified_stats.team_name,
            'region': unified_stats.region,
            'win_rate': unified_stats.win_rate,
            'matches_played': unified_stats.matches_played,
            'composite_team_rating': unified_stats.composite_team_rating,
            'vlr_rating': unified_stats.vlr_rating,
            'vlr_round_win_rate': unified_stats.vlr_round_win_rate,
            'vlr_recent_matches': unified_stats.vlr_recent_matches,
            'rib_first_blood_rate': unified_stats.rib_first_blood_rate,
            'rib_clutch_success_rate': unified_stats.rib_clutch_success_rate,
            'rib_eco_round_win_rate': unified_stats.rib_eco_round_win_rate,
            'rib_tactical_timeout_efficiency': unified_stats.rib_tactical_timeout_efficiency,
            'rib_comeback_factor': unified_stats.rib_comeback_factor,
            'rib_consistency_rating': unified_stats.rib_consistency_rating,
            'rib_current_streak': unified_stats.rib_current_streak,
            'rib_streak_type': unified_stats.rib_streak_type,
            'momentum_index': unified_stats.momentum_index,
            'tactical_diversity': unified_stats.tactical_diversity,
            'pressure_performance': unified_stats.pressure_performance,
            'adaptability_score': unified_stats.adaptability_score,
            'data_confidence_score': unified_stats.data_confidence_score,
            'cross_validation_score': unified_stats.cross_validation_score
        }

    def train_enhanced_model(self) -> bool:
        """
        Train the enhanced ML model with Bayesian optimization and dual-source features.

        Returns:
            bool: True if training was successful
        """
        if self.training_features is None or len(self.training_features) == 0:
            print("❌ No training data available")
            return False

        try:
            print("🚀 Starting enhanced model training with 58+ features...")
            print(f"📊 Training on {len(self.training_features)} samples")

            X = self.training_features
            y = self.training_labels


            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )


            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)


            tscv = TimeSeriesSplit(n_splits=5)

            print("🔍 Bayesian hyperparameter optimization...")


            print("  🌲 Optimizing Random Forest...")
            rf_space = {
                'n_estimators': Integer(200, 800),
                'max_depth': Integer(10, 40),
                'min_samples_split': Integer(2, 15),
                'max_features': Categorical(['sqrt', 'log2', 0.3]),
                'min_samples_leaf': Integer(1, 8)
            }

            rf_bayes = BayesSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                rf_space,
                n_iter=60,
                cv=tscv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            rf_bayes.fit(X_train_scaled, y_train)
            self.rf_model = rf_bayes.best_estimator_
            print(f"    Best RF score: {rf_bayes.best_score_:.4f}")


            print("  🌳 Optimizing Gradient Boosting...")
            gb_space = {
                'n_estimators': Integer(200, 1200),
                'learning_rate': Real(0.005, 0.2, prior='log-uniform'),
                'max_depth': Integer(4, 12),
                'subsample': Real(0.7, 1.0),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 15),
                'max_features': Categorical(['sqrt', 'log2', 0.5])
            }

            gb_bayes = BayesSearchCV(
                GradientBoostingClassifier(random_state=42),
                gb_space,
                n_iter=70,
                cv=tscv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            gb_bayes.fit(X_train_scaled, y_train)
            self.gb_model = gb_bayes.best_estimator_
            print(f"    Best GB score: {gb_bayes.best_score_:.4f}")


            print("  🧠 Optimizing Neural Network...")
            mlp_space = {
                'alpha': Real(1e-6, 1e-1, prior='log-uniform'),
                'learning_rate_init': Real(0.0001, 0.01, prior='log-uniform'),
                'hidden_layer_sizes': Categorical([
                    (150,), (200,), (100, 50), (200, 100), 
                    (150, 100, 50), (200, 150, 75)
                ]),
                'batch_size': Categorical([32, 64, 128]),
                'beta_1': Real(0.8, 0.99),
                'beta_2': Real(0.9, 0.999)
            }

            mlp_bayes = BayesSearchCV(
                MLPClassifier(random_state=42, max_iter=2000, early_stopping=True, validation_fraction=0.1),
                mlp_space,
                n_iter=40,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            mlp_bayes.fit(X_train_scaled, y_train)
            self.mlp_model = mlp_bayes.best_estimator_
            print(f"    Best MLP score: {mlp_bayes.best_score_:.4f}")


            print("  ⚡ Optimizing SVM...")
            svm_space = {
                'C': Real(0.01, 1000, prior='log-uniform'),
                'gamma': Real(1e-6, 1e-1, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'poly']),
                'degree': Integer(2, 4),
                'coef0': Real(0.0, 10.0)
            }

            svm_bayes = BayesSearchCV(
                SVC(probability=True, random_state=42),
                svm_space,
                n_iter=35,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            svm_bayes.fit(X_train_scaled, y_train)
            self.svm_model = svm_bayes.best_estimator_
            print(f"    Best SVM score: {svm_bayes.best_score_:.4f}")


            print("🏗️  Creating advanced StackingClassifier...")
            base_models = [
                ('rf_enhanced', self.rf_model),
                ('gb_enhanced', self.gb_model),
                ('mlp_enhanced', self.mlp_model),
                ('svm_enhanced', self.svm_model)
            ]


            meta_learner = LogisticRegression(
                random_state=42, 
                max_iter=2000,
                C=1.0,
                solver='lbfgs'
            )

            self.stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=tscv,
                stack_method='predict_proba',
                n_jobs=-1,
                passthrough=True
            )

            print("🎯 Training StackingClassifier...")
            self.stacking_model.fit(X_train_scaled, y_train)


            print("🎲 Applying probability calibration...")
            self.calibrated_model = CalibratedClassifierCV(
                self.stacking_model,
                method='isotonic',
                cv=5
            )

            self.calibrated_model.fit(X_train_scaled, y_train)


            print("\n📈 Enhanced Model Evaluation Results:")
            print("=" * 70)

            models = {
                'Random Forest (Enhanced)': self.rf_model,
                'Gradient Boosting (Enhanced)': self.gb_model,
                'Neural Network (Enhanced)': self.mlp_model,
                'SVM (Enhanced)': self.svm_model,
                'Stacking Ensemble': self.stacking_model,
                'Calibrated Stacking (Final)': self.calibrated_model
            }

            best_accuracy = 0
            for model_name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                brier = brier_score_loss(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)


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


            final_pred_proba = self.calibrated_model.predict_proba(X_test_scaled)[:, 1]
            self.model_accuracy = accuracy_score(y_test, self.calibrated_model.predict(X_test_scaled))
            self.brier_score = brier_score_loss(y_test, final_pred_proba)
            self.roc_auc_score = roc_auc_score(y_test, final_pred_proba)


            print("🔍 Enhanced Feature Importance Analysis:")
            print("-" * 50)

            if hasattr(self.gb_model, 'feature_importances_'):

                importance_pairs = list(zip(self.feature_names, self.gb_model.feature_importances_))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)

                print("Top 15 Most Important Features:")
                for i, (feature_name, importance) in enumerate(importance_pairs[:15], 1):
                    print(f"{i:2d}. {feature_name:<40} {importance:.4f}")

                self.feature_importance = dict(importance_pairs)


                feature_groups = self.feature_engineer.get_feature_importance_groups()
                group_importance = {}

                for group_name, features in feature_groups.items():
                    group_score = sum(
                        self.feature_importance.get(f, 0.0) for f in features
                    )
                    group_importance[group_name] = group_score

                print(f"\nFeature Group Importance:")
                for group, score in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {group:<25} {score:.4f}")

            print(f"\n🎉 Enhanced model training completed!")
            print(f"🎯 Final Enhanced Model Performance:")
            print(f"   Accuracy:           {self.model_accuracy:.4f}")
            print(f"   Brier Score:        {self.brier_score:.4f}")
            print(f"   ROC-AUC:           {self.roc_auc_score:.4f}")
            print(f"   Data Quality:       {self.data_quality_score:.4f}")
            print(f"   Features Used:      {len(self.feature_names)}")

            return True

        except Exception as e:
            print(f"❌ Error in enhanced model training: {e}")
            return False

    def predict_match_enhanced(self, team1: str, team2: str, 
                              match_context: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Make enhanced prediction using dual-source 58+ features.

        Args:
            team1: First team name
            team2: Second team name
            match_context: Additional match context

        Returns:
            Dictionary with enhanced prediction results
        """
        try:
            if not self.calibrated_model:
                print("❌ Enhanced model not trained")
                return None


            team1_key = team1.lower().replace(' ', '_')
            team2_key = team2.lower().replace(' ', '_')

            team1_stats = self.unified_team_data.get(team1_key)
            team2_stats = self.unified_team_data.get(team2_key)

            if not team1_stats or not team2_stats:
                print(f"❌ Team data not found for {team1} or {team2}")
                return None


            team1_dict = self._unified_stats_to_dict(team1_stats)
            team2_dict = self._unified_stats_to_dict(team2_stats)


            context = match_context or {
                'tournament_importance': 0.8,
                'stage_importance': 0.7,
                'tournament_region': 'International'
            }

            features = self.feature_engineer.create_enhanced_features(
                team1_dict, team2_dict, context
            )


            feature_vector = np.array(list(features.values())).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)


            proba = self.calibrated_model.predict_proba(feature_vector_scaled)[0]
            team1_prob = proba[1]
            team2_prob = proba[0]


            if team1_prob > team2_prob:
                predicted_winner = team1
                confidence = team1_prob
            else:
                predicted_winner = team2
                confidence = team2_prob


            if confidence >= 0.95:
                confidence_level = "Extremely High"
            elif confidence >= 0.85:
                confidence_level = "Very High"
            elif confidence >= 0.75:
                confidence_level = "High"
            elif confidence >= 0.65:
                confidence_level = "Medium"
            elif confidence >= 0.55:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"


            uncertainty = abs(0.5 - confidence)


            prediction_quality = min(
                team1_stats.data_confidence_score,
                team2_stats.data_confidence_score
            )

            return {
                'team1': team1,
                'team2': team2,
                'predicted_winner': predicted_winner,
                'team1_probability': float(team1_prob),
                'team2_probability': float(team2_prob),
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'prediction_uncertainty': float(uncertainty),
                'prediction_quality': float(prediction_quality),


                'model_accuracy': float(self.model_accuracy),
                'brier_score': float(self.brier_score),
                'roc_auc': float(self.roc_auc_score),
                'data_quality_score': float(self.data_quality_score),


                'features_used': len(self.feature_names),
                'dual_source_features': True,
                'vlr_data_available': team1_stats.vlr_data_available and team2_stats.vlr_data_available,
                'rib_data_available': team1_stats.rib_data_available and team2_stats.rib_data_available,


                'team1_momentum': float(team1_dict.get('momentum_index', 0.5)),
                'team2_momentum': float(team2_dict.get('momentum_index', 0.5)),
                'team1_pressure_performance': float(team1_dict.get('pressure_performance', 0.5)),
                'team2_pressure_performance': float(team2_dict.get('pressure_performance', 0.5)),
                'tactical_advantage': team1_dict.get('tactical_diversity', 0.5) - team2_dict.get('tactical_diversity', 0.5),


                'enhanced_model': True,
                'calibrated_probabilities': True,
                'bayesian_optimized': True,
                'prediction_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Enhanced prediction error: {e}")
            return None

    def save_enhanced_model(self, filepath: str) -> bool:
        """Save the enhanced trained model."""
        try:
            model_data = {
                'calibrated_model': self.calibrated_model,
                'stacking_model': self.stacking_model,
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'mlp_model': self.mlp_model,
                'svm_model': self.svm_model,
                'meta_learner': self.meta_learner,
                'scaler': self.scaler,
                'unified_team_data': self.unified_team_data,
                'feature_names': self.feature_names,
                'model_accuracy': self.model_accuracy,
                'brier_score': self.brier_score,
                'roc_auc_score': self.roc_auc_score,
                'data_quality_score': self.data_quality_score,
                'feature_importance': self.feature_importance,
                'validation_scores': self.validation_scores,
                'training_timestamp': datetime.now().isoformat(),
                'model_version': 'dual_source_enhanced_v1.0'
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"🚀 Enhanced model saved to {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error saving enhanced model: {e}")
            return False

    def load_enhanced_model(self, filepath: str) -> bool:
        """Load enhanced trained model."""
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)


            self.calibrated_model = model_data.get('calibrated_model')
            self.stacking_model = model_data.get('stacking_model')
            self.rf_model = model_data.get('rf_model')
            self.gb_model = model_data.get('gb_model')
            self.mlp_model = model_data.get('mlp_model')
            self.svm_model = model_data.get('svm_model')
            self.meta_learner = model_data.get('meta_learner')
            self.scaler = model_data.get('scaler')


            self.unified_team_data = model_data.get('unified_team_data', {})
            self.feature_names = model_data.get('feature_names', [])
            self.model_accuracy = model_data.get('model_accuracy', 0.0)
            self.brier_score = model_data.get('brier_score', 0.0)
            self.roc_auc_score = model_data.get('roc_auc_score', 0.0)
            self.data_quality_score = model_data.get('data_quality_score', 0.0)
            self.feature_importance = model_data.get('feature_importance', {})
            self.validation_scores = model_data.get('validation_scores', {})

            print(f"🚀 Enhanced model loaded from {filepath}")
            print(f"📊 Model Performance:")
            print(f"   Accuracy:           {self.model_accuracy:.4f}")
            print(f"   Brier Score:        {self.brier_score:.4f}")
            print(f"   ROC-AUC:           {self.roc_auc_score:.4f}")
            print(f"   Data Quality:       {self.data_quality_score:.4f}")
            print(f"   Features:          {len(self.feature_names)}")
            print(f"   Teams Available:    {len(self.unified_team_data)}")
            print(f"   Model Version:      {model_data.get('model_version', 'unknown')}")

            return True

        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'model_type': 'dual_source_enhanced',
            'features_count': len(self.feature_names),
            'teams_available': len(self.unified_team_data),
            'performance_metrics': {
                'accuracy': self.model_accuracy,
                'brier_score': self.brier_score,
                'roc_auc': self.roc_auc_score,
                'data_quality': self.data_quality_score
            },
            'validation_scores': self.validation_scores,
            'feature_importance_top10': dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ) if self.feature_importance else {},
            'model_components': {
                'random_forest': self.rf_model is not None,
                'gradient_boosting': self.gb_model is not None,
                'neural_network': self.mlp_model is not None,
                'svm': self.svm_model is not None,
                'stacking_ensemble': self.stacking_model is not None,
                'calibrated_probabilities': self.calibrated_model is not None
            },
            'data_sources': {
                'vlr_gg': True,
                'rib_gg': True,
                'dual_source_integration': True
            }
        }


def main():
    """Test the enhanced dual-source predictor."""
    print("🚀 Testing Dual-Source Enhanced VCT Predictor")


    predictor = DualSourceMLPredictor()


    print("\n1. Testing data collection...")





    print("✅ Dual-source enhanced predictor initialization complete")
    print("🔧 Ready for training with 58+ features")

if __name__ == "__main__":
    main()
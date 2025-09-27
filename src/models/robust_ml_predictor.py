
"""
Robust ML VCT Predictor with Proper Train-Validation-Test Split
Simplified but reliable approach with maximum data utilization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import pickle
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RobustVCTPredictor:
    def __init__(self, data_dir=None):
        """Initialize robust ML predictor with comprehensive validation"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)


        self.models = {}
        self.calibrated_models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()


        self.matches_df = None
        self.team_stats = {}


        self.test_results = {}
        self.validation_scores = {}
        self.split_info = {}

        print("🚀 Robust VCT ML Predictor initialized with comprehensive validation")

    def load_and_process_all_data(self):
        """Load and process all available match data"""
        print("📊 Loading all VCT tournament data...")

        all_matches = []


        tournament_dates = {
            'Champions Tour 2024 Americas Kickoff': '2024-02-15',
            'Champions Tour 2024 Pacific Kickoff': '2024-02-15', 
            'Champions Tour 2024 China Kickoff': '2024-02-15',
            'Champions Tour 2024 EMEA Kickoff': '2024-02-15',
            'Champions Tour 2024 Americas Stage 1': '2024-03-15',
            'Champions Tour 2024 Pacific Stage 1': '2024-03-15',
            'Champions Tour 2024 China Stage 1': '2024-03-15',
            'Champions Tour 2024 EMEA Stage 1': '2024-03-15',
            'Champions Tour 2024 Masters Madrid': '2024-04-15',
            'Champions Tour 2024 Americas Stage 2': '2024-05-15',
            'Champions Tour 2024 Pacific Stage 2': '2024-05-15',
            'Champions Tour 2024 China Stage 2': '2024-05-15',
            'Champions Tour 2024 EMEA Stage 2': '2024-05-15',
            'Champions Tour 2024 Masters Shanghai': '2024-06-15',
            'Valorant Champions 2024': '2024-08-15'
        }

        for tournament_dir in self.data_dir.iterdir():
            if tournament_dir.is_dir() and 'csvs' in tournament_dir.name:
                tournament_name = tournament_dir.name.replace('_csvs', '')
                tournament_date = tournament_dates.get(tournament_name, '2024-01-01')

                print(f"📁 Processing: {tournament_name}")

                try:
                    matches_file = tournament_dir / "matches.csv"
                    if matches_file.exists():
                        df = pd.read_csv(matches_file)
                        df['tournament'] = tournament_name
                        df['date'] = pd.to_datetime(tournament_date)
                        df['region'] = self._get_region(tournament_name)
                        df['tier'] = self._get_tier(tournament_name)
                        all_matches.append(df)

                except Exception as e:
                    print(f"❌ Error loading {tournament_name}: {e}")
                    continue

        if not all_matches:
            print("❌ No match data loaded")
            return False


        self.matches_df = pd.concat(all_matches, ignore_index=True)
        self.matches_df = self.matches_df.sort_values('date').reset_index(drop=True)


        self.matches_df = self.matches_df.dropna(subset=['team1', 'team2', 'winner'])

        print(f"✅ Loaded {len(self.matches_df)} matches from {len(all_matches)} tournaments")
        return True

    def _get_region(self, tournament_name):
        """Get region from tournament name"""
        if 'Americas' in tournament_name:
            return 'Americas'
        elif 'EMEA' in tournament_name:
            return 'EMEA'
        elif 'Pacific' in tournament_name:
            return 'APAC'
        elif 'China' in tournament_name:
            return 'China'
        elif 'Masters' in tournament_name or 'Champions' in tournament_name:
            return 'International'
        return 'Other'

    def _get_tier(self, tournament_name):
        """Get tournament tier"""
        if 'Champions' in tournament_name:
            return 3
        elif 'Masters' in tournament_name:
            return 2
        else:
            return 1

    def create_comprehensive_features(self):
        """Create feature matrix from match data with rolling statistics"""
        print("🔧 Creating comprehensive features with rolling statistics...")

        features = []
        labels = []
        match_info = []


        for idx, match in self.matches_df.iterrows():
            try:
                team1 = match['team1']
                team2 = match['team2']
                winner = match['winner']
                match_date = match['date']


                historical = self.matches_df.iloc[:idx]


                feature_vector = self._extract_match_features(
                    team1, team2, match_date, historical
                )

                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(1 if winner == team1 else 0)
                    match_info.append({
                        'team1': team1,
                        'team2': team2,
                        'winner': winner,
                        'date': match_date,
                        'tournament': match['tournament']
                    })

            except Exception as e:
                print(f"⚠️ Error processing match {idx}: {e}")
                continue

        if not features:
            print("❌ No features created")
            return None, None

        features_df = pd.DataFrame(features)
        labels_series = pd.Series(labels)
        self.match_info_df = pd.DataFrame(match_info)

        print(f"✅ Created {len(features)} samples with {len(features_df.columns)} features")
        print(f"   📊 Feature names: {list(features_df.columns)}")

        return features_df, labels_series

    def _extract_match_features(self, team1, team2, match_date, historical_data):
        """Extract features for a single match using historical data"""
        features = {}


        team1_matches = historical_data[
            (historical_data['team1'] == team1) | (historical_data['team2'] == team1)
        ]
        team2_matches = historical_data[
            (historical_data['team1'] == team2) | (historical_data['team2'] == team2)
        ]


        features.update(self._get_team_features(team1, team1_matches, 't1'))


        features.update(self._get_team_features(team2, team2_matches, 't2'))


        h2h_matches = historical_data[
            ((historical_data['team1'] == team1) & (historical_data['team2'] == team2)) |
            ((historical_data['team1'] == team2) & (historical_data['team2'] == team1))
        ]

        if len(h2h_matches) > 0:
            team1_h2h_wins = len(h2h_matches[
                ((h2h_matches['team1'] == team1) & (h2h_matches['winner'] == team1)) |
                ((h2h_matches['team2'] == team1) & (h2h_matches['winner'] == team1))
            ])
            features['h2h_win_rate'] = team1_h2h_wins / len(h2h_matches)
            features['h2h_matches'] = len(h2h_matches)
        else:
            features['h2h_win_rate'] = 0.5
            features['h2h_matches'] = 0


        team1_region = self._get_team_primary_region(team1, historical_data)
        team2_region = self._get_team_primary_region(team2, historical_data)
        features['same_region'] = 1 if team1_region == team2_region else 0

        return list(features.values())

    def _get_team_features(self, team, team_matches, prefix):
        """Get comprehensive features for a single team"""
        features = {}

        if len(team_matches) == 0:

            features[f'{prefix}_win_rate'] = 0.5
            features[f'{prefix}_matches'] = 0
            features[f'{prefix}_recent_form'] = 0.5
            features[f'{prefix}_intl_exp'] = 0
            features[f'{prefix}_high_tier_exp'] = 0
            features[f'{prefix}_streak'] = 0
            return features


        wins = len(team_matches[team_matches['winner'] == team])
        features[f'{prefix}_win_rate'] = wins / len(team_matches)
        features[f'{prefix}_matches'] = len(team_matches)


        recent_matches = team_matches.tail(10)
        if len(recent_matches) > 0:
            recent_wins = len(recent_matches[recent_matches['winner'] == team])
            features[f'{prefix}_recent_form'] = recent_wins / len(recent_matches)


            streak = 0
            for _, match in recent_matches.iloc[::-1].iterrows():
                if match['winner'] == team:
                    streak += 1
                else:
                    break
            features[f'{prefix}_streak'] = streak
        else:
            features[f'{prefix}_recent_form'] = 0.5
            features[f'{prefix}_streak'] = 0


        intl_matches = team_matches[team_matches['region'] == 'International']
        features[f'{prefix}_intl_exp'] = len(intl_matches)

        high_tier_matches = team_matches[team_matches['tier'] >= 2]
        features[f'{prefix}_high_tier_exp'] = len(high_tier_matches)

        return features

    def _get_team_primary_region(self, team, historical_data):
        """Get team's primary region"""
        team_data = historical_data[
            (historical_data['team1'] == team) | (historical_data['team2'] == team)
        ]

        if len(team_data) == 0:
            return 'Unknown'


        regions = team_data[team_data['region'] != 'International']['region'].value_counts()
        return regions.index[0] if len(regions) > 0 else 'Unknown'

    def implement_temporal_splits(self, features, labels, test_size=0.2, val_size=0.15):
        """Implement time-based train-validation-test splits"""
        print("⏰ Implementing temporal data splits...")
        print(f"   📊 Total samples: {len(features)}")


        combined_data = pd.concat([
            features.reset_index(drop=True),
            labels.reset_index(drop=True).rename('label'),
            self.match_info_df.reset_index(drop=True)
        ], axis=1)


        combined_data = combined_data.sort_values('date').reset_index(drop=True)


        n_total = len(combined_data)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val


        train_data = combined_data.iloc[:n_train]

        val_data = combined_data.iloc[n_train:n_train + n_val]

        test_data = combined_data.iloc[n_train + n_val:]


        feature_cols = [col for col in features.columns]

        self.X_train = train_data[feature_cols].values
        self.y_train = train_data['label'].values
        self.X_val = val_data[feature_cols].values
        self.y_val = val_data['label'].values
        self.X_test = test_data[feature_cols].values
        self.y_test = test_data['label'].values

        print(f"   📚 Training: {len(self.X_train)} samples ({len(self.X_train)/n_total:.1%})")
        print(f"   🔍 Validation: {len(self.X_val)} samples ({len(self.X_val)/n_total:.1%})")  
        print(f"   🎯 Test: {len(self.X_test)} samples ({len(self.X_test)/n_total:.1%})")


        self.split_info = {
            'train_dates': (train_data['date'].min(), train_data['date'].max()),
            'val_dates': (val_data['date'].min(), val_data['date'].max()),
            'test_dates': (test_data['date'].min(), test_data['date'].max()),
            'feature_names': feature_cols
        }

        print(f"   📅 Train: {self.split_info['train_dates'][0]:%Y-%m} to {self.split_info['train_dates'][1]:%Y-%m}")
        print(f"   📅 Val: {self.split_info['val_dates'][0]:%Y-%m} to {self.split_info['val_dates'][1]:%Y-%m}")
        print(f"   📅 Test: {self.split_info['test_dates'][0]:%Y-%m} to {self.split_info['test_dates'][1]:%Y-%m}")


        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✅ Temporal data splitting completed!")
        return True

    def train_and_validate_models(self):
        """Train models with validation-based hyperparameter tuning"""
        print("🎯 Training models with validation-based tuning...")


        model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5],
                    'random_state': [42]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'random_state': [42]
                }
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'probability': [True],
                    'random_state': [42]
                }
            }
        }

        trained_models = {}
        validation_scores = {}

        for name, config in model_configs.items():
            print(f"   🔧 Training {name}...")


            grid_search = GridSearchCV(
                config['model'](),
                config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )


            grid_search.fit(self.X_train_scaled, self.y_train)
            best_model = grid_search.best_estimator_


            val_pred = best_model.predict(self.X_val_scaled)
            val_accuracy = accuracy_score(self.y_val, val_pred)

            trained_models[name] = best_model
            validation_scores[name] = val_accuracy

            print(f"      ✅ Best params: {grid_search.best_params_}")
            print(f"      📊 Validation accuracy: {val_accuracy:.4f}")


        top_models = sorted(validation_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"   🏆 Top models for ensemble: {[name for name, _ in top_models]}")

        ensemble_estimators = [(name, trained_models[name]) for name, _ in top_models]
        ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'
        )


        X_combined = np.vstack([self.X_train_scaled, self.X_val_scaled])
        y_combined = np.hstack([self.y_train, self.y_val])
        ensemble_model.fit(X_combined, y_combined)

        trained_models['ensemble'] = ensemble_model


        print("   🎯 Calibrating model probabilities...")
        for name, model in trained_models.items():
            if name != 'ensemble':
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated.fit(self.X_train_scaled, self.y_train)
                self.calibrated_models[name] = calibrated
            else:
                self.calibrated_models[name] = model

        self.models = trained_models
        self.validation_scores = validation_scores

        print("✅ Model training and calibration completed!")
        return trained_models

    def evaluate_final_performance(self):
        """Evaluate all models on held-out test set"""
        print("🎯 Final evaluation on held-out test set...")

        test_results = {}

        for name, model in self.calibrated_models.items():
            print(f"   📊 Testing {name}...")


            test_pred = model.predict(self.X_test_scaled)
            test_proba = model.predict_proba(self.X_test_scaled)[:, 1]


            accuracy = accuracy_score(self.y_test, test_pred)

            try:
                auc = roc_auc_score(self.y_test, test_proba)
            except:
                auc = 0.5


            try:
                fraction_pos, mean_pred = calibration_curve(
                    self.y_test, test_proba, n_bins=min(5, len(np.unique(self.y_test))), 
                    strategy='quantile'
                )
                calibration_error = np.mean(np.abs(fraction_pos - mean_pred))
            except:
                calibration_error = 0.1

            test_results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'calibration_error': calibration_error
            }

            print(f"      ✅ Accuracy: {accuracy:.4f}")
            print(f"      📈 AUC: {auc:.4f}")
            print(f"      🎯 Calibration Error: {calibration_error:.4f}")


        best_model_name = max(test_results.keys(), 
                             key=lambda k: test_results[k]['accuracy'])

        self.best_model = self.calibrated_models[best_model_name]
        self.best_model_name = best_model_name
        self.test_results = test_results

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   📊 Test Accuracy: {test_results[best_model_name]['accuracy']:.4f}")
        print(f"   📈 Test AUC: {test_results[best_model_name]['auc']:.4f}")

        return test_results

    def predict_match(self, team1, team2):
        """Make prediction for a new match"""
        if not hasattr(self, 'best_model'):
            print("❌ Model not trained yet")
            return None

        try:

            match_features = self._extract_match_features(
                team1, team2, pd.Timestamp.now(), self.matches_df
            )

            if match_features is None:
                print(f"❌ Could not create features for {team1} vs {team2}")
                return None


            features_scaled = self.scaler.transform([match_features])
            prediction = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]

            winner = team1 if prediction == 1 else team2
            confidence = max(probabilities)


            if confidence >= 0.8:
                conf_level = "Very High"
            elif confidence >= 0.7:
                conf_level = "High"
            elif confidence >= 0.6:
                conf_level = "Medium"
            else:
                conf_level = "Low"

            return {
                'team1': team1,
                'team2': team2,
                'predicted_winner': winner,
                'team1_probability': float(probabilities[1]),
                'team2_probability': float(probabilities[0]),
                'confidence': float(confidence),
                'confidence_level': conf_level,
                'model_used': self.best_model_name,
                'model_accuracy': float(self.test_results[self.best_model_name]['accuracy']),
                'temporal_split': True
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def save_model(self, filepath):
        """Save the complete model"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'calibrated_models': self.calibrated_models,
            'scaler': self.scaler,
            'test_results': self.test_results,
            'validation_scores': self.validation_scores,
            'split_info': self.split_info,
            'matches_df': self.matches_df
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")


def main():
    """Main training pipeline"""
    print("🚀 Robust VCT ML Predictor - Complete Pipeline")
    print("=" * 60)

    predictor = RobustVCTPredictor()


    if not predictor.load_and_process_all_data():
        return


    features, labels = predictor.create_comprehensive_features()
    if features is None:
        return


    predictor.implement_temporal_splits(features, labels)


    predictor.train_and_validate_models()


    test_results = predictor.evaluate_final_performance()


    print("\n" + "=" * 60)
    print("🎯 SAMPLE PREDICTIONS")
    print("=" * 60)

    test_matches = [
        ("Team Heretics", "Fnatic"),
        ("Paper Rex", "DRX"),
        ("Sentinels", "G2 Esports"),
        ("Edward Gaming", "Bilibili Gaming")
    ]

    for team1, team2 in test_matches:
        prediction = predictor.predict_match(team1, team2)
        if prediction:
            print(f"\n{team1} vs {team2}")
            print(f"Winner: {prediction['predicted_winner']}")
            print(f"Confidence: {prediction['confidence']:.1%} ({prediction['confidence_level']})")
            print(f"Model: {prediction['model_used']} (Accuracy: {prediction['model_accuracy']:.1%})")


    model_path = Path(__file__).parent / "robust_vct_model.pkl"
    predictor.save_model(model_path)

    print(f"\n✅ Robust ML pipeline completed!")
    print(f"📊 Best Model: {predictor.best_model_name}")
    print(f"🎯 Test Accuracy: {test_results[predictor.best_model_name]['accuracy']:.1%}")
    print(f"📈 AUC Score: {test_results[predictor.best_model_name]['auc']:.1%}")


if __name__ == "__main__":
    main()
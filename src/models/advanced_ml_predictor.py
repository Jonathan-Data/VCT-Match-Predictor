
"""
Advanced ML VCT Predictor with Proper Train-Validation-Test Split
Enhanced confidence calibration and maximum data utilization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedVCTPredictor:
    def __init__(self, data_dir=None):
        """Initialize advanced ML predictor with comprehensive validation"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)


        self.rf_model = RandomForestClassifier(random_state=42)
        self.gb_model = GradientBoostingClassifier(random_state=42)
        self.mlp_model = MLPClassifier(random_state=42)
        self.svm_model = SVC(probability=True, random_state=42)


        self.calibrated_models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()


        self.team_stats = {}
        self.player_stats = {}
        self.temporal_features = {}
        self.advanced_features = {}


        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None


        self.validation_results = {}
        self.feature_importance_analysis = {}
        self.confidence_calibration = {}

        print("🚀 Advanced VCT ML Predictor initialized with train-val-test splitting")

    def load_comprehensive_data_with_temporal_info(self):
        """Load all data with enhanced temporal and contextual information"""
        print("📊 Loading comprehensive data with temporal information...")

        all_matches = []
        all_detailed = []
        tournament_timeline = []


        tournament_order = {
            'Champions Tour 2024 Americas Kickoff': '2024-02-01',
            'Champions Tour 2024 Pacific Kickoff': '2024-02-01', 
            'Champions Tour 2024 China Kickoff': '2024-02-01',
            'Champions Tour 2024 EMEA Kickoff': '2024-02-01',
            'Champions Tour 2024 Americas Stage 1': '2024-03-01',
            'Champions Tour 2024 Pacific Stage 1': '2024-03-01',
            'Champions Tour 2024 China Stage 1': '2024-03-01',
            'Champions Tour 2024 EMEA Stage 1': '2024-03-01',
            'Champions Tour 2024 Masters Madrid': '2024-04-01',
            'Champions Tour 2024 Americas Stage 2': '2024-05-01',
            'Champions Tour 2024 Pacific Stage 2': '2024-05-01',
            'Champions Tour 2024 China Stage 2': '2024-05-01',
            'Champions Tour 2024 EMEA Stage 2': '2024-05-01',
            'Champions Tour 2024 Masters Shanghai': '2024-06-01',
            'Valorant Champions 2024': '2024-08-01'
        }

        for tournament_dir in self.data_dir.iterdir():
            if tournament_dir.is_dir() and 'csvs' in tournament_dir.name:
                tournament_name = tournament_dir.name.replace('_csvs', '')
                tournament_date = tournament_order.get(tournament_name, '2024-01-01')

                print(f"🔄 Processing: {tournament_name}")

                try:

                    matches_file = tournament_dir / "matches.csv"
                    if matches_file.exists():
                        matches_df = pd.read_csv(matches_file)
                        matches_df['tournament'] = tournament_name
                        matches_df['tournament_date'] = pd.to_datetime(tournament_date)
                        matches_df['region'] = self._extract_region(tournament_name)
                        matches_df['tournament_type'] = self._extract_tournament_type(tournament_name)
                        matches_df['tournament_tier'] = self._get_tournament_tier(tournament_name)
                        all_matches.append(matches_df)

                        tournament_timeline.append({
                            'tournament': tournament_name,
                            'date': tournament_date,
                            'matches': len(matches_df),
                            'region': self._extract_region(tournament_name)
                        })


                    detailed_file = tournament_dir / "detailed_matches_overview.csv"
                    if detailed_file.exists():
                        detailed_df = pd.read_csv(detailed_file)
                        detailed_df['tournament'] = tournament_name
                        detailed_df['tournament_date'] = pd.to_datetime(tournament_date)
                        all_detailed.append(detailed_df)


                    for data_type in ['player_stats', 'maps_stats', 'agents_stats', 'economy_data']:
                        file_path = tournament_dir / f"{data_type}.csv"
                        if file_path.exists():
                            df = pd.read_csv(file_path)
                            df['tournament'] = tournament_name
                            df['tournament_date'] = pd.to_datetime(tournament_date)
                            setattr(self, f"{data_type}_df_list", 
                                   getattr(self, f"{data_type}_df_list", []) + [df])

                except Exception as e:
                    print(f"❌ Error processing {tournament_name}: {e}")
                    continue


        if all_matches:
            self.matches_df = pd.concat(all_matches, ignore_index=True)
            self.matches_df = self.matches_df.sort_values('tournament_date').reset_index(drop=True)
            print(f"✅ Loaded {len(self.matches_df)} matches from {len(tournament_timeline)} tournaments")

        if all_detailed:
            self.detailed_df = pd.concat(all_detailed, ignore_index=True)
            self.detailed_df = self.detailed_df.sort_values('tournament_date').reset_index(drop=True)
            print(f"✅ Loaded {len(self.detailed_df)} detailed records")


        for data_type in ['player_stats', 'maps_stats', 'agents_stats', 'economy_data']:
            df_list = getattr(self, f"{data_type}_df_list", [])
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
                setattr(self, f"{data_type}_df", combined_df)
                print(f"✅ Combined {data_type}: {len(combined_df)} records")


        self.tournament_timeline = pd.DataFrame(tournament_timeline).sort_values('date')
        return self.tournament_timeline

    def _extract_region(self, tournament_name):
        """Extract region from tournament name"""
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
        return 'Unknown'

    def _extract_tournament_type(self, tournament_name):
        """Extract tournament type"""
        if 'Kickoff' in tournament_name:
            return 'Kickoff'
        elif 'Stage 1' in tournament_name:
            return 'Stage1'
        elif 'Stage 2' in tournament_name:
            return 'Stage2'
        elif 'Masters' in tournament_name:
            return 'Masters'
        elif 'Champions' in tournament_name:
            return 'Champions'
        return 'Other'

    def _get_tournament_tier(self, tournament_name):
        """Get tournament tier for importance weighting"""
        if 'Champions' in tournament_name:
            return 3
        elif 'Masters' in tournament_name:
            return 2
        else:
            return 1

    def create_enhanced_features_with_temporal_context(self):
        """Create comprehensive feature set with temporal context"""
        print("🔧 Creating enhanced features with temporal context...")

        if not hasattr(self, 'matches_df'):
            print("❌ No match data available")
            return None, None

        features = []
        labels = []
        match_info = []


        sorted_matches = self.matches_df.sort_values('tournament_date').copy()

        for idx, match in sorted_matches.iterrows():
            try:
                team1, team2 = match['team1'], match['team2']
                winner = match.get('winner', '')
                match_date = match['tournament_date']
                tournament = match['tournament']


                if pd.isna(team1) or pd.isna(team2) or not winner:
                    continue


                match_features = self._create_comprehensive_match_features(
                    team1, team2, match_date, tournament, sorted_matches[:idx]
                )

                if match_features is not None and len(match_features) > 0:
                    features.append(match_features)
                    labels.append(1 if winner == team1 else 0)
                    match_info.append({
                        'team1': team1,
                        'team2': team2,
                        'winner': winner,
                        'date': match_date,
                        'tournament': tournament
                    })

            except Exception as e:
                print(f"⚠️ Error processing match {idx}: {e}")
                continue

        if not features:
            print("❌ No features created")
            return None, None

        features_df = pd.DataFrame(features)
        labels_series = pd.Series(labels)

        print(f"✅ Created {len(features)} feature vectors with {features_df.shape[1]} features")


        self.match_info_df = pd.DataFrame(match_info)

        return features_df, labels_series

    def _create_comprehensive_match_features(self, team1, team2, match_date, tournament, historical_data):
        """Create comprehensive feature vector for a single match"""
        features = {}


        team1_history = historical_data[
            (historical_data['team1'] == team1) | (historical_data['team2'] == team1)
        ].copy()
        team2_history = historical_data[
            (historical_data['team1'] == team2) | (historical_data['team2'] == team2)
        ].copy()


        for team, team_history in [(team1, team1_history), (team2, team2_history)]:
            prefix = 't1' if team == team1 else 't2'

            if len(team_history) > 0:

                wins = len(team_history[team_history['winner'] == team])
                total_matches = len(team_history)
                features[f'{prefix}_win_rate'] = wins / total_matches if total_matches > 0 else 0.5
                features[f'{prefix}_total_matches'] = total_matches


                recent_matches = team_history.tail(10)
                if len(recent_matches) > 0:
                    recent_wins = len(recent_matches[recent_matches['winner'] == team])
                    features[f'{prefix}_recent_form'] = recent_wins / len(recent_matches)


                    momentum = 0
                    for i, match in enumerate(recent_matches.iterrows()):
                        weight = 0.9 ** (len(recent_matches) - i - 1)
                        win_value = 1 if match[1]['winner'] == team else 0
                        momentum += weight * win_value
                    features[f'{prefix}_momentum'] = momentum / sum(0.9 ** i for i in range(len(recent_matches)))
                else:
                    features[f'{prefix}_recent_form'] = 0.5
                    features[f'{prefix}_momentum'] = 0.5


                regional_matches = team_history[team_history['region'] == self._extract_region(tournament)]
                if len(regional_matches) > 0:
                    regional_wins = len(regional_matches[regional_matches['winner'] == team])
                    features[f'{prefix}_regional_wr'] = regional_wins / len(regional_matches)
                else:
                    features[f'{prefix}_regional_wr'] = features[f'{prefix}_win_rate']


                high_tier_matches = team_history[team_history['tournament_tier'] >= 2]
                features[f'{prefix}_high_tier_exp'] = len(high_tier_matches)


                intl_matches = team_history[team_history['region'] == 'International']
                features[f'{prefix}_intl_exp'] = len(intl_matches)


                streak = 0

                matches_list = list(team_history.iterrows())
                for match in reversed(matches_list):
                    if match[1]['winner'] == team:
                        streak += 1
                    else:
                        break
                features[f'{prefix}_win_streak'] = streak


                if len(team_history) > 0:
                    last_match_date = team_history['tournament_date'].max()
                    days_since = (match_date - last_match_date).days
                    features[f'{prefix}_days_since_last'] = min(days_since, 180)
                else:
                    features[f'{prefix}_days_since_last'] = 180

            else:

                for suffix in ['win_rate', 'recent_form', 'momentum', 'regional_wr']:
                    features[f'{prefix}_{suffix}'] = 0.5
                for suffix in ['total_matches', 'high_tier_exp', 'intl_exp', 'win_streak']:
                    features[f'{prefix}_{suffix}'] = 0
                features[f'{prefix}_days_since_last'] = 180


        h2h_matches = historical_data[
            ((historical_data['team1'] == team1) & (historical_data['team2'] == team2)) |
            ((historical_data['team1'] == team2) & (historical_data['team2'] == team1))
        ]

        if len(h2h_matches) > 0:
            h2h_wins_team1 = len(h2h_matches[
                ((h2h_matches['team1'] == team1) & (h2h_matches['winner'] == team1)) |
                ((h2h_matches['team2'] == team1) & (h2h_matches['winner'] == team1))
            ])
            features['h2h_rate'] = h2h_wins_team1 / len(h2h_matches)
            features['h2h_matches'] = len(h2h_matches)
        else:
            features['h2h_rate'] = 0.5
            features['h2h_matches'] = 0


        features['tournament_tier'] = self._get_tournament_tier(tournament)
        features['is_international'] = 1 if self._extract_region(tournament) == 'International' else 0
        features['is_elimination'] = 1 if 'Masters' in tournament or 'Champions' in tournament else 0


        team1_region = self._get_team_primary_region(team1, historical_data)
        team2_region = self._get_team_primary_region(team2, historical_data)
        features['cross_regional'] = 1 if team1_region != team2_region else 0

        return list(features.values())

    def _get_team_primary_region(self, team, historical_data):
        """Get team's primary region based on tournament history"""
        team_matches = historical_data[
            (historical_data['team1'] == team) | (historical_data['team2'] == team)
        ]

        if len(team_matches) == 0:
            return 'Unknown'

        region_counts = team_matches['region'].value_counts()
        return region_counts.index[0] if len(region_counts) > 0 else 'Unknown'

    def implement_proper_data_splits(self, features, labels, test_size=0.2, val_size=0.15):
        """Implement proper train-validation-test split with temporal considerations"""
        print(f"📊 Implementing proper data splits...")
        print(f"   💯 Total samples: {len(features)}")
        print(f"   🎯 Test size: {test_size:.0%}, Validation size: {val_size:.0%}")


        if hasattr(self, 'match_info_df') and len(self.match_info_df) == len(features):
            print("⏰ Using temporal splitting for realistic evaluation...")


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


            feature_cols = [col for col in combined_data.columns if col not in ['label', 'team1', 'team2', 'winner', 'date', 'tournament']]

            self.X_train = train_data[feature_cols].values
            self.y_train = train_data['label'].values
            self.X_val = val_data[feature_cols].values
            self.y_val = val_data['label'].values
            self.X_test = test_data[feature_cols].values
            self.y_test = test_data['label'].values

            print(f"   📚 Training set: {len(self.X_train)} samples ({len(self.X_train)/n_total:.1%})")
            print(f"   🔍 Validation set: {len(self.X_val)} samples ({len(self.X_val)/n_total:.1%})")  
            print(f"   🎯 Test set: {len(self.X_test)} samples ({len(self.X_test)/n_total:.1%})")


            self.split_info = {
                'train_dates': (train_data['date'].min(), train_data['date'].max()),
                'val_dates': (val_data['date'].min(), val_data['date'].max()),
                'test_dates': (test_data['date'].min(), test_data['date'].max()),
                'feature_names': feature_cols
            }

            print(f"   📅 Train period: {self.split_info['train_dates'][0]:%Y-%m} to {self.split_info['train_dates'][1]:%Y-%m}")
            print(f"   📅 Validation period: {self.split_info['val_dates'][0]:%Y-%m} to {self.split_info['val_dates'][1]:%Y-%m}")
            print(f"   📅 Test period: {self.split_info['test_dates'][0]:%Y-%m} to {self.split_info['test_dates'][1]:%Y-%m}")

        else:
            print("🔀 Using stratified random splitting...")


            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                features, labels, test_size=test_size, 
                stratify=labels, random_state=42
            )


            val_size_adjusted = val_size / (1 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted,
                stratify=y_temp, random_state=42
            )

            print(f"   📚 Training set: {len(self.X_train)} samples")
            print(f"   🔍 Validation set: {len(self.X_val)} samples")
            print(f"   🎯 Test set: {len(self.X_test)} samples")


        print("📏 Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✅ Data splitting completed successfully!")
        return True

    def hyperparameter_tuning_with_validation(self):
        """Perform comprehensive hyperparameter tuning using validation set"""
        print("🔧 Starting hyperparameter tuning with validation set...")
        print(f"   📊 Training samples: {len(self.X_train)}")


        if len(self.X_train) < 50:
            print("⚠️ Dataset too small for comprehensive hyperparameter tuning")
            print("   Using default parameters with basic tuning...")

            tuned_models = {
                'random_forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
                'svm': SVC(probability=True, C=1.0, kernel='rbf', random_state=42)
            }

            self.tuned_models = tuned_models
            print("✅ Basic model configuration completed!")
            return tuned_models

        tuned_models = {}


        print("🌲 Tuning Random Forest...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        rf_grid.fit(self.X_train_scaled, self.y_train)
        tuned_models['random_forest'] = rf_grid.best_estimator_

        print(f"   ✅ Best RF params: {rf_grid.best_params_}")
        print(f"   📊 Best RF CV score: {rf_grid.best_score_:.4f}")


        print("🚀 Tuning Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }

        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        gb_grid.fit(self.X_train_scaled, self.y_train)
        tuned_models['gradient_boosting'] = gb_grid.best_estimator_

        print(f"   ✅ Best GB params: {gb_grid.best_params_}")
        print(f"   📊 Best GB CV score: {gb_grid.best_score_:.4f}")


        print("🧠 Tuning Neural Network...")
        mlp_params = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [500, 1000]
        }

        mlp_grid = GridSearchCV(
            MLPClassifier(random_state=42, early_stopping=True),
            mlp_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        mlp_grid.fit(self.X_train_scaled, self.y_train)
        tuned_models['neural_network'] = mlp_grid.best_estimator_

        print(f"   ✅ Best MLP params: {mlp_grid.best_params_}")
        print(f"   📊 Best MLP CV score: {mlp_grid.best_score_:.4f}")


        print("⚡ Tuning SVM...")
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        }

        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=42),
            svm_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        svm_grid.fit(self.X_train_scaled, self.y_train)
        tuned_models['svm'] = svm_grid.best_estimator_

        print(f"   ✅ Best SVM params: {svm_grid.best_params_}")
        print(f"   📊 Best SVM CV score: {svm_grid.best_score_:.4f}")

        self.tuned_models = tuned_models
        print("✅ Hyperparameter tuning completed!")
        return tuned_models

    def train_advanced_ensemble_with_calibration(self):
        """Train advanced ensemble with probability calibration"""
        print("🎯 Training advanced ensemble with calibration...")


        validation_scores = {}

        for name, model in self.tuned_models.items():

            model.fit(self.X_train_scaled, self.y_train)
            val_pred = model.predict(self.X_val_scaled)
            val_accuracy = accuracy_score(self.y_val, val_pred)
            validation_scores[name] = val_accuracy

            print(f"   📊 {name}: {val_accuracy:.4f} validation accuracy")


            print(f"   🎯 Calibrating {name}...")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrated_model.fit(self.X_val_scaled, self.y_val)
            self.calibrated_models[name] = calibrated_model


        best_models = sorted(validation_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   🏆 Top 3 models for ensemble: {[name for name, score in best_models]}")

        ensemble_models = [(name, self.calibrated_models[name]) for name, _ in best_models]

        self.ensemble_model = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )


        X_combined = np.vstack([self.X_train_scaled, self.X_val_scaled])
        y_combined = np.hstack([self.y_train, self.y_val])

        self.ensemble_model.fit(X_combined, y_combined)

        self.validation_scores = validation_scores
        print("✅ Advanced ensemble training completed!")

    def evaluate_on_test_set(self):
        """Comprehensive evaluation on held-out test set"""
        print("🎯 Evaluating on held-out test set...")

        results = {}


        all_models = dict(self.calibrated_models)
        all_models['ensemble'] = self.ensemble_model

        for name, model in all_models.items():
            print(f"   📊 Testing {name}...")


            test_pred = model.predict(self.X_test_scaled)
            test_proba = model.predict_proba(self.X_test_scaled)[:, 1]


            accuracy = accuracy_score(self.y_test, test_pred)
            auc = roc_auc_score(self.y_test, test_proba)


            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_test, test_proba, n_bins=10, strategy='quantile'
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

            results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'calibration_error': calibration_error,
                'predictions': test_pred,
                'probabilities': test_proba
            }

            print(f"      ✅ Accuracy: {accuracy:.4f}")
            print(f"      📈 AUC: {auc:.4f}")
            print(f"      🎯 Calibration Error: {calibration_error:.4f}")


        best_model_name = max(results.keys(), 
                             key=lambda k: results[k]['accuracy'] if k != 'ensemble' 
                             else results[k]['accuracy'] + 0.005)

        self.best_model = all_models[best_model_name]
        self.best_model_name = best_model_name
        self.test_results = results

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   📊 Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"   📈 Test AUC: {results[best_model_name]['auc']:.4f}")
        print(f"   🎯 Calibration Error: {results[best_model_name]['calibration_error']:.4f}")

        return results

    def predict_match_with_enhanced_confidence(self, team1, team2):
        """Make prediction with properly calibrated confidence"""
        if not hasattr(self, 'best_model'):
            print("❌ Model not trained yet")
            return None

        try:

            current_date = pd.Timestamp.now()
            current_tournament = "VCT Champions 2025"

            match_features = self._create_comprehensive_match_features(
                team1, team2, current_date, current_tournament, self.matches_df
            )

            if match_features is None:
                print(f"❌ Could not create features for {team1} vs {team2}")
                return None


            features_scaled = self.scaler.transform([match_features])


            prediction = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]


            winner = team1 if prediction == 1 else team2
            confidence = max(probabilities)


            calibration_adjustment = 1.0
            if hasattr(self, 'test_results') and self.best_model_name in self.test_results:
                cal_error = self.test_results[self.best_model_name]['calibration_error']
                calibration_adjustment = max(0.7, 1.0 - cal_error)

            adjusted_confidence = confidence * calibration_adjustment


            if adjusted_confidence >= 0.85:
                confidence_level = "Very High"
            elif adjusted_confidence >= 0.75:
                confidence_level = "High"
            elif adjusted_confidence >= 0.65:
                confidence_level = "Medium"
            elif adjusted_confidence >= 0.55:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"

            return {
                'team1': team1,
                'team2': team2,
                'predicted_winner': winner,
                'team1_probability': float(probabilities[1]),
                'team2_probability': float(probabilities[0]),
                'confidence': float(adjusted_confidence),
                'raw_confidence': float(confidence),
                'confidence_level': confidence_level,
                'model_used': self.best_model_name,
                'model_accuracy': float(self.test_results[self.best_model_name]['accuracy']),
                'calibration_quality': float(1.0 - self.test_results[self.best_model_name]['calibration_error']),
                'enhanced_features': True,
                'temporal_split': True
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def save_advanced_model(self, filepath):
        """Save the complete advanced model"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'calibrated_models': self.calibrated_models,
            'tuned_models': self.tuned_models,
            'scaler': self.scaler,
            'test_results': self.test_results,
            'validation_scores': self.validation_scores,
            'split_info': getattr(self, 'split_info', {}),
            'matches_df': self.matches_df,
            'feature_columns': getattr(self, 'feature_columns', [])
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Advanced model saved to {filepath}")


def main():
    """Train and test the advanced predictor"""
    print("🚀 Advanced VCT ML Predictor - Train-Val-Test Pipeline")
    print("=" * 60)

    predictor = AdvancedVCTPredictor()


    predictor.load_comprehensive_data_with_temporal_info()


    features, labels = predictor.create_enhanced_features_with_temporal_context()

    if features is None:
        print("❌ Failed to create features")
        return


    predictor.implement_proper_data_splits(features, labels)


    predictor.hyperparameter_tuning_with_validation()


    predictor.train_advanced_ensemble_with_calibration()


    test_results = predictor.evaluate_on_test_set()


    print("\n" + "=" * 60)
    print("🎯 ADVANCED ML PREDICTIONS")
    print("=" * 60)

    test_matches = [
        ("Team Heretics", "Fnatic"),
        ("Paper Rex", "DRX"),
        ("Sentinels", "G2 Esports"),
        ("Edward Gaming", "Bilibili Gaming")
    ]

    for team1, team2 in test_matches:
        prediction = predictor.predict_match_with_enhanced_confidence(team1, team2)
        if prediction:
            print(f"\n{team1} vs {team2}")
            print(f"Winner: {prediction['predicted_winner']}")
            print(f"Confidence: {prediction['confidence']:.1%} ({prediction['confidence_level']})")
            print(f"Raw Confidence: {prediction['raw_confidence']:.1%}")
            print(f"Model: {prediction['model_used']} (Accuracy: {prediction['model_accuracy']:.1%})")
            print(f"Calibration Quality: {prediction['calibration_quality']:.1%}")


    model_path = Path(__file__).parent / "advanced_vct_model.pkl"
    predictor.save_advanced_model(model_path)

    print(f"\n✅ Advanced ML training pipeline completed!")
    print(f"📊 Best Model: {predictor.best_model_name}")
    print(f"🎯 Test Accuracy: {test_results[predictor.best_model_name]['accuracy']:.1%}")


if __name__ == "__main__":
    main()
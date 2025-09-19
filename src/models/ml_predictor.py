#!/usr/bin/env python3
"""
ML-based VCT prediction system using 2024 historical data and 2025 roster adjustments
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VCTMLPredictor:
    def __init__(self, data_dir=None):
        """Initialize ML predictor with VCT 2024 data"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # Model components
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Data storage
        self.team_stats = {}
        self.h2h_records = {}
        self.map_performance = {}
        self.recent_form = {}
        self.roster_adjustments = {}
        
        # Model performance
        self.model_accuracy = 0.0
        self.confidence_threshold = 0.6
        
        print("VCT ML Predictor initialized")
    
    def load_and_process_data(self):
        """Load and process all VCT 2024 tournament data"""
        print("Loading VCT 2024 tournament data...")
        
        all_matches = []
        all_player_stats = []
        all_team_performance = []
        
        # Get all tournament directories
        tournament_dirs = [d for d in self.data_dir.iterdir() 
                          if d.is_dir() and 'csvs' in d.name]
        
        print(f"Found {len(tournament_dirs)} tournaments to process")
        
        for tournament_dir in tournament_dirs:
            tournament_name = tournament_dir.name.replace('_csvs', '')
            print(f"Processing: {tournament_name}")
            
            try:
                # Load matches
                matches_file = tournament_dir / "matches.csv"
                if matches_file.exists():
                    matches_df = pd.read_csv(matches_file)
                    matches_df['tournament'] = tournament_name
                    all_matches.append(matches_df)
                
                # Load player stats
                player_stats_file = tournament_dir / "player_stats.csv"
                if player_stats_file.exists():
                    player_df = pd.read_csv(player_stats_file)
                    player_df['tournament'] = tournament_name
                    all_player_stats.append(player_df)
                
                # Load detailed match data
                detailed_overview_file = tournament_dir / "detailed_matches_overview.csv"
                if detailed_overview_file.exists():
                    detailed_df = pd.read_csv(detailed_overview_file)
                    detailed_df['tournament'] = tournament_name
                    all_team_performance.append(detailed_df)
                    
            except Exception as e:
                print(f"Error processing {tournament_name}: {e}")
                continue
        
        # Combine all data
        if all_matches:
            self.matches_df = pd.concat(all_matches, ignore_index=True)
            print(f"Loaded {len(self.matches_df)} matches")
        
        if all_player_stats:
            self.player_stats_df = pd.concat(all_player_stats, ignore_index=True)
            print(f"Loaded {len(self.player_stats_df)} player records")
        
        if all_team_performance:
            self.team_performance_df = pd.concat(all_team_performance, ignore_index=True)
            print(f"Loaded {len(self.team_performance_df)} detailed match records")
        
        # Process the data
        self._calculate_team_stats()
        self._build_h2h_records()
        self._analyze_recent_form()
        
        print("Data processing completed!")
    
    def _calculate_team_stats(self):
        """Calculate comprehensive team statistics from 2024 data"""
        print("Calculating team statistics...")
        
        if not hasattr(self, 'matches_df'):
            print("No matches data loaded")
            return
        
        # Calculate team performance metrics
        team_stats = {}
        
        for _, match in self.matches_df.iterrows():
            team1, team2 = match['team1'], match['team2']
            winner = match.get('winner', '')
            
            # Initialize team stats if not exists
            for team in [team1, team2]:
                if team not in team_stats:
                    team_stats[team] = {
                        'matches_played': 0,
                        'wins': 0,
                        'losses': 0,
                        'maps_won': 0,
                        'maps_lost': 0,
                        'tournaments': set(),
                        'recent_matches': [],
                        'win_rate': 0.0,
                        'map_win_rate': 0.0
                    }
            
            # Update match stats
            team_stats[team1]['matches_played'] += 1
            team_stats[team2]['matches_played'] += 1
            
            if winner == team1:
                team_stats[team1]['wins'] += 1
                team_stats[team2]['losses'] += 1
            elif winner == team2:
                team_stats[team2]['wins'] += 1
                team_stats[team1]['losses'] += 1
            
            # Track tournament participation
            tournament = match.get('tournament', 'Unknown')
            team_stats[team1]['tournaments'].add(tournament)
            team_stats[team2]['tournaments'].add(tournament)
            
            # Store recent match info
            match_info = {
                'date': match.get('date', ''),
                'opponent': team2 if winner == team1 else team1,
                'result': 'W' if winner == team1 else 'L',
                'tournament': tournament
            }
            team_stats[team1]['recent_matches'].append(match_info)
            
            match_info_2 = {
                'date': match.get('date', ''),
                'opponent': team1 if winner == team2 else team2,
                'result': 'W' if winner == team2 else 'L',
                'tournament': tournament
            }
            team_stats[team2]['recent_matches'].append(match_info_2)
        
        # Calculate win rates
        for team, stats in team_stats.items():
            if stats['matches_played'] > 0:
                stats['win_rate'] = stats['wins'] / stats['matches_played']
            
            # Keep only recent 10 matches
            stats['recent_matches'] = sorted(
                stats['recent_matches'], 
                key=lambda x: x['date'], 
                reverse=True
            )[:10]
        
        self.team_stats = team_stats
        print(f"Calculated stats for {len(team_stats)} teams")
    
    def _build_h2h_records(self):
        """Build head-to-head records between teams"""
        print("Building head-to-head records...")
        
        h2h = {}
        
        if hasattr(self, 'matches_df'):
            for _, match in self.matches_df.iterrows():
                team1, team2 = match['team1'], match['team2']
                winner = match.get('winner', '')
                
                # Create bidirectional h2h key
                teams_key = tuple(sorted([team1, team2]))
                
                if teams_key not in h2h:
                    h2h[teams_key] = {
                        'matches': 0,
                        team1: 0,
                        team2: 0,
                        'last_meeting': '',
                        'recent_winner': ''
                    }
                
                h2h[teams_key]['matches'] += 1
                if winner:
                    h2h[teams_key][winner] += 1
                    h2h[teams_key]['recent_winner'] = winner
                
                h2h[teams_key]['last_meeting'] = match.get('date', '')
        
        self.h2h_records = h2h
        print(f"Built H2H records for {len(h2h)} team pairings")
    
    def _analyze_recent_form(self):
        """Analyze recent form and momentum for each team"""
        print("Analyzing recent team form...")
        
        for team, stats in self.team_stats.items():
            recent_matches = stats['recent_matches'][:5]  # Last 5 matches
            
            if len(recent_matches) >= 3:
                wins = sum(1 for match in recent_matches if match['result'] == 'W')
                form_rating = wins / len(recent_matches)
                
                # Bonus for recent tournament wins
                recent_tournaments = set(match['tournament'] for match in recent_matches)
                tournament_bonus = len(recent_tournaments) * 0.1
                
                stats['form_rating'] = min(1.0, form_rating + tournament_bonus)
                stats['recent_form'] = 'Hot' if form_rating >= 0.7 else 'Cold' if form_rating <= 0.3 else 'Stable'
            else:
                stats['form_rating'] = 0.5
                stats['recent_form'] = 'Unknown'
    
    def apply_roster_adjustments(self):
        """Apply known roster changes for 2025 season"""
        print("Applying 2025 roster adjustments...")
        
        # Known roster changes and their impact on team strength
        roster_changes_2025 = {
            # Major roster upgrades
            "Team Heretics": {"adjustment": +8, "reason": "Strong 2025 roster additions"},
            "Sentinels": {"adjustment": +5, "reason": "Consistent roster with TenZ"},
            "Paper Rex": {"adjustment": +3, "reason": "Stable core roster"},
            
            # Moderate changes
            "G2 Esports": {"adjustment": +4, "reason": "Good roster stability"},
            "Edward Gaming": {"adjustment": +2, "reason": "Minor roster tweaks"},
            "Fnatic": {"adjustment": +1, "reason": "Experienced roster"},
            
            # Teams with challenges
            "MIBR": {"adjustment": -2, "reason": "Roster development needed"},
            "Rex Regum Qeon": {"adjustment": -1, "reason": "Regional competition"}
        }
        
        # Apply adjustments to team stats
        for team, adjustment in roster_changes_2025.items():
            if team in self.team_stats:
                current_rating = self.team_stats[team]['win_rate']
                adjusted_rating = min(1.0, max(0.0, current_rating + (adjustment['adjustment'] * 0.01)))
                
                self.team_stats[team]['adjusted_win_rate'] = adjusted_rating
                self.team_stats[team]['roster_adjustment'] = adjustment
                
                print(f"{team}: {current_rating:.3f} -> {adjusted_rating:.3f} ({adjustment['reason']})")
        
        self.roster_adjustments = roster_changes_2025
    
    def create_features(self, team1, team2):
        """Create feature vector for prediction"""
        features = []
        feature_names = []
        
        # Team 1 stats
        t1_stats = self.team_stats.get(team1, {})
        t2_stats = self.team_stats.get(team2, {})
        
        # Basic performance features
        features.extend([
            t1_stats.get('win_rate', 0.5),
            t2_stats.get('win_rate', 0.5),
            t1_stats.get('adjusted_win_rate', t1_stats.get('win_rate', 0.5)),
            t2_stats.get('adjusted_win_rate', t2_stats.get('win_rate', 0.5)),
        ])
        feature_names.extend(['t1_win_rate', 't2_win_rate', 't1_adj_win_rate', 't2_adj_win_rate'])
        
        # Form features
        features.extend([
            t1_stats.get('form_rating', 0.5),
            t2_stats.get('form_rating', 0.5),
        ])
        feature_names.extend(['t1_form', 't2_form'])
        
        # Head-to-head features
        teams_key = tuple(sorted([team1, team2]))
        h2h = self.h2h_records.get(teams_key, {})
        
        if h2h and h2h.get('matches', 0) > 0:
            t1_h2h_rate = h2h.get(team1, 0) / h2h['matches']
            features.append(t1_h2h_rate)
        else:
            features.append(0.5)  # No history
        feature_names.append('t1_h2h_rate')
        
        # Experience features (tournament count)
        features.extend([
            len(t1_stats.get('tournaments', set())),
            len(t2_stats.get('tournaments', set())),
        ])
        feature_names.extend(['t1_tournaments', 't2_tournaments'])
        
        # Roster adjustment impact
        t1_roster_adj = 0
        t2_roster_adj = 0
        if team1 in self.roster_adjustments:
            t1_roster_adj = self.roster_adjustments[team1]['adjustment']
        if team2 in self.roster_adjustments:
            t2_roster_adj = self.roster_adjustments[team2]['adjustment']
        
        features.extend([t1_roster_adj, t2_roster_adj])
        feature_names.extend(['t1_roster_adj', 't2_roster_adj'])
        
        return np.array(features).reshape(1, -1), feature_names
    
    def train_model(self):
        """Train the ML model using historical match data"""
        print("Training ML prediction model...")
        
        if not hasattr(self, 'matches_df') or len(self.matches_df) == 0:
            print("No training data available")
            return
        
        # Prepare training data
        X_data = []
        y_data = []
        
        for _, match in self.matches_df.iterrows():
            team1, team2 = match['team1'], match['team2']
            winner = match.get('winner', '')
            
            if not winner or winner not in [team1, team2]:
                continue
            
            # Create features
            features, _ = self.create_features(team1, team2)
            X_data.append(features.flatten())
            
            # Label: 1 if team1 wins, 0 if team2 wins
            y_data.append(1 if winner == team1 else 0)
        
        if len(X_data) < 10:
            print("Insufficient training data")
            return
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"Training with {len(X)} matches")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble of models
        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = self.rf_model.predict(X_test_scaled)
        gb_pred = self.gb_model.predict(X_test_scaled)
        
        rf_accuracy = accuracy_score(y_test, rf_pred)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        
        print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"Gradient Boosting Accuracy: {gb_accuracy:.3f}")
        
        # Cross-validation
        rf_cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=5)
        gb_cv_scores = cross_val_score(self.gb_model, X_train_scaled, y_train, cv=5)
        
        print(f"RF Cross-val: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
        print(f"GB Cross-val: {gb_cv_scores.mean():.3f} (+/- {gb_cv_scores.std() * 2:.3f})")
        
        # Set model accuracy for confidence calculation
        self.model_accuracy = max(rf_accuracy, gb_accuracy)
        
        print("Model training completed!")
    
    def predict_match(self, team1, team2):
        """Predict match outcome with confidence intervals"""
        try:
            # Create features
            features, feature_names = self.create_features(team1, team2)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from both models
            rf_prob = self.rf_model.predict_proba(features_scaled)[0]
            gb_prob = self.gb_model.predict_proba(features_scaled)[0]
            
            # Ensemble prediction (average)
            team1_prob = (rf_prob[1] + gb_prob[1]) / 2
            team2_prob = 1 - team1_prob
            
            # Determine winner and confidence
            if team1_prob > team2_prob:
                predicted_winner = team1
                confidence = team1_prob
            else:
                predicted_winner = team2
                confidence = team2_prob
            
            # Adjust confidence based on model accuracy
            adjusted_confidence = min(0.95, confidence * self.model_accuracy)
            
            # Determine confidence level
            if adjusted_confidence >= 0.75:
                confidence_level = "Very High"
            elif adjusted_confidence >= 0.65:
                confidence_level = "High"
            elif adjusted_confidence >= 0.55:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            return {
                'team1': team1,
                'team2': team2,
                'predicted_winner': predicted_winner,
                'team1_probability': float(team1_prob),
                'team2_probability': float(team2_prob),
                'confidence': float(adjusted_confidence),
                'confidence_level': confidence_level,
                'model_accuracy': float(self.model_accuracy),
                'features_used': len(feature_names)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_team_analysis(self, team_name):
        """Get detailed team analysis"""
        if team_name not in self.team_stats:
            return None
        
        stats = self.team_stats[team_name]
        analysis = {
            'team': team_name,
            'matches_played': stats.get('matches_played', 0),
            'win_rate': stats.get('win_rate', 0.0),
            'recent_form': stats.get('recent_form', 'Unknown'),
            'form_rating': stats.get('form_rating', 0.5),
            'tournaments_played': len(stats.get('tournaments', set())),
            'recent_matches': stats.get('recent_matches', [])[:3]
        }
        
        if team_name in self.roster_adjustments:
            analysis['roster_changes'] = self.roster_adjustments[team_name]
        
        return analysis
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        model_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'scaler': self.scaler,
            'team_stats': self.team_stats,
            'h2h_records': self.h2h_records,
            'roster_adjustments': self.roster_adjustments,
            'model_accuracy': self.model_accuracy
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf_model = model_data['rf_model']
        self.gb_model = model_data['gb_model']
        self.scaler = model_data['scaler']
        self.team_stats = model_data['team_stats']
        self.h2h_records = model_data['h2h_records']
        self.roster_adjustments = model_data['roster_adjustments']
        self.model_accuracy = model_data['model_accuracy']
        
        print(f"Model loaded from {filepath}")
        return True


def main():
    """Test the ML predictor"""
    predictor = VCTMLPredictor()
    
    # Load and process data
    predictor.load_and_process_data()
    predictor.apply_roster_adjustments()
    
    # Train model
    predictor.train_model()
    
    # Test predictions
    test_matches = [
        ("Team Heretics", "Fnatic"),
        ("Paper Rex", "DRX"),
        ("Sentinels", "G2 Esports"),
        ("Edward Gaming", "Bilibili Gaming")
    ]
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    for team1, team2 in test_matches:
        prediction = predictor.predict_match(team1, team2)
        if prediction:
            print(f"\n{team1} vs {team2}")
            print(f"Winner: {prediction['predicted_winner']}")
            print(f"Confidence: {prediction['confidence']:.1%} ({prediction['confidence_level']})")
            print(f"Probabilities: {team1} {prediction['team1_probability']:.1%}, {team2} {prediction['team2_probability']:.1%}")
    
    # Save model
    model_path = Path(__file__).parent / "vct_ml_model.pkl"
    predictor.save_model(model_path)


if __name__ == "__main__":
    main()
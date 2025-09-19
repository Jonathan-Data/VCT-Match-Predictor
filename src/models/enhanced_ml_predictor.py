#!/usr/bin/env python3
"""
Enhanced ML-based VCT prediction system with comprehensive player and regional data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import map picking functionality
sys.path.append(str(Path(__file__).parent.parent))
from prediction.map_picker import VCTMapPicker, SeriesFormat

class EnhancedVCTPredictor:
    def __init__(self, data_dir=None):
        """Initialize enhanced ML predictor with comprehensive VCT data"""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # Enhanced model ensemble
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        self.svm_model = SVC(probability=True, kernel='rbf', random_state=42)
        
        # Ensemble model
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Enhanced data storage
        self.team_stats = {}
        self.player_stats = {}
        self.regional_performance = {}
        self.map_statistics = {}
        self.agent_meta = {}
        self.h2h_records = {}
        self.tournament_performance = {}
        self.recent_form = {}
        
        # Model performance tracking
        self.model_accuracy = 0.0
        self.feature_importance = {}
        self.validation_scores = {}
        
        # Initialize map picking system
        self.map_picker = VCTMapPicker(self.data_dir)
        self.map_features_enabled = False
        
        print("Enhanced VCT ML Predictor initialized")
    
    def load_comprehensive_data(self):
        """Load and process all available VCT data"""
        print("Loading comprehensive VCT data...")
        
        all_matches = []
        all_player_stats = []
        all_detailed_matches = []
        all_map_data = []
        all_agent_data = []
        all_performance_data = []
        
        # Get all tournament directories
        tournament_dirs = [d for d in self.data_dir.iterdir() 
                          if d.is_dir() and 'csvs' in d.name]
        
        print(f"Processing {len(tournament_dirs)} tournaments...")
        
        for tournament_dir in tournament_dirs:
            tournament_name = tournament_dir.name.replace('_csvs', '')
            region = self.extract_region_from_tournament(tournament_name)
            tournament_type = self.extract_tournament_type(tournament_name)
            
            print(f"Processing: {tournament_name} ({region}, {tournament_type})")
            
            try:
                # Load all data types
                data_files = {
                    'matches': tournament_dir / "matches.csv",
                    'player_stats': tournament_dir / "player_stats.csv", 
                    'detailed_matches_overview': tournament_dir / "detailed_matches_overview.csv",
                    'detailed_matches_player_stats': tournament_dir / "detailed_matches_player_stats.csv",
                    'maps_stats': tournament_dir / "maps_stats.csv",
                    'agents_stats': tournament_dir / "agents_stats.csv",
                    'performance_data': tournament_dir / "performance_data.csv"
                }
                
                for data_type, file_path in data_files.items():
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        df['tournament'] = tournament_name
                        df['region'] = region
                        df['tournament_type'] = tournament_type
                        
                        if data_type == 'matches':
                            all_matches.append(df)
                        elif data_type == 'player_stats':
                            all_player_stats.append(df)
                        elif data_type == 'detailed_matches_overview':
                            all_detailed_matches.append(df)
                        elif data_type == 'maps_stats':
                            all_map_data.append(df)
                        elif data_type == 'agents_stats':
                            all_agent_data.append(df)
                        elif data_type == 'performance_data':
                            all_performance_data.append(df)
                        elif data_type == 'detailed_matches_player_stats':
                            all_detailed_matches.append(df)
                    
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
        
        if all_detailed_matches:
            self.detailed_matches_df = pd.concat(all_detailed_matches, ignore_index=True)
            print(f"Loaded {len(self.detailed_matches_df)} detailed match records")
        
        if all_map_data:
            self.maps_df = pd.concat(all_map_data, ignore_index=True)
            print(f"Loaded {len(self.maps_df)} map statistics")
        
        if all_agent_data:
            self.agents_df = pd.concat(all_agent_data, ignore_index=True)
            print(f"Loaded {len(self.agents_df)} agent statistics")
        
        if all_performance_data:
            self.performance_df = pd.concat(all_performance_data, ignore_index=True)
            print(f"Loaded {len(self.performance_df)} performance records")
        
        # Process all the data
        self._calculate_enhanced_team_stats()
        self._analyze_player_performance()
        self._calculate_regional_strength()
        self._analyze_map_performance()
        self._build_comprehensive_h2h()
        self._calculate_tournament_performance()
        
        # Load map picking data
        try:
            self.map_picker.load_map_data()
            self.map_features_enabled = True
            print("Map picking system loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load map data: {e}")
            self.map_features_enabled = False
        
        print("Comprehensive data processing completed!")
    
    def extract_region_from_tournament(self, tournament_name):
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
        else:
            return 'Unknown'
    
    def extract_tournament_type(self, tournament_name):
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
        else:
            return 'Other'
    
    def _calculate_enhanced_team_stats(self):
        """Calculate comprehensive team statistics"""
        print("Calculating enhanced team statistics...")
        
        if not hasattr(self, 'matches_df'):
            return
        
        team_stats = {}
        
        for _, match in self.matches_df.iterrows():
            team1, team2 = match['team1'], match['team2']
            winner = match.get('winner', '')
            region = match.get('region', 'Unknown')
            tournament_type = match.get('tournament_type', 'Other')
            
            for team in [team1, team2]:
                if team not in team_stats:
                    team_stats[team] = {
                        'matches_played': 0,
                        'wins': 0,
                        'losses': 0,
                        'regional_wins': {},
                        'tournament_type_performance': {},
                        'international_experience': 0,
                        'recent_matches': [],
                        'win_streak': 0,
                        'loss_streak': 0,
                        'current_streak': 0,
                        'clutch_factor': 0.0,
                        'comeback_ability': 0.0,
                        'consistency_rating': 0.0
                    }
                
                # Initialize regional and tournament type tracking
                if region not in team_stats[team]['regional_wins']:
                    team_stats[team]['regional_wins'][region] = {'wins': 0, 'matches': 0}
                if tournament_type not in team_stats[team]['tournament_type_performance']:
                    team_stats[team]['tournament_type_performance'][tournament_type] = {'wins': 0, 'matches': 0}
                
                team_stats[team]['matches_played'] += 1
                team_stats[team]['regional_wins'][region]['matches'] += 1
                team_stats[team]['tournament_type_performance'][tournament_type]['matches'] += 1
                
                if region == 'International':
                    team_stats[team]['international_experience'] += 1
                
                # Update win/loss records
                if winner == team:
                    team_stats[team]['wins'] += 1
                    team_stats[team]['regional_wins'][region]['wins'] += 1
                    team_stats[team]['tournament_type_performance'][tournament_type]['wins'] += 1
                    
                    # Update streaks
                    if team_stats[team]['current_streak'] >= 0:
                        team_stats[team]['current_streak'] += 1
                        team_stats[team]['win_streak'] = max(team_stats[team]['win_streak'], 
                                                           team_stats[team]['current_streak'])
                    else:
                        team_stats[team]['current_streak'] = 1
                else:
                    team_stats[team]['losses'] += 1
                    
                    # Update streaks
                    if team_stats[team]['current_streak'] <= 0:
                        team_stats[team]['current_streak'] -= 1
                        team_stats[team]['loss_streak'] = max(team_stats[team]['loss_streak'], 
                                                            abs(team_stats[team]['current_streak']))
                    else:
                        team_stats[team]['current_streak'] = -1
                
                # Store match details for recent form analysis
                match_detail = {
                    'opponent': team2 if team == team1 else team1,
                    'result': 'W' if winner == team else 'L',
                    'region': region,
                    'tournament_type': tournament_type,
                    'date': match.get('date', ''),
                    'score': match.get('score', '')
                }
                team_stats[team]['recent_matches'].append(match_detail)
        
        # Calculate advanced metrics
        for team, stats in team_stats.items():
            if stats['matches_played'] > 0:
                stats['win_rate'] = stats['wins'] / stats['matches_played']
                
                # Regional strength calculation
                regional_performance = {}
                for region, performance in stats['regional_wins'].items():
                    if performance['matches'] > 0:
                        regional_performance[region] = performance['wins'] / performance['matches']
                stats['regional_performance'] = regional_performance
                
                # Tournament type performance
                tournament_performance = {}
                for tournament_type, performance in stats['tournament_type_performance'].items():
                    if performance['matches'] > 0:
                        tournament_performance[tournament_type] = performance['wins'] / performance['matches']
                stats['tournament_performance'] = tournament_performance
                
                # Sort recent matches by date (most recent first)
                stats['recent_matches'] = sorted(
                    stats['recent_matches'], 
                    key=lambda x: x['date'], 
                    reverse=True
                )[:15]  # Keep last 15 matches
                
                # Calculate form metrics
                if len(stats['recent_matches']) >= 5:
                    recent_5 = stats['recent_matches'][:5]
                    stats['recent_form_rating'] = sum(1 for match in recent_5 if match['result'] == 'W') / len(recent_5)
                    
                    # Consistency rating (variance in performance)
                    recent_results = [1 if match['result'] == 'W' else 0 for match in stats['recent_matches'][:10]]
                    if len(recent_results) > 1:
                        stats['consistency_rating'] = 1 - np.var(recent_results)
                else:
                    stats['recent_form_rating'] = stats['win_rate']
                    stats['consistency_rating'] = 0.5
        
        self.team_stats = team_stats
        print(f"Enhanced statistics calculated for {len(team_stats)} teams")
    
    def _analyze_player_performance(self):
        """Analyze individual player performance and team synergy"""
        print("Analyzing player performance and team synergy...")
        
        if not hasattr(self, 'player_stats_df'):
            return
        
        player_stats = {}
        team_player_stats = {}
        
        for _, player_row in self.player_stats_df.iterrows():
            player_name = player_row.get('player_name', '')
            team = player_row.get('team', '')
            
            if not player_name or not team:
                continue
            
            # Individual player stats
            if player_name not in player_stats:
                player_stats[player_name] = {
                    'current_team': team,
                    'rating': player_row.get('rating', 0),
                    'acs': player_row.get('acs', 0),
                    'kd_ratio': player_row.get('kd_ratio', 0),
                    'kast': player_row.get('kast', 0),
                    'adr': player_row.get('adr', 0),
                    'fkpr': player_row.get('fkpr', 0),
                    'clutch_success': player_row.get('cl_percent', 0),
                    'consistency': 0.0,
                    'tournament_experience': 0
                }
            
            # Team-level player aggregation
            if team not in team_player_stats:
                team_player_stats[team] = {
                    'players': [],
                    'avg_rating': 0,
                    'avg_acs': 0,
                    'avg_kd': 0,
                    'team_synergy': 0,
                    'star_player_factor': 0,
                    'depth_factor': 0
                }
            
            team_player_stats[team]['players'].append({
                'name': player_name,
                'rating': player_row.get('rating', 0),
                'acs': player_row.get('acs', 0),
                'kd_ratio': player_row.get('kd_ratio', 0),
                'clutch_factor': player_row.get('cl_percent', 0)
            })
        
        # Calculate team-level player metrics
        for team, team_data in team_player_stats.items():
            players = team_data['players']
            if len(players) > 0:
                # Average team stats
                team_data['avg_rating'] = np.mean([p['rating'] for p in players])
                team_data['avg_acs'] = np.mean([p['acs'] for p in players])
                team_data['avg_kd'] = np.mean([p['kd_ratio'] for p in players])
                
                # Star player factor (impact of best player)
                ratings = [p['rating'] for p in players]
                if len(ratings) > 0:
                    team_data['star_player_factor'] = max(ratings) - np.mean(ratings)
                
                # Team depth (consistency across players)
                team_data['depth_factor'] = 1 - (np.std(ratings) / np.mean(ratings)) if np.mean(ratings) > 0 else 0
                
                # Team synergy approximation (based on consistent performance)
                acs_values = [p['acs'] for p in players]
                team_data['team_synergy'] = 1 - (np.std(acs_values) / np.mean(acs_values)) if np.mean(acs_values) > 0 else 0
        
        self.player_stats = player_stats
        self.team_player_stats = team_player_stats
        print(f"Player analysis completed for {len(player_stats)} players across {len(team_player_stats)} teams")
    
    def _calculate_regional_strength(self):
        """Calculate regional strength and cross-regional performance"""
        print("Calculating regional strength metrics...")
        
        regional_performance = {
            'Americas': {'wins': 0, 'matches': 0, 'international_wins': 0, 'international_matches': 0},
            'EMEA': {'wins': 0, 'matches': 0, 'international_wins': 0, 'international_matches': 0},
            'APAC': {'wins': 0, 'matches': 0, 'international_wins': 0, 'international_matches': 0},
            'China': {'wins': 0, 'matches': 0, 'international_wins': 0, 'international_matches': 0}
        }
        
        if hasattr(self, 'matches_df'):
            for _, match in self.matches_df.iterrows():
                team1, team2 = match['team1'], match['team2']
                winner = match.get('winner', '')
                region = match.get('region', 'Unknown')
                tournament_type = match.get('tournament_type', 'Other')
                
                # Get team regions
                team1_region = self.get_team_region(team1)
                team2_region = self.get_team_region(team2)
                
                if tournament_type in ['Masters', 'Champions']:  # International tournaments
                    for team, team_region in [(team1, team1_region), (team2, team2_region)]:
                        if team_region in regional_performance:
                            regional_performance[team_region]['international_matches'] += 1
                            if winner == team:
                                regional_performance[team_region]['international_wins'] += 1
                
                # Regional tournament performance
                if region != 'International':
                    for team, team_region in [(team1, team1_region), (team2, team2_region)]:
                        if team_region in regional_performance and team_region == region:
                            regional_performance[team_region]['matches'] += 1
                            if winner == team:
                                regional_performance[team_region]['wins'] += 1
        
        # Calculate regional strength ratings
        for region, stats in regional_performance.items():
            if stats['matches'] > 0:
                stats['regional_win_rate'] = stats['wins'] / stats['matches']
            else:
                stats['regional_win_rate'] = 0.5
            
            if stats['international_matches'] > 0:
                stats['international_win_rate'] = stats['international_wins'] / stats['international_matches']
            else:
                stats['international_win_rate'] = 0.5
            
            # Combined strength rating
            stats['strength_rating'] = (stats['regional_win_rate'] * 0.3 + 
                                       stats['international_win_rate'] * 0.7)
        
        self.regional_performance = regional_performance
        print("Regional strength analysis completed")
    
    def get_team_region(self, team_name):
        """Get team's region based on known team regions"""
        team_regions = {
            # EMEA
            'Team Heretics': 'EMEA', 'Fnatic': 'EMEA', 'Team Liquid': 'EMEA', 'GIANTX': 'EMEA',
            'FUT Esports': 'EMEA', 'KOI': 'EMEA', 'Karmine Corp': 'EMEA', 'Vitality': 'EMEA',
            
            # Americas
            'Sentinels': 'Americas', 'G2 Esports': 'Americas', 'NRG': 'Americas', 'MIBR': 'Americas',
            'LOUD': 'Americas', 'KRÜ Esports': 'Americas', 'Cloud9': 'Americas', 'Evil Geniuses': 'Americas',
            'Leviatán': 'Americas', 'FURIA': 'Americas', '100 Thieves': 'Americas',
            
            # APAC
            'Paper Rex': 'APAC', 'DRX': 'APAC', 'T1': 'APAC', 'Rex Regum Qeon': 'APAC',
            'Gen.G': 'APAC', 'Global Esports': 'APAC', 'ZETA DIVISION': 'APAC', 'DetonationFocusMe': 'APAC',
            
            # China
            'Edward Gaming': 'China', 'Bilibili Gaming': 'China', 'Dragon Ranger Gaming': 'China', 
            'Xi Lai Gaming': 'China', 'JD Gaming': 'China', 'Trace Esports': 'China'
        }
        
        return team_regions.get(team_name, 'Unknown')
    
    def _analyze_map_performance(self):
        """Analyze team performance on different maps"""
        print("Analyzing map-specific performance...")
        
        map_performance = {}
        
        if hasattr(self, 'maps_df'):
            for _, map_row in self.maps_df.iterrows():
                map_name = map_row.get('map_name', '')
                if map_name and map_name not in map_performance:
                    # Handle potential string values in percentage columns
                    attack_pct = map_row.get('attack_win_percent', 50)
                    defense_pct = map_row.get('defense_win_percent', 50)
                    times_played = map_row.get('times_played', 0)
                    
                    # Convert to float if string
                    try:
                        attack_pct = float(str(attack_pct).replace('%', '')) if isinstance(attack_pct, str) else float(attack_pct)
                        defense_pct = float(str(defense_pct).replace('%', '')) if isinstance(defense_pct, str) else float(defense_pct)
                        times_played = int(times_played) if times_played else 0
                    except (ValueError, TypeError):
                        attack_pct = 50.0
                        defense_pct = 50.0
                        times_played = 0
                    
                    map_performance[map_name] = {
                        'attack_win_rate': attack_pct / 100,
                        'defense_win_rate': defense_pct / 100,
                        'times_played': times_played,
                        'balance_factor': abs(50 - attack_pct) / 50
                    }
        
        self.map_statistics = map_performance
        print(f"Map analysis completed for {len(map_performance)} maps")
    
    def _build_comprehensive_h2h(self):
        """Build enhanced head-to-head records with context"""
        print("Building comprehensive head-to-head records...")
        
        h2h = {}
        
        if hasattr(self, 'matches_df'):
            for _, match in self.matches_df.iterrows():
                team1, team2 = match['team1'], match['team2']
                winner = match.get('winner', '')
                region = match.get('region', 'Unknown')
                tournament_type = match.get('tournament_type', 'Other')
                
                teams_key = tuple(sorted([team1, team2]))
                
                if teams_key not in h2h:
                    h2h[teams_key] = {
                        'total_matches': 0,
                        team1: 0,
                        team2: 0,
                        'recent_matches': [],
                        'regional_context': {},
                        'tournament_context': {},
                        'momentum': 0  # Recent winner advantage
                    }
                
                h2h[teams_key]['total_matches'] += 1
                
                # Track regional and tournament context
                if region not in h2h[teams_key]['regional_context']:
                    h2h[teams_key]['regional_context'][region] = {team1: 0, team2: 0, 'matches': 0}
                if tournament_type not in h2h[teams_key]['tournament_context']:
                    h2h[teams_key]['tournament_context'][tournament_type] = {team1: 0, team2: 0, 'matches': 0}
                
                h2h[teams_key]['regional_context'][region]['matches'] += 1
                h2h[teams_key]['tournament_context'][tournament_type]['matches'] += 1
                
                if winner:
                    h2h[teams_key][winner] += 1
                    h2h[teams_key]['regional_context'][region][winner] += 1
                    h2h[teams_key]['tournament_context'][tournament_type][winner] += 1
                
                # Track recent matches for momentum
                match_detail = {
                    'winner': winner,
                    'date': match.get('date', ''),
                    'region': region,
                    'tournament_type': tournament_type
                }
                h2h[teams_key]['recent_matches'].append(match_detail)
        
        # Calculate momentum and recent trends
        for teams_key, record in h2h.items():
            recent_matches = sorted(record['recent_matches'], key=lambda x: x['date'], reverse=True)[:5]
            if recent_matches:
                # Calculate momentum based on recent 5 matches
                team1, team2 = teams_key
                team1_recent_wins = sum(1 for match in recent_matches if match['winner'] == team1)
                record['momentum'] = (team1_recent_wins - 2.5) / 2.5  # Normalized between -1 and 1
            
            record['recent_matches'] = recent_matches
        
        self.h2h_records = h2h
        print(f"Comprehensive H2H records built for {len(h2h)} team pairings")
    
    def _calculate_tournament_performance(self):
        """Calculate performance in different tournament types"""
        print("Calculating tournament-specific performance...")
        
        tournament_performance = {}
        
        for team, stats in self.team_stats.items():
            tournament_performance[team] = {
                'kickoff_rating': 0,
                'stage_rating': 0,
                'masters_rating': 0,
                'champions_rating': 0,
                'clutch_tournament_factor': 0,
                'big_match_experience': 0
            }
            
            # Extract tournament performance from team stats
            for tournament_type, performance in stats.get('tournament_performance', {}).items():
                if tournament_type == 'Kickoff':
                    tournament_performance[team]['kickoff_rating'] = performance
                elif 'Stage' in tournament_type:
                    tournament_performance[team]['stage_rating'] = performance
                elif tournament_type == 'Masters':
                    tournament_performance[team]['masters_rating'] = performance
                    tournament_performance[team]['big_match_experience'] += 1
                elif tournament_type == 'Champions':
                    tournament_performance[team]['champions_rating'] = performance
                    tournament_performance[team]['big_match_experience'] += 2
        
        self.tournament_performance = tournament_performance
        print("Tournament performance analysis completed")
    
    def create_enhanced_features(self, team1, team2):
        """Create comprehensive feature vector with all available data"""
        features = []
        feature_names = []
        
        # Get team data
        t1_stats = self.team_stats.get(team1, {})
        t2_stats = self.team_stats.get(team2, {})
        t1_players = self.team_player_stats.get(team1, {})
        t2_players = self.team_player_stats.get(team2, {})
        t1_tournament = self.tournament_performance.get(team1, {})
        t2_tournament = self.tournament_performance.get(team2, {})
        
        # Basic performance features (enhanced)
        basic_features = [
            t1_stats.get('win_rate', 0.5),
            t2_stats.get('win_rate', 0.5),
            t1_stats.get('recent_form_rating', 0.5),
            t2_stats.get('recent_form_rating', 0.5),
            t1_stats.get('consistency_rating', 0.5),
            t2_stats.get('consistency_rating', 0.5),
            t1_stats.get('international_experience', 0) / 20,  # Normalized
            t2_stats.get('international_experience', 0) / 20,
        ]
        features.extend(basic_features)
        feature_names.extend([
            't1_win_rate', 't2_win_rate', 't1_recent_form', 't2_recent_form',
            't1_consistency', 't2_consistency', 't1_intl_exp', 't2_intl_exp'
        ])
        
        # Streak and momentum features
        streak_features = [
            max(-5, min(5, t1_stats.get('current_streak', 0))) / 5,  # Normalized
            max(-5, min(5, t2_stats.get('current_streak', 0))) / 5,
            t1_stats.get('win_streak', 0) / 10,  # Normalized
            t2_stats.get('win_streak', 0) / 10,
        ]
        features.extend(streak_features)
        feature_names.extend(['t1_current_streak', 't2_current_streak', 't1_win_streak', 't2_win_streak'])
        
        # Player-level features
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
        
        # Regional strength features
        t1_region = self.get_team_region(team1)
        t2_region = self.get_team_region(team2)
        regional_features = [
            self.regional_performance.get(t1_region, {}).get('strength_rating', 0.5),
            self.regional_performance.get(t2_region, {}).get('strength_rating', 0.5),
            1.0 if t1_region == t2_region else 0.0,  # Same region matchup
        ]
        features.extend(regional_features)
        feature_names.extend(['t1_regional_strength', 't2_regional_strength', 'same_region'])
        
        # Tournament performance features
        tournament_features = [
            t1_tournament.get('masters_rating', 0.5),
            t2_tournament.get('masters_rating', 0.5),
            t1_tournament.get('champions_rating', 0.5),
            t2_tournament.get('champions_rating', 0.5),
            t1_tournament.get('big_match_experience', 0) / 5,  # Normalized
            t2_tournament.get('big_match_experience', 0) / 5,
        ]
        features.extend(tournament_features)
        feature_names.extend([
            't1_masters_rating', 't2_masters_rating', 't1_champions_rating', 
            't2_champions_rating', 't1_big_match_exp', 't2_big_match_exp'
        ])
        
        # Head-to-head features (enhanced)
        teams_key = tuple(sorted([team1, team2]))
        h2h = self.h2h_records.get(teams_key, {})
        
        if h2h and h2h.get('total_matches', 0) > 0:
            h2h_features = [
                h2h.get(team1, 0) / h2h['total_matches'],  # H2H win rate
                h2h.get('momentum', 0),  # Recent momentum
                h2h.get('total_matches', 0) / 20,  # History depth (normalized)
            ]
        else:
            h2h_features = [0.5, 0.0, 0.0]  # No history
        
        features.extend(h2h_features)
        feature_names.extend(['t1_h2h_rate', 'h2h_momentum', 'h2h_depth'])
        
        # Ensure all features are numeric and handle NaN values
        features = [float(f) if not np.isnan(float(f)) else 0.5 for f in features]
        
        return np.array(features).reshape(1, -1), feature_names
    
    def train_enhanced_model(self):
        """Train enhanced ensemble model with hyperparameter tuning"""
        print("Training enhanced ML model with comprehensive features...")
        
        if not hasattr(self, 'matches_df') or len(self.matches_df) == 0:
            print("No training data available")
            return
        
        # Prepare enhanced training data
        X_data = []
        y_data = []
        
        print("Preparing training data...")
        for _, match in self.matches_df.iterrows():
            team1, team2 = match['team1'], match['team2']
            winner = match.get('winner', '')
            
            if not winner or winner not in [team1, team2]:
                continue
            
            try:
                features, feature_names = self.create_enhanced_features(team1, team2)
                X_data.append(features.flatten())
                y_data.append(1 if winner == team1 else 0)
            except Exception as e:
                # Skip matches with missing data
                continue
        
        if len(X_data) < 50:
            print("Insufficient training data for enhanced model")
            return
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"Training enhanced model with {len(X)} matches and {len(feature_names)} features")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning for individual models
        print("Performing hyperparameter tuning...")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_grid = GridSearchCV(self.rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        self.rf_model = rf_grid.best_estimator_
        
        # Gradient Boosting tuning
        gb_params = {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7]
        }
        gb_grid = GridSearchCV(self.gb_model, gb_params, cv=5, scoring='accuracy', n_jobs=-1)
        gb_grid.fit(X_train_scaled, y_train)
        self.gb_model = gb_grid.best_estimator_
        
        # Train MLP and SVM with default params (for speed)
        self.mlp_model.fit(X_train_scaled, y_train)
        self.svm_model.fit(X_train_scaled, y_train)
        
        # Create weighted ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model),
                ('mlp', self.mlp_model),
                ('svm', self.svm_model)
            ],
            voting='soft',
            weights=[0.3, 0.35, 0.2, 0.15]  # GB gets highest weight
        )
        
        print("Training ensemble model...")
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Evaluate all models
        models = {
            'Random Forest': self.rf_model,
            'Gradient Boosting': self.gb_model,
            'Neural Network': self.mlp_model,
            'SVM': self.svm_model,
            'Ensemble': self.ensemble_model
        }
        
        print("\nModel Evaluation Results:")
        print("=" * 50)
        
        best_accuracy = 0
        for model_name, model in models.items():
            # Test accuracy
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            print(f"{model_name}:")
            print(f"  Test Accuracy: {accuracy:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            self.validation_scores[model_name] = {
                'test_accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        self.model_accuracy = best_accuracy
        
        # Feature importance analysis
        if hasattr(self.rf_model, 'feature_importances_'):
            importance_pairs = list(zip(feature_names, self.rf_model.feature_importances_))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 10 Most Important Features:")
            print("-" * 40)
            for feature_name, importance in importance_pairs[:10]:
                print(f"{feature_name:.<25} {importance:.3f}")
            
            self.feature_importance = dict(importance_pairs)
        
        print(f"\nEnhanced model training completed!")
        print(f"Best model accuracy: {best_accuracy:.3f}")
    
    def predict_match_enhanced(self, team1, team2):
        """Enhanced prediction with comprehensive analysis"""
        try:
            if not self.ensemble_model:
                print("Enhanced model not trained")
                return None
            
            features, feature_names = self.create_enhanced_features(team1, team2)
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            proba = self.ensemble_model.predict_proba(features_scaled)[0]
            team1_prob = proba[1]
            team2_prob = 1 - team1_prob
            
            # Determine winner and confidence
            if team1_prob > team2_prob:
                predicted_winner = team1
                confidence = team1_prob
            else:
                predicted_winner = team2
                confidence = team2_prob
            
            # Enhanced confidence adjustment based on model performance
            confidence_adjusted = confidence * self.model_accuracy
            
            # Determine confidence level with higher thresholds
            if confidence_adjusted >= 0.85:
                confidence_level = "Very High"
            elif confidence_adjusted >= 0.75:
                confidence_level = "High"  
            elif confidence_adjusted >= 0.65:
                confidence_level = "Medium"
            elif confidence_adjusted >= 0.55:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            return {
                'team1': team1,
                'team2': team2,
                'predicted_winner': predicted_winner,
                'team1_probability': float(team1_prob),
                'team2_probability': float(team2_prob),
                'confidence': float(confidence_adjusted),
                'confidence_level': confidence_level,
                'model_accuracy': float(self.model_accuracy),
                'features_used': len(feature_names),
                'enhanced_features': True
            }
            
        except Exception as e:
            print(f"Enhanced prediction error: {e}")
            return None
    
    def predict_match_with_maps(self, team1: str, team2: str, 
                               series_format: SeriesFormat = SeriesFormat.BO3,
                               include_map_simulation: bool = True):
        """Enhanced prediction with map picking simulation"""
        # Get base prediction first
        base_prediction = self.predict_match_enhanced(team1, team2)
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
                
                # Combine base prediction with map analysis (weighted average)
                base_weight = 0.6
                map_weight = 0.4
                
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
    
    def get_team_map_strengths(self, team: str) -> Dict[str, float]:
        """Get team's strength ratings across all maps"""
        if not self.map_features_enabled:
            return {}
        
        strengths = {}
        for map_name in self.map_picker.current_map_pool:
            strengths[map_name] = self.map_picker.get_team_map_strength(team, map_name)
        
        return strengths
    
    def compare_teams_on_maps(self, team1: str, team2: str) -> Dict[str, Dict]:
        """Compare two teams' strengths across all maps"""
        if not self.map_features_enabled:
            return {}
        
        comparison = {}
        team1_strengths = self.get_team_map_strengths(team1)
        team2_strengths = self.get_team_map_strengths(team2)
        
        for map_name in self.map_picker.current_map_pool:
            team1_strength = team1_strengths.get(map_name, 50.0)
            team2_strength = team2_strengths.get(map_name, 50.0)
            
            comparison[map_name] = {
                'team1_strength': team1_strength,
                'team2_strength': team2_strength,
                'advantage': team1_strength - team2_strength,
                'favored_team': team1 if team1_strength > team2_strength else team2,
                'confidence': abs(team1_strength - team2_strength) / 100
            }
        
        return comparison
    
    def save_enhanced_model(self, filepath):
        """Save the enhanced trained model"""
        model_data = {
            'ensemble_model': self.ensemble_model,
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'mlp_model': self.mlp_model,
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'team_stats': self.team_stats,
            'team_player_stats': self.team_player_stats,
            'regional_performance': self.regional_performance,
            'h2h_records': self.h2h_records,
            'tournament_performance': self.tournament_performance,
            'model_accuracy': self.model_accuracy,
            'feature_importance': self.feature_importance,
            'validation_scores': self.validation_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Enhanced model saved to {filepath}")
    
    def load_enhanced_model(self, filepath):
        """Load enhanced trained model"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ensemble_model = model_data['ensemble_model']
        self.rf_model = model_data['rf_model']
        self.gb_model = model_data['gb_model']
        self.mlp_model = model_data['mlp_model']
        self.svm_model = model_data['svm_model']
        self.scaler = model_data['scaler']
        self.team_stats = model_data['team_stats']
        self.team_player_stats = model_data.get('team_player_stats', {})
        self.regional_performance = model_data['regional_performance']
        self.h2h_records = model_data['h2h_records']
        self.tournament_performance = model_data.get('tournament_performance', {})
        self.model_accuracy = model_data['model_accuracy']
        self.feature_importance = model_data.get('feature_importance', {})
        self.validation_scores = model_data.get('validation_scores', {})
        
        print(f"Enhanced model loaded from {filepath}")
        print(f"Model accuracy: {self.model_accuracy:.3f}")
        return True


def main():
    """Train and test the enhanced predictor"""
    predictor = EnhancedVCTPredictor()
    
    # Load comprehensive data
    predictor.load_comprehensive_data()
    
    # Train enhanced model
    predictor.train_enhanced_model()
    
    # Test predictions
    test_matches = [
        ("Team Heretics", "Fnatic"),
        ("Paper Rex", "DRX"), 
        ("Sentinels", "G2 Esports"),
        ("Edward Gaming", "Bilibili Gaming")
    ]
    
    print("\n" + "="*70)
    print("ENHANCED ML PREDICTIONS")
    print("="*70)
    
    for team1, team2 in test_matches:
        prediction = predictor.predict_match_enhanced(team1, team2)
        if prediction:
            print(f"\n{team1} vs {team2}")
            print(f"Winner: {prediction['predicted_winner']}")
            print(f"Confidence: {prediction['confidence']:.1%} ({prediction['confidence_level']})")
            print(f"Probabilities: {team1} {prediction['team1_probability']:.1%}, {team2} {prediction['team2_probability']:.1%}")
    
    # Save enhanced model
    model_path = Path(__file__).parent / "enhanced_vct_model.pkl"
    predictor.save_enhanced_model(model_path)


if __name__ == "__main__":
    main()
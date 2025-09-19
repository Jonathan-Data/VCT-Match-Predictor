"""
Enhanced Feature Engineering Module
Incorporates RIB.gg advanced analytics to expand the feature set beyond the existing 32 features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """Enhanced feature engineering incorporating RIB.gg advanced analytics."""
    
    def __init__(self):
        """Initialize the enhanced feature engineer."""
        # Original 32 feature names for compatibility
        self.base_features = [
            'team1_win_rate', 'team2_win_rate', 
            'team1_rating', 'team2_rating',
            'team1_recent_form', 'team2_recent_form',
            'team1_head_to_head_rate', 'team2_head_to_head_rate',
            'team1_international_experience', 'team2_international_experience',
            'team1_big_match_experience', 'team2_big_match_experience',
            'team1_current_streak', 'team2_current_streak',
            'team1_round_win_rate', 'team2_round_win_rate',
            'team1_consistency', 'team2_consistency',
            'team1_regional_dominance', 'team2_regional_dominance',
            'team1_tournament_experience', 'team2_tournament_experience',
            'team1_clutch_factor', 'team2_clutch_factor',
            'team1_comeback_ability', 'team2_comeback_ability',
            'team1_player_experience', 'team2_player_experience',
            'team1_roster_stability', 'team2_roster_stability',
            'cross_regional_matchup', 'regional_advantage',
            'tournament_importance', 'stage_importance'
        ]
        
        # New RIB.gg enhanced features (expanding to 50+ total features)
        self.rib_enhanced_features = [
            'team1_first_blood_rate', 'team2_first_blood_rate',
            'team1_clutch_success_rate', 'team2_clutch_success_rate',
            'team1_eco_round_win_rate', 'team2_eco_round_win_rate',
            'team1_tactical_timeout_efficiency', 'team2_tactical_timeout_efficiency',
            'team1_comeback_factor_rib', 'team2_comeback_factor_rib',
            'team1_consistency_rating_rib', 'team2_consistency_rating_rib',
            'team1_momentum_index', 'team2_momentum_index',
            'team1_tactical_diversity', 'team2_tactical_diversity',
            'team1_pressure_performance', 'team2_pressure_performance',
            'team1_adaptability_score', 'team2_adaptability_score',
            'team1_composite_rating', 'team2_composite_rating',
            'team1_data_confidence', 'team2_data_confidence',
            'cross_validation_agreement', 'dual_source_reliability'
        ]
        
        # Meta features derived from both sources
        self.meta_features = [
            'rating_differential', 'momentum_differential', 
            'tactical_advantage', 'pressure_readiness_diff',
            'adaptability_gap', 'experience_weighted_rating',
            'form_momentum_composite', 'clutch_reliability_index'
        ]
        
        # All features combined (50+ total)
        self.all_features = self.base_features + self.rib_enhanced_features + self.meta_features
        
        logger.info(f"Enhanced feature engineer initialized with {len(self.all_features)} total features")
    
    def create_enhanced_features(self, team1_stats: Dict[str, Any], team2_stats: Dict[str, Any],
                               match_context: Dict[str, Any] = None) -> Dict[str, float]:
        """Create enhanced feature set incorporating RIB.gg analytics."""
        features = {}
        
        # Create base features (existing 32 features)
        base_features = self._create_base_features(team1_stats, team2_stats, match_context)
        features.update(base_features)
        
        # Create RIB.gg enhanced features
        rib_features = self._create_rib_enhanced_features(team1_stats, team2_stats)
        features.update(rib_features)
        
        # Create meta features combining insights from both sources
        meta_features = self._create_meta_features(team1_stats, team2_stats, features)
        features.update(meta_features)
        
        # Ensure all features are present (fill missing with defaults)
        features = self._ensure_complete_feature_set(features)
        
        logger.debug(f"Generated {len(features)} enhanced features")
        return features
    
    def _create_base_features(self, team1_stats: Dict[str, Any], team2_stats: Dict[str, Any],
                             match_context: Dict[str, Any] = None) -> Dict[str, float]:
        """Create the original 32 base features for backward compatibility."""
        features = {}
        
        # Basic team performance features
        features['team1_win_rate'] = team1_stats.get('win_rate', 0.0)
        features['team2_win_rate'] = team2_stats.get('win_rate', 0.0)
        
        # Team ratings (use composite if available, fallback to VLR)
        features['team1_rating'] = team1_stats.get('composite_team_rating', team1_stats.get('vlr_rating', 0.0))
        features['team2_rating'] = team2_stats.get('composite_team_rating', team2_stats.get('vlr_rating', 0.0))
        
        # Recent form and streaks
        features['team1_recent_form'] = self._calculate_recent_form(team1_stats)
        features['team2_recent_form'] = self._calculate_recent_form(team2_stats)
        
        features['team1_current_streak'] = self._normalize_streak(team1_stats.get('rib_current_streak', 0))
        features['team2_current_streak'] = self._normalize_streak(team2_stats.get('rib_current_streak', 0))
        
        # Head-to-head features (placeholder - would need H2H data)
        features['team1_head_to_head_rate'] = 0.5  # Default neutral
        features['team2_head_to_head_rate'] = 0.5
        
        # Experience features
        features['team1_international_experience'] = self._calculate_international_experience(team1_stats)
        features['team2_international_experience'] = self._calculate_international_experience(team2_stats)
        
        features['team1_big_match_experience'] = self._calculate_big_match_experience(team1_stats)
        features['team2_big_match_experience'] = self._calculate_big_match_experience(team2_stats)
        
        # Round performance
        features['team1_round_win_rate'] = team1_stats.get('vlr_round_win_rate', 0.0)
        features['team2_round_win_rate'] = team2_stats.get('vlr_round_win_rate', 0.0)
        
        # Consistency and stability
        features['team1_consistency'] = team1_stats.get('rib_consistency_rating', 0.5) or 0.5
        features['team2_consistency'] = team2_stats.get('rib_consistency_rating', 0.5) or 0.5
        
        # Regional and tournament features
        features['team1_regional_dominance'] = self._calculate_regional_dominance(team1_stats)
        features['team2_regional_dominance'] = self._calculate_regional_dominance(team2_stats)
        
        features['team1_tournament_experience'] = self._calculate_tournament_experience(team1_stats)
        features['team2_tournament_experience'] = self._calculate_tournament_experience(team2_stats)
        
        # Clutch and comeback abilities (use RIB data if available)
        features['team1_clutch_factor'] = team1_stats.get('rib_clutch_success_rate', 0.5)
        features['team2_clutch_factor'] = team2_stats.get('rib_clutch_success_rate', 0.5)
        
        features['team1_comeback_ability'] = team1_stats.get('rib_comeback_factor', 0.5) or 0.5
        features['team2_comeback_ability'] = team2_stats.get('rib_comeback_factor', 0.5) or 0.5
        
        # Player features
        features['team1_player_experience'] = self._calculate_player_experience(team1_stats)
        features['team2_player_experience'] = self._calculate_player_experience(team2_stats)
        
        features['team1_roster_stability'] = self._calculate_roster_stability(team1_stats)
        features['team2_roster_stability'] = self._calculate_roster_stability(team2_stats)
        
        # Match context features
        context = match_context or {}
        features['cross_regional_matchup'] = 1.0 if team1_stats.get('region') != team2_stats.get('region') else 0.0
        features['regional_advantage'] = self._calculate_regional_advantage(team1_stats, team2_stats, context)
        features['tournament_importance'] = context.get('tournament_importance', 0.5)
        features['stage_importance'] = context.get('stage_importance', 0.5)
        
        return features
    
    def _create_rib_enhanced_features(self, team1_stats: Dict[str, Any], 
                                     team2_stats: Dict[str, Any]) -> Dict[str, float]:
        """Create RIB.gg specific enhanced features."""
        features = {}
        
        # Direct RIB.gg analytics
        features['team1_first_blood_rate'] = team1_stats.get('rib_first_blood_rate', 0.5)
        features['team2_first_blood_rate'] = team2_stats.get('rib_first_blood_rate', 0.5)
        
        features['team1_clutch_success_rate'] = team1_stats.get('rib_clutch_success_rate', 0.5)
        features['team2_clutch_success_rate'] = team2_stats.get('rib_clutch_success_rate', 0.5)
        
        features['team1_eco_round_win_rate'] = team1_stats.get('rib_eco_round_win_rate', 0.3)  # Eco rounds are harder
        features['team2_eco_round_win_rate'] = team2_stats.get('rib_eco_round_win_rate', 0.3)
        
        features['team1_tactical_timeout_efficiency'] = team1_stats.get('rib_tactical_timeout_efficiency', 0.5) or 0.5
        features['team2_tactical_timeout_efficiency'] = team2_stats.get('rib_tactical_timeout_efficiency', 0.5) or 0.5
        
        features['team1_comeback_factor_rib'] = team1_stats.get('rib_comeback_factor', 0.5) or 0.5
        features['team2_comeback_factor_rib'] = team2_stats.get('rib_comeback_factor', 0.5) or 0.5
        
        features['team1_consistency_rating_rib'] = team1_stats.get('rib_consistency_rating', 0.5) or 0.5
        features['team2_consistency_rating_rib'] = team2_stats.get('rib_consistency_rating', 0.5) or 0.5
        
        # Enhanced composite metrics
        features['team1_momentum_index'] = team1_stats.get('momentum_index', 0.5)
        features['team2_momentum_index'] = team2_stats.get('momentum_index', 0.5)
        
        features['team1_tactical_diversity'] = team1_stats.get('tactical_diversity', 0.5)
        features['team2_tactical_diversity'] = team2_stats.get('tactical_diversity', 0.5)
        
        features['team1_pressure_performance'] = team1_stats.get('pressure_performance', 0.5)
        features['team2_pressure_performance'] = team2_stats.get('pressure_performance', 0.5)
        
        features['team1_adaptability_score'] = team1_stats.get('adaptability_score', 0.5)
        features['team2_adaptability_score'] = team2_stats.get('adaptability_score', 0.5)
        
        features['team1_composite_rating'] = team1_stats.get('composite_team_rating', 0.0)
        features['team2_composite_rating'] = team2_stats.get('composite_team_rating', 0.0)
        
        # Data quality indicators
        features['team1_data_confidence'] = team1_stats.get('data_confidence_score', 0.5)
        features['team2_data_confidence'] = team2_stats.get('data_confidence_score', 0.5)
        
        features['cross_validation_agreement'] = (
            team1_stats.get('cross_validation_score', 0.5) + 
            team2_stats.get('cross_validation_score', 0.5)
        ) / 2.0
        
        features['dual_source_reliability'] = min(
            features['team1_data_confidence'], 
            features['team2_data_confidence']
        )
        
        return features
    
    def _create_meta_features(self, team1_stats: Dict[str, Any], team2_stats: Dict[str, Any],
                             existing_features: Dict[str, float]) -> Dict[str, float]:
        """Create meta features that combine insights from multiple sources."""
        meta_features = {}
        
        # Rating differentials
        meta_features['rating_differential'] = (
            existing_features.get('team1_composite_rating', 0) - 
            existing_features.get('team2_composite_rating', 0)
        ) / 1000.0  # Normalize
        
        # Momentum differential
        meta_features['momentum_differential'] = (
            existing_features.get('team1_momentum_index', 0.5) - 
            existing_features.get('team2_momentum_index', 0.5)
        )
        
        # Tactical advantage (diversity vs opponent)
        team1_diversity = existing_features.get('team1_tactical_diversity', 0.5)
        team2_diversity = existing_features.get('team2_tactical_diversity', 0.5)
        meta_features['tactical_advantage'] = team1_diversity - team2_diversity
        
        # Pressure readiness differential
        team1_pressure = existing_features.get('team1_pressure_performance', 0.5)
        team2_pressure = existing_features.get('team2_pressure_performance', 0.5)
        meta_features['pressure_readiness_diff'] = team1_pressure - team2_pressure
        
        # Adaptability gap
        team1_adapt = existing_features.get('team1_adaptability_score', 0.5)
        team2_adapt = existing_features.get('team2_adaptability_score', 0.5)
        meta_features['adaptability_gap'] = team1_adapt - team2_adapt
        
        # Experience-weighted rating
        team1_exp = existing_features.get('team1_international_experience', 0.5)
        team2_exp = existing_features.get('team2_international_experience', 0.5)
        team1_rating = existing_features.get('team1_rating', 0)
        team2_rating = existing_features.get('team2_rating', 0)
        
        exp_weight = 0.3
        meta_features['experience_weighted_rating'] = (
            (team1_rating * (1 + exp_weight * team1_exp)) - 
            (team2_rating * (1 + exp_weight * team2_exp))
        ) / 1000.0
        
        # Form-momentum composite
        team1_form = existing_features.get('team1_recent_form', 0.5)
        team2_form = existing_features.get('team2_recent_form', 0.5)
        team1_momentum = existing_features.get('team1_momentum_index', 0.5)
        team2_momentum = existing_features.get('team2_momentum_index', 0.5)
        
        meta_features['form_momentum_composite'] = (
            (team1_form * 0.6 + team1_momentum * 0.4) - 
            (team2_form * 0.6 + team2_momentum * 0.4)
        )
        
        # Clutch reliability index
        team1_clutch = existing_features.get('team1_clutch_success_rate', 0.5)
        team2_clutch = existing_features.get('team2_clutch_success_rate', 0.5)
        team1_consistency = existing_features.get('team1_consistency_rating_rib', 0.5)
        team2_consistency = existing_features.get('team2_consistency_rating_rib', 0.5)
        
        meta_features['clutch_reliability_index'] = (
            (team1_clutch * team1_consistency) - 
            (team2_clutch * team2_consistency)
        )
        
        return meta_features
    
    # Helper methods for feature calculation
    def _calculate_recent_form(self, team_stats: Dict[str, Any]) -> float:
        """Calculate recent form from match history."""
        if 'vlr_recent_matches' in team_stats and team_stats['vlr_recent_matches']:
            recent_matches = team_stats['vlr_recent_matches'][:5]  # Last 5 matches
            wins = sum(1 for match in recent_matches 
                      if 'win' in match.get('result', '').lower() or match.get('result') == 'W')
            return wins / len(recent_matches) if recent_matches else 0.5
        
        # Fallback to momentum index if available
        return team_stats.get('momentum_index', 0.5)
    
    def _normalize_streak(self, streak: int) -> float:
        """Normalize streak value to 0-1 range."""
        # Positive for win streaks, negative for loss streaks
        # Normalize to -1 to +1, then shift to 0-1
        normalized = max(-1.0, min(1.0, streak / 5.0))  # Cap at 5-game streaks
        return (normalized + 1.0) / 2.0  # Shift to 0-1 range
    
    def _calculate_international_experience(self, team_stats: Dict[str, Any]) -> float:
        """Calculate international tournament experience."""
        # Use tournament performance data if available
        tournament_perf = team_stats.get('tournament_performance', {})
        if 'international_matches' in tournament_perf:
            return min(tournament_perf['international_matches'] / 20.0, 1.0)  # Cap at 20 matches
        
        # Fallback based on team rating and region
        rating = team_stats.get('composite_team_rating', team_stats.get('vlr_rating', 0))
        region = team_stats.get('region', '')
        
        # Higher rated teams from major regions typically have more international experience
        base_experience = min(rating / 1200.0, 1.0) if rating > 0 else 0.5
        
        # Regional modifiers (major regions get slight boost)
        regional_bonus = 0.1 if region in ['Americas', 'EMEA', 'APAC'] else 0.0
        
        return min(base_experience + regional_bonus, 1.0)
    
    def _calculate_big_match_experience(self, team_stats: Dict[str, Any]) -> float:
        """Calculate experience in high-stakes matches."""
        # Use tournament performance data if available
        tournament_perf = team_stats.get('tournament_performance', {})
        if 'playoffs_matches' in tournament_perf:
            return min(tournament_perf['playoffs_matches'] / 15.0, 1.0)
        
        # Estimate based on pressure performance and consistency
        pressure_perf = team_stats.get('pressure_performance', 0.5)
        consistency = team_stats.get('rib_consistency_rating', 0.5) or 0.5
        
        return (pressure_perf * 0.6 + consistency * 0.4)
    
    def _calculate_regional_dominance(self, team_stats: Dict[str, Any]) -> float:
        """Calculate dominance within the team's region."""
        win_rate = team_stats.get('win_rate', 0.0)
        rating = team_stats.get('composite_team_rating', team_stats.get('vlr_rating', 0))
        
        # Regional dominance based on win rate and rating
        rating_factor = min(rating / 1300.0, 1.0) if rating > 0 else 0.5
        return (win_rate * 0.7 + rating_factor * 0.3)
    
    def _calculate_tournament_experience(self, team_stats: Dict[str, Any]) -> float:
        """Calculate overall tournament experience."""
        matches_played = team_stats.get('matches_played', 0)
        
        # Experience based on total matches played
        experience = min(matches_played / 50.0, 1.0)  # Cap at 50 matches for full experience
        
        # Boost for teams with international experience
        intl_exp = self._calculate_international_experience(team_stats)
        return min(experience + (intl_exp * 0.2), 1.0)
    
    def _calculate_player_experience(self, team_stats: Dict[str, Any]) -> float:
        """Calculate average player experience on the team."""
        players = team_stats.get('vlr_players', [])
        if not players:
            return 0.5  # Default neutral
        
        # If player experience data is available, use it
        if players and isinstance(players[0], dict) and 'experience' in players[0]:
            avg_experience = sum(player.get('experience', 0) for player in players) / len(players)
            return min(avg_experience / 100.0, 1.0)  # Normalize
        
        # Fallback: estimate based on team performance
        win_rate = team_stats.get('win_rate', 0.0)
        consistency = team_stats.get('rib_consistency_rating', 0.5) or 0.5
        
        return (win_rate * 0.4 + consistency * 0.6)
    
    def _calculate_roster_stability(self, team_stats: Dict[str, Any]) -> float:
        """Calculate roster stability."""
        # If we had roster change data, we'd use it here
        # For now, estimate based on consistency
        consistency = team_stats.get('rib_consistency_rating', 0.5) or 0.5
        return consistency  # More consistent teams typically have more stable rosters
    
    def _calculate_regional_advantage(self, team1_stats: Dict[str, Any], 
                                    team2_stats: Dict[str, Any], 
                                    context: Dict[str, Any]) -> float:
        """Calculate regional advantage in the match."""
        team1_region = team1_stats.get('region', '')
        team2_region = team2_stats.get('region', '')
        tournament_region = context.get('tournament_region', '')
        
        if team1_region == tournament_region and team2_region != tournament_region:
            return 0.3  # Team1 has home advantage
        elif team2_region == tournament_region and team1_region != tournament_region:
            return -0.3  # Team2 has home advantage
        else:
            return 0.0  # No regional advantage
    
    def _ensure_complete_feature_set(self, features: Dict[str, float]) -> Dict[str, float]:
        """Ensure all expected features are present with default values."""
        complete_features = {}
        
        for feature_name in self.all_features:
            if feature_name in features:
                # Ensure numeric and not NaN
                value = features[feature_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    complete_features[feature_name] = float(value)
                else:
                    complete_features[feature_name] = 0.5 if 'rate' in feature_name or 'score' in feature_name or 'index' in feature_name else 0.0
            else:
                # Default values for missing features
                if any(keyword in feature_name for keyword in ['rate', 'score', 'index', 'efficiency', 'factor', 'diversity', 'performance']):
                    complete_features[feature_name] = 0.5  # Neutral default for ratios/scores
                elif 'differential' in feature_name or 'gap' in feature_name or 'advantage' in feature_name:
                    complete_features[feature_name] = 0.0  # Neutral default for differences
                else:
                    complete_features[feature_name] = 0.0  # Zero default for others
        
        return complete_features
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names in order."""
        return self.all_features.copy()
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis."""
        return {
            'base_performance': [f for f in self.base_features if 'win_rate' in f or 'rating' in f],
            'experience': [f for f in self.base_features if 'experience' in f],
            'form_momentum': [f for f in self.base_features if 'form' in f or 'streak' in f] + 
                           [f for f in self.rib_enhanced_features if 'momentum' in f],
            'tactical_analysis': [f for f in self.rib_enhanced_features if 'tactical' in f or 'diversity' in f],
            'pressure_clutch': [f for f in self.rib_enhanced_features if 'clutch' in f or 'pressure' in f],
            'advanced_analytics': [f for f in self.rib_enhanced_features if 'first_blood' in f or 'eco' in f or 'timeout' in f],
            'meta_features': self.meta_features,
            'data_quality': [f for f in self.rib_enhanced_features if 'confidence' in f or 'validation' in f]
        }

def main():
    """Demonstration of enhanced feature engineering."""
    # Sample team stats (would come from dual-source integrator)
    team1_stats = {
        'team_name': 'Sentinels',
        'region': 'Americas',
        'win_rate': 0.73,
        'composite_team_rating': 1247.0,
        'vlr_rating': 1200.0,
        'vlr_round_win_rate': 0.68,
        'rib_first_blood_rate': 0.62,
        'rib_clutch_success_rate': 0.58,
        'rib_eco_round_win_rate': 0.35,
        'rib_current_streak': 3,
        'momentum_index': 0.75,
        'tactical_diversity': 0.68,
        'pressure_performance': 0.72,
        'adaptability_score': 0.65,
        'data_confidence_score': 0.85,
        'cross_validation_score': 0.78
    }
    
    team2_stats = {
        'team_name': 'Fnatic',
        'region': 'EMEA',
        'win_rate': 0.68,
        'composite_team_rating': 1189.0,
        'vlr_rating': 1150.0,
        'vlr_round_win_rate': 0.65,
        'rib_first_blood_rate': 0.58,
        'rib_clutch_success_rate': 0.55,
        'rib_eco_round_win_rate': 0.32,
        'rib_current_streak': -1,
        'momentum_index': 0.45,
        'tactical_diversity': 0.72,
        'pressure_performance': 0.68,
        'adaptability_score': 0.78,
        'data_confidence_score': 0.82,
        'cross_validation_score': 0.75
    }
    
    match_context = {
        'tournament_importance': 0.9,
        'stage_importance': 0.8,
        'tournament_region': 'International'
    }
    
    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer()
    
    # Create enhanced features
    features = engineer.create_enhanced_features(team1_stats, team2_stats, match_context)
    
    print(f"Enhanced Feature Engineering Demo")
    print(f"=================================")
    print(f"Total features generated: {len(features)}")
    print(f"Feature groups: {list(engineer.get_feature_importance_groups().keys())}")
    print(f"\nSample features:")
    
    # Show some key features
    key_features = [
        'team1_win_rate', 'team2_win_rate',
        'team1_composite_rating', 'team2_composite_rating',
        'team1_momentum_index', 'team2_momentum_index',
        'rating_differential', 'momentum_differential',
        'tactical_advantage', 'clutch_reliability_index'
    ]
    
    for feature in key_features:
        if feature in features:
            print(f"  {feature}: {features[feature]:.3f}")
    
    print(f"\nEnhanced feature engineering successfully demonstrated!")
    print(f"This expanded feature set should improve prediction accuracy beyond the current >83%")

if __name__ == "__main__":
    main()
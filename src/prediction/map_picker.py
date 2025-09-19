#!/usr/bin/env python3
"""
VCT Map Picking System for Best-Of Series
Analyzes team strengths per map and provides strategic map selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class MapType(Enum):
    """VALORANT map types"""
    ASCENT = "Ascent"
    BIND = "Bind"
    BREEZE = "Breeze"
    FRACTURE = "Fracture"
    HAVEN = "Haven"
    ICEBOX = "Icebox"
    LOTUS = "Lotus"
    PEARL = "Pearl"
    SPLIT = "Split"
    SUNSET = "Sunset"


class SeriesFormat(Enum):
    """Best-Of series formats"""
    BO1 = "bo1"
    BO3 = "bo3"
    BO5 = "bo5"


@dataclass
class TeamMapStats:
    """Team statistics on a specific map"""
    team_name: str
    map_name: str
    matches_played: int
    wins: int
    losses: int
    win_rate: float
    avg_rounds_won: float
    avg_rounds_lost: float
    attack_side_win_rate: float
    defense_side_win_rate: float
    clutch_success_rate: float
    first_blood_rate: float
    economy_rating: float
    recent_form: float  # Win rate in last 5 matches on this map


@dataclass
class MapPickResult:
    """Result of map picking simulation"""
    picked_maps: List[str]
    team1_advantages: Dict[str, float]
    team2_advantages: Dict[str, float]
    predicted_series_outcome: Dict[str, float]
    pick_sequence: List[Dict[str, str]]
    strategic_analysis: Dict[str, str]


class VCTMapPicker:
    """VALORANT Champions Tour Map Picking System"""
    
    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # Current VCT map pool (2025)
        self.current_map_pool = [
            MapType.ASCENT.value,
            MapType.BIND.value,
            MapType.BREEZE.value,
            MapType.HAVEN.value,
            MapType.ICEBOX.value,
            MapType.LOTUS.value,
            MapType.SUNSET.value
        ]
        
        # Team map statistics
        self.team_map_stats: Dict[str, Dict[str, TeamMapStats]] = {}
        self.map_meta_analysis = {}
        
        print("VCT Map Picker initialized")
    
    def load_map_data(self):
        """Load and process map statistics from all available data"""
        print("Loading map statistics...")
        
        all_map_data = []
        all_detailed_matches = []
        
        # Get all tournament directories
        tournament_dirs = [d for d in self.data_dir.iterdir() 
                          if d.is_dir() and 'csvs' in d.name]
        
        for tournament_dir in tournament_dirs:
            try:
                # Load map statistics
                maps_file = tournament_dir / "maps_stats.csv"
                if maps_file.exists():
                    df = pd.read_csv(maps_file)
                    df['tournament'] = tournament_dir.name.replace('_csvs', '')
                    all_map_data.append(df)
                
                # Load detailed match data for side-specific analysis
                detailed_file = tournament_dir / "detailed_matches_overview.csv"
                if detailed_file.exists():
                    df = pd.read_csv(detailed_file)
                    df['tournament'] = tournament_dir.name.replace('_csvs', '')
                    all_detailed_matches.append(df)
                    
            except Exception as e:
                print(f"Error loading data from {tournament_dir}: {e}")
                continue
        
        if all_map_data:
            self.maps_df = pd.concat(all_map_data, ignore_index=True)
            print(f"Loaded {len(self.maps_df)} map statistics records")
        
        if all_detailed_matches:
            self.detailed_matches_df = pd.concat(all_detailed_matches, ignore_index=True)
            print(f"Loaded {len(self.detailed_matches_df)} detailed match records")
        
        self._calculate_team_map_statistics()
        self._analyze_map_meta()
    
    def _calculate_team_map_statistics(self):
        """Calculate comprehensive team statistics for each map"""
        print("Calculating team map statistics...")
        
        team_map_stats = {}
        
        if hasattr(self, 'maps_df'):
            for _, row in self.maps_df.iterrows():
                team = row.get('team', '')
                map_name = row.get('map', '')
                
                if not team or not map_name or map_name not in self.current_map_pool:
                    continue
                
                if team not in team_map_stats:
                    team_map_stats[team] = {}
                
                if map_name not in team_map_stats[team]:
                    team_map_stats[team][map_name] = {
                        'matches': [],
                        'wins': 0,
                        'total_rounds_won': 0,
                        'total_rounds_lost': 0,
                        'attack_wins': 0,
                        'defense_wins': 0,
                        'attack_rounds': 0,
                        'defense_rounds': 0,
                        'clutch_successes': 0,
                        'clutch_attempts': 0,
                        'first_bloods': 0,
                        'total_matches': 0
                    }
                
                stats = team_map_stats[team][map_name]
                
                # Basic match statistics
                result = row.get('result', '')
                if result == 'W':
                    stats['wins'] += 1
                
                stats['total_matches'] += 1
                stats['total_rounds_won'] += row.get('rounds_won', 0)
                stats['total_rounds_lost'] += row.get('rounds_lost', 0)
                
                # Side-specific statistics (if available)
                attack_rounds = row.get('attack_rounds_won', 0)
                defense_rounds = row.get('defense_rounds_won', 0)
                
                stats['attack_rounds'] += attack_rounds
                stats['defense_rounds'] += defense_rounds
                
                # Additional statistics
                stats['clutch_successes'] += row.get('clutches_won', 0)
                stats['clutch_attempts'] += row.get('clutches_attempted', 0)
                stats['first_bloods'] += row.get('first_bloods', 0)
                
                # Store match details for recent form analysis
                stats['matches'].append({
                    'result': result,
                    'rounds_won': row.get('rounds_won', 0),
                    'rounds_lost': row.get('rounds_lost', 0),
                    'date': row.get('date', ''),
                    'tournament': row.get('tournament', '')
                })
        
        # Convert to TeamMapStats objects
        for team, maps in team_map_stats.items():
            self.team_map_stats[team] = {}
            
            for map_name, stats in maps.items():
                if stats['total_matches'] > 0:
                    # Calculate rates and averages
                    win_rate = stats['wins'] / stats['total_matches']
                    avg_rounds_won = stats['total_rounds_won'] / stats['total_matches']
                    avg_rounds_lost = stats['total_rounds_lost'] / stats['total_matches']
                    
                    # Side-specific win rates
                    total_attack_rounds = stats['attack_rounds'] + stats.get('attack_rounds_lost', 0)
                    total_defense_rounds = stats['defense_rounds'] + stats.get('defense_rounds_lost', 0)
                    
                    attack_win_rate = (stats['attack_rounds'] / total_attack_rounds 
                                     if total_attack_rounds > 0 else 0.5)
                    defense_win_rate = (stats['defense_rounds'] / total_defense_rounds 
                                      if total_defense_rounds > 0 else 0.5)
                    
                    # Clutch success rate
                    clutch_rate = (stats['clutch_successes'] / stats['clutch_attempts'] 
                                 if stats['clutch_attempts'] > 0 else 0.0)
                    
                    # First blood rate (approximate)
                    fb_rate = stats['first_bloods'] / stats['total_matches'] if stats['total_matches'] > 0 else 0.0
                    
                    # Recent form (last 5 matches)
                    recent_matches = sorted(stats['matches'], key=lambda x: x['date'], reverse=True)[:5]
                    recent_form = (sum(1 for match in recent_matches if match['result'] == 'W') / 
                                 len(recent_matches) if recent_matches else win_rate)
                    
                    # Economy rating (based on round differential)
                    economy_rating = avg_rounds_won - avg_rounds_lost
                    
                    self.team_map_stats[team][map_name] = TeamMapStats(
                        team_name=team,
                        map_name=map_name,
                        matches_played=stats['total_matches'],
                        wins=stats['wins'],
                        losses=stats['total_matches'] - stats['wins'],
                        win_rate=win_rate,
                        avg_rounds_won=avg_rounds_won,
                        avg_rounds_lost=avg_rounds_lost,
                        attack_side_win_rate=attack_win_rate,
                        defense_side_win_rate=defense_win_rate,
                        clutch_success_rate=clutch_rate,
                        first_blood_rate=fb_rate,
                        economy_rating=economy_rating,
                        recent_form=recent_form
                    )
        
        print(f"Calculated map statistics for {len(self.team_map_stats)} teams")
    
    def _analyze_map_meta(self):
        """Analyze current map meta and strategic considerations"""
        print("Analyzing map meta...")
        
        meta_analysis = {}
        
        for map_name in self.current_map_pool:
            map_data = []
            
            # Collect all team stats for this map
            for team_stats in self.team_map_stats.values():
                if map_name in team_stats:
                    stats = team_stats[map_name]
                    map_data.append({
                        'win_rate': stats.win_rate,
                        'attack_win_rate': stats.attack_side_win_rate,
                        'defense_win_rate': stats.defense_side_win_rate,
                        'avg_rounds': stats.avg_rounds_won,
                        'clutch_rate': stats.clutch_success_rate
                    })
            
            if map_data:
                df = pd.DataFrame(map_data)
                
                meta_analysis[map_name] = {
                    'avg_attack_advantage': df['attack_win_rate'].mean() - 0.5,
                    'avg_defense_advantage': df['defense_win_rate'].mean() - 0.5,
                    'competitiveness': 1 - df['win_rate'].std(),  # Lower std = more competitive
                    'avg_round_count': df['avg_rounds'].mean() * 2,  # Approximate total rounds
                    'clutch_frequency': df['clutch_rate'].mean(),
                    'strategic_type': self._classify_map_type(map_name, df)
                }
        
        self.map_meta_analysis = meta_analysis
    
    def _classify_map_type(self, map_name: str, stats_df: pd.DataFrame) -> str:
        """Classify map based on strategic characteristics"""
        attack_bias = stats_df['attack_win_rate'].mean() - 0.5
        
        if attack_bias > 0.1:
            return "attacker_favored"
        elif attack_bias < -0.1:
            return "defender_favored"
        else:
            return "balanced"
    
    def get_team_map_strength(self, team: str, map_name: str) -> float:
        """Get team's strength rating on a specific map (0-100)"""
        if team not in self.team_map_stats or map_name not in self.team_map_stats[team]:
            return 50.0  # Neutral if no data
        
        stats = self.team_map_stats[team][map_name]
        
        # Weighted strength calculation
        base_strength = stats.win_rate * 100
        form_adjustment = (stats.recent_form - stats.win_rate) * 10
        experience_bonus = min(stats.matches_played * 2, 10)
        clutch_bonus = stats.clutch_success_rate * 5
        
        strength = base_strength + form_adjustment + experience_bonus + clutch_bonus
        return max(0, min(100, strength))
    
    def simulate_map_pick_ban(self, team1: str, team2: str, 
                            series_format: SeriesFormat) -> MapPickResult:
        """Simulate the map pick/ban process for a Best-Of series"""
        print(f"Simulating {series_format.value.upper()} map pick/ban: {team1} vs {team2}")
        
        available_maps = self.current_map_pool.copy()
        picked_maps = []
        pick_sequence = []
        team1_advantages = {}
        team2_advantages = {}
        
        # Determine number of maps needed
        maps_needed = {
            SeriesFormat.BO1: 1,
            SeriesFormat.BO3: 3,
            SeriesFormat.BO5: 5
        }[series_format]
        
        # Calculate team preferences
        team1_preferences = self._calculate_team_map_preferences(team1)
        team2_preferences = self._calculate_team_map_preferences(team2)
        
        # Simulate pick/ban process based on series format
        if series_format == SeriesFormat.BO1:
            picked_maps, pick_sequence = self._simulate_bo1_pickban(
                team1, team2, available_maps, team1_preferences, team2_preferences
            )
        elif series_format == SeriesFormat.BO3:
            picked_maps, pick_sequence = self._simulate_bo3_pickban(
                team1, team2, available_maps, team1_preferences, team2_preferences
            )
        elif series_format == SeriesFormat.BO5:
            picked_maps, pick_sequence = self._simulate_bo5_pickban(
                team1, team2, available_maps, team1_preferences, team2_preferences
            )
        
        # Calculate team advantages on picked maps
        for map_name in picked_maps:
            team1_strength = self.get_team_map_strength(team1, map_name)
            team2_strength = self.get_team_map_strength(team2, map_name)
            
            team1_advantages[map_name] = team1_strength - team2_strength
            team2_advantages[map_name] = team2_strength - team1_strength
        
        # Predict series outcome
        predicted_outcome = self._predict_series_outcome(
            team1, team2, picked_maps, series_format
        )
        
        # Strategic analysis
        strategic_analysis = self._generate_strategic_analysis(
            team1, team2, picked_maps, team1_advantages, team2_advantages
        )
        
        return MapPickResult(
            picked_maps=picked_maps,
            team1_advantages=team1_advantages,
            team2_advantages=team2_advantages,
            predicted_series_outcome=predicted_outcome,
            pick_sequence=pick_sequence,
            strategic_analysis=strategic_analysis
        )
    
    def _calculate_team_map_preferences(self, team: str) -> Dict[str, float]:
        """Calculate team's preferences for each map"""
        preferences = {}
        
        for map_name in self.current_map_pool:
            strength = self.get_team_map_strength(team, map_name)
            preferences[map_name] = strength
        
        return preferences
    
    def _simulate_bo1_pickban(self, team1: str, team2: str, available_maps: List[str],
                            team1_prefs: Dict[str, float], team2_prefs: Dict[str, float]
                            ) -> Tuple[List[str], List[Dict]]:
        """Simulate BO1 pick/ban process"""
        # Standard BO1: Team1 ban, Team2 ban, Team1 ban, Team2 ban, Team1 ban, Team2 ban, remaining map
        pick_sequence = []
        maps = available_maps.copy()
        
        # Teams alternate bans (6 total)
        for i in range(6):
            banning_team = team1 if i % 2 == 0 else team2
            opponent_prefs = team2_prefs if i % 2 == 0 else team1_prefs
            
            # Ban opponent's strongest remaining map
            strongest_opponent_map = max(maps, key=lambda m: opponent_prefs[m])
            maps.remove(strongest_opponent_map)
            
            pick_sequence.append({
                'action': 'ban',
                'team': banning_team,
                'map': strongest_opponent_map
            })
        
        # Remaining map is played
        remaining_map = maps[0]
        pick_sequence.append({
            'action': 'remaining',
            'team': 'decider',
            'map': remaining_map
        })
        
        return [remaining_map], pick_sequence
    
    def _simulate_bo3_pickban(self, team1: str, team2: str, available_maps: List[str],
                            team1_prefs: Dict[str, float], team2_prefs: Dict[str, float]
                            ) -> Tuple[List[str], List[Dict]]:
        """Simulate BO3 pick/ban process"""
        # Standard BO3: Team1 ban, Team2 ban, Team1 pick, Team2 pick, Team1 ban, Team2 ban, remaining map
        pick_sequence = []
        maps = available_maps.copy()
        picked_maps = []
        
        # Initial bans
        for i in range(2):
            banning_team = team1 if i % 2 == 0 else team2
            opponent_prefs = team2_prefs if i % 2 == 0 else team1_prefs
            
            strongest_opponent_map = max(maps, key=lambda m: opponent_prefs[m])
            maps.remove(strongest_opponent_map)
            
            pick_sequence.append({
                'action': 'ban',
                'team': banning_team,
                'map': strongest_opponent_map
            })
        
        # Picks
        for i in range(2):
            picking_team = team1 if i % 2 == 0 else team2
            team_prefs = team1_prefs if i % 2 == 0 else team2_prefs
            
            strongest_team_map = max(maps, key=lambda m: team_prefs[m])
            maps.remove(strongest_team_map)
            picked_maps.append(strongest_team_map)
            
            pick_sequence.append({
                'action': 'pick',
                'team': picking_team,
                'map': strongest_team_map
            })
        
        # Final bans
        for i in range(2):
            banning_team = team1 if i % 2 == 0 else team2
            opponent_prefs = team2_prefs if i % 2 == 0 else team1_prefs
            
            if maps:  # Check if there are still maps to ban
                strongest_opponent_map = max(maps, key=lambda m: opponent_prefs[m])
                maps.remove(strongest_opponent_map)
                
                pick_sequence.append({
                    'action': 'ban',
                    'team': banning_team,
                    'map': strongest_opponent_map
                })
        
        # Remaining map is decider
        if maps:
            remaining_map = maps[0]
            picked_maps.append(remaining_map)
            pick_sequence.append({
                'action': 'decider',
                'team': 'remaining',
                'map': remaining_map
            })
        
        return picked_maps, pick_sequence
    
    def _simulate_bo5_pickban(self, team1: str, team2: str, available_maps: List[str],
                            team1_prefs: Dict[str, float], team2_prefs: Dict[str, float]
                            ) -> Tuple[List[str], List[Dict]]:
        """Simulate BO5 pick/ban process"""
        # BO5: Team1 ban, Team2 ban, Team1 pick, Team2 pick, Team1 pick, Team2 pick, remaining maps
        pick_sequence = []
        maps = available_maps.copy()
        picked_maps = []
        
        # Initial bans
        for i in range(2):
            banning_team = team1 if i % 2 == 0 else team2
            opponent_prefs = team2_prefs if i % 2 == 0 else team1_prefs
            
            strongest_opponent_map = max(maps, key=lambda m: opponent_prefs[m])
            maps.remove(strongest_opponent_map)
            
            pick_sequence.append({
                'action': 'ban',
                'team': banning_team,
                'map': strongest_opponent_map
            })
        
        # Picks (4 maps picked, 1 remains)
        for i in range(4):
            picking_team = team1 if i % 2 == 0 else team2
            team_prefs = team1_prefs if i % 2 == 0 else team2_prefs
            
            strongest_team_map = max(maps, key=lambda m: team_prefs[m])
            maps.remove(strongest_team_map)
            picked_maps.append(strongest_team_map)
            
            pick_sequence.append({
                'action': 'pick',
                'team': picking_team,
                'map': strongest_team_map
            })
        
        # Remaining map is decider
        if maps:
            remaining_map = maps[0]
            picked_maps.append(remaining_map)
            pick_sequence.append({
                'action': 'decider',
                'team': 'remaining',
                'map': remaining_map
            })
        
        return picked_maps, pick_sequence
    
    def _predict_series_outcome(self, team1: str, team2: str, 
                              picked_maps: List[str], series_format: SeriesFormat
                              ) -> Dict[str, float]:
        """Predict series outcome based on map strengths"""
        team1_map_wins = 0
        team2_map_wins = 0
        total_confidence = 0
        
        for map_name in picked_maps:
            team1_strength = self.get_team_map_strength(team1, map_name)
            team2_strength = self.get_team_map_strength(team2, map_name)
            
            # Convert to win probability
            strength_diff = team1_strength - team2_strength
            team1_win_prob = 0.5 + (strength_diff / 200)  # Normalize to 0-1
            team1_win_prob = max(0.1, min(0.9, team1_win_prob))  # Clamp
            
            if team1_win_prob > 0.5:
                team1_map_wins += team1_win_prob
            else:
                team2_map_wins += (1 - team1_win_prob)
            
            total_confidence += abs(team1_win_prob - 0.5) * 2
        
        # Series win probability
        total_maps = len(picked_maps)
        team1_series_prob = team1_map_wins / total_maps if total_maps > 0 else 0.5
        
        return {
            'team1_win_probability': team1_series_prob,
            'team2_win_probability': 1 - team1_series_prob,
            'confidence': total_confidence / total_maps if total_maps > 0 else 0.5,
            'predicted_winner': team1 if team1_series_prob > 0.5 else team2
        }
    
    def _generate_strategic_analysis(self, team1: str, team2: str, 
                                   picked_maps: List[str], 
                                   team1_advantages: Dict[str, float],
                                   team2_advantages: Dict[str, float]) -> Dict[str, str]:
        """Generate strategic analysis of the map pool"""
        analysis = {}
        
        # Overall map pool analysis
        team1_favored = sum(1 for adv in team1_advantages.values() if adv > 0)
        team2_favored = sum(1 for adv in team2_advantages.values() if adv > 0)
        
        analysis['overall'] = f"{team1} favored on {team1_favored} maps, {team2} favored on {team2_favored} maps"
        
        # Strongest advantages
        team1_strongest = max(team1_advantages.items(), key=lambda x: x[1]) if team1_advantages else (None, 0)
        team2_strongest = max(team2_advantages.items(), key=lambda x: x[1]) if team2_advantages else (None, 0)
        
        if team1_strongest[0]:
            analysis['team1_strength'] = f"{team1}'s strongest map: {team1_strongest[0]} (+{team1_strongest[1]:.1f})"
        
        if team2_strongest[0]:
            analysis['team2_strength'] = f"{team2}'s strongest map: {team2_strongest[0]} (+{team2_strongest[1]:.1f})"
        
        # Map meta considerations
        attacker_maps = []
        defender_maps = []
        
        for map_name in picked_maps:
            if map_name in self.map_meta_analysis:
                meta = self.map_meta_analysis[map_name]
                if meta['strategic_type'] == 'attacker_favored':
                    attacker_maps.append(map_name)
                elif meta['strategic_type'] == 'defender_favored':
                    defender_maps.append(map_name)
        
        if attacker_maps:
            analysis['attacker_maps'] = f"Attacker-favored maps: {', '.join(attacker_maps)}"
        if defender_maps:
            analysis['defender_maps'] = f"Defender-favored maps: {', '.join(defender_maps)}"
        
        return analysis
    
    def get_map_pool_summary(self) -> Dict[str, Dict]:
        """Get summary of current map pool and meta"""
        summary = {}
        
        for map_name in self.current_map_pool:
            map_data = {
                'teams_with_data': len([team for team in self.team_map_stats 
                                      if map_name in self.team_map_stats[team]]),
                'avg_team_strength': np.mean([
                    self.get_team_map_strength(team, map_name) 
                    for team in self.team_map_stats 
                    if map_name in self.team_map_stats[team]
                ]) if any(map_name in stats for stats in self.team_map_stats.values()) else 50.0
            }
            
            if map_name in self.map_meta_analysis:
                map_data.update(self.map_meta_analysis[map_name])
            
            summary[map_name] = map_data
        
        return summary


def main():
    """Test the map picker system"""
    map_picker = VCTMapPicker()
    map_picker.load_map_data()
    
    # Test map picking simulation
    result = map_picker.simulate_map_pick_ban("Sentinels", "Fnatic", SeriesFormat.BO3)
    
    print("\n=== Map Pick/Ban Simulation ===")
    print(f"Picked maps: {result.picked_maps}")
    print(f"Predicted winner: {result.predicted_series_outcome['predicted_winner']}")
    print(f"Win probability: {result.predicted_series_outcome['team1_win_probability']:.2%}")
    
    print("\nStrategic Analysis:")
    for key, analysis in result.strategic_analysis.items():
        print(f"  {key}: {analysis}")


if __name__ == "__main__":
    main()
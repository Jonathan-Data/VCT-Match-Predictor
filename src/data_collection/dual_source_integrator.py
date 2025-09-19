"""
Dual-Source Data Integrator
Combines and harmonizes data from VLR.gg and RIB.gg for enhanced prediction accuracy
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .vlr_scraper import VLRScraper, TeamStats as VLRTeamStats
from .rib_scraper import RibScraper, RibTeamStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedTeamStats:
    """Unified team statistics combining VLR.gg and RIB.gg data."""
    team_name: str
    region: str
    
    # Basic statistics (consensus from both sources)
    matches_played: int
    wins: int
    losses: int
    win_rate: float
    
    # VLR.gg specific data
    vlr_rating: float
    vlr_rounds_won: int
    vlr_rounds_lost: int
    vlr_round_win_rate: float
    vlr_players: List[Dict[str, Any]]
    vlr_recent_matches: List[Dict[str, Any]]
    
    # RIB.gg specific enhanced analytics
    rib_first_blood_rate: float
    rib_clutch_success_rate: float
    rib_eco_round_win_rate: float
    rib_tactical_timeout_efficiency: Optional[float]
    rib_comeback_factor: Optional[float]
    rib_consistency_rating: Optional[float]
    rib_map_pool_strength: Dict[str, float]
    rib_agent_composition_meta: Dict[str, float]
    rib_current_streak: int
    rib_streak_type: str
    
    # Composite metrics (derived from both sources)
    composite_team_rating: float  # Weighted average of both ratings
    data_confidence_score: float  # Quality score based on data availability
    cross_validation_score: float  # Agreement between sources
    
    # Enhanced feature set for ML
    momentum_index: float  # Recent performance trend
    tactical_diversity: float  # Agent and map pool diversity
    pressure_performance: float  # Performance in high-stakes situations
    adaptability_score: float  # Cross-regional and meta adaptation
    
    # Data source metadata
    vlr_data_available: bool
    rib_data_available: bool
    last_updated: str
    data_sources_used: List[str]

class DualSourceIntegrator:
    """Integrates data from multiple sources to create enhanced team profiles."""
    
    def __init__(self, config_path: str = None):
        """Initialize the dual-source integrator."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "teams.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "unified_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scrapers
        self.vlr_scraper = VLRScraper(config_path, delay=2.0)
        self.rib_scraper = RibScraper(config_path, use_selenium=True, delay=3.0)
        
        # Data validation thresholds
        self.min_matches_for_reliability = 5
        self.max_data_age_days = 7
        
        logger.info("Dual-source integrator initialized")
    
    def collect_all_data(self) -> Dict[str, UnifiedTeamStats]:
        """Collect data from both VLR.gg and RIB.gg, then integrate."""
        logger.info("Starting dual-source data collection...")
        
        # Collect VLR.gg data
        logger.info("Collecting VLR.gg data...")
        vlr_data = {}
        try:
            vlr_data = self.vlr_scraper.scrape_all_teams()
            logger.info(f"Collected VLR.gg data for {len(vlr_data)} teams")
        except Exception as e:
            logger.error(f"Error collecting VLR.gg data: {e}")
        
        # Collect RIB.gg data
        logger.info("Collecting RIB.gg data...")
        rib_data = {}
        try:
            rib_data = self.rib_scraper.scrape_all_teams()
            logger.info(f"Collected RIB.gg data for {len(rib_data)} teams")
        except Exception as e:
            logger.error(f"Error collecting RIB.gg data: {e}")
        
        # Integrate the data
        logger.info("Integrating data from both sources...")
        unified_data = self.integrate_team_data(vlr_data, rib_data)
        
        logger.info(f"Successfully created unified profiles for {len(unified_data)} teams")
        return unified_data
    
    def integrate_team_data(self, vlr_data: Dict[str, VLRTeamStats], 
                          rib_data: Dict[str, RibTeamStats]) -> Dict[str, UnifiedTeamStats]:
        """Integrate team data from both sources."""
        unified_teams = {}
        
        # Get all unique team names
        all_team_keys = set(vlr_data.keys()) | set(rib_data.keys())
        
        for team_key in all_team_keys:
            vlr_stats = vlr_data.get(team_key)
            rib_stats = rib_data.get(team_key)
            
            # Determine team name (prefer VLR if available)
            team_name = vlr_stats.team_name if vlr_stats else rib_stats.team_name
            region = vlr_stats.region if vlr_stats else rib_stats.region
            
            # Integrate the stats
            unified_stats = self._create_unified_stats(team_name, region, vlr_stats, rib_stats)
            unified_teams[team_key] = unified_stats
            
            logger.debug(f"Integrated data for {team_name}: "
                        f"VLR={'✓' if vlr_stats else '✗'}, "
                        f"RIB={'✓' if rib_stats else '✗'}")
        
        return unified_teams
    
    def _create_unified_stats(self, team_name: str, region: str,
                             vlr_stats: Optional[VLRTeamStats], 
                             rib_stats: Optional[RibTeamStats]) -> UnifiedTeamStats:
        """Create unified team statistics from available sources."""
        
        # Determine consensus statistics
        consensus_stats = self._determine_consensus_stats(vlr_stats, rib_stats)
        
        # Calculate composite metrics
        composite_rating = self._calculate_composite_rating(vlr_stats, rib_stats)
        confidence_score = self._calculate_data_confidence(vlr_stats, rib_stats)
        cross_validation_score = self._calculate_cross_validation_score(vlr_stats, rib_stats)
        
        # Calculate enhanced features
        momentum_index = self._calculate_momentum_index(vlr_stats, rib_stats)
        tactical_diversity = self._calculate_tactical_diversity(rib_stats)
        pressure_performance = self._calculate_pressure_performance(rib_stats)
        adaptability_score = self._calculate_adaptability_score(vlr_stats, rib_stats)
        
        # Determine data sources used
        data_sources = []
        if vlr_stats:
            data_sources.append("vlr.gg")
        if rib_stats:
            data_sources.append("rib.gg")
        
        return UnifiedTeamStats(
            team_name=team_name,
            region=region,
            
            # Consensus basic stats
            matches_played=consensus_stats['matches_played'],
            wins=consensus_stats['wins'],
            losses=consensus_stats['losses'],
            win_rate=consensus_stats['win_rate'],
            
            # VLR.gg specific data
            vlr_rating=vlr_stats.rating if vlr_stats else 0.0,
            vlr_rounds_won=vlr_stats.rounds_won if vlr_stats else 0,
            vlr_rounds_lost=vlr_stats.rounds_lost if vlr_stats else 0,
            vlr_round_win_rate=vlr_stats.round_win_rate if vlr_stats else 0.0,
            vlr_players=vlr_stats.players if vlr_stats else [],
            vlr_recent_matches=vlr_stats.recent_matches if vlr_stats else [],
            
            # RIB.gg specific data
            rib_first_blood_rate=rib_stats.first_blood_rate if rib_stats else 0.0,
            rib_clutch_success_rate=rib_stats.clutch_success_rate if rib_stats else 0.0,
            rib_eco_round_win_rate=rib_stats.eco_round_win_rate if rib_stats else 0.0,
            rib_tactical_timeout_efficiency=rib_stats.tactical_timeout_efficiency if rib_stats else None,
            rib_comeback_factor=rib_stats.comeback_factor if rib_stats else None,
            rib_consistency_rating=rib_stats.consistency_rating if rib_stats else None,
            rib_map_pool_strength=rib_stats.map_pool_strength if rib_stats else {},
            rib_agent_composition_meta=rib_stats.agent_composition_meta if rib_stats else {},
            rib_current_streak=rib_stats.current_streak if rib_stats else 0,
            rib_streak_type=rib_stats.streak_type if rib_stats else "none",
            
            # Composite metrics
            composite_team_rating=composite_rating,
            data_confidence_score=confidence_score,
            cross_validation_score=cross_validation_score,
            
            # Enhanced features
            momentum_index=momentum_index,
            tactical_diversity=tactical_diversity,
            pressure_performance=pressure_performance,
            adaptability_score=adaptability_score,
            
            # Metadata
            vlr_data_available=vlr_stats is not None,
            rib_data_available=rib_stats is not None,
            last_updated=datetime.now().isoformat(),
            data_sources_used=data_sources
        )
    
    def _determine_consensus_stats(self, vlr_stats: Optional[VLRTeamStats], 
                                  rib_stats: Optional[RibTeamStats]) -> Dict[str, Any]:
        """Determine consensus basic statistics between sources."""
        consensus = {
            'matches_played': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0
        }
        
        # If both sources available, use weighted average (prefer VLR for basic stats)
        if vlr_stats and rib_stats:
            # Use VLR as primary for basic stats, validate with RIB
            consensus['matches_played'] = vlr_stats.wins + vlr_stats.losses
            consensus['wins'] = vlr_stats.wins
            consensus['losses'] = vlr_stats.losses
            consensus['win_rate'] = vlr_stats.win_rate
            
            # Cross-validate with RIB data (log discrepancies)
            rib_matches = rib_stats.matches_played
            if abs(consensus['matches_played'] - rib_matches) > 2:
                logger.warning(f"Match count discrepancy for {vlr_stats.team_name}: "
                             f"VLR={consensus['matches_played']}, RIB={rib_matches}")
        
        elif vlr_stats:
            consensus['matches_played'] = vlr_stats.wins + vlr_stats.losses
            consensus['wins'] = vlr_stats.wins
            consensus['losses'] = vlr_stats.losses
            consensus['win_rate'] = vlr_stats.win_rate
        
        elif rib_stats:
            consensus['matches_played'] = rib_stats.matches_played
            consensus['wins'] = rib_stats.wins
            consensus['losses'] = rib_stats.losses
            consensus['win_rate'] = rib_stats.win_rate
        
        return consensus
    
    def _calculate_composite_rating(self, vlr_stats: Optional[VLRTeamStats], 
                                   rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate composite team rating from available sources."""
        if vlr_stats and rib_stats:
            # Weighted combination: VLR rating (60%) + RIB rating (40%)
            vlr_weight = 0.6
            rib_weight = 0.4
            return (vlr_stats.rating * vlr_weight) + (rib_stats.team_rating * rib_weight)
        elif vlr_stats:
            return vlr_stats.rating
        elif rib_stats:
            return rib_stats.team_rating
        else:
            return 0.0
    
    def _calculate_data_confidence(self, vlr_stats: Optional[VLRTeamStats], 
                                  rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate confidence score based on data availability and quality."""
        confidence = 0.0
        
        # Base confidence from data availability
        if vlr_stats:
            confidence += 0.5
            # Bonus for having recent matches
            if vlr_stats.recent_matches and len(vlr_stats.recent_matches) >= 3:
                confidence += 0.1
        
        if rib_stats:
            confidence += 0.4
            # Bonus for advanced metrics
            if rib_stats.consistency_rating is not None:
                confidence += 0.1
        
        # Penalty for insufficient match data
        total_matches = 0
        if vlr_stats:
            total_matches = max(total_matches, vlr_stats.wins + vlr_stats.losses)
        if rib_stats:
            total_matches = max(total_matches, rib_stats.matches_played)
        
        if total_matches < self.min_matches_for_reliability:
            confidence *= 0.7
        
        return min(confidence, 1.0)
    
    def _calculate_cross_validation_score(self, vlr_stats: Optional[VLRTeamStats], 
                                        rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate how well the two sources agree with each other."""
        if not (vlr_stats and rib_stats):
            return 0.5  # Default when only one source available
        
        agreements = []
        
        # Compare win rates
        vlr_win_rate = vlr_stats.win_rate
        rib_win_rate = rib_stats.win_rate
        if vlr_win_rate > 0 and rib_win_rate > 0:
            win_rate_agreement = 1.0 - abs(vlr_win_rate - rib_win_rate)
            agreements.append(win_rate_agreement)
        
        # Compare team ratings (normalized)
        if vlr_stats.rating > 0 and rib_stats.team_rating > 0:
            # Normalize ratings to 0-1 scale for comparison
            vlr_norm = min(vlr_stats.rating / 1500, 1.0)  # Assume max rating ~1500
            rib_norm = min(rib_stats.team_rating / 2.0, 1.0)  # Assume max rating ~2.0
            rating_agreement = 1.0 - abs(vlr_norm - rib_norm)
            agreements.append(rating_agreement)
        
        return np.mean(agreements) if agreements else 0.5
    
    def _calculate_momentum_index(self, vlr_stats: Optional[VLRTeamStats], 
                                 rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate team momentum based on recent performance."""
        momentum = 0.0
        
        # Use RIB streak data if available
        if rib_stats and rib_stats.current_streak > 0:
            streak_multiplier = min(rib_stats.current_streak / 5.0, 1.0)  # Max at 5 game streak
            if rib_stats.streak_type == "win":
                momentum = 0.5 + (0.5 * streak_multiplier)
            else:  # loss streak
                momentum = 0.5 - (0.5 * streak_multiplier)
        
        # Enhance with VLR recent matches if available
        if vlr_stats and vlr_stats.recent_matches:
            recent_results = []
            for match in vlr_stats.recent_matches[:5]:  # Last 5 matches
                result = match.get('result', '').lower()
                if 'win' in result or 'w' == result:
                    recent_results.append(1.0)
                elif 'loss' in result or 'l' == result:
                    recent_results.append(0.0)
            
            if recent_results:
                recent_win_rate = np.mean(recent_results)
                # Weight recent VLR data with existing momentum
                if momentum > 0:
                    momentum = (momentum * 0.7) + (recent_win_rate * 0.3)
                else:
                    momentum = recent_win_rate
        
        return max(0.0, min(1.0, momentum))  # Clamp to [0, 1]
    
    def _calculate_tactical_diversity(self, rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate tactical diversity from map pool and agent usage."""
        if not rib_stats:
            return 0.5  # Default
        
        diversity_score = 0.0
        
        # Map pool diversity
        map_pool = rib_stats.map_pool_strength
        if map_pool:
            # Calculate entropy of map win rates
            win_rates = [wr for wr in map_pool.values() if wr > 0]
            if win_rates:
                # Higher entropy = more diverse/balanced map pool
                win_rates_norm = np.array(win_rates) / sum(win_rates)
                entropy = -sum(p * np.log(p) for p in win_rates_norm if p > 0)
                max_entropy = np.log(len(win_rates))  # Max possible entropy
                map_diversity = entropy / max_entropy if max_entropy > 0 else 0
                diversity_score += map_diversity * 0.6
        
        # Agent composition diversity
        agent_meta = rib_stats.agent_composition_meta
        if agent_meta:
            usage_rates = [ur for ur in agent_meta.values() if ur > 0]
            if usage_rates:
                # Balanced agent usage indicates tactical flexibility
                usage_norm = np.array(usage_rates) / sum(usage_rates)
                agent_entropy = -sum(p * np.log(p) for p in usage_norm if p > 0)
                max_agent_entropy = np.log(len(usage_rates))
                agent_diversity = agent_entropy / max_agent_entropy if max_agent_entropy > 0 else 0
                diversity_score += agent_diversity * 0.4
        
        return max(0.0, min(1.0, diversity_score))
    
    def _calculate_pressure_performance(self, rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate performance under pressure from clutch and comeback stats."""
        if not rib_stats:
            return 0.5  # Default
        
        pressure_score = 0.0
        weights_sum = 0.0
        
        # Clutch success rate (high-pressure 1vX situations)
        if rib_stats.clutch_success_rate > 0:
            pressure_score += rib_stats.clutch_success_rate * 0.4
            weights_sum += 0.4
        
        # Comeback factor (recovering from disadvantaged positions)
        if rib_stats.comeback_factor is not None and rib_stats.comeback_factor > 0:
            pressure_score += rib_stats.comeback_factor * 0.3
            weights_sum += 0.3
        
        # Eco round win rate (winning with limited economy)
        if rib_stats.eco_round_win_rate > 0:
            pressure_score += rib_stats.eco_round_win_rate * 0.3
            weights_sum += 0.3
        
        # Normalize by actual weights used
        return pressure_score / weights_sum if weights_sum > 0 else 0.5
    
    def _calculate_adaptability_score(self, vlr_stats: Optional[VLRTeamStats], 
                                     rib_stats: Optional[RibTeamStats]) -> float:
        """Calculate adaptability based on cross-regional performance and meta adaptation."""
        adaptability = 0.5  # Default neutral score
        
        # Use cross-regional record if available from RIB
        if rib_stats and rib_stats.cross_regional_record:
            cross_regional = rib_stats.cross_regional_record
            if 'win_rate' in cross_regional:
                adaptability = cross_regional['win_rate']
        
        # Factor in tactical timeout efficiency (adaptation during matches)
        if rib_stats and rib_stats.tactical_timeout_efficiency is not None:
            timeout_factor = rib_stats.tactical_timeout_efficiency
            adaptability = (adaptability * 0.7) + (timeout_factor * 0.3)
        
        return max(0.0, min(1.0, adaptability))
    
    def save_unified_data(self, unified_data: Dict[str, UnifiedTeamStats], 
                         filename: str = "unified_team_stats.json") -> Path:
        """Save unified team statistics to JSON file."""
        output_file = self.data_dir / filename
        
        # Convert to dictionaries for JSON serialization
        unified_dict = {}
        for team_key, stats in unified_data.items():
            unified_dict[team_key] = asdict(stats)
        
        with open(output_file, 'w') as f:
            json.dump(unified_dict, f, indent=2, default=str)
        
        logger.info(f"Unified team statistics saved to {output_file}")
        return output_file
    
    def export_to_csv(self, unified_data: Dict[str, UnifiedTeamStats]) -> Path:
        """Export unified team statistics to CSV format for ML training."""
        data = []
        for team_key, stats in unified_data.items():
            row = {
                'team_key': team_key,
                'team_name': stats.team_name,
                'region': stats.region,
                'matches_played': stats.matches_played,
                'win_rate': stats.win_rate,
                'composite_team_rating': stats.composite_team_rating,
                'vlr_rating': stats.vlr_rating,
                'vlr_round_win_rate': stats.vlr_round_win_rate,
                'rib_first_blood_rate': stats.rib_first_blood_rate,
                'rib_clutch_success_rate': stats.rib_clutch_success_rate,
                'rib_eco_round_win_rate': stats.rib_eco_round_win_rate,
                'rib_consistency_rating': stats.rib_consistency_rating or 0.0,
                'rib_comeback_factor': stats.rib_comeback_factor or 0.0,
                'rib_tactical_timeout_efficiency': stats.rib_tactical_timeout_efficiency or 0.0,
                'momentum_index': stats.momentum_index,
                'tactical_diversity': stats.tactical_diversity,
                'pressure_performance': stats.pressure_performance,
                'adaptability_score': stats.adaptability_score,
                'data_confidence_score': stats.data_confidence_score,
                'cross_validation_score': stats.cross_validation_score,
                'vlr_data_available': stats.vlr_data_available,
                'rib_data_available': stats.rib_data_available,
                'current_streak': stats.rib_current_streak,
                'streak_type': stats.rib_streak_type
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_file = self.data_dir / "unified_team_stats.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Unified team statistics exported to {csv_file}")
        return csv_file
    
    def generate_integration_report(self, unified_data: Dict[str, UnifiedTeamStats]) -> str:
        """Generate a report on the data integration process."""
        total_teams = len(unified_data)
        vlr_only = sum(1 for stats in unified_data.values() if stats.vlr_data_available and not stats.rib_data_available)
        rib_only = sum(1 for stats in unified_data.values() if stats.rib_data_available and not stats.vlr_data_available)
        both_sources = sum(1 for stats in unified_data.values() if stats.vlr_data_available and stats.rib_data_available)
        
        avg_confidence = np.mean([stats.data_confidence_score for stats in unified_data.values()])
        avg_cross_validation = np.mean([stats.cross_validation_score for stats in unified_data.values() if stats.cross_validation_score != 0.5])
        
        report = f"""
        === Dual-Source Data Integration Report ===
        
        Total Teams Processed: {total_teams}
        
        Data Source Coverage:
        - Both VLR.gg & RIB.gg: {both_sources} ({both_sources/total_teams*100:.1f}%)
        - VLR.gg only: {vlr_only} ({vlr_only/total_teams*100:.1f}%)
        - RIB.gg only: {rib_only} ({rib_only/total_teams*100:.1f}%)
        
        Quality Metrics:
        - Average Data Confidence: {avg_confidence:.3f}
        - Average Cross-Validation Score: {avg_cross_validation:.3f}
        
        Enhanced Features Generated:
        - Momentum Index
        - Tactical Diversity
        - Pressure Performance
        - Adaptability Score
        - Composite Team Rating
        
        The integration process has successfully combined traditional statistics from VLR.gg
        with advanced analytics from RIB.gg to create a comprehensive dataset for ML training.
        """
        
        return report
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.rib_scraper, 'close'):
            self.rib_scraper.close()

def main():
    """Main function to demonstrate dual-source integration."""
    integrator = DualSourceIntegrator()
    
    try:
        print("Starting dual-source data integration...")
        print("This will collect data from both VLR.gg and RIB.gg")
        
        # Collect and integrate data
        unified_data = integrator.collect_all_data()
        
        if unified_data:
            # Save results
            json_file = integrator.save_unified_data(unified_data)
            csv_file = integrator.export_to_csv(unified_data)
            
            # Generate report
            report = integrator.generate_integration_report(unified_data)
            
            print("\nDual-source integration completed!")
            print(f"Results saved to:")
            print(f"  - JSON: {json_file}")
            print(f"  - CSV: {csv_file}")
            print(report)
        else:
            print("No team data was successfully integrated.")
    
    except Exception as e:
        logger.error(f"Error during integration: {e}")
    finally:
        integrator.close()

if __name__ == "__main__":
    main()
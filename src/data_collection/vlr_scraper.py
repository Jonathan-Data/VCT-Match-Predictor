"""
VLR.gg Team Statistics Scraper
Scrapes team statistics and match data from VLR.gg
"""

import requests
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TeamStats:
    """Data class for team statistics."""
    team_name: str
    region: str
    vlr_id: int
    rating: float
    wins: int
    losses: int
    win_rate: float
    rounds_won: int
    rounds_lost: int
    round_win_rate: float
    avg_combat_score: float
    players: List[Dict[str, Any]]
    recent_matches: List[Dict[str, Any]]

class VLRScraper:
    """Scraper for VLR.gg team statistics and match data."""
    
    def __init__(self, config_path: str = None, delay: float = 2.0):
        """Initialize the scraper with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "teams.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "external"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make a request to VLR.gg with error handling."""
        try:
            time.sleep(self.delay)  # Be respectful to the server
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def scrape_team_stats(self, team_key: str) -> Optional[TeamStats]:
        """Scrape statistics for a specific team."""
        team_config = self.config['teams'][team_key]
        url = team_config['vlr_url']
        
        logger.info(f"Scraping stats for {team_config['name']} from {url}")
        soup = self._make_request(url)
        
        if not soup:
            return None
        
        try:
            # Extract basic team info
            team_name = team_config['name']
            region = team_config['region']
            vlr_id = team_config['vlr_id']
            
            # Try to extract statistics (these selectors may need adjustment)
            stats = self._extract_team_statistics(soup)
            players = self._extract_player_info(soup)
            recent_matches = self._extract_recent_matches(soup, vlr_id)
            
            return TeamStats(
                team_name=team_name,
                region=region,
                vlr_id=vlr_id,
                rating=stats.get('rating', 0.0),
                wins=stats.get('wins', 0),
                losses=stats.get('losses', 0),
                win_rate=stats.get('win_rate', 0.0),
                rounds_won=stats.get('rounds_won', 0),
                rounds_lost=stats.get('rounds_lost', 0),
                round_win_rate=stats.get('round_win_rate', 0.0),
                avg_combat_score=stats.get('avg_combat_score', 0.0),
                players=players,
                recent_matches=recent_matches
            )
            
        except Exception as e:
            logger.error(f"Error parsing team stats for {team_name}: {e}")
            return None
    
    def _extract_team_statistics(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract team statistics from the page."""
        stats = {}
        
        try:
            # These selectors would need to be updated based on VLR.gg's actual HTML structure
            # This is a placeholder implementation
            stats_section = soup.find('div', class_='team-stats')
            if stats_section:
                # Extract wins/losses
                record_elem = stats_section.find('span', text=lambda x: x and 'W' in x and 'L' in x)
                if record_elem:
                    record_text = record_elem.get_text().strip()
                    # Parse "15W - 3L" format
                    if '-' in record_text:
                        wins = int(record_text.split('W')[0].strip())
                        losses = int(record_text.split('-')[1].split('L')[0].strip())
                        stats['wins'] = wins
                        stats['losses'] = losses
                        stats['win_rate'] = wins / (wins + losses) if (wins + losses) > 0 else 0.0
            
            # Extract rating if available
            rating_elem = soup.find('span', class_='rating')
            if rating_elem:
                stats['rating'] = float(rating_elem.get_text().strip())
                
        except Exception as e:
            logger.warning(f"Could not extract all statistics: {e}")
        
        return stats
    
    def _extract_player_info(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract player information from the team page."""
        players = []
        
        try:
            # This would need to be updated based on VLR.gg's actual HTML structure
            player_elements = soup.find_all('div', class_='player-card')
            
            for player_elem in player_elements:
                player_info = {
                    'name': '',
                    'role': '',
                    'rating': 0.0,
                    'acs': 0.0,
                    'kd_ratio': 0.0
                }
                
                name_elem = player_elem.find('span', class_='player-name')
                if name_elem:
                    player_info['name'] = name_elem.get_text().strip()
                
                players.append(player_info)
                
        except Exception as e:
            logger.warning(f"Could not extract player info: {e}")
        
        return players
    
    def _extract_recent_matches(self, soup: BeautifulSoup, team_id: int) -> List[Dict[str, Any]]:
        """Extract recent match results."""
        matches = []
        
        try:
            # This would need to be updated based on VLR.gg's actual HTML structure
            match_elements = soup.find_all('div', class_='match-item')[:10]  # Last 10 matches
            
            for match_elem in match_elements:
                match_info = {
                    'date': '',
                    'opponent': '',
                    'result': '',
                    'score': '',
                    'map': ''
                }
                matches.append(match_info)
                
        except Exception as e:
            logger.warning(f"Could not extract recent matches: {e}")
        
        return matches
    
    def scrape_all_teams(self) -> Dict[str, TeamStats]:
        """Scrape statistics for all teams."""
        all_teams_stats = {}
        
        for team_key in tqdm(self.config['teams'].keys(), desc="Scraping team stats"):
            stats = self.scrape_team_stats(team_key)
            if stats:
                all_teams_stats[team_key] = stats
            
        logger.info(f"Successfully scraped stats for {len(all_teams_stats)} teams")
        return all_teams_stats
    
    def save_team_stats(self, team_stats: Dict[str, TeamStats], filename: str = "team_stats.json"):
        """Save team statistics to a JSON file."""
        output_file = self.data_dir / filename
        
        # Convert TeamStats objects to dictionaries for JSON serialization
        stats_dict = {}
        for team_key, stats in team_stats.items():
            stats_dict[team_key] = {
                'team_name': stats.team_name,
                'region': stats.region,
                'vlr_id': stats.vlr_id,
                'rating': stats.rating,
                'wins': stats.wins,
                'losses': stats.losses,
                'win_rate': stats.win_rate,
                'rounds_won': stats.rounds_won,
                'rounds_lost': stats.rounds_lost,
                'round_win_rate': stats.round_win_rate,
                'avg_combat_score': stats.avg_combat_score,
                'players': stats.players,
                'recent_matches': stats.recent_matches
            }
        
        with open(output_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        logger.info(f"Team statistics saved to {output_file}")
        return output_file
    
    def export_to_csv(self, team_stats: Dict[str, TeamStats]) -> Path:
        """Export team statistics to CSV format."""
        data = []
        for team_key, stats in team_stats.items():
            data.append({
                'team_key': team_key,
                'team_name': stats.team_name,
                'region': stats.region,
                'vlr_id': stats.vlr_id,
                'rating': stats.rating,
                'wins': stats.wins,
                'losses': stats.losses,
                'win_rate': stats.win_rate,
                'rounds_won': stats.rounds_won,
                'rounds_lost': stats.rounds_lost,
                'round_win_rate': stats.round_win_rate,
                'avg_combat_score': stats.avg_combat_score,
                'num_players': len(stats.players),
                'recent_matches_count': len(stats.recent_matches)
            })
        
        df = pd.DataFrame(data)
        csv_file = self.data_dir / "team_stats.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Team statistics exported to {csv_file}")
        return csv_file

def main():
    """Main function to scrape all team statistics."""
    scraper = VLRScraper()
    
    print("Scraping VLR.gg team statistics...")
    print("Note: This will take some time to be respectful to the server")
    
    team_stats = scraper.scrape_all_teams()
    
    # Save results
    json_file = scraper.save_team_stats(team_stats)
    csv_file = scraper.export_to_csv(team_stats)
    
    print(f"\nScraping completed!")
    print(f"Results saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - CSV: {csv_file}")
    print(f"\nScraped data for {len(team_stats)} teams")

if __name__ == "__main__":
    main()
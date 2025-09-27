"""
RIB.gg Team Statistics and Match Data Scraper
Enhanced data collection from rib.gg to complement vlr.gg data
"""

import requests
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RibTeamStats:
    """Data class for rib.gg team statistics."""
    team_name: str
    region: str
    rib_id: Optional[int]

    matches_played: int
    wins: int
    losses: int
    win_rate: float


    avg_rounds_per_match: float
    first_blood_rate: float
    clutch_success_rate: float
    eco_round_win_rate: float


    tactical_timeout_efficiency: Optional[float]
    comeback_factor: Optional[float]
    consistency_rating: Optional[float]


    map_pool_strength: Dict[str, float]
    agent_composition_meta: Dict[str, float]


    team_rating: float
    avg_combat_score: float
    avg_kd_ratio: float


    recent_form: List[Dict[str, Any]]
    current_streak: int
    streak_type: str


    tournament_performance: Dict[str, Any]
    cross_regional_record: Optional[Dict[str, Any]]

@dataclass
class RibMatchData:
    """Enhanced match data from rib.gg."""
    match_id: str
    date: str
    team1: str
    team2: str
    score: str
    winner: str


    match_duration: Optional[int]
    total_rounds: int
    overtime_rounds: int


    first_blood_advantage: Dict[str, float]
    eco_round_conversions: Dict[str, int]
    clutch_situations: Dict[str, Dict[str, Any]]


    maps_played: List[Dict[str, Any]]
    agent_compositions: Dict[str, List[str]]


    team_ratings: Dict[str, float]
    individual_performances: List[Dict[str, Any]]

class RibScraper:
    """Advanced scraper for rib.gg with Cloudflare bypass capabilities."""

    def __init__(self, config_path: str = None, use_selenium: bool = True, delay: float = 3.0):
        """Initialize the rib.gg scraper."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "teams.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(__file__).parent.parent.parent / "data" / "rib_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.delay = delay
        self.use_selenium = use_selenium
        self.driver = None

        if use_selenium:
            self._setup_selenium()
        else:
            self._setup_requests_session()

    def _setup_selenium(self):
        """Setup Selenium WebDriver to handle Cloudflare protection."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")


        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        try:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            logger.info("Falling back to requests session")
            self.use_selenium = False
            self._setup_requests_session()

    def _setup_requests_session(self):
        """Setup requests session as fallback."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def _get_page_content(self, url: str, wait_for_element: str = None) -> Optional[BeautifulSoup]:
        """Get page content with Cloudflare handling."""
        time.sleep(self.delay)

        if self.use_selenium and self.driver:
            try:
                logger.info(f"Loading {url} with Selenium")
                self.driver.get(url)


                WebDriverWait(self.driver, 30).until(
                    lambda driver: "Just a moment" not in driver.title
                )

                if wait_for_element:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                    )

                html = self.driver.page_source
                return BeautifulSoup(html, 'html.parser')

            except TimeoutException:
                logger.warning(f"Timeout waiting for page to load: {url}")
                return None
            except Exception as e:
                logger.error(f"Selenium error for {url}: {e}")
                return None

        else:

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                if "Just a moment" in response.text:
                    logger.warning(f"Cloudflare challenge detected for {url}")
                    return None

                return BeautifulSoup(response.content, 'html.parser')

            except requests.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

    def scrape_team_stats(self, team_name: str) -> Optional[RibTeamStats]:
        """Scrape enhanced team statistics from rib.gg."""

        team_url = f"https://rib.gg/teams/{team_name.lower().replace(' ', '-')}"

        logger.info(f"Scraping rib.gg stats for {team_name}")
        soup = self._get_page_content(team_url, wait_for_element=".team-stats")

        if not soup:
            logger.warning(f"Could not load team page for {team_name}")
            return None

        try:

            region = self._extract_team_region(soup)


            matches_played, wins, losses = self._extract_basic_stats(soup)
            win_rate = wins / matches_played if matches_played > 0 else 0.0


            enhanced_metrics = self._extract_enhanced_metrics(soup)


            map_data = self._extract_map_performance(soup)
            agent_data = self._extract_agent_composition(soup)


            recent_matches = self._extract_recent_matches(soup, limit=10)
            current_streak, streak_type = self._calculate_current_streak(recent_matches)

            return RibTeamStats(
                team_name=team_name,
                region=region or "Unknown",
                rib_id=None,
                matches_played=matches_played,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                avg_rounds_per_match=enhanced_metrics.get('avg_rounds_per_match', 0.0),
                first_blood_rate=enhanced_metrics.get('first_blood_rate', 0.0),
                clutch_success_rate=enhanced_metrics.get('clutch_success_rate', 0.0),
                eco_round_win_rate=enhanced_metrics.get('eco_round_win_rate', 0.0),
                tactical_timeout_efficiency=enhanced_metrics.get('tactical_timeout_efficiency'),
                comeback_factor=enhanced_metrics.get('comeback_factor'),
                consistency_rating=enhanced_metrics.get('consistency_rating'),
                map_pool_strength=map_data,
                agent_composition_meta=agent_data,
                team_rating=enhanced_metrics.get('team_rating', 0.0),
                avg_combat_score=enhanced_metrics.get('avg_combat_score', 0.0),
                avg_kd_ratio=enhanced_metrics.get('avg_kd_ratio', 0.0),
                recent_form=recent_matches,
                current_streak=current_streak,
                streak_type=streak_type,
                tournament_performance=enhanced_metrics.get('tournament_performance', {}),
                cross_regional_record=enhanced_metrics.get('cross_regional_record')
            )

        except Exception as e:
            logger.error(f"Error extracting team stats for {team_name}: {e}")
            return None

    def _extract_basic_stats(self, soup: BeautifulSoup) -> tuple[int, int, int]:
        """Extract basic win/loss statistics."""

        try:
            stats_container = soup.find('div', class_='team-record') or soup.find('div', class_='stats-summary')

            if stats_container:

                record_text = stats_container.get_text()


                if 'W' in record_text and 'L' in record_text:
                    import re
                    match = re.search(r'(\d+)W.*?(\d+)L', record_text)
                    if match:
                        wins, losses = int(match.group(1)), int(match.group(2))
                        return wins + losses, wins, losses


            wins_elem = soup.find(text=re.compile(r'Wins?:?\s*(\d+)', re.I))
            losses_elem = soup.find(text=re.compile(r'Losses?:?\s*(\d+)', re.I))

            if wins_elem and losses_elem:
                wins = int(re.search(r'(\d+)', wins_elem).group(1))
                losses = int(re.search(r'(\d+)', losses_elem).group(1))
                return wins + losses, wins, losses

        except Exception as e:
            logger.warning(f"Could not extract basic stats: {e}")

        return 0, 0, 0

    def _extract_enhanced_metrics(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract rib.gg specific enhanced metrics."""
        metrics = {}

        try:

            analytics_section = soup.find('div', class_=['analytics', 'advanced-stats', 'team-metrics'])

            if analytics_section:

                metrics.update({
                    'avg_rounds_per_match': self._extract_float_stat(analytics_section, 'rounds-per-match'),
                    'first_blood_rate': self._extract_float_stat(analytics_section, 'first-blood-rate'),
                    'clutch_success_rate': self._extract_float_stat(analytics_section, 'clutch-rate'),
                    'eco_round_win_rate': self._extract_float_stat(analytics_section, 'eco-win-rate'),
                    'team_rating': self._extract_float_stat(analytics_section, 'team-rating'),
                    'avg_combat_score': self._extract_float_stat(analytics_section, 'avg-acs'),
                    'avg_kd_ratio': self._extract_float_stat(analytics_section, 'avg-kd'),
                })


                metrics.update({
                    'tactical_timeout_efficiency': self._extract_float_stat(analytics_section, 'timeout-efficiency'),
                    'comeback_factor': self._extract_float_stat(analytics_section, 'comeback-factor'),
                    'consistency_rating': self._extract_float_stat(analytics_section, 'consistency'),
                })

        except Exception as e:
            logger.warning(f"Could not extract enhanced metrics: {e}")

        return metrics

    def _extract_float_stat(self, container: BeautifulSoup, stat_class: str) -> float:
        """Helper to extract float statistics."""
        try:
            stat_elem = container.find(class_=stat_class) or container.find(attrs={'data-stat': stat_class})
            if stat_elem:
                text = stat_elem.get_text().strip()

                import re
                number_match = re.search(r'([\d.]+)', text.replace('%', ''))
                if number_match:
                    return float(number_match.group(1))
        except:
            pass
        return 0.0

    def _extract_team_region(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract team region information."""
        try:
            region_elem = soup.find(class_=['team-region', 'region']) or soup.find(attrs={'data-region': True})
            if region_elem:
                return region_elem.get_text().strip()
        except:
            pass
        return None

    def _extract_map_performance(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract map-specific performance data."""
        map_data = {}
        try:
            maps_section = soup.find('div', class_=['map-stats', 'map-performance'])
            if maps_section:
                map_elements = maps_section.find_all(class_=['map-stat', 'map-item'])
                for map_elem in map_elements:
                    map_name = map_elem.find(class_='map-name')
                    win_rate = map_elem.find(class_='map-winrate')

                    if map_name and win_rate:
                        map_data[map_name.get_text().strip()] = self._extract_float_stat(map_elem, 'win-rate')
        except Exception as e:
            logger.warning(f"Could not extract map performance: {e}")

        return map_data

    def _extract_agent_composition(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract agent composition and usage statistics."""
        agent_data = {}
        try:
            agents_section = soup.find('div', class_=['agent-stats', 'composition-stats'])
            if agents_section:
                agent_elements = agents_section.find_all(class_=['agent-stat', 'agent-item'])
                for agent_elem in agent_elements:
                    agent_name = agent_elem.find(class_='agent-name')
                    usage_rate = agent_elem.find(class_='usage-rate')

                    if agent_name and usage_rate:
                        agent_data[agent_name.get_text().strip()] = self._extract_float_stat(agent_elem, 'usage-rate')
        except Exception as e:
            logger.warning(f"Could not extract agent composition: {e}")

        return agent_data

    def _extract_recent_matches(self, soup: BeautifulSoup, limit: int = 10) -> List[Dict[str, Any]]:
        """Extract recent match results with detailed statistics."""
        matches = []
        try:
            matches_section = soup.find('div', class_=['recent-matches', 'match-history'])
            if matches_section:
                match_elements = matches_section.find_all(class_=['match-item', 'match-row'])[:limit]

                for match_elem in match_elements:
                    match_data = {
                        'date': self._extract_text(match_elem, 'match-date'),
                        'opponent': self._extract_text(match_elem, 'opponent-name'),
                        'result': self._extract_text(match_elem, 'match-result'),
                        'score': self._extract_text(match_elem, 'match-score'),
                        'map': self._extract_text(match_elem, 'match-map'),
                        'performance_rating': self._extract_float_stat(match_elem, 'team-rating')
                    }
                    matches.append(match_data)
        except Exception as e:
            logger.warning(f"Could not extract recent matches: {e}")

        return matches

    def _extract_text(self, container: BeautifulSoup, class_name: str) -> str:
        """Helper to extract text from elements."""
        elem = container.find(class_=class_name)
        return elem.get_text().strip() if elem else ""

    def _calculate_current_streak(self, recent_matches: List[Dict[str, Any]]) -> tuple[int, str]:
        """Calculate current win/loss streak."""
        if not recent_matches:
            return 0, "none"

        streak = 0
        streak_type = "none"

        for match in recent_matches:
            result = match.get('result', '').lower()
            if 'win' in result or 'w' == result:
                if streak_type == "win" or streak_type == "none":
                    streak += 1
                    streak_type = "win"
                else:
                    break
            elif 'loss' in result or 'l' == result:
                if streak_type == "loss" or streak_type == "none":
                    streak += 1
                    streak_type = "loss"
                else:
                    break
            else:
                break

        return streak, streak_type

    def scrape_all_teams(self) -> Dict[str, RibTeamStats]:
        """Scrape statistics for all configured teams."""
        all_team_stats = {}

        team_names = [team_info['name'] for team_info in self.config['teams'].values()]

        for team_name in tqdm(team_names, desc="Scraping rib.gg team stats"):
            stats = self.scrape_team_stats(team_name)
            if stats:
                all_team_stats[team_name.lower().replace(' ', '_')] = stats
            else:
                logger.warning(f"Failed to scrape stats for {team_name}")

        logger.info(f"Successfully scraped rib.gg stats for {len(all_team_stats)} teams")
        return all_team_stats

    def save_team_stats(self, team_stats: Dict[str, RibTeamStats], filename: str = "rib_team_stats.json"):
        """Save rib.gg team statistics to JSON file."""
        output_file = self.data_dir / filename


        stats_dict = {}
        for team_key, stats in team_stats.items():
            stats_dict[team_key] = asdict(stats)

        with open(output_file, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)

        logger.info(f"Rib.gg team statistics saved to {output_file}")
        return output_file

    def export_to_csv(self, team_stats: Dict[str, RibTeamStats]) -> Path:
        """Export rib.gg team statistics to CSV format."""
        data = []
        for team_key, stats in team_stats.items():
            row = {
                'team_key': team_key,
                'team_name': stats.team_name,
                'region': stats.region,
                'matches_played': stats.matches_played,
                'wins': stats.wins,
                'losses': stats.losses,
                'win_rate': stats.win_rate,
                'team_rating': stats.team_rating,
                'avg_combat_score': stats.avg_combat_score,
                'avg_kd_ratio': stats.avg_kd_ratio,
                'first_blood_rate': stats.first_blood_rate,
                'clutch_success_rate': stats.clutch_success_rate,
                'eco_round_win_rate': stats.eco_round_win_rate,
                'current_streak': stats.current_streak,
                'streak_type': stats.streak_type,
                'consistency_rating': stats.consistency_rating,
                'comeback_factor': stats.comeback_factor,
                'tactical_timeout_efficiency': stats.tactical_timeout_efficiency
            }
            data.append(row)

        df = pd.DataFrame(data)
        csv_file = self.data_dir / "rib_team_stats.csv"
        df.to_csv(csv_file, index=False)

        logger.info(f"Rib.gg team statistics exported to {csv_file}")
        return csv_file

    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            logger.info("Selenium WebDriver closed")

def main():
    """Main function to scrape all rib.gg team statistics."""
    scraper = RibScraper(use_selenium=True)

    try:
        print("Scraping rib.gg team statistics...")
        print("Note: This uses Selenium to bypass Cloudflare protection")

        team_stats = scraper.scrape_all_teams()

        if team_stats:

            json_file = scraper.save_team_stats(team_stats)
            csv_file = scraper.export_to_csv(team_stats)

            print(f"\nRib.gg scraping completed!")
            print(f"Results saved to:")
            print(f"  - JSON: {json_file}")
            print(f"  - CSV: {csv_file}")
            print(f"\nScraped enhanced data for {len(team_stats)} teams")
        else:
            print("No team statistics were successfully scraped.")

    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
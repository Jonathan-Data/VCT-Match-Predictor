#!/usr/bin/env python3
"""
VLR.gg Tournament Scraper - Optimized Version
====================================================

A high-performance, production-ready scraper for VLR.gg tournament data.
Includes aggressive filtering to ensure only valid team matchups are returned.

Features:
- Comprehensive team name cleaning and validation
- Robust error handling and retry logic
- Performance optimizations and caching
- Extensive logging and monitoring
- Type hints throughout

Author: VCT Predictor Team
Version: 2.0.0
"""

from __future__ import annotations

import re
import time
import logging
import unicodedata
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchData:
    """Data class for match information"""
    team1: str
    team2: str
    match_time: str = ""
    match_stage: str = ""
    status: str = "upcoming"
    match_url: str = ""
    confidence: float = 0.0

@dataclass
class TournamentData:
    """Data class for tournament information"""
    name: str
    event_id: str
    dates: str = ""
    prize: str = ""
    location: str = ""
    url: str = ""

class TeamNameProcessor:
    """Optimized team name processing with constants and caching"""
    
    # Pre-compiled regex patterns for better performance
    INCOMPLETE_MATCH_PATTERNS = [
        re.compile(r'^[^\s]+\s+vs\s*$', re.IGNORECASE),
        re.compile(r'^\s*vs\s+[^\s]+$', re.IGNORECASE), 
        re.compile(r'^\s*vs\s+', re.IGNORECASE),
        re.compile(r'^[^\s]+\s+v\s*$', re.IGNORECASE),
        re.compile(r'^\s*v\s+[^\s]+$', re.IGNORECASE),
    ]
    
    NATIONALITY_PATTERNS = [
        re.compile(r'\b(united states|usa)\b', re.IGNORECASE),
        re.compile(r'\bchina\b', re.IGNORECASE),
        re.compile(r'\b(korea|south korea)\b', re.IGNORECASE),
        re.compile(r'\bjapan\b', re.IGNORECASE),
        re.compile(r'\bbrazil\b', re.IGNORECASE),
        re.compile(r'\bsingapore\b', re.IGNORECASE),
        re.compile(r'\b(united kingdom|uk)\b', re.IGNORECASE),
        re.compile(r'\bgermany\b', re.IGNORECASE),
        re.compile(r'\bthailand\b', re.IGNORECASE),
        re.compile(r'\bphilippines\b', re.IGNORECASE),
        re.compile(r'\bindonesia\b', re.IGNORECASE),
        re.compile(r'\bmalaysia\b', re.IGNORECASE),
        re.compile(r'\bvietnam\b', re.IGNORECASE),
        re.compile(r'\baustralia\b', re.IGNORECASE),
        re.compile(r'\b(emea|apac|europe|asia|na)\b', re.IGNORECASE),
    ]
    
    QUALIFICATION_PATTERNS = [
        re.compile(r'\bqualified\b', re.IGNORECASE),
        re.compile(r'\beliminated\b', re.IGNORECASE),
        re.compile(r'\bbye\b', re.IGNORECASE),
        re.compile(r'\btbd\b', re.IGNORECASE),
        re.compile(r'\bto be determined\b', re.IGNORECASE),
        re.compile(r'\bwinner of\b', re.IGNORECASE),
        re.compile(r'\bloser of\b', re.IGNORECASE),
        re.compile(r'\bupper bracket\b', re.IGNORECASE),
        re.compile(r'\blower bracket\b', re.IGNORECASE),
        re.compile(r'\b(quarterfinal|semifinal|grand final)\b', re.IGNORECASE),
        re.compile(r'^\s*[wl]\d+\s*$', re.IGNORECASE),  # W1, L2, etc
        re.compile(r'\bgroup\s+stage\b', re.IGNORECASE),
        re.compile(r'\bplayoffs\b', re.IGNORECASE),
        re.compile(r'\bquarterfinals\b', re.IGNORECASE),
        re.compile(r'\belimination\s+match\b', re.IGNORECASE),
        re.compile(r'\bdecider\s+match\b', re.IGNORECASE),
        re.compile(r'\badvanced\b', re.IGNORECASE),
        re.compile(r'^(elimination|decider|winner|loser)\s+', re.IGNORECASE),  # Stage prefixes
        re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', re.IGNORECASE),  # Dates in team names
        re.compile(r'\d+\s*$', re.IGNORECASE),  # Names ending with numbers
    ]
    
    CLEANING_PATTERNS = [
        # Country/region removal
        (re.compile(r'\[[A-Z]{2,4}\]\s*'), ''),                    # [US] format
        (re.compile(r'\([A-Z]{2,4}\)\s*'), ''),                    # (US) format
        (re.compile(r'^[A-Z]{2,4}\s+'), ''),                       # US Team format
        (re.compile(r'\b(United States|China|Korea|South Korea|Japan|Brazil|Singapore|United Kingdom|Germany|Thailand|Philippines|Indonesia|Malaysia|Vietnam|Australia)\b', re.I), ''),  # Country names
        (re.compile(r'\b(USA|EMEA|APAC|Europe|Asia|NA|EU|BR|KR|JP|SG|GB|UK|DE|FR|TH|PH|ID|MY|VN|AU)\b', re.I), ''),  # Region/country codes
        
        # Numeric suffixes and prefixes
        (re.compile(r'\d+$'), ''),                                 # Trailing numbers
        (re.compile(r'^\d+\s*'), ''),                              # Leading numbers
        (re.compile(r'\s*#\d+'), ''),                              # Seed numbers
        
        # Date and time patterns
        (re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b', re.I), ''),  # Dates
        (re.compile(r'\s*\d{1,2}/\d{1,2}(/\d{2,4})?'), ''),        # Date formats
        (re.compile(r'\s*\d{1,2}:\d{2}\s*(AM|PM)?', re.I), ''),   # Times
        (re.compile(r'\s*\d+d\s*\d*h?'), ''),                     # Time durations
        (re.compile(r'\s*\d+h\s*\d*m?'), ''),                     # Time durations
        (re.compile(r'\s*\d+m'), ''),                              # Minutes
        
        # Tournament stage prefixes/suffixes
        (re.compile(r'^(Elimination|Decider|Winner|Loser|Upper|Lower)\s+', re.I), ''),  # Stage prefixes
        (re.compile(r'\s+(Elimination|Decider|Winner|Loser|Upper|Lower)$', re.I), ''),  # Stage suffixes
        
        # Score and bracket information
        (re.compile(r'\s*\d+[-–—]\d+'), ''),                       # Scores
        (re.compile(r'\s*\(\d+[-–—]\d+\)'), ''),                   # (2-1) format
        
        # Punctuation and spacing
        (re.compile(r'[•·‧‹›\-–—]', re.I), ' '),                  # Replace punctuation
        (re.compile(r'\s+'), ' '),                                 # Normalize whitespace
    ]
    
    # Known valid VCT teams (cached for performance)
    KNOWN_VCT_TEAMS: Set[str] = {
        # Major teams
        'sentinels', 'paper rex', 'fnatic', 'drx', 'team heretics', 'edward gaming',
        'g2 esports', 'nrg', 't1', 'team liquid', 'giantx', 'fut esports',
        'bilibili gaming', 'rex regum qeon', 'mibr', 'cloud9', 'gen.g',
        'trace esports', 'dragon ranger gaming', 'xi lai gaming',
        'loud', '100 thieves', 'evil geniuses', 'leviatan', 'kru esports',
        
        # Abbreviations and variations
        'prx', 'edg', 'ths', 'tl', 'blg', 'rrq', 'c9', 'geng', 'te', 'drg', 'xlg',
        'eg', 'lev', 'kru', 'sen', 'fnc', 'fut', 'gx', 'th', 'ge', 'fs',
        'ts', 'dk', 'on', 'sr', 'og', 'mibr',
        
        # Direct validation for special cases (removed 'vs' as it's not a team name)
        't1', 'drx', 'loud', 'nrg', '100 thieves', 'gen.g', 'c9', 'cloud9',
        'g2', 'g2 esports', 'team heretics', 'team liquid', 'edward gaming',
        'paper rex', 'sentinels', 'fnatic'
    }
    
    # Explicitly invalid terms that should never be considered valid teams
    INVALID_TERMS: Set[str] = {
        'tbd', 'vs', 'v', 'winner', 'loser', 'qualified', 'eliminated', 'bye',
        'match', 'game', 'stage', 'group', 'playoffs', 'final', 'semifinal',
        'quarterfinal', 'upper', 'lower', 'bracket', 'usa', 'br', 'kr', 'jp',
        'eu', 'na', 'emea', 'apac', 'asia', 'europe', 'china', 'korea',
        'elimination', 'decider', 'united states', 'south korea', 'japan',
        'brazil', 'singapore', 'united kingdom', 'germany', 'thailand',
        'philippines', 'indonesia', 'malaysia', 'vietnam', 'australia',
        'sentinelsunited states', 'edward gamingchina', 'paper rex2',
        'bilibili gaming0', 'team liquid0', 'g2 esports0', 'sentinels1',
        'evry', 'courcouronnes'  # Known invalid entries from scraping
    }
    
    COUNTRY_CODE_MAP = {
        'us': 'usa', 'usa': 'usa', 'br': 'brazil', 'kr': 'korea',
        'jp': 'japan', 'sg': 'singapore', 'gb': 'united kingdom',
        'uk': 'united kingdom', 'de': 'germany', 'fr': 'france'
    }
    
    @classmethod
    @lru_cache(maxsize=1000)
    def clean_team_name(cls, team_name: str) -> str:
        """
        Clean and standardize team name with aggressive filtering.
        
        Args:
            team_name: Raw team name from scraping
            
        Returns:
            Cleaned team name or empty string if invalid
        """
        if not team_name or len(team_name.strip()) < 2:
            return ""
        
        original = team_name.strip()
        full_text_lower = original.lower()
        
        # Step 1: Immediate rejection for problematic patterns
        if cls._should_reject_immediately(full_text_lower, original):
            return ""
        
        # Step 2: Clean the team name
        cleaned = cls._apply_cleaning_patterns(team_name)
        
        # Step 3: Final validation
        if not cleaned or len(cleaned.strip()) < 2:
            return ""
        
        return cleaned.strip()
    
    @classmethod
    def _should_reject_immediately(cls, full_text_lower: str, original: str) -> bool:
        """Check if team name should be rejected immediately"""
        
        # Check incomplete match patterns
        for pattern in cls.INCOMPLETE_MATCH_PATTERNS:
            if pattern.search(full_text_lower):
                return True
        
        # Check qualification patterns
        for pattern in cls.QUALIFICATION_PATTERNS:
            if pattern.search(full_text_lower):
                return True
        
        # Check nationality-heavy text
        country_matches = sum(1 for pattern in cls.NATIONALITY_PATTERNS 
                            if pattern.search(full_text_lower))
        
        if country_matches > 0:
            word_count = len(original.split())
            # Reject if multiple countries or single country with multiple words
            if country_matches > 1 or (country_matches == 1 and word_count > 2):
                return True
            
            # Check bracket + country combinations
            if re.search(r'\[[A-Z]{2,4}\].*\b(usa|china|korea|japan|brazil|emea|apac)\b', 
                        full_text_lower):
                return True
            
            # Check for redundant country information
            bracket_match = re.search(r'\[([A-Z]{2,4})\]', original)
            if bracket_match:
                bracket_code = bracket_match.group(1).lower()
                if bracket_code in cls.COUNTRY_CODE_MAP:
                    country_name = cls.COUNTRY_CODE_MAP[bracket_code]
                    if country_name in full_text_lower or bracket_code in full_text_lower:
                        return True
        
        return False
    
    @classmethod
    def _apply_cleaning_patterns(cls, team_name: str) -> str:
        """Apply cleaning patterns to team name"""
        
        # Remove emojis first
        team_name = ''.join(char for char in team_name 
                          if unicodedata.category(char) not in ['So', 'Sc', 'Sk', 'Sm'])
        
        # Apply regex cleaning patterns
        for pattern, replacement in cls.CLEANING_PATTERNS:
            team_name = pattern.sub(replacement, team_name)
        
        return team_name.strip()
    
    @classmethod
    @lru_cache(maxsize=500)
    def is_valid_team_name(cls, team_name: str) -> bool:
        """
        Validate if a team name is legitimate.
        
        Args:
            team_name: Cleaned team name to validate
            
        Returns:
            True if valid VCT team name, False otherwise
        """
        if not team_name or len(team_name.strip()) < 2:
            return False
        
        team_lower = team_name.lower().strip()
        
        # Check explicitly invalid terms first
        if team_lower in cls.INVALID_TERMS:
            return False
            
        # Additional checks for problematic patterns
        # Reject names ending with digits
        if re.search(r'\d+$', team_name.strip()):
            return False
            
        # Reject names with country concatenations
        country_concat_patterns = [
            r'\b(sentinels|paper rex|bilibili gaming|edward gaming|g2 esports|team liquid)(united states|china|korea|japan|brazil)\b',
            r'\b\w+(usa|china|korea|japan|brazil|emea|apac)$'
        ]
        
        for pattern in country_concat_patterns:
            if re.search(pattern, team_lower, re.IGNORECASE):
                return False
                
        # Reject names starting with stage/date information
        if re.search(r'^(elimination|decider|sep|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug)\s+', team_lower):
            return False
        
        # Direct check for known teams
        if team_lower in cls.KNOWN_VCT_TEAMS:
            return True
        
        # Check letter count and composition
        letter_count = len(re.findall(r'[a-zA-Z]', team_name))
        if letter_count < 2:
            return False
        
        # Length constraints
        if len(team_name) > 30:
            return False
        
        # Pattern validation for unknown teams
        if len(team_name) >= 3 and letter_count >= 3:
            # Check for reasonable letter-to-total ratio
            letter_ratio = letter_count / len(team_name)
            if letter_ratio >= 0.6:
                return True
        
        return False


class VLRScraperOptimized:
    """
    Optimized VLR.gg tournament scraper with enhanced performance and reliability.
    """
    
    BASE_URL = "https://www.vlr.gg"
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.3
    
    def __init__(self, enable_caching: bool = True):
        """
        Initialize the VLR scraper.
        
        Args:
            enable_caching: Enable response caching for better performance
        """
        self.enable_caching = enable_caching
        self.session = self._create_session()
        self.team_processor = TeamNameProcessor()
        
        logger.info("VLR Scraper initialized successfully")
    
    def _create_session(self) -> requests.Session:
        """Create optimized requests session with retry strategy"""
        session = requests.Session()
        
        # Set headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=self.BACKOFF_FACTOR,
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10, pool_block=True)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get_tournament_info(self, event_id: str) -> Optional[TournamentData]:
        """
        Get tournament information from event ID.
        
        Args:
            event_id: VLR event ID
            
        Returns:
            TournamentData object or None if failed
        """
        url = f"{self.BASE_URL}/event/{event_id}"
        
        try:
            logger.info(f"Fetching tournament info for event {event_id}")
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract tournament information
            name = self._extract_tournament_name(soup, event_id)
            dates = self._extract_tournament_dates(soup)
            prize = self._extract_prize_pool(soup)
            location = self._extract_location(soup)
            
            tournament_data = TournamentData(
                name=name,
                event_id=event_id,
                dates=dates,
                prize=prize,
                location=location,
                url=url
            )
            
            logger.info(f"Successfully retrieved tournament: {name}")
            return tournament_data
            
        except Exception as e:
            logger.error(f"Error fetching tournament info for {event_id}: {e}")
            return None
    
    def get_upcoming_matches(self, event_id: str) -> List[MatchData]:
        """
        Get upcoming matches for a tournament.
        
        Args:
            event_id: VLR event ID
            
        Returns:
            List of MatchData objects
        """
        url = f"{self.BASE_URL}/event/{event_id}"
        
        try:
            logger.info(f"Fetching matches for event {event_id}")
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            matches = self._extract_matches(soup)
            
            # Filter and validate matches
            valid_matches = self._filter_valid_matches(matches)
            
            logger.info(f"Found {len(valid_matches)} valid matches for event {event_id}")
            return valid_matches
            
        except Exception as e:
            logger.error(f"Error fetching matches for {event_id}: {e}")
            return []
    
    def _extract_tournament_name(self, soup: BeautifulSoup, event_id: str) -> str:
        """Extract tournament name from soup"""
        title_element = soup.find('h1', class_='wf-title')
        if title_element:
            return title_element.get_text(strip=True)
        
        # Fallback
        title_element = soup.find('title')
        if title_element:
            title = title_element.get_text(strip=True)
            if title and title != 'VLR.gg':
                return title.replace(' - VLR.gg', '')
        
        return f"Tournament {event_id}"
    
    def _extract_tournament_dates(self, soup: BeautifulSoup) -> str:
        """Extract tournament dates from soup"""
        date_selectors = [
            'div.event-desc-item',
            'div.event-desc',
            'div.event-date',
            'span.date'
        ]
        
        for selector in date_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                # Look for date patterns
                if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}/\d{1,2}|\d{4})', text, re.I):
                    return text
        
        return ""
    
    def _extract_prize_pool(self, soup: BeautifulSoup) -> str:
        """Extract prize pool from soup"""
        prize_elements = soup.find_all('div', class_='event-desc-item')
        for element in prize_elements:
            text = element.get_text(strip=True)
            if '$' in text or 'USD' in text:
                return text
        return ""
    
    def _extract_location(self, soup: BeautifulSoup) -> str:
        """Extract location from soup"""
        location_keywords = ['arena', 'venue', 'location', 'city', 'country']
        prize_elements = soup.find_all('div', class_='event-desc-item')
        
        for element in prize_elements:
            text = element.get_text(strip=True)
            if any(keyword in text.lower() for keyword in location_keywords):
                # Skip if it looks like a date or prize
                if not re.search(r'(\$|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', text, re.I):
                    return text
        return ""
    
    def _extract_matches(self, soup: BeautifulSoup) -> List[MatchData]:
        """Extract matches from tournament page"""
        matches = []
        
        # Multiple strategies for finding matches
        strategies = [
            self._extract_matches_from_elements,
            self._extract_matches_from_links,
            self._extract_matches_from_text
        ]
        
        for strategy in strategies:
            try:
                strategy_matches = strategy(soup)
                if strategy_matches:
                    matches.extend(strategy_matches)
                    logger.debug(f"Strategy {strategy.__name__} found {len(strategy_matches)} matches")
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        return matches
    
    def _extract_matches_from_elements(self, soup: BeautifulSoup) -> List[MatchData]:
        """Extract matches from match elements"""
        matches = []
        match_selectors = [
            'div[class*="match"]',
            'a[class*="match"]',
            'div[class*="wf-card"]',
            'div[class*="game"]'
        ]
        
        for selector in match_selectors:
            elements = soup.select(selector)
            for element in elements:
                match_data = self._parse_match_element(element)
                if match_data:
                    matches.append(match_data)
        
        return matches
    
    def _extract_matches_from_links(self, soup: BeautifulSoup) -> List[MatchData]:
        """Extract matches from match links"""
        matches = []
        match_links = soup.find_all('a', href=re.compile(r'/match/'))
        
        for link in match_links:
            text = link.get_text(strip=True)
            teams = self._extract_teams_from_text(text)
            
            if len(teams) >= 2:
                team1_clean = self.team_processor.clean_team_name(teams[0])
                team2_clean = self.team_processor.clean_team_name(teams[1])
                
                if (team1_clean and team2_clean and
                    self.team_processor.is_valid_team_name(team1_clean) and
                    self.team_processor.is_valid_team_name(team2_clean)):
                    
                    matches.append(MatchData(
                        team1=team1_clean,
                        team2=team2_clean,
                        match_url=link.get('href', ''),
                        confidence=0.8
                    ))
        
        return matches
    
    def _extract_matches_from_text(self, soup: BeautifulSoup) -> List[MatchData]:
        """Extract matches using text pattern matching"""
        matches = []
        page_text = soup.get_text()
        
        # Pattern matching for team vs team
        patterns = [
            re.compile(r'([A-Za-z0-9\s]+)\s*–\s*([A-Za-z0-9\s]+)', re.MULTILINE),
            re.compile(r'([A-Za-z0-9\s]+)\s*vs\s*([A-Za-z0-9\s]+)', re.MULTILINE | re.IGNORECASE),
            re.compile(r'([A-Za-z0-9\s]+)\s*-\s*([A-Za-z0-9\s]+)', re.MULTILINE)
        ]
        
        for pattern in patterns:
            pattern_matches = pattern.findall(page_text)
            
            for team1_raw, team2_raw in pattern_matches:
                team1_clean = self.team_processor.clean_team_name(team1_raw.strip())
                team2_clean = self.team_processor.clean_team_name(team2_raw.strip())
                
                if (team1_clean and team2_clean and
                    self.team_processor.is_valid_team_name(team1_clean) and
                    self.team_processor.is_valid_team_name(team2_clean) and
                    team1_clean != team2_clean):
                    
                    # Check for duplicates
                    if not any(m.team1 == team1_clean and m.team2 == team2_clean for m in matches):
                        matches.append(MatchData(
                            team1=team1_clean,
                            team2=team2_clean,
                            confidence=0.6
                        ))
        
        return matches
    
    def _parse_match_element(self, element) -> Optional[MatchData]:
        """Parse individual match element"""
        try:
            # Extract team names
            team_elements = element.find_all(['div', 'span'], 
                                           class_=lambda x: x and 'team' in x.lower())
            
            teams = []
            if len(team_elements) >= 2:
                teams = [el.get_text(strip=True) for el in team_elements[:2]]
            else:
                # Try extracting from text
                text = element.get_text(strip=True)
                teams = self._extract_teams_from_text(text)
            
            if len(teams) < 2:
                return None
            
            # Clean and validate teams
            team1_clean = self.team_processor.clean_team_name(teams[0])
            team2_clean = self.team_processor.clean_team_name(teams[1])
            
            if not (team1_clean and team2_clean and
                   self.team_processor.is_valid_team_name(team1_clean) and
                   self.team_processor.is_valid_team_name(team2_clean) and
                   team1_clean != team2_clean):
                return None
            
            # Extract additional information
            match_time = self._extract_match_time(element)
            match_stage = self._extract_match_stage(element)
            match_url = element.get('href', '') if element.name == 'a' else ''
            
            return MatchData(
                team1=team1_clean,
                team2=team2_clean,
                match_time=match_time,
                match_stage=match_stage,
                match_url=match_url,
                confidence=0.9
            )
            
        except Exception as e:
            logger.debug(f"Error parsing match element: {e}")
            return None
    
    def _extract_teams_from_text(self, text: str) -> List[str]:
        """Extract team names from text"""
        # Split on common separators
        separators = ['–', 'vs', '-', 'v', '|']
        
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                if len(parts) >= 2:
                    return [part.strip() for part in parts[:2]]
        
        return []
    
    def _extract_match_time(self, element) -> str:
        """Extract match time from element"""
        time_selectors = [
            '[class*="time"]',
            '[class*="date"]', 
            '.match-time',
            '.time'
        ]
        
        for selector in time_selectors:
            time_element = element.select_one(selector)
            if time_element:
                return time_element.get_text(strip=True)
        
        return ""
    
    def _extract_match_stage(self, element) -> str:
        """Extract match stage from element"""
        stage_selectors = [
            '[class*="stage"]',
            '[class*="round"]',
            '.match-stage',
            '.stage'
        ]
        
        for selector in stage_selectors:
            stage_element = element.select_one(selector)
            if stage_element:
                return stage_element.get_text(strip=True)
        
        return ""
    
    def _filter_valid_matches(self, matches: List[MatchData]) -> List[MatchData]:
        """Filter matches to remove duplicates and invalid entries"""
        seen_matches = set()
        valid_matches = []
        
        for match in matches:
            # Create a normalized match key for deduplication
            match_key = tuple(sorted([match.team1.lower(), match.team2.lower()]))
            
            if match_key not in seen_matches:
                seen_matches.add(match_key)
                valid_matches.append(match)
        
        # Sort by confidence score (highest first)
        valid_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to reasonable number
        return valid_matches[:15]
    
    def predict_matches(self, matches: List[MatchData], predictor) -> List[Dict]:
        """
        Add predictions to match data using the ML predictor.
        
        Args:
            matches: List of MatchData objects
            predictor: ML predictor instance
            
        Returns:
            List of match dictionaries with predictions
        """
        predicted_matches = []
        
        for match in matches:
            try:
                logger.debug(f"Predicting: {match.team1} vs {match.team2}")
                
                prediction = predictor.predict_match(match.team1, match.team2)
                
                match_dict = {
                    'team1': match.team1,
                    'team2': match.team2,
                    'match_time': match.match_time,
                    'match_stage': match.match_stage,
                    'status': match.status,
                    'match_url': match.match_url,
                    'scraper_confidence': match.confidence
                }
                
                if prediction:
                    match_dict.update({
                        'predicted_winner': prediction['predicted_winner'],
                        'confidence': prediction['confidence'],
                        'team1_probability': prediction['team1_probability'],
                        'team2_probability': prediction['team2_probability'],
                        'confidence_level': prediction.get('confidence_level', 'Medium')
                    })
                else:
                    match_dict['prediction_error'] = 'Failed to predict'
                
                predicted_matches.append(match_dict)
                
            except Exception as e:
                logger.error(f"Error predicting match {match.team1} vs {match.team2}: {e}")
                match_dict['prediction_error'] = str(e)
                predicted_matches.append(match_dict)
        
        return predicted_matches
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        logger.info("VLR Scraper closed successfully")


# Backward compatibility alias
VLRScraper = VLRScraperOptimized
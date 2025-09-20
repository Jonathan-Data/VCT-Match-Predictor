#!/usr/bin/env python3
"""
VLR.gg Tournament Scraper
Fetches upcoming matches from VLR.gg tournament pages
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

class VLRScraper:
    """Scraper for VLR.gg tournament data"""
    
    def __init__(self):
        self.base_url = "https://www.vlr.gg"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_tournament_info(self, event_id: str) -> Optional[Dict]:
        """Get tournament information from event ID"""
        url = f"{self.base_url}/event/{event_id}"
        
        try:
            print(f"🔍 Fetching tournament info from: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract tournament name
            title_element = soup.find('h1', class_='wf-title')
            tournament_name = title_element.get_text(strip=True) if title_element else f"Tournament {event_id}"
            
            # Extract dates
            dates_text = ""
            dates_element = soup.find('div', class_='event-desc-item')
            if dates_element:
                dates_text = dates_element.get_text(strip=True)
            
            # Extract prize pool
            prize_text = ""
            prize_elements = soup.find_all('div', class_='event-desc-item')
            for element in prize_elements:
                text = element.get_text(strip=True)
                if '$' in text or 'USD' in text:
                    prize_text = text
                    break
            
            # Extract location
            location_text = ""
            for element in prize_elements:
                text = element.get_text(strip=True)
                if any(keyword in text.lower() for keyword in ['arena', 'venue', 'location', ',']):
                    if '$' not in text and 'sep' not in text.lower() and 'oct' not in text.lower():
                        location_text = text
                        break
            
            tournament_info = {
                'name': tournament_name,
                'event_id': event_id,
                'dates': dates_text,
                'prize': prize_text,
                'location': location_text,
                'url': url
            }
            
            print(f"✅ Tournament: {tournament_name}")
            return tournament_info
            
        except Exception as e:
            print(f"❌ Error fetching tournament info: {e}")
            return None
    
    def get_upcoming_matches(self, event_id: str) -> List[Dict]:
        """Get upcoming matches for a tournament"""
        url = f"{self.base_url}/event/{event_id}"
        
        try:
            print(f"🔍 Fetching upcoming matches from: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            upcoming_matches = []
            
            # Find match elements - VLR uses different selectors for matches
            match_elements = soup.find_all(['div', 'a'], class_=lambda x: x and ('match-item' in x or 'wf-card' in x))
            
            if not match_elements:
                # Try alternative selectors
                match_elements = soup.find_all('a', href=re.compile(r'/match/'))
            
            print(f"📊 Found {len(match_elements)} potential match elements")
            
            for match_element in match_elements:
                try:
                    match_data = self._parse_match_element(match_element)
                    if match_data and match_data.get('status') == 'upcoming':
                        upcoming_matches.append(match_data)
                except Exception as e:
                    continue
            
            # If no matches found through normal parsing, try text-based approach
            if not upcoming_matches:
                upcoming_matches = self._parse_matches_from_text(soup)
            
            print(f"✅ Found {len(upcoming_matches)} upcoming matches")
            return upcoming_matches
            
        except Exception as e:
            print(f"❌ Error fetching upcoming matches: {e}")
            return []
    
    def _parse_match_element(self, element) -> Optional[Dict]:
        """Parse individual match element"""
        try:
            # Extract team names
            team_elements = element.find_all(['div', 'span'], class_=lambda x: x and 'team' in x.lower())
            
            if len(team_elements) < 2:
                # Try finding team names in text
                text = element.get_text(strip=True)
                teams = self._extract_teams_from_text(text)
                if len(teams) < 2:
                    return None
                team1, team2 = teams[0], teams[1]
            else:
                team1 = team_elements[0].get_text(strip=True)
                team2 = team_elements[1].get_text(strip=True)
            
            # Clean team names
            team1 = self._clean_team_name(team1)
            team2 = self._clean_team_name(team2)
            
            # Skip if teams are not real team names
            if not self._is_valid_team_name(team1) or not self._is_valid_team_name(team2):
                return None
            
            # Additional validation: ensure both teams are different and substantial
            if team1 == team2 or len(team1) < 3 or len(team2) < 3:
                return None
                
            # Final check: ensure no qualification/bracket language remains
            combined_text = f"{team1} {team2}".lower()
            forbidden_terms = ['qualified', 'eliminated', 'tbd', 'winner', 'loser', 'vs', 'match']
            if any(term in combined_text for term in forbidden_terms):
                return None
            
            # Extract match time/status
            time_element = element.find(['div', 'span'], class_=lambda x: x and 'time' in x.lower())
            match_time = ""
            if time_element:
                match_time = time_element.get_text(strip=True)
            
            # Extract match stage/round
            stage_element = element.find(['div', 'span'], class_=lambda x: x and ('stage' in x.lower() or 'round' in x.lower()))
            match_stage = ""
            if stage_element:
                match_stage = stage_element.get_text(strip=True)
            
            # Determine if match is upcoming
            status = 'upcoming'
            if any(keyword in element.get_text().lower() for keyword in ['live', 'completed', 'finished']):
                status = 'live' if 'live' in element.get_text().lower() else 'completed'
            
            return {
                'team1': team1,
                'team2': team2,
                'match_time': match_time,
                'match_stage': match_stage,
                'status': status,
                'match_url': element.get('href') if element.get('href') else ''
            }
            
        except Exception as e:
            return None
    
    def _parse_matches_from_text(self, soup) -> List[Dict]:
        """Fallback method to parse matches from page text using known team names"""
        matches = []
        
        try:
            # Known VCT team names for better matching
            known_teams = [
                'Sentinels', 'Paper Rex', 'Fnatic', 'DRX', 'Team Heretics', 'Edward Gaming', 
                'G2 Esports', 'NRG', 'T1', 'Team Liquid', 'GIANTX', 'FUT Esports',
                'Bilibili Gaming', 'Rex Regum Qeon', 'MIBR', 'Cloud9', 'Gen.G',
                'Trace Esports', 'Dragon Ranger Gaming', 'Xi Lai Gaming',
                'LOUD', '100 Thieves', 'Evil Geniuses', 'Leviatán', 'KRU Esports'
            ]
            
            # Get all text content
            page_text = soup.get_text()
            
            # Look for team vs team patterns with known teams
            for i, team1 in enumerate(known_teams):
                for team2 in known_teams[i+1:]:  # Avoid duplicate combinations
                    # Check for various patterns
                    patterns = [
                        rf'{re.escape(team1)}\s*[–-]\s*{re.escape(team2)}',
                        rf'{re.escape(team2)}\s*[–-]\s*{re.escape(team1)}',
                        rf'{re.escape(team1)}\s*vs\s*{re.escape(team2)}',
                        rf'{re.escape(team2)}\s*vs\s*{re.escape(team1)}'
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, page_text, re.IGNORECASE):
                            # Determine the correct order (team1 comes first in the text)
                            team1_pos = page_text.lower().find(team1.lower())
                            team2_pos = page_text.lower().find(team2.lower())
                            
                            if team1_pos < team2_pos:
                                first_team, second_team = team1, team2
                            else:
                                first_team, second_team = team2, team1
                            
                            # Check if we already have this match
                            match_exists = any(
                                (m['team1'] == first_team and m['team2'] == second_team) or
                                (m['team1'] == second_team and m['team2'] == first_team)
                                for m in matches
                            )
                            
                            if not match_exists:
                                matches.append({
                                    'team1': first_team,
                                    'team2': second_team,
                                    'match_time': 'TBD',
                                    'match_stage': 'Tournament Match',
                                    'status': 'upcoming',
                                    'match_url': ''
                                })
                                break  # Found a match for this team combination
            
            # If still no matches, try the original text-based approach with better cleaning
            if not matches:
                matches = self._parse_with_text_patterns(page_text)
            
            # Limit to reasonable number of matches
            return matches[:10]
            
        except Exception as e:
            print(f"❌ Error parsing matches from text: {e}")
            return []
    
    def _parse_with_text_patterns(self, page_text: str) -> List[Dict]:
        """Original text pattern parsing as fallback"""
        matches = []
        
        # Look for patterns that indicate upcoming matches
        match_patterns = [
            r'([A-Za-z0-9\s]+)\s*–\s*([A-Za-z0-9\s]+)',
            r'([A-Za-z0-9\s]+)\s*vs\s*([A-Za-z0-9\s]+)',
            r'([A-Za-z0-9\s]+)\s*-\s*([A-Za-z0-9\s]+)'
        ]
        
        for pattern in match_patterns:
            pattern_matches = re.findall(pattern, page_text, re.MULTILINE | re.IGNORECASE)
            
            for team1, team2 in pattern_matches:
                team1 = self._clean_team_name(team1.strip())
                team2 = self._clean_team_name(team2.strip())
                
                if self._is_valid_team_name(team1) and self._is_valid_team_name(team2):
                    if team1 != team2:  # Avoid duplicates
                        match_exists = any(m['team1'] == team1 and m['team2'] == team2 for m in matches)
                        if not match_exists:
                            matches.append({
                                'team1': team1,
                                'team2': team2,
                                'match_time': 'TBD',
                                'match_stage': 'Tournament Match',
                                'status': 'upcoming',
                                'match_url': ''
                            })
        
        return matches
    
    def _extract_teams_from_text(self, text: str) -> List[str]:
        """Extract team names from text"""
        # Split on common separators
        for separator in ['–', 'vs', '-', 'v']:
            if separator in text:
                parts = text.split(separator)
                if len(parts) >= 2:
                    return [part.strip() for part in parts[:2]]
        return []
    
    def _clean_team_name(self, team_name: str) -> str:
        """Clean and standardize team name - AGGRESSIVE version"""
        if not team_name:
            return ""
        
        original = team_name
        
        # STEP 1: Immediately reject problematic patterns from ORIGINAL text
        full_text_lower = original.lower()
        
        # Check for incomplete match patterns (has "vs" but missing opponent)
        incomplete_match_patterns = [
            r'^[^\s]+\s+vs\s*$',      # "Team vs" (ends with vs)
            r'^\s*vs\s+[^\s]+$',      # "vs Team" (starts with vs)
            r'^\s*vs\s+',            # "vs Team" (starts with vs - more general)
            r'^[^\s]+\s+v\s*$',       # "Team v" (ends with v)
            r'^\s*v\s+[^\s]+$',       # "v Team" (starts with v)
        ]
        
        for pattern in incomplete_match_patterns:
            if re.search(pattern, full_text_lower.strip()):
                return ""  # Reject incomplete matches
        
        # Check for nationality-heavy text (team name + multiple country indicators)
        nationality_heavy_patterns = [
            r'\b(united states|usa)\b',
            r'\bchina\b', r'\bkorea\b', r'\bjapan\b', r'\bbrazil\b',
            r'\bsingapore\b', r'\bunited kingdom\b', r'\bgermany\b',
            r'\bthailand\b', r'\bphilippines\b', r'\bindonesia\b',
            r'\bmalaysia\b', r'\bvietnam\b', r'\baustralia\b',
            r'\bemea\b', r'\bapac\b', r'\beurope\b', r'\basia\b',
            r'\bna\b'  # North America
        ]
        
        # If original text contains team name + country, it's likely scraped metadata
        country_match_count = sum(1 for pattern in nationality_heavy_patterns 
                                if re.search(pattern, full_text_lower))
        if country_match_count > 0:
            # Always reject if contains full country names (not just codes)
            full_country_patterns = [
                r'\bunited states\b', r'\bunited kingdom\b', r'\bsingapore\b',
                r'\bthailand\b', r'\bphilippines\b', r'\bindonesia\b',
                r'\bmalaysia\b', r'\bvietnam\b', r'\baustralia\b'
            ]
            
            has_full_country = any(re.search(pattern, full_text_lower) 
                                 for pattern in full_country_patterns)
            
            # Reject if has full country name OR multiple country indicators OR single indicator with multiple words
            word_count = len(original.split())
            if has_full_country or country_match_count > 1 or (country_match_count == 1 and word_count > 2):
                return ""  # Reject nationality-heavy text
            
            # Also reject bracket format with country names like "[US] Team USA"
            if re.search(r'\[[A-Z]{2,4}\].*\b(usa|china|korea|japan|brazil|emea|apac)\b', full_text_lower):
                return ""  # Reject bracket + country name combinations
            
            # Special case: reject if it's bracket format + same country mentioned again
            # like "[US] Cloud9 USA" or "[BR] LOUD Brazil" 
            bracket_match = re.search(r'\[([A-Z]{2,4})\]', original)
            if bracket_match:
                bracket_code = bracket_match.group(1).lower()
                # Map common codes to country names
                code_to_country = {
                    'us': 'usa', 'usa': 'usa', 'br': 'brazil', 'kr': 'korea',
                    'jp': 'japan', 'sg': 'singapore', 'gb': 'united kingdom',
                    'uk': 'united kingdom', 'de': 'germany', 'fr': 'france'
                }
                
                if bracket_code in code_to_country:
                    country_name = code_to_country[bracket_code]
                    if country_name in full_text_lower or bracket_code in full_text_lower:
                        return ""  # Reject redundant country information
        
        # Check for exact qualification patterns
        immediate_reject_patterns = [
            r'\bqualified\b', r'\beliminated\b', r'\bbye\b', r'\btbd\b',
            r'\bto be determined\b', r'\bwinner of\b', r'\bloser of\b',
            r'\bupper bracket\b', r'\blower bracket\b',
            r'\bquarterfinal\b', r'\bsemifinal\b', r'\bgrand final\b'
        ]
        
        for pattern in immediate_reject_patterns:
            if re.search(pattern, full_text_lower):
                return ""  # Immediately reject
        
        # STEP 2: Remove emojis aggressively
        import unicodedata
        # Remove all symbols, other symbols, and currency symbols
        team_name = ''.join(char for char in team_name if unicodedata.category(char) not in ['So', 'Sc', 'Sk', 'Sm'])
        
        # STEP 3: Remove nationality indicators AGGRESSIVELY
        # Remove bracket formats
        team_name = re.sub(r'\[[A-Z]{2,4}\]\s*', '', team_name)  # [US], [USA] format
        team_name = re.sub(r'\([A-Z]{2,4}\)\s*', '', team_name)  # (US), (USA) format
        team_name = re.sub(r'^[A-Z]{2,4}\s+', '', team_name)     # US Team, USA Team format
        
        # Remove country names more aggressively
        all_countries = [
            # Major countries
            'United States', 'United Kingdom', 'China', 'Korea', 'South Korea', 'Japan', 'Brazil', 'Argentina', 'Chile', 'Peru',
            'Turkey', 'Russia', 'Germany', 'France', 'Spain', 'Sweden', 'Finland', 'Denmark', 'Norway', 'Netherlands',
            'Thailand', 'Philippines', 'Indonesia', 'Singapore', 'Malaysia', 'Vietnam', 'Australia', 'New Zealand', 'India',
            # Common abbreviations and variations
            'USA', 'UK', 'KR', 'JP', 'BR', 'TR', 'RU', 'DE', 'FR', 'ES', 'SE', 'FI', 'DK', 'NO', 'NL',
            'TH', 'PH', 'ID', 'SG', 'MY', 'VN', 'AU', 'NZ', 'IN', 'NA', 'EU', 'APAC', 'EMEA'
        ]
        
        for country in all_countries:
            # Remove anywhere in the string, not just at the end
            team_name = re.sub(rf'\b{re.escape(country)}\b', '', team_name, flags=re.IGNORECASE)
            team_name = re.sub(rf'\s+{re.escape(country)}\s+', ' ', team_name, flags=re.IGNORECASE)
            team_name = re.sub(rf'^{re.escape(country)}\s+', '', team_name, flags=re.IGNORECASE)
            team_name = re.sub(rf'\s+{re.escape(country)}\s*$', '', team_name, flags=re.IGNORECASE)
        
        # STEP 4: Remove times, scores, and other noise aggressively
        team_name = re.sub(r'\s*\d{1,2}:\d{2}\s*(AM|PM)?', '', team_name, flags=re.IGNORECASE)  # Remove times
        team_name = re.sub(r'\s*\d+[-–—]\d+', '', team_name)  # Remove scores like 2-1
        team_name = re.sub(r'\s*\(\d+[-–—]\d+\)', '', team_name)  # Remove (2-1) format
        team_name = re.sub(r'\s*\d{1,2}/\d{1,2}', '', team_name)  # Remove dates like 10/15
        team_name = re.sub(r'\s*#\d+', '', team_name)  # Remove seed numbers
        
        # STEP 5: Remove time patterns anywhere in the string
        team_name = re.sub(r'\s*\d+d\s*\d*h?', '', team_name)  # "1d 4h" or "1d"
        team_name = re.sub(r'\s*\d+h\s*\d*m?', '', team_name)  # "1h 7m" or "1h"
        team_name = re.sub(r'\s*\d+m', '', team_name)  # "30m"
        
        # STEP 6: Remove qualification and bracket indicators carefully
        # Remove these patterns but preserve the core team name
        removal_patterns = [
            r'\s*\(qualified\)\s*', r'\s*\(eliminated\)\s*', r'\s*\(advanced\)\s*',
            r'\s*qualified\s*$', r'\s*eliminated\s*$', r'\s*advanced\s*$',
            r'\s*\bbye\b\s*', r'\s*\btbd\b\s*', r'\s*\bvs\b\s*',
            r'\s*winner\s+of\s*', r'\s*loser\s+of\s*',
            r'\s*\bw\d+\b\s*', r'\s*\bl\d+\b\s*',  # W1, L2 etc
            r'\s*match\s+\d+\s*', r'\s*game\s+\d+\s*',
            r'\s*upper\s+bracket\s*', r'\s*lower\s+bracket\s*',
            r'\s*quarterfinal\s*', r'\s*semifinal\s*', r'\s*grand\s+final\s*',
            r'\s*group\s+stage\s*', r'\s*playoff\s*', r'\s*elimination\s*'
        ]
        
        for pattern in removal_patterns:
            team_name = re.sub(pattern, ' ', team_name, flags=re.IGNORECASE)
        
        # STEP 7: Clean up punctuation and whitespace (but preserve dots for Gen.G etc)
        team_name = re.sub(r'[•·‧‹›\-–—]', ' ', team_name)  # Replace bullets/dashes with spaces
        team_name = re.sub(r'[\s\t\r\n]+', ' ', team_name)  # Normalize all whitespace
        team_name = team_name.strip()
        
        # STEP 8: Final validation - reject if still contains problematic content
        if not team_name or len(team_name) < 2:
            return ""
        
        # Known valid short team names should bypass strict checks
        known_short_teams = ['T1', 'C9', 'G2', 'DRX', 'NRG', 'LOUD', 'MIBR', 'Gen.G']
        if team_name in known_short_teams:
            return team_name
        
        # Check for remaining problematic patterns
        final_reject_patterns = [
            r'^\s*vs\s*$', r'^\s*v\s*$',  # Just "vs" or "v"
            r'^\s*\d+\s*$',  # Just numbers
            r'^\s*[A-Z]{2,4}\s*$',  # Just country codes like "US", "BR" (but not valid teams)
        ]
        
        # Don't apply country code pattern to known teams
        if team_name.lower() not in ['t1', 'c9', 'g2', 'drx', 'nrg', 'loud', 'mibr', 'gen.g']:
            for pattern in final_reject_patterns:
                if re.search(pattern, team_name, flags=re.IGNORECASE):
                    return ""
        
        # Must contain at least 2 letters (lowered from 3 for short team names)
        if len(re.findall(r'[a-zA-Z]', team_name)) < 2:
            return ""
            
        return team_name
    
    def _is_valid_team_name(self, team_name: str) -> bool:
        """Check if string is a valid team name - VERY STRICT version"""
        if not team_name or len(team_name.strip()) < 2:  # Minimum 2 characters (for T1, etc.)
            return False
        
        original_name = team_name
        team_name = team_name.lower().strip()
        
        # Direct check for specific valid teams that might fail other checks
        direct_valid_teams = [
            't1', 'drx', 'loud', 'nrg', '100 thieves', 'gen.g', 'mibr',
            'c9', 'cloud9', 'g2', 'g2 esports', 'team heretics', 'team liquid',
            'edward gaming', 'paper rex', 'sentinels', 'fnatic', 'prx', 'sen'
        ]
        
        if team_name in direct_valid_teams:
            return True
        
        # IMMEDIATE REJECTION for qualification/bracket terms
        instant_reject_terms = [
            'tbd', 'to be determined', 'winner', 'loser', 'elimination', 
            'qualified', 'eliminated', 'advanced', 'bye', 'seed', 'playoff',
            'quarterfinal', 'semifinal', 'grand final', 'final',
            'upper bracket', 'lower bracket', 'group stage',
            'match', 'game', 'vs', 'v', 'against',
            'w1', 'w2', 'w3', 'w4', 'l1', 'l2', 'l3', 'l4',
            'group', 'stage', 'decider', 'elimination',
        ]
        
        for term in instant_reject_terms:
            if term in team_name:
                return False
        
        # IMMEDIATE REJECTION for country names/codes
        country_terms = [
            'united states', 'usa', 'china', 'korea', 'south korea', 'kr',
            'japan', 'jp', 'brazil', 'br', 'argentina', 'chile', 'peru',
            'turkey', 'tr', 'russia', 'ru', 'germany', 'de', 'france', 'fr',
            'spain', 'es', 'sweden', 'se', 'finland', 'fi', 'denmark', 'dk',
            'thailand', 'th', 'philippines', 'ph', 'indonesia', 'id',
            'singapore', 'sg', 'malaysia', 'my', 'vietnam', 'vn',
            'australia', 'au', 'new zealand', 'nz', 'india', 'in',
            'na', 'eu', 'apac', 'emea', 'uk', 'netherlands', 'nl'
        ]
        
        # Reject if it's ONLY a country name
        if team_name in country_terms:
            return False
        
        # STRICT pattern matching for invalid formats
        invalid_patterns = [
            r'^\s*[a-z]{2,4}\s*$',  # Just country codes like "us", "br"
            r'^\s*\d+\s*$',         # Just numbers
            r'\d{1,2}:\d{2}',       # Time patterns
            r'\d+[-–—]\d+',       # Score patterns
            r'\d+d\s*\d*h',         # Time duration patterns
            r'\d+h\s*\d*m',         # Time duration patterns
            r'#\d+',                # Seed numbers
            r'\bmatch\s+\d+',       # Match numbers
            r'\bw\d+\b|\bl\d+\b',   # Winner/Loser brackets
            r'^\s*vs?\s*$',         # Just "vs" or "v"
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, team_name):
                return False
        
        # Must contain at least 3 letters
        letter_count = len(re.findall(r'[a-zA-Z]', team_name))
        if letter_count < 3:
            return False
        
        # Must not be too long 
        if len(original_name) > 30:  # Stricter length limit
            return False
        
        # Known VCT teams - expanded list (include abbreviations and variations)
        known_teams = [
            'sentinels', 'paper rex', 'fnatic', 'drx', 'team heretics', 'edward gaming',
            'g2 esports', 'nrg', 't1', 'team liquid', 'giantx', 'fut esports',
            'bilibili gaming', 'rex regum qeon', 'mibr', 'cloud9', 'gen.g',
            'trace esports', 'dragon ranger gaming', 'xi lai gaming',
            'loud', '100 thieves', 'evil geniuses', 'leviatan', 'kru esports',
            'prx', 'edg', 'ths', 'tl', 'blg', 'rrq', 'c9', 'geng', 'te', 'drg', 'xlg',
            'eg', 'lev', 'kru', 'sen', 'fnc', 'fut', 'gx',
            # Short team names that are valid
            'th', 'ge', 'fs', 'vs', 'ts', 'dk', 'on', 'sr', 'og'
        ]
        
        # If it matches a known team, it's valid (even if short)
        if team_name.lower() in known_teams:
            return True
            
        # Special case for very short but legitimate team names and special formats
        short_valid_teams = ['t1', 'c9', 'g2', 'ts', 'dk', 'og', 'th', 'ge', 'drx', 'nrg', 'loud', 'mibr', 'gen.g', 'geng']
        if team_name.lower() in short_valid_teams:
            return True
            
        # Handle special valid team formats
        special_valid_patterns = [
            r'^\d+\s+thieves$',     # "100 Thieves"
            r'^[a-z]+\s+gaming$',   # "Edward Gaming", etc.
            r'^team\s+[a-z]+$',     # "Team Liquid", "Team Heretics"
            r'^g2\s+esports$',      # "G2 Esports"
        ]
        
        for pattern in special_valid_patterns:
            if re.match(pattern, team_name, re.IGNORECASE):
                return True
        
        # For unknown teams, be strict but not overly so
        # Must be at least 4 characters and have good letter-to-total ratio
        if len(team_name) >= 4 and letter_count >= 3:
            # Allow numbers in team names (many teams have numbers)
            if re.search(r'\d', team_name):
                # Check for reasonable letter-to-total ratio
                letter_ratio = len(re.findall(r'[a-zA-Z]', team_name)) / len(team_name)
                if letter_ratio >= 0.3:  # At least 30% letters
                    return True
                else:
                    return False
            
            # For teams without numbers, check letter ratio
            letter_ratio = len(re.findall(r'[a-zA-Z]', team_name)) / len(team_name)
            if letter_ratio >= 0.7:  # At least 70% letters for non-numeric teams
                return True
        
        return False
    
    def predict_matches(self, matches: List[Dict], predictor) -> List[Dict]:
        """Add predictions to match data"""
        predicted_matches = []
        
        for match in matches:
            try:
                team1 = match['team1']
                team2 = match['team2']
                
                print(f"🔮 Predicting: {team1} vs {team2}")
                
                # Get prediction from the ML model
                prediction = predictor.predict_match(team1, team2)
                
                if prediction:
                    match_with_prediction = match.copy()
                    match_with_prediction.update({
                        'predicted_winner': prediction['predicted_winner'],
                        'confidence': prediction['confidence'],
                        'team1_probability': prediction['team1_probability'],
                        'team2_probability': prediction['team2_probability'],
                        'confidence_level': prediction.get('confidence_level', 'Medium')
                    })
                    predicted_matches.append(match_with_prediction)
                else:
                    match['prediction_error'] = 'Failed to predict'
                    predicted_matches.append(match)
                    
            except Exception as e:
                match['prediction_error'] = str(e)
                predicted_matches.append(match)
        
        return predicted_matches

def test_scraper():
    """Test the VLR scraper"""
    scraper = VLRScraper()
    
    # Test with Champions 2025
    event_id = "2283"
    
    # Get tournament info
    tournament_info = scraper.get_tournament_info(event_id)
    if tournament_info:
        print(f"\n🏆 Tournament: {tournament_info['name']}")
        print(f"📅 Dates: {tournament_info['dates']}")
        print(f"💰 Prize: {tournament_info['prize']}")
        print(f"📍 Location: {tournament_info['location']}")
    
    # Get upcoming matches
    matches = scraper.get_upcoming_matches(event_id)
    
    if matches:
        print(f"\n📋 Upcoming Matches ({len(matches)}):")
        for i, match in enumerate(matches, 1):
            print(f"{i:2}. {match['team1']} vs {match['team2']}")
            if match['match_time']:
                print(f"    ⏰ Time: {match['match_time']}")
            if match['match_stage']:
                print(f"    🎯 Stage: {match['match_stage']}")
    else:
        print("\n❌ No upcoming matches found")

if __name__ == "__main__":
    test_scraper()
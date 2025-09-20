#!/usr/bin/env python3
"""
Test script for the improved VLR scraper
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vlr_scraper import VLRScraper

def test_team_name_cleaning():
    """Test the improved team name cleaning"""
    scraper = VLRScraper()
    
    print("🧪 Testing team name cleaning...")
    
    # Test cases with various problematic inputs
    test_cases = [
        ("🇺🇸 Sentinels", "Sentinels"),
        ("[US] Cloud9", "Cloud9"),
        ("(BR) LOUD 2-1", "LOUD"),
        ("Paper Rex • 10:30 AM", "Paper Rex"),
        ("Fnatic (qualified)", "Fnatic"),
        ("Winner of Match 1", ""),
        ("TBD vs Team Heretics", ""),
        ("G2 Esports #1", "G2 Esports"),
        ("DRX 1d 4h", "DRX"),
        ("Team Liquid Thailand", "Team Liquid"),
        ("NRG • United States", "NRG"),
        ("Bye", ""),
        ("Qualified", ""),
        ("W1", ""),
        ("L2", ""),
        ("EDG Edward Gaming", "EDG Edward Gaming"),
        ("100 Thieves 🇺🇸", "100 Thieves"),
    ]
    
    print("\n📝 Team name cleaning results:")
    for input_name, expected in test_cases:
        cleaned = scraper._clean_team_name(input_name)
        status = "✅" if cleaned == expected else "❌"
        print(f"{status} '{input_name}' → '{cleaned}' (expected: '{expected}')")
    
    print()

def test_team_name_validation():
    """Test the improved team name validation"""
    scraper = VLRScraper()
    
    print("🔍 Testing team name validation...")
    
    # Test cases for validation
    test_cases = [
        ("Sentinels", True),
        ("Paper Rex", True),
        ("G2 Esports", True),
        ("", False),
        ("TBD", False),
        ("Winner of Match 1", False),
        ("Qualified", False),
        ("Bye", False),
        ("W1", False),
        ("L2", False),
        ("123", False),
        ("Team Heretics", True),
        ("Edward Gaming", True),
        ("a", False),  # Too short
        ("T1", True),  # Valid short team name
    ]
    
    print("\n📝 Team name validation results:")
    for team_name, expected in test_cases:
        is_valid = scraper._is_valid_team_name(team_name)
        status = "✅" if is_valid == expected else "❌"
        print(f"{status} '{team_name}' → {is_valid} (expected: {expected})")
    
    print()

def test_scraper():
    """Test the actual scraper"""
    scraper = VLRScraper()
    
    print("🌐 Testing VLR scraper with improved filtering...")
    
    try:
        # Test getting tournament info for a known tournament
        print("\n1. Testing get_tournament_info with a sample event ID...")
        
        # Test with a sample event ID (you might need to update this with current ones)
        sample_event_id = "2097"  # VCT Champions 2024 or similar
        tournament_info = scraper.get_tournament_info(f"https://vlr.gg/event/{sample_event_id}")
        
        if tournament_info:
            print(f"✅ Tournament info retrieved: {tournament_info['name']}")
            print(f"   Event ID: {tournament_info.get('event_id', 'N/A')}")
            print(f"   Dates: {tournament_info.get('dates', 'N/A')}")
            
            # Test getting matches for this tournament
            if tournament_info.get('event_id'):
                print(f"\n2. Getting upcoming matches for: {tournament_info['name']}")
                matches = scraper.get_upcoming_matches(tournament_info['event_id'])
                
                if matches:
                    print(f"✅ Found {len(matches)} potential matches")
                    
                    # Show match details
                    valid_matches = []
                    for i, match in enumerate(matches[:10]):  # Check first 10
                        team1 = match.get('team1', 'N/A')
                        team2 = match.get('team2', 'N/A')
                        stage = match.get('match_stage', 'N/A')
                        time = match.get('match_time', 'N/A')
                        
                        # Check if both teams are valid
                        if scraper._is_valid_team_name(team1) and scraper._is_valid_team_name(team2):
                            valid_matches.append(match)
                            print(f"   ✅ {team1} vs {team2}")
                            print(f"      Stage: {stage}, Time: {time}")
                        else:
                            print(f"   ❌ {team1} vs {team2} (invalid team names)")
                    
                    print(f"\n📊 {len(valid_matches)}/{len(matches)} matches have valid team names")
                    
                    if len(valid_matches) > 0:
                        print("\n✅ Scraper successfully filtered out invalid matches!")
                    else:
                        print("\n⚠️ No valid matches found - might need to try different event IDs")
                else:
                    print("❌ No matches found for this tournament")
            else:
                print("❌ No event ID found in tournament info")
        else:
            print("❌ Could not retrieve tournament info")
            
        # Test with manual team name examples
        print("\n3. Testing with manual team name examples...")
        test_matches = [
            {'team1': 'Sentinels', 'team2': 'Fnatic'},
            {'team1': 'TBD', 'team2': 'Paper Rex'},
            {'team1': 'G2 Esports', 'team2': 'Winner of Match 1'},
            {'team1': 'Cloud9', 'team2': 'Team Heretics'}
        ]
        
        for match in test_matches:
            team1_valid = scraper._is_valid_team_name(match['team1'])
            team2_valid = scraper._is_valid_team_name(match['team2'])
            status = "✅" if team1_valid and team2_valid else "❌"
            print(f"   {status} {match['team1']} vs {match['team2']} (Valid: {team1_valid}, {team2_valid})")
            
    except Exception as e:
        print(f"❌ Error testing scraper: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 VLR Scraper Improvement Test")
    print("=" * 50)
    
    # Test individual components
    test_team_name_cleaning()
    test_team_name_validation()
    
    # Test the full scraper
    test_scraper()
    
    print("\n✅ Testing complete!")
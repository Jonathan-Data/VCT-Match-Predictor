#!/usr/bin/env python3
"""
Final test of the VLR scraper filtering with all fixes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vlr_scraper import VLRScraper

def final_test():
    """Final comprehensive test"""
    scraper = VLRScraper()
    
    print("FINAL VLR Scraper Test - All Problematic Cases")
    print("=" * 60)
    
    # All problematic cases that should be filtered out
    problematic_cases = [
        "PRX (qualified)",
        "Paper Rex qualified", 
        "Paper Rex (qualified)",
        "[SG] PRX qualified for Champions",
        "Sentinels vs",
        "vs Paper Rex", 
        "Team Heretics vs TBD",
        "TBD vs Cloud9",
        "Winner of Match 1 vs Fnatic",
        "G2 Esports vs Loser of UB Final",
        "Sentinels United States",
        "[US] Cloud9 USA", 
        "(BR) LOUD Brazil",
        "Paper Rex Singapore",
        "Fnatic United Kingdom",
        "DRX Korea",
        "Team Liquid EMEA",
        "W1", "L2", 
        "Winner of Match 3",
        "Loser of Semifinal 1",
        "Upper Bracket Final Winner",
        "Lower Bracket R2",
        "Group Stage", "Playoffs", "Quarterfinals", 
        "Grand Final", "Elimination Match", "Decider Match",
        "Sentinels (qualified) vs TBD",
        "[KR] DRX 2-1 (advanced)",
        "Paper Rex 1d 4h qualified",
        "Fnatic (eliminated)",
    ]
    
    # Valid cases that should be preserved
    valid_cases = [
        "Sentinels", "Paper Rex", "Fnatic", "Team Heretics", "Edward Gaming",
        "G2 Esports", "100 Thieves", "T1", "Cloud9", "DRX", "LOUD", "NRG", 
        "Gen.G", "Team Liquid"
    ]
    
    print("\nProblematic cases (should be FILTERED):")
    print("-" * 45)
    
    filtered_count = 0
    for case in problematic_cases:
        cleaned = scraper.team_processor.clean_team_name(case)
        is_valid = scraper.team_processor.is_valid_team_name(cleaned) if cleaned else False
        
        if not cleaned or not is_valid:
            status = "FILTERED"
            filtered_count += 1
        else:
            status = "NOT FILTERED - ERROR"
        
        print(f"{status:15} | {case}")
    
    print(f"\nFiltered: {filtered_count}/{len(problematic_cases)} ({filtered_count/len(problematic_cases)*100:.1f}%)")
    
    print(f"\nValid cases (should be PRESERVED):")
    print("-" * 45)
    
    preserved_count = 0
    for case in valid_cases:
        cleaned = scraper.team_processor.clean_team_name(case)
        is_valid = scraper.team_processor.is_valid_team_name(cleaned) if cleaned else False
        
        if cleaned and is_valid:
            status = "PRESERVED"
            preserved_count += 1
        else:
            status = "FILTERED - ERROR"
        
        print(f"{status:15} | {case} -> {cleaned}")
    
    print(f"\nPreserved: {preserved_count}/{len(valid_cases)} ({preserved_count/len(valid_cases)*100:.1f}%)")
    
    # Test sample matches
    print(f"\nSample match validation:")
    print("-" * 45)
    
    test_matches = [
        ("Sentinels", "Fnatic"),
        ("Paper Rex", "DRX"),
        ("100 Thieves", "Team Heretics"),
        ("T1", "Gen.G"),
    ]
    
    valid_matches = 0
    for team1, team2 in test_matches:
        clean1 = scraper.team_processor.clean_team_name(team1)
        clean2 = scraper.team_processor.clean_team_name(team2)
        valid1 = scraper.team_processor.is_valid_team_name(clean1) if clean1 else False
        valid2 = scraper.team_processor.is_valid_team_name(clean2) if clean2 else False
        
        match_valid = valid1 and valid2 and clean1 != clean2 and len(clean1) >= 2 and len(clean2) >= 2
        
        if match_valid:
            status = "VALID MATCH"
            valid_matches += 1
        else:
            status = "INVALID"
        
        print(f"{status:15} | {team1} vs {team2}")
    
    print(f"\nValid matches: {valid_matches}/{len(test_matches)} ({valid_matches/len(test_matches)*100:.1f}%)")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"  Problematic cases filtered: {filtered_count}/{len(problematic_cases)} ({filtered_count/len(problematic_cases)*100:.1f}%)")
    print(f"  Valid teams preserved:      {preserved_count}/{len(valid_cases)} ({preserved_count/len(valid_cases)*100:.1f}%)")
    print(f"  Sample matches valid:       {valid_matches}/{len(test_matches)} ({valid_matches/len(test_matches)*100:.1f}%)")
    
    if filtered_count == len(problematic_cases) and preserved_count == len(valid_cases):
        print(f"\n*** SUCCESS! ALL ISSUES FIXED! ***")
        print(f"  - 100% of problematic cases filtered")
        print(f"  - 100% of valid teams preserved")
        print(f"  - Ready for production use!")
    else:
        print(f"\nStill needs work on remaining cases.")

if __name__ == "__main__":
    final_test()
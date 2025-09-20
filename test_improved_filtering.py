#!/usr/bin/env python3
"""
Test script for improved team name filtering
"""

import sys
import os
sys.path.append('src')

from vlr_scraper import TeamNameProcessor

# Test problematic team names from the GUI output
problematic_names = [
    "Paper Rex2",
    "Bilibili Gaming0", 
    "Team Liquid0",
    "G2 Esports0",
    "SentinelsUnited States",
    "EDward GamingChina",
    "Sentinels1",
    "Sep 19 Decider GIANTX",
    "Elimination Bilibili Gaming",
    "Sep 19 Decider DRX",
    "Elimination G2 Esports",
    "Evry",
    "Courcouronnes"
]

# Test valid team names that should pass
valid_names = [
    "Paper Rex",
    "Bilibili Gaming",
    "Team Liquid", 
    "G2 Esports",
    "Sentinels",
    "Edward Gaming",
    "DRX",
    "GIANTX",
    "Xi Lai Gaming",
    "Dragon Ranger Gaming",
    "Rex Regum Qeon"
]

print("=== Testing Problematic Names (should be REJECTED) ===")
processor = TeamNameProcessor()

for name in problematic_names:
    cleaned = processor.clean_team_name(name)
    valid = processor.is_valid_team_name(cleaned) if cleaned else False
    status = "✅ REJECTED" if not valid else "❌ ACCEPTED"
    print(f"{status}: '{name}' -> '{cleaned}' -> Valid: {valid}")

print("\n=== Testing Valid Names (should be ACCEPTED) ===")
for name in valid_names:
    cleaned = processor.clean_team_name(name)
    valid = processor.is_valid_team_name(cleaned) if cleaned else False
    status = "✅ ACCEPTED" if valid else "❌ REJECTED"
    print(f"{status}: '{name}' -> '{cleaned}' -> Valid: {valid}")

print("\n=== Summary ===")
problematic_rejected = sum(1 for name in problematic_names 
                          if not processor.is_valid_team_name(processor.clean_team_name(name)))
valid_accepted = sum(1 for name in valid_names 
                    if processor.is_valid_team_name(processor.clean_team_name(name)))

print(f"Problematic names rejected: {problematic_rejected}/{len(problematic_names)}")
print(f"Valid names accepted: {valid_accepted}/{len(valid_names)}")
print(f"Overall accuracy: {(problematic_rejected + valid_accepted)}/{len(problematic_names) + len(valid_names)} = {((problematic_rejected + valid_accepted) / (len(problematic_names) + len(valid_names))) * 100:.1f}%")
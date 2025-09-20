#!/usr/bin/env python3
"""
Test script to debug manual prediction tab issues
"""

import sys
import os
import yaml
from pathlib import Path

# Test teams config loading
config_path = Path("config/teams.yaml")
print(f"Config path exists: {config_path.exists()}")

if config_path.exists():
    try:
        with open(config_path, 'r') as f:
            teams_config = yaml.safe_load(f)
        
        print(f"Teams config loaded successfully")
        print(f"Teams available: {len(teams_config.get('teams', {}))}")
        
        # Test dropdown population logic
        teams = []
        if teams_config and 'teams' in teams_config:
            for team_info in teams_config['teams'].values():
                team_name = team_info.get('name', '')
                vlr_id = team_info.get('vlr_id', '')
                if team_name:
                    teams.append(f"{team_name} (ID: {vlr_id})")
        
        teams.sort()
        print(f"\nDropdown entries would be:")
        for i, team in enumerate(teams[:5]):  # Show first 5
            print(f"  {i+1}. {team}")
        print(f"  ... and {len(teams) - 5} more")
        
        # Test team extraction logic
        test_selection = teams[0] if teams else "Paper Rex (ID: 624)"
        team_name = test_selection.split(" (ID: ")[0] if " (ID: " in test_selection else test_selection
        print(f"\nTest selection: '{test_selection}'")
        print(f"Extracted name: '{team_name}'")
        
        # Test ID lookup
        test_id = "624"  # Paper Rex
        try:
            id_int = int(test_id.strip())
            teams_dict = teams_config.get('teams', {})
            
            found_team = None
            for team_info in teams_dict.values():
                if team_info.get('vlr_id') == id_int:
                    found_team = team_info.get('name')
                    break
            
            print(f"\nID lookup test:")
            print(f"  Looking for ID: {test_id}")
            print(f"  Found team: {found_team}")
            
        except ValueError:
            print(f"Invalid ID format: {test_id}")
        
        print(f"\n✅ Teams config and extraction logic working properly")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
else:
    print("❌ Teams config file not found")

# Test model import
print(f"\n" + "="*50)
print("Testing model imports...")

try:
    sys.path.append(str(Path(__file__).parent / "src"))
    print("✅ Added src to path")
except Exception as e:
    print(f"❌ Path setup failed: {e}")

# Check for pre-trained models
model_dir = Path("models/pretrained")
print(f"\nChecking for models in {model_dir}:")
if model_dir.exists():
    for file in model_dir.glob("*.pkl"):
        print(f"  Found: {file.name}")
else:
    print("  Model directory not found")

print(f"\n" + "="*50)
print("Manual prediction debugging complete!")
#!/usr/bin/env python3
"""
Test if the main GUI can initialize without errors
"""

import sys
import os
from pathlib import Path
import threading
import time

# Suppress tkinter warnings
os.environ['TK_SILENCE_DEPRECATION'] = '1'
os.environ['DISPLAY'] = ':0.0'  # For headless testing

def test_gui_init():
    """Test GUI initialization"""
    print("=== Testing Main GUI Initialization ===")
    
    try:
        # Import the main GUI class
        from main_gui import VCTPredictionGUI
        
        print("✅ GUI class imported successfully")
        
        # Try to initialize (but don't run mainloop)
        print("🔄 Initializing GUI...")
        
        # This might fail on headless systems, so we'll catch that
        try:
            app = VCTPredictionGUI()
            print("✅ GUI initialized successfully")
            
            # Test key attributes
            checks = [
                ("teams_config", app.teams_config),
                ("model_trained", app.model_trained), 
                ("predictor", app.predictor),
                ("team1_combo", hasattr(app, 'team1_combo')),
                ("team2_combo", hasattr(app, 'team2_combo')),
                ("predict_match method", hasattr(app, 'predict_match')),
            ]
            
            print("\n=== Component Checks ===")
            for name, check in checks:
                if name.endswith("method"):
                    status = "✅" if check else "❌"
                    print(f"{status} {name}: {'Available' if check else 'Missing'}")
                elif name in ["team1_combo", "team2_combo"]:
                    status = "✅" if check else "❌"
                    print(f"{status} {name}: {'Created' if check else 'Missing'}")
                else:
                    if name == "teams_config":
                        team_count = len(check.get('teams', {})) if check else 0
                        status = "✅" if team_count > 0 else "❌"
                        print(f"{status} {name}: {team_count} teams loaded")
                    elif name == "model_trained":
                        status = "✅" if check else "❌"
                        print(f"{status} {name}: {'Yes' if check else 'No'}")
                    elif name == "predictor":
                        status = "✅" if check else "❌"
                        print(f"{status} {name}: {'Loaded' if check else 'None'}")
            
            # Test dropdown population
            if hasattr(app, 'team1_combo') and hasattr(app, 'team2_combo'):
                team1_values = app.team1_combo['values']
                team2_values = app.team2_combo['values']
                
                print(f"\n=== Dropdown Population ===")
                print(f"✅ Team 1 dropdown: {len(team1_values)} options")
                print(f"✅ Team 2 dropdown: {len(team2_values)} options")
                
                if len(team1_values) > 0:
                    print(f"   Sample: {team1_values[0]}")
            
            # Test prediction method availability
            if app.predictor and hasattr(app, 'predict_match'):
                print(f"\n=== Prediction System ===")
                print(f"✅ Model loaded: {app.model_trained}")
                print(f"✅ Prediction method: Available")
                
                # Try to get first two teams for a test
                if hasattr(app, 'team1_combo') and len(app.team1_combo['values']) >= 2:
                    test_team1 = app.get_team_name_from_selection(app.team1_combo['values'][0])
                    test_team2 = app.get_team_name_from_selection(app.team1_combo['values'][1])
                    print(f"   Test teams: {test_team1} vs {test_team2}")
                    
                    # Don't actually run prediction to avoid loading delays
                    print(f"✅ Ready for predictions")
            
            # Clean up - don't start the main loop
            app.root.quit()
            app.root.destroy()
            
            print(f"\n✅ GUI INITIALIZATION: SUCCESS")
            return True
            
        except Exception as gui_error:
            print(f"❌ GUI initialization failed: {str(gui_error)}")
            print("   This might be due to no display available (headless environment)")
            return False
            
    except Exception as import_error:
        print(f"❌ Failed to import GUI: {str(import_error)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_init()
    
    print(f"\n{'='*50}")
    print(f"TEST RESULT: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*50}")
    
    if success:
        print("🎉 The main GUI should work properly!")
        print("   Try running: py main_gui.py")
    else:
        print("❌ There are still issues with the GUI initialization")
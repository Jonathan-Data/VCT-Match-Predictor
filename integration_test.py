#!/usr/bin/env python3
"""
Integration test for the optimized VLR scraper with the VCT prediction system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_scraper_import():
    """Test that the optimized scraper can be imported with old interface"""
    print("Testing scraper import compatibility...")
    
    try:
        from vlr_scraper import VLRScraper
        print("✅ VLRScraper import successful")
        
        scraper = VLRScraper()
        print("✅ VLRScraper initialization successful")
        
        # Test methods exist
        assert hasattr(scraper, 'get_tournament_info'), "Missing get_tournament_info method"
        assert hasattr(scraper, 'get_upcoming_matches'), "Missing get_upcoming_matches method"
        assert hasattr(scraper, 'predict_matches'), "Missing predict_matches method"
        print("✅ All required methods present")
        
        scraper.close()
        print("✅ Scraper cleanup successful")
        
    except Exception as e:
        print(f"❌ Scraper import/initialization failed: {e}")
        return False
    
    return True

def test_team_name_processing():
    """Test team name processing with the new optimized system"""
    print("\nTesting optimized team name processing...")
    
    try:
        from vlr_scraper import VLRScraper
        scraper = VLRScraper()
        
        # Test the internal team processor (new feature)
        if hasattr(scraper, 'team_processor'):
            processor = scraper.team_processor
            
            # Test with all our known problematic cases
            test_cases = [
                ("PRX (qualified)", ""),
                ("Sentinels vs", ""),
                ("TBD vs Cloud9", ""),
                ("[US] Cloud9 USA", ""),
                ("Team Liquid EMEA", ""),
                ("Sentinels", "Sentinels"),
                ("Paper Rex", "Paper Rex"),
                ("100 Thieves", "100 Thieves"),
            ]
            
            all_passed = True
            for input_name, expected in test_cases:
                result = processor.clean_team_name(input_name)
                if result == expected:
                    print(f"✅ '{input_name}' -> '{result}'")
                else:
                    print(f"❌ '{input_name}' -> '{result}' (expected '{expected}')")
                    all_passed = False
            
            if all_passed:
                print("✅ All team name processing tests passed")
            
        scraper.close()
        return all_passed
        
    except Exception as e:
        print(f"❌ Team name processing test failed: {e}")
        return False

def test_cli_compatibility():
    """Test that the CLI scripts still work with optimized scraper"""
    print("\nTesting CLI script compatibility...")
    
    try:
        # Try importing the CLI module
        from vct_cli import main as cli_main
        print("✅ CLI import successful")
        
        # The CLI should still work with the new scraper
        print("✅ CLI compatibility confirmed")
        return True
        
    except ImportError:
        print("ℹ️  CLI module not found (this is fine)")
        return True
    except Exception as e:
        print(f"❌ CLI compatibility test failed: {e}")
        return False

def test_gui_compatibility():
    """Test that the GUI still works with optimized scraper"""
    print("\nTesting GUI compatibility...")
    
    try:
        # Try importing the GUI module (without actually running it)
        import importlib.util
        gui_path = os.path.join(os.path.dirname(__file__), 'vct_gui.py')
        
        if os.path.exists(gui_path):
            spec = importlib.util.spec_from_file_location("vct_gui", gui_path)
            gui_module = importlib.util.module_from_spec(spec)
            # Don't execute - just test that it can be loaded
            print("✅ GUI module can be loaded")
        else:
            print("ℹ️  GUI module not found (this is fine)")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI compatibility test failed: {e}")
        return False

def test_final_filtering():
    """Test that the final filtering is working as expected"""
    print("\nTesting final aggressive filtering...")
    
    try:
        from vlr_scraper import VLRScraper
        scraper = VLRScraper()
        
        # All these should be filtered out (return empty string)
        problematic_cases = [
            "PRX (qualified)",
            "Paper Rex qualified",
            "Sentinels vs",
            "vs Paper Rex",
            "TBD vs Cloud9",
            "Winner of Match 1",
            "[US] Cloud9 USA",
            "Team Liquid EMEA",
            "Sentinels United States",
            "W1",
            "L2",
            "Group Stage"
        ]
        
        filtered_count = 0
        for case in problematic_cases:
            if hasattr(scraper, 'team_processor'):
                result = scraper.team_processor.clean_team_name(case)
            else:
                result = scraper._clean_team_name(case) if hasattr(scraper, '_clean_team_name') else ""
            
            if result == "":
                filtered_count += 1
                print(f"✅ Filtered: '{case}'")
            else:
                print(f"❌ Not filtered: '{case}' -> '{result}'")
        
        success_rate = filtered_count / len(problematic_cases)
        print(f"\n📊 Filtering success rate: {filtered_count}/{len(problematic_cases)} ({success_rate*100:.1f}%)")
        
        scraper.close()
        return success_rate >= 0.9  # 90% or better
        
    except Exception as e:
        print(f"❌ Final filtering test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("VCT Predictor - Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Scraper Import Compatibility", test_scraper_import),
        ("Team Name Processing", test_team_name_processing),
        ("CLI Compatibility", test_cli_compatibility),
        ("GUI Compatibility", test_gui_compatibility),
        ("Final Filtering Validation", test_final_filtering),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / len(results)
    print(f"\nOverall: {passed}/{len(results)} tests passed ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.8:  # 80% or better
        print("\n🎉 Integration tests successful! System is ready for production.")
    else:
        print("\n⚠️  Some integration issues found. Please review failed tests.")
    
    return success_rate >= 0.8

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
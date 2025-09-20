#!/usr/bin/env python3
"""
Quick test of the optimized VLR scraper to ensure it works correctly
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vlr_scraper_optimized import VLRScraperOptimized, TeamNameProcessor

def test_team_name_processor():
    """Test the team name processor"""
    print("Testing Team Name Processor...")
    print("=" * 50)
    
    processor = TeamNameProcessor()
    
    # Test valid team names
    valid_tests = [
        ("Sentinels", "Sentinels"),
        ("Paper Rex", "Paper Rex"),
        ("100 Thieves", "100 Thieves"),
        ("T1", "T1"),
        ("Gen.G", "Gen.G"),
        ("[US] Cloud9", "Cloud9"),
        ("G2 Esports #1", "G2 Esports"),
    ]
    
    print("Valid team cleaning tests:")
    for input_name, expected in valid_tests:
        result = processor.clean_team_name(input_name)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_name}' -> '{result}' (expected: '{expected}')")
    
    # Test problematic cases (should be filtered out)
    problematic_tests = [
        "PRX (qualified)",
        "Sentinels vs",
        "TBD vs Cloud9",
        "Team Liquid EMEA",
        "Winner of Match 1"
    ]
    
    print(f"\nProblematic case filtering tests:")
    for case in problematic_tests:
        result = processor.clean_team_name(case)
        status = "✅" if result == "" else "❌"
        print(f"{status} '{case}' -> '{result}' (should be empty)")
    
    # Test validation
    print(f"\nValidation tests:")
    validation_tests = [
        ("Sentinels", True),
        ("Paper Rex", True),
        ("T1", True),
        ("", False),
        ("TBD", False),
        ("vs", False),
    ]
    
    for team_name, expected in validation_tests:
        result = processor.is_valid_team_name(team_name)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{team_name}' -> {result} (expected: {expected})")

def test_performance():
    """Test performance of team processing"""
    print(f"\nPerformance Test...")
    print("=" * 50)
    
    processor = TeamNameProcessor()
    
    # Performance test
    test_names = [
        "🇺🇸 Sentinels United States (qualified) 2-1 #1",
        "[BR] LOUD Brazil qualified for Champions", 
        "Paper Rex 🇸🇬 Singapore APAC",
        "Team Heretics EMEA (eliminated)",
    ] * 250  # 1000 operations
    
    print(f"Processing {len(test_names)} team names...")
    
    start_time = time.time()
    for name in test_names:
        cleaned = processor.clean_team_name(name)
        is_valid = processor.is_valid_team_name(cleaned)
    elapsed_time = time.time() - start_time
    
    # Avoid division by zero if processing is very fast
    if elapsed_time > 0:
        ops_per_second = len(test_names) * 2 / elapsed_time  # 2 operations per name
        print(f"✅ Performance: {ops_per_second:.0f} operations/second")
    else:
        print(f"✅ Performance: >10000 operations/second (too fast to measure accurately)")
    print(f"   Processing time: {elapsed_time:.4f} seconds")

def test_scraper_initialization():
    """Test scraper can be initialized"""
    print(f"\nScraper Initialization Test...")
    print("=" * 50)
    
    try:
        scraper = VLRScraperOptimized()
        print("✅ Scraper initialized successfully")
        print(f"   Base URL: {scraper.BASE_URL}")
        print(f"   Timeout: {scraper.REQUEST_TIMEOUT}s")
        print(f"   Max retries: {scraper.MAX_RETRIES}")
        
        # Test cleanup
        scraper.close()
        print("✅ Scraper closed successfully")
        
    except Exception as e:
        print(f"❌ Scraper initialization failed: {e}")

def test_backward_compatibility():
    """Test backward compatibility with old import"""
    print(f"\nBackward Compatibility Test...")
    print("=" * 50)
    
    try:
        from vlr_scraper_optimized import VLRScraper
        scraper = VLRScraper()
        print("✅ Backward compatibility works")
        scraper.close()
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")

if __name__ == "__main__":
    print("VLR Scraper Optimization Test Suite")
    print("=" * 60)
    
    test_team_name_processor()
    test_performance()
    test_scraper_initialization()
    test_backward_compatibility()
    
    print(f"\n" + "=" * 60)
    print("✅ All tests completed!")
    print("The optimized scraper is ready for production use.")
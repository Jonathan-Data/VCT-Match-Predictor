#!/usr/bin/env python3
"""
Comprehensive Test Suite for VLR Scraper
==========================================

Tests all functionality including:
- Team name cleaning and validation
- Tournament information extraction
- Match extraction and filtering
- Performance benchmarks
- Error handling and edge cases
- Integration with prediction system

Run with: python -m pytest tests/test_vlr_scraper_comprehensive.py -v
"""

import pytest
import time
import logging
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlr_scraper_optimized import VLRScraperOptimized, TeamNameProcessor, MatchData, TournamentData

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

class TestTeamNameProcessor:
    """Test suite for TeamNameProcessor"""
    
    @pytest.fixture
    def processor(self):
        return TeamNameProcessor()
    
    def test_clean_team_name_valid_teams(self, processor):
        """Test cleaning of valid team names"""
        test_cases = [
            ("Sentinels", "Sentinels"),
            ("Paper Rex", "Paper Rex"),
            ("Team Heretics", "Team Heretics"),
            ("100 Thieves", "100 Thieves"),
            ("G2 Esports", "G2 Esports"),
            ("Gen.G", "Gen.G"),
            ("T1", "T1"),
            ("DRX", "DRX"),
            ("LOUD", "LOUD"),
            ("NRG", "NRG"),
        ]
        
        for input_name, expected in test_cases:
            result = processor.clean_team_name(input_name)
            assert result == expected, f"Failed for {input_name}: got '{result}', expected '{expected}'"
    
    def test_clean_team_name_removes_emojis(self, processor):
        """Test removal of flag emojis and symbols"""
        test_cases = [
            ("🇺🇸 Sentinels", "Sentinels"),
            ("Paper Rex 🇸🇬", "Paper Rex"),
            ("🇧🇷 LOUD Brazil", ""),  # Should reject due to country name
            ("Fnatic • United Kingdom", ""),  # Should reject due to country name
        ]
        
        for input_name, expected in test_cases:
            result = processor.clean_team_name(input_name)
            assert result == expected, f"Failed for {input_name}: got '{result}', expected '{expected}'"
    
    def test_clean_team_name_removes_nationality_indicators(self, processor):
        """Test removal of nationality brackets and codes"""
        test_cases = [
            ("[US] Cloud9", "Cloud9"),
            ("(BR) LOUD", "LOUD"), 
            ("KR DRX", "DRX"),
            ("Team Liquid EMEA", ""),  # Should reject due to region indicator
            ("[US] Cloud9 USA", ""),   # Should reject due to redundant country info
        ]
        
        for input_name, expected in test_cases:
            result = processor.clean_team_name(input_name)
            assert result == expected, f"Failed for {input_name}: got '{result}', expected '{expected}'"
    
    def test_clean_team_name_removes_scores_and_times(self, processor):
        """Test removal of scores, times, and other noise"""
        test_cases = [
            ("Sentinels 2-1", "Sentinels"),
            ("Paper Rex (2-0)", "Paper Rex"),
            ("Team Heretics 10:30 AM", "Team Heretics"),
            ("G2 Esports #1", "G2 Esports"),
            ("DRX 1d 4h", "DRX"),
        ]
        
        for input_name, expected in test_cases:
            result = processor.clean_team_name(input_name)
            assert result == expected, f"Failed for {input_name}: got '{result}', expected '{expected}'"
    
    def test_clean_team_name_rejects_problematic_cases(self, processor):
        """Test rejection of problematic cases"""
        problematic_cases = [
            "PRX (qualified)",
            "Paper Rex qualified",
            "TBD vs Cloud9",
            "Winner of Match 1",
            "Sentinels vs",
            "vs Paper Rex",
            "W1",
            "L2",
            "Group Stage",
            "Playoffs",
            "Grand Final",
            "Elimination Match",
            "Sentinels United States",
            "Paper Rex Singapore",
            "Fnatic United Kingdom",
        ]
        
        for case in problematic_cases:
            result = processor.clean_team_name(case)
            assert result == "", f"Should reject '{case}' but got '{result}'"
    
    def test_is_valid_team_name(self, processor):
        """Test team name validation"""
        valid_cases = [
            "Sentinels",
            "Paper Rex", 
            "T1",
            "100 Thieves",
            "Gen.G",
            "Team Heretics",
            "Edward Gaming"
        ]
        
        invalid_cases = [
            "",
            "A",
            "12",
            "TBD",
            "Winner",
            "Qualified",
            "vs",
            "USA",
            "BR"
        ]
        
        for case in valid_cases:
            assert processor.is_valid_team_name(case), f"Should validate '{case}'"
        
        for case in invalid_cases:
            assert not processor.is_valid_team_name(case), f"Should reject '{case}'"
    
    def test_caching_performance(self, processor):
        """Test that caching improves performance"""
        test_name = "Team Heretics with lots of extra text and emojis 🇪🇸 Spain EMEA"
        
        # First call (cache miss)
        start_time = time.time()
        result1 = processor.clean_team_name(test_name)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit) 
        start_time = time.time()
        result2 = processor.clean_team_name(test_name)
        second_call_time = time.time() - start_time
        
        assert result1 == result2
        assert second_call_time < first_call_time * 0.5  # Cache should be significantly faster


class TestVLRScraperOptimized:
    """Test suite for VLRScraperOptimized"""
    
    @pytest.fixture
    def scraper(self):
        return VLRScraperOptimized(enable_caching=True)
    
    @pytest.fixture
    def mock_session(self):
        """Mock requests session"""
        session = Mock()
        response = Mock()
        response.status_code = 200
        response.content = b"""
        <html>
            <head><title>VCT Champions 2024 - VLR.gg</title></head>
            <body>
                <h1 class="wf-title">VCT Champions 2024</h1>
                <div class="event-desc-item">Aug 1-11, 2024</div>
                <div class="event-desc-item">$1,000,000 USD</div>
                <div class="event-desc-item">Seoul, South Korea</div>
                <div class="match-item">
                    <span class="team">Sentinels</span>
                    <span class="team">Paper Rex</span>
                    <span class="time">14:00 UTC</span>
                    <span class="stage">Grand Final</span>
                </div>
                <a href="/match/12345">Team Heretics vs G2 Esports</a>
            </body>
        </html>
        """
        response.raise_for_status = Mock()
        session.get.return_value = response
        return session
    
    def test_initialization(self, scraper):
        """Test scraper initialization"""
        assert scraper.BASE_URL == "https://www.vlr.gg"
        assert scraper.REQUEST_TIMEOUT == 15
        assert scraper.MAX_RETRIES == 3
        assert hasattr(scraper, 'session')
        assert hasattr(scraper, 'team_processor')
    
    @patch('vlr_scraper_optimized.requests.Session')
    def test_session_configuration(self, mock_session_class, scraper):
        """Test session configuration with retries"""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Create new scraper to test session setup
        new_scraper = VLRScraperOptimized()
        
        # Verify session configuration
        mock_session.headers.update.assert_called()
        mock_session.mount.assert_called()
    
    def test_get_tournament_info(self, scraper, mock_session):
        """Test tournament information extraction"""
        scraper.session = mock_session
        
        result = scraper.get_tournament_info("2024")
        
        assert result is not None
        assert isinstance(result, TournamentData)
        assert result.name == "VCT Champions 2024"
        assert result.event_id == "2024"
        assert "Aug" in result.dates
        assert "$1,000,000" in result.prize
        assert "Seoul" in result.location
    
    def test_get_tournament_info_error_handling(self, scraper):
        """Test error handling in tournament info extraction"""
        # Mock a failed request
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Network error")
        scraper.session = mock_session
        
        result = scraper.get_tournament_info("invalid")
        assert result is None
    
    def test_get_upcoming_matches(self, scraper, mock_session):
        """Test match extraction"""
        scraper.session = mock_session
        
        matches = scraper.get_upcoming_matches("2024")
        
        assert isinstance(matches, list)
        # Should find at least one valid match from our mock HTML
        valid_matches = [m for m in matches if m.team1 and m.team2]
        assert len(valid_matches) > 0
    
    def test_extract_teams_from_text(self, scraper):
        """Test team extraction from text"""
        test_cases = [
            ("Sentinels vs Paper Rex", ["Sentinels", "Paper Rex"]),
            ("Team Heretics – G2 Esports", ["Team Heretics", "G2 Esports"]),
            ("DRX - T1", ["DRX", "T1"]),
            ("Invalid text", []),
        ]
        
        for text, expected in test_cases:
            result = scraper._extract_teams_from_text(text)
            assert result == expected, f"Failed for '{text}': got {result}, expected {expected}"
    
    def test_filter_valid_matches(self, scraper):
        """Test match filtering and deduplication"""
        matches = [
            MatchData("Sentinels", "Paper Rex", confidence=0.9),
            MatchData("Paper Rex", "Sentinels", confidence=0.8),  # Duplicate
            MatchData("Team Heretics", "G2 Esports", confidence=0.7),
            MatchData("DRX", "T1", confidence=0.6),
        ]
        
        filtered = scraper._filter_valid_matches(matches)
        
        # Should remove duplicate and sort by confidence
        assert len(filtered) == 3
        assert filtered[0].confidence >= filtered[1].confidence >= filtered[2].confidence
    
    def test_predict_matches_integration(self, scraper):
        """Test integration with prediction system"""
        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.predict_match.return_value = {
            'predicted_winner': 'Sentinels',
            'confidence': 0.75,
            'team1_probability': 0.75,
            'team2_probability': 0.25,
            'confidence_level': 'High'
        }
        
        matches = [
            MatchData("Sentinels", "Paper Rex"),
            MatchData("Team Heretics", "G2 Esports"),
        ]
        
        results = scraper.predict_matches(matches, mock_predictor)
        
        assert len(results) == 2
        assert all('predicted_winner' in match for match in results)
        assert all('confidence' in match for match in results)
    
    def test_predict_matches_error_handling(self, scraper):
        """Test error handling in predictions"""
        # Mock predictor that fails
        mock_predictor = Mock()
        mock_predictor.predict_match.side_effect = Exception("Prediction error")
        
        matches = [MatchData("Sentinels", "Paper Rex")]
        
        results = scraper.predict_matches(matches, mock_predictor)
        
        assert len(results) == 1
        assert 'prediction_error' in results[0]
    
    def test_resource_cleanup(self, scraper):
        """Test proper resource cleanup"""
        mock_session = Mock()
        scraper.session = mock_session
        
        scraper.close()
        
        mock_session.close.assert_called_once()


class TestPerformance:
    """Performance and benchmark tests"""
    
    @pytest.fixture
    def scraper(self):
        return VLRScraperOptimized()
    
    def test_team_name_processing_performance(self, scraper):
        """Test team name processing performance"""
        test_names = [
            "🇺🇸 Sentinels United States (qualified) 2-1 #1",
            "[BR] LOUD Brazil AMERICAS qualified for Champions",
            "Paper Rex 🇸🇬 Singapore APAC • 10:30 AM",
            "Team Heretics 🇪🇸 Spain EMEA (eliminated)",
            "G2 Esports • Europe • 1d 4h"
        ] * 100  # 500 total operations
        
        start_time = time.time()
        
        for name in test_names:
            cleaned = scraper.team_processor.clean_team_name(name)
            is_valid = scraper.team_processor.is_valid_team_name(cleaned)
        
        elapsed_time = time.time() - start_time
        operations_per_second = len(test_names) * 2 / elapsed_time  # 2 operations per name
        
        print(f"Team processing: {operations_per_second:.0f} ops/sec")
        assert operations_per_second > 1000, "Team processing should be fast enough for real-time use"
    
    def test_memory_usage_stability(self, scraper):
        """Test that memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many team names
        for i in range(1000):
            name = f"Team Name {i} with lots of extra text and emojis 🇺🇸 United States"
            cleaned = scraper.team_processor.clean_team_name(name)
            scraper.team_processor.is_valid_team_name(cleaned)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024, f"Memory growth too high: {memory_growth / 1024 / 1024:.1f}MB"


class TestErrorHandling:
    """Error handling and resilience tests"""
    
    @pytest.fixture
    def scraper(self):
        return VLRScraperOptimized()
    
    def test_network_timeout_handling(self, scraper):
        """Test handling of network timeouts"""
        import requests
        
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")
        scraper.session = mock_session
        
        # Should not raise exception
        result = scraper.get_tournament_info("test")
        assert result is None
        
        matches = scraper.get_upcoming_matches("test")
        assert matches == []
    
    def test_malformed_html_handling(self, scraper):
        """Test handling of malformed HTML"""
        mock_session = Mock()
        response = Mock()
        response.content = b"<html><invalid>malformed html</invalid>"
        response.raise_for_status = Mock()
        mock_session.get.return_value = response
        scraper.session = mock_session
        
        # Should handle gracefully without crashing
        result = scraper.get_tournament_info("test")
        assert result is not None  # Should create fallback data
        
        matches = scraper.get_upcoming_matches("test")
        assert isinstance(matches, list)  # Should return empty list
    
    def test_invalid_team_names(self, scraper):
        """Test handling of various invalid inputs"""
        invalid_inputs = [
            None,
            "",
            "   ",
            123,
            [],
            {},
            "a" * 1000,  # Very long string
        ]
        
        for invalid_input in invalid_inputs:
            try:
                if isinstance(invalid_input, str) or invalid_input is None:
                    result = scraper.team_processor.clean_team_name(invalid_input or "")
                    assert result == ""
                    
                    is_valid = scraper.team_processor.is_valid_team_name(result)
                    assert not is_valid
            except Exception as e:
                pytest.fail(f"Should handle invalid input gracefully: {invalid_input}, but got {e}")
    
    def test_concurrent_access(self, scraper):
        """Test thread safety of caching"""
        import threading
        import concurrent.futures
        
        def process_team_name(name):
            cleaned = scraper.team_processor.clean_team_name(name)
            return scraper.team_processor.is_valid_team_name(cleaned)
        
        test_names = ["Sentinels", "Paper Rex", "Team Heretics", "G2 Esports"] * 25
        
        # Test concurrent access
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_team_name, name) for name in test_names]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == len(test_names)
        assert all(isinstance(result, bool) for result in results)


class TestIntegration:
    """Integration tests with the full pipeline"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from scraping to prediction"""
        scraper = VLRScraperOptimized()
        
        # Mock a complete workflow
        with patch.object(scraper.session, 'get') as mock_get:
            # Mock tournament page
            response = Mock()
            response.content = b"""
            <html>
                <head><title>VCT Champions 2024</title></head>
                <body>
                    <h1 class="wf-title">VCT Champions 2024</h1>
                    <div class="event-desc-item">Aug 1-11, 2024</div>
                    <a href="/match/123">Sentinels vs Paper Rex</a>
                    <a href="/match/456">Team Heretics vs G2 Esports</a>
                </body>
            </html>
            """
            response.raise_for_status = Mock()
            mock_get.return_value = response
            
            # Get tournament info
            tournament = scraper.get_tournament_info("2024")
            assert tournament is not None
            
            # Get matches
            matches = scraper.get_upcoming_matches("2024")
            assert len(matches) > 0
            
            # Mock predictor
            mock_predictor = Mock()
            mock_predictor.predict_match.return_value = {
                'predicted_winner': 'Sentinels',
                'confidence': 0.75,
                'team1_probability': 0.75,
                'team2_probability': 0.25
            }
            
            # Get predictions
            predictions = scraper.predict_matches(matches, mock_predictor)
            assert len(predictions) > 0
            assert all('predicted_winner' in pred for pred in predictions)
        
        scraper.close()


# Pytest configuration and utilities
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Disable verbose logging during tests
    logging.getLogger('vlr_scraper_optimized').setLevel(logging.WARNING)
    
    yield
    
    # Cleanup after tests
    pass


def run_all_tests():
    """Run all tests with detailed output"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-x"  # Stop on first failure
    ])


if __name__ == "__main__":
    # Run tests when called directly
    run_all_tests()
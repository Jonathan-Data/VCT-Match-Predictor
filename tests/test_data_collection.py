"""
Unit tests for data collection modules.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_collection import KaggleDataDownloader, VLRScraper, TeamStats


class TestKaggleDataDownloader:
    """Test cases for KaggleDataDownloader."""
    
    @pytest.fixture
    def config_file(self):
        """Create a temporary config file."""
        config_data = {
            'kaggle_datasets': [
                {
                    'name': 'test-dataset',
                    'dataset_id': 'user/test-dataset',
                    'url': 'https://www.kaggle.com/datasets/user/test-dataset'
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink()
    
    @patch('src.data_collection.kaggle_downloader.kaggle.api')
    def test_init(self, mock_kaggle, config_file):
        """Test KaggleDataDownloader initialization."""
        downloader = KaggleDataDownloader(config_file)
        
        assert downloader.config is not None
        assert 'kaggle_datasets' in downloader.config
        mock_kaggle.authenticate.assert_called_once()
    
    @patch('src.data_collection.kaggle_downloader.kaggle.api')
    def test_list_available_datasets(self, mock_kaggle, config_file):
        """Test listing available datasets."""
        downloader = KaggleDataDownloader(config_file)
        datasets = downloader.list_available_datasets()
        
        assert len(datasets) == 1
        assert datasets[0]['name'] == 'test-dataset'
    
    @patch('src.data_collection.kaggle_downloader.kaggle.api')
    def test_get_dataset_info(self, mock_kaggle, config_file):
        """Test getting dataset info."""
        # Mock the kaggle API response
        mock_dataset_info = Mock()
        mock_dataset_info.title = "Test Dataset"
        mock_dataset_info.description = "A test dataset"
        mock_dataset_info.totalBytes = 1024
        mock_dataset_info.lastUpdated = "2025-01-01"
        mock_dataset_info.downloadCount = 100
        mock_dataset_info.url = "https://kaggle.com/test"
        
        mock_kaggle.dataset_view.return_value = mock_dataset_info
        
        downloader = KaggleDataDownloader(config_file)
        info = downloader.get_dataset_info('user/test-dataset')
        
        assert info['title'] == "Test Dataset"
        assert info['size'] == 1024
        mock_kaggle.dataset_view.assert_called_once_with('user/test-dataset')


class TestVLRScraper:
    """Test cases for VLRScraper."""
    
    @pytest.fixture
    def config_file(self):
        """Create a temporary config file."""
        config_data = {
            'teams': {
                'test_team': {
                    'name': 'Test Team',
                    'region': 'Test Region',
                    'vlr_url': 'https://www.vlr.gg/team/123/test-team',
                    'vlr_id': 123
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_init(self, config_file):
        """Test VLRScraper initialization."""
        scraper = VLRScraper(config_file)
        
        assert scraper.config is not None
        assert 'teams' in scraper.config
        assert scraper.delay == 2.0
    
    @patch('src.data_collection.vlr_scraper.requests.Session.get')
    def test_make_request_success(self, mock_get, config_file):
        """Test successful HTTP request."""
        mock_response = Mock()
        mock_response.content = b'<html><body>Test</body></html>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        scraper = VLRScraper(config_file, delay=0.1)  # Reduce delay for testing
        soup = scraper._make_request('https://example.com')
        
        assert soup is not None
        assert soup.find('body').get_text() == 'Test'
    
    @patch('src.data_collection.vlr_scraper.requests.Session.get')
    def test_make_request_failure(self, mock_get, config_file):
        """Test failed HTTP request."""
        mock_get.side_effect = Exception("Network error")
        
        scraper = VLRScraper(config_file, delay=0.1)
        soup = scraper._make_request('https://example.com')
        
        assert soup is None
    
    def test_extract_team_statistics(self, config_file):
        """Test team statistics extraction."""
        scraper = VLRScraper(config_file)
        
        # Mock BeautifulSoup object
        mock_soup = Mock()
        stats_section = Mock()
        record_elem = Mock()
        record_elem.get_text.return_value = "15W - 3L"
        stats_section.find.return_value = record_elem
        mock_soup.find.return_value = stats_section
        
        stats = scraper._extract_team_statistics(mock_soup)
        
        assert stats['wins'] == 15
        assert stats['losses'] == 3
        assert stats['win_rate'] == 15 / 18
    
    def test_save_team_stats(self, config_file):
        """Test saving team stats to file."""
        scraper = VLRScraper(config_file)
        
        # Create test data
        team_stats = {
            'test_team': TeamStats(
                team_name='Test Team',
                region='Test Region',
                vlr_id=123,
                rating=1000.0,
                wins=10,
                losses=5,
                win_rate=0.67,
                rounds_won=250,
                rounds_lost=200,
                round_win_rate=0.56,
                avg_combat_score=220.5,
                players=[],
                recent_matches=[]
            )
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper.data_dir = Path(temp_dir)
            output_file = scraper.save_team_stats(team_stats)
            
            assert output_file.exists()
            
            # Verify the saved data
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert 'test_team' in saved_data
            assert saved_data['test_team']['team_name'] == 'Test Team'
            assert saved_data['test_team']['rating'] == 1000.0


class TestTeamStats:
    """Test cases for TeamStats dataclass."""
    
    def test_create_team_stats(self):
        """Test creating TeamStats object."""
        stats = TeamStats(
            team_name='Test Team',
            region='Test Region',
            vlr_id=123,
            rating=1000.0,
            wins=10,
            losses=5,
            win_rate=0.67,
            rounds_won=250,
            rounds_lost=200,
            round_win_rate=0.56,
            avg_combat_score=220.5,
            players=[{'name': 'Player1'}],
            recent_matches=[{'opponent': 'Team2', 'result': 'W'}]
        )
        
        assert stats.team_name == 'Test Team'
        assert stats.rating == 1000.0
        assert stats.wins == 10
        assert len(stats.players) == 1
        assert len(stats.recent_matches) == 1


if __name__ == '__main__':
    pytest.main([__file__])
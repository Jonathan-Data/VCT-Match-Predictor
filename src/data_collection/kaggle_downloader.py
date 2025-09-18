"""
Kaggle Dataset Downloader for VCT 2025 Tournament Data
"""

import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any
import kaggle
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDataDownloader:
    """Download and manage Kaggle datasets for VCT 2025 tournaments."""
    
    def __init__(self, config_path: str = None):
        """Initialize the downloader with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "teams.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        self._setup_kaggle()
    
    def _setup_kaggle(self):
        """Setup Kaggle API credentials."""
        try:
            kaggle.api.authenticate()
            logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            logger.error(f"Failed to authenticate Kaggle API: {e}")
            logger.error("Please ensure you have set up your Kaggle credentials")
            logger.error("Visit: https://www.kaggle.com/docs/api#getting-started-installation-&-authentication")
            sys.exit(1)
    
    def download_dataset(self, dataset_id: str, dataset_name: str) -> Path:
        """Download a specific dataset from Kaggle."""
        output_dir = self.data_dir / dataset_name
        output_dir.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Downloading dataset: {dataset_id}")
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(output_dir),
                unzip=True,
                quiet=False
            )
            logger.info(f"Successfully downloaded {dataset_name} to {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            raise
    
    def download_all_datasets(self) -> Dict[str, Path]:
        """Download all VCT 2025 datasets."""
        downloaded_datasets = {}
        
        for dataset in tqdm(self.config['kaggle_datasets'], desc="Downloading datasets"):
            try:
                path = self.download_dataset(
                    dataset['dataset_id'],
                    dataset['name']
                )
                downloaded_datasets[dataset['name']] = path
                
            except Exception as e:
                logger.error(f"Failed to download {dataset['name']}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_datasets)} datasets")
        return downloaded_datasets
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets in the configuration."""
        return self.config['kaggle_datasets']
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        try:
            dataset_info = kaggle.api.dataset_view(dataset_id)
            return {
                'title': dataset_info.title,
                'description': dataset_info.description,
                'size': dataset_info.totalBytes,
                'last_updated': dataset_info.lastUpdated,
                'download_count': dataset_info.downloadCount,
                'url': dataset_info.url
            }
        except Exception as e:
            logger.error(f"Failed to get info for dataset {dataset_id}: {e}")
            return {}

def main():
    """Main function to download all datasets."""
    downloader = KaggleDataDownloader()
    
    print("Available datasets:")
    for dataset in downloader.list_available_datasets():
        print(f"  - {dataset['name']}: {dataset['dataset_id']}")
    
    print("\nDownloading all datasets...")
    downloaded = downloader.download_all_datasets()
    
    print(f"\nDownload completed. {len(downloaded)} datasets available in:")
    for name, path in downloaded.items():
        print(f"  - {name}: {path}")

if __name__ == "__main__":
    main()
"""
Data Collection Module for VCT 2025 Predictor

This module handles downloading Kaggle datasets and scraping team statistics from VLR.gg
"""

from .kaggle_downloader import KaggleDataDownloader
from .vlr_scraper import VLRScraper, TeamStats

__all__ = ['KaggleDataDownloader', 'VLRScraper', 'TeamStats']
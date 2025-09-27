"""
Command Line Interface for VCT 2025 Match Predictor

This module provides a CLI for making match predictions and viewing team analysis.
"""

import click
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


from src.data_collection import KaggleDataDownloader, VLRScraper
from src.preprocessing import VCTDataProcessor
from src.models import VCTMatchPredictor

class VCTPredictorCLI:
    """Main CLI class for the VCT predictor."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = self.project_root / "config" / "teams.yaml"


        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)


        self.kaggle_downloader = None
        self.vlr_scraper = None
        self.data_processor = None
        self.predictor = None

    def get_team_names(self) -> List[str]:
        """Get list of available team names."""
        return [team_info['name'] for team_info in self.config['teams'].values()]

    def find_team_by_name(self, team_name: str) -> Dict[str, Any]:
        """Find team configuration by name (case-insensitive)."""
        team_name_lower = team_name.lower()

        for team_key, team_info in self.config['teams'].items():
            if team_info['name'].lower() == team_name_lower:
                return {'key': team_key, **team_info}


        for team_key, team_info in self.config['teams'].items():
            if team_name_lower in team_info['name'].lower():
                return {'key': team_key, **team_info}

        return None

@click.group()
@click.pass_context
def cli(ctx):
    """VCT 2025 Champions Match Predictor CLI"""
    ctx.ensure_object(dict)
    ctx.obj['predictor_cli'] = VCTPredictorCLI()



#!/usr/bin/env python3
"""
VCT 2025 Champions Match Predictor - Main Entry Point

Usage:
    python main.py --help
    python main.py predict "Sentinels" "Fnatic"
    python main.py teams
    python main.py setup
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_collection import KaggleDataDownloader, VLRScraper
from src.preprocessing import VCTDataProcessor
from src.models import VCTMatchPredictor

class VCTPredictor:
    """Main VCT Predictor application class."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_path = self.project_root / "config" / "teams.yaml"
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_team_names(self) -> List[str]:
        """Get list of available team names."""
        return [team_info['name'] for team_info in self.config['teams'].values()]
    
    def find_team_by_name(self, team_name: str) -> Dict[str, Any]:
        """Find team configuration by name (case-insensitive)."""
        team_name_lower = team_name.lower()
        
        for team_key, team_info in self.config['teams'].items():
            if team_info['name'].lower() == team_name_lower:
                return {'key': team_key, **team_info}
        
        # Try partial matching
        for team_key, team_info in self.config['teams'].items():
            if team_name_lower in team_info['name'].lower():
                return {'key': team_key, **team_info}
        
        return None
    
    def setup(self):
        """Run the complete setup process."""
        print("🎮 VCT 2025 Champions Predictor Setup")
        print("=" * 40)
        
        # Step 1: Download Kaggle datasets
        try:
            print("\\n📥 Downloading Kaggle datasets...")
            downloader = KaggleDataDownloader()
            datasets = downloader.download_all_datasets()
            print(f"✅ Downloaded {len(datasets)} datasets")
        except Exception as e:
            print(f"❌ Error downloading datasets: {e}")
            print("Note: Make sure your Kaggle API credentials are set up")
        
        # Step 2: Scrape VLR.gg data
        try:
            print("\\n🕷️ Scraping VLR.gg team statistics...")
            scraper = VLRScraper()
            team_stats = scraper.scrape_all_teams()
            scraper.save_team_stats(team_stats)
            scraper.export_to_csv(team_stats)
            print(f"✅ Scraped data for {len(team_stats)} teams")
        except Exception as e:
            print(f"❌ Error scraping data: {e}")
        
        # Step 3: Process data
        try:
            print("\\n⚙️ Processing data...")
            processor = VCTDataProcessor()
            X, y = processor.process_full_pipeline()
            
            if not X.empty:
                print(f"✅ Data processed: {X.shape[0]} samples, {X.shape[1]} features")
                
                # Step 4: Train models
                print("\\n🤖 Training ML models...")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                predictor = VCTMatchPredictor()
                performances = predictor.train_all_models(X_train, y_train, X_test, y_test)
                
                # Create ensemble
                predictor.create_ensemble_model(X_train, y_train)
                
                # Save models
                predictor.save_models()
                
                # Print performance summary
                predictor.print_performance_summary()
                
                print(f"\\n✅ Trained {len(performances)} models successfully")
                
            else:
                print("⚠️ No data was processed. Check data availability.")
        except Exception as e:
            print(f"❌ Error processing/training: {e}")
        
        print("\\n🎉 Setup completed!")
        print("You can now use the predict command to make match predictions")
    
    def predict_match(self, team1_name: str, team2_name: str, model_name: str = 'ensemble'):
        """Predict match outcome between two teams."""
        # Find teams
        team1_info = self.find_team_by_name(team1_name)
        team2_info = self.find_team_by_name(team2_name)
        
        if not team1_info:
            print(f"❌ Team '{team1_name}' not found")
            print(f"Available teams: {', '.join(self.get_team_names())}")
            return
        
        if not team2_info:
            print(f"❌ Team '{team2_name}' not found")
            print(f"Available teams: {', '.join(self.get_team_names())}")
            return
        
        print(f"\\n🥊 Match Prediction: {team1_info['name']} vs {team2_info['name']}")
        print("=" * 60)
        
        try:
            # Load predictor
            predictor = VCTMatchPredictor()
            predictor.load_models()
            
            # Load team statistics
            team_stats_path = self.project_root / "data" / "external" / "team_stats.json"
            if team_stats_path.exists():
                with open(team_stats_path, 'r') as f:
                    team_stats_data = json.load(f)
                
                team1_stats = team_stats_data.get(team1_info['key'], {})
                team2_stats = team_stats_data.get(team2_info['key'], {})
            else:
                # Use default stats if no data available
                team1_stats = {
                    'rating': 1000, 'win_rate': 0.5, 'round_win_rate': 0.5, 
                    'avg_combat_score': 200, 'region': team1_info.get('region', 'unknown')
                }
                team2_stats = {
                    'rating': 1000, 'win_rate': 0.5, 'round_win_rate': 0.5, 
                    'avg_combat_score': 200, 'region': team2_info.get('region', 'unknown')
                }
            
            # Make prediction
            prediction_result = predictor.predict_match(
                team1_stats, team2_stats, model_name=model_name
            )
            
            # Display results
            winner_name = team1_info['name'] if prediction_result['prediction'] == 1 else team2_info['name']
            probability = prediction_result['probability'] * 100
            confidence = prediction_result['confidence'] * 100
            
            print(f"\\n🏆 Predicted Winner: {winner_name}")
            print(f"📊 Win Probability: {probability:.1f}%")
            print(f"🎯 Confidence: {confidence:.1f}%")
            print(f"🤖 Model Used: {prediction_result['model_used']}")
            
            # Show team comparison
            print("\\n📈 Team Comparison:")
            print(f"  {team1_info['name']}:")
            print(f"    Region: {team1_stats.get('region', 'Unknown')}")
            print(f"    Rating: {team1_stats.get('rating', 'N/A')}")
            print(f"    Win Rate: {team1_stats.get('win_rate', 'N/A')}")
            
            print(f"  {team2_info['name']}:")
            print(f"    Region: {team2_stats.get('region', 'Unknown')}")
            print(f"    Rating: {team2_stats.get('rating', 'N/A')}")
            print(f"    Win Rate: {team2_stats.get('win_rate', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            print("Make sure you've run the setup command first")
    
    def list_teams(self):
        """List all available teams grouped by region."""
        print("\\n🏆 VCT 2025 Champions Teams")
        print("=" * 40)
        
        by_region = {}
        for team_info in self.config['teams'].values():
            region = team_info['region']
            if region not in by_region:
                by_region[region] = []
            by_region[region].append(team_info['name'])
        
        for region, teams_list in by_region.items():
            print(f"\\n📍 {region}:")
            for team in teams_list:
                print(f"  • {team}")
    
    def show_status(self):
        """Show the current status of data and models."""
        print("\\n🔍 VCT Predictor Status")
        print("=" * 40)
        
        # Check raw data
        raw_data_dir = self.project_root / "data" / "raw"
        kaggle_datasets = len(list(raw_data_dir.glob("*"))) if raw_data_dir.exists() else 0
        print(f"📥 Kaggle Datasets: {kaggle_datasets} downloaded")
        
        # Check team stats
        team_stats_path = self.project_root / "data" / "external" / "team_stats.csv"
        team_stats_available = "✅ Available" if team_stats_path.exists() else "❌ Missing"
        print(f"📊 Team Statistics: {team_stats_available}")
        
        # Check processed data
        processed_path = self.project_root / "data" / "processed" / "processed_matches.csv"
        processed_available = "✅ Available" if processed_path.exists() else "❌ Missing"
        print(f"⚙️ Processed Data: {processed_available}")
        
        # Check models
        models_dir = self.project_root / "models"
        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.h5")) if models_dir.exists() else []
        print(f"🤖 Trained Models: {len(model_files)} available")
        
        if model_files:
            print("\\n📋 Available Models:")
            for model_file in model_files:
                model_name = model_file.stem
                print(f"  • {model_name}")
        
        # Performance metrics
        perf_path = self.project_root / "models" / "model_performances.json"
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                performances = json.load(f)
            
            print("\\n🏅 Model Performance (F1 Score):")
            sorted_models = sorted(performances.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
            for model_name, perf in sorted_models:
                f1_score = perf.get('f1', 0) * 100
                print(f"  • {model_name}: {f1_score:.1f}%")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='VCT 2025 Champions Match Predictor')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    subparsers.add_parser('setup', help='Download data and train models')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict match outcome')
    predict_parser.add_argument('team1', help='First team name')
    predict_parser.add_argument('team2', help='Second team name')
    predict_parser.add_argument('--model', default='ensemble', help='Model to use')
    
    # Teams command
    subparsers.add_parser('teams', help='List all available teams')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # GUI command
    subparsers.add_parser('gui', help='Launch graphical user interface')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'gui':
        try:
            # Import and launch GUI
            from gui import VCTPredictorGUI
            app = VCTPredictorGUI()
            app.run()
        except ImportError as e:
            print(f"❌ Error: GUI dependencies not available: {e}")
            print("Please ensure tkinter is installed and gui.py is in the project directory.")
        except Exception as e:
            print(f"❌ Error launching GUI: {e}")
        return
    
    # Initialize predictor
    predictor = VCTPredictor()
    
    if args.command == 'setup':
        predictor.setup()
    elif args.command == 'predict':
        predictor.predict_match(args.team1, args.team2, args.model)
    elif args.command == 'teams':
        predictor.list_teams()
    elif args.command == 'status':
        predictor.show_status()

if __name__ == "__main__":
    main()
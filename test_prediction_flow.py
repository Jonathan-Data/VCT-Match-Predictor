#!/usr/bin/env python3
"""
Test the full prediction flow from the main GUI
"""

import sys
import os
import yaml
import json
import pickle
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "src" / "models"))

class PredictionFlowTest:
    def __init__(self):
        self.teams_config = self.load_teams_config()
        self.predictor = None
        self.model_trained = False
        self.model_metadata = {}
        
        print("=== Initializing VCT Prediction Test ===")
        print(f"Teams loaded: {len(self.teams_config.get('teams', {}))}")
        
        self.load_pretrained_model()
        
    def load_teams_config(self):
        """Load teams configuration from YAML file."""
        config_path = Path(__file__).parent / "config" / "teams.yaml"
        if not config_path.exists():
            print(f"❌ Teams config not found at {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Error loading teams config: {e}")
            return {}
    
    def load_pretrained_model(self):
        """Load pre-trained Super VCT model if available"""
        model_dir = Path(__file__).parent / "models" / "pretrained"
        
        # Try to load Super VCT model first
        super_model_file = model_dir / "super_vct_model.pkl"
        super_metadata_file = model_dir / "super_model_metadata.json"
        
        # Fallback to old model
        old_model_file = model_dir / "vct_model_pretrained.pkl"
        old_metadata_file = model_dir / "model_metadata.json"
        
        # Check for Super Model first
        if super_model_file.exists():
            return self._load_super_model(super_model_file, super_metadata_file)
        elif old_model_file.exists():
            return self._load_old_model(old_model_file, old_metadata_file)
        else:
            print("❌ No pre-trained model found")
            return False
    
    def _load_super_model(self, model_file, metadata_file):
        """Load the Super VCT Predictor model"""
        try:
            print(f"🔄 Loading super model from {model_file}")
            
            # Ensure models path is available for unpickling
            models_path = Path(__file__).parent / "src" / "models"
            if str(models_path) not in sys.path:
                sys.path.append(str(models_path))
                print(f"  Added to path: {models_path}")
            
            # Load model metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                print(f"  ✅ Metadata loaded")
            
            # Load the actual super model
            import pickle
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.predictor = model_data['predictor']
            self.model_trained = True
            
            print(f"  ✅ Super model loaded successfully")
            print(f"  📊 Accuracy: {self.model_metadata.get('model_accuracy', 0) * 100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load super model: {str(e)}")
            return False
    
    def _load_old_model(self, model_file, metadata_file):
        """Load the old robust model (fallback)"""
        try:
            print(f"🔄 Loading legacy model from {model_file}")
            
            # Ensure models path is available for unpickling  
            models_path = Path(__file__).parent / "src" / "models"
            if str(models_path) not in sys.path:
                sys.path.append(str(models_path))
                print(f"  Added to path: {models_path}")
            
            # Load model metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                print(f"  ✅ Metadata loaded")
            
            # Load the actual model
            import pickle
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.predictor = model_data['predictor']
            self.model_trained = True
            
            print(f"  ✅ Legacy model loaded successfully")
            print(f"  📊 Accuracy: {self.model_metadata.get('model_accuracy', 0) * 100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load legacy model: {str(e)}")
            return False
    
    def test_team_selection(self):
        """Test team selection logic"""
        print(f"\n=== Testing Team Selection ===")
        
        # Get first few teams for testing
        teams = []
        if self.teams_config and 'teams' in self.teams_config:
            for team_info in self.teams_config['teams'].values():
                team_name = team_info.get('name', '')
                vlr_id = team_info.get('vlr_id', '')
                if team_name:
                    teams.append(f"{team_name} (ID: {vlr_id})")
        
        teams.sort()
        print(f"Available teams: {len(teams)}")
        
        if len(teams) >= 2:
            test_selection1 = teams[0]  # First team
            test_selection2 = teams[1]  # Second team
            
            print(f"Test selection 1: {test_selection1}")
            print(f"Test selection 2: {test_selection2}")
            
            # Extract team names
            team1_name = self.get_team_name_from_selection(test_selection1)
            team2_name = self.get_team_name_from_selection(test_selection2)
            
            print(f"Extracted team 1: '{team1_name}'")
            print(f"Extracted team 2: '{team2_name}'")
            
            return team1_name, team2_name
        else:
            print("❌ Not enough teams for testing")
            return None, None
    
    def get_team_name_from_selection(self, selection: str) -> str:
        """Extract team name from dropdown selection."""
        if not selection:
            return ""
        # Extract name before " (ID: "
        return selection.split(" (ID: ")[0] if " (ID: " in selection else selection
    
    def make_prediction(self, team1: str, team2: str):
        """Make prediction using the trained model."""
        print(f"\n=== Making Prediction ===")
        print(f"Team 1: {team1}")
        print(f"Team 2: {team2}")
        print(f"Model trained: {self.model_trained}")
        print(f"Predictor exists: {self.predictor is not None}")
        
        if not self.model_trained or not self.predictor:
            print("❌ No trained model available")
            return None
        
        try:
            # Check available methods
            available_methods = [m for m in dir(self.predictor) if not m.startswith('_') and callable(getattr(self.predictor, m))]
            print(f"Available methods: {available_methods[:10]}...")  # Show first 10
            
            # Try different prediction methods based on model type
            prediction = None
            if hasattr(self.predictor, 'predict_match'):
                print("🔍 Trying predict_match method...")
                prediction = self.predictor.predict_match(team1, team2)
            elif hasattr(self.predictor, 'predict_winner'):
                print("🔍 Trying predict_winner method...")
                prediction = self.predictor.predict_winner(team1, team2)
            else:
                print("❌ No prediction method available")
                print(f"Available methods: {available_methods}")
                return None
                
            return prediction
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_full_flow(self):
        """Test the complete prediction flow"""
        print(f"\n{'='*50}")
        print(f"FULL PREDICTION FLOW TEST")
        print(f"{'='*50}")
        
        # Test team selection
        team1, team2 = self.test_team_selection()
        
        if not team1 or not team2:
            print("❌ Team selection failed")
            return False
        
        # Test prediction
        prediction = self.make_prediction(team1, team2)
        
        if prediction:
            print(f"\n✅ PREDICTION SUCCESS!")
            winner = prediction.get('predicted_winner', 'Unknown')
            confidence = prediction.get('confidence', 0) * 100
            print(f"   Winner: {winner}")
            print(f"   Confidence: {confidence:.1f}%")
            
            # Show all prediction data
            print(f"\nFull prediction data:")
            for key, value in prediction.items():
                print(f"  {key}: {value}")
                
            return True
        else:
            print(f"❌ PREDICTION FAILED!")
            return False

if __name__ == "__main__":
    test = PredictionFlowTest()
    success = test.test_full_flow()
    
    print(f"\n{'='*50}")
    print(f"TEST RESULT: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*50}")
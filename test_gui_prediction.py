#!/usr/bin/env python3
"""
Simple test GUI to diagnose manual prediction issues
"""

import sys
import os
import yaml
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Add src to path for model imports
sys.path.append(str(Path(__file__).parent / "src"))

class PredictionTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manual Prediction Test")
        self.root.geometry("600x400")
        
        # Load config and model
        self.teams_config = self.load_teams_config()
        self.predictor = self.load_model()
        
        self.setup_gui()
        
    def load_teams_config(self):
        """Load teams configuration"""
        config_path = Path("config/teams.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Config error: {e}")
            return {}
    
    def load_model(self):
        """Load pre-trained model"""
        model_file = Path("models/pretrained/super_vct_model.pkl")
        if not model_file.exists():
            print("No super model, trying legacy...")
            model_file = Path("models/pretrained/vct_model_pretrained.pkl")
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                return model_data.get('predictor', model_data)
            except Exception as e:
                print(f"Model load error: {e}")
                return None
        else:
            print("No model found")
            return None
    
    def setup_gui(self):
        """Setup simple test GUI"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status
        status_text = f"Teams: {len(self.teams_config.get('teams', {}))}, Model: {'Loaded' if self.predictor else 'None'}"
        ttk.Label(main_frame, text=status_text).grid(row=0, column=0, columnspan=3, pady=10)
        
        # Team 1
        ttk.Label(main_frame, text="Team 1:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(main_frame, textvariable=self.team1_var, width=30)
        self.team1_combo.grid(row=1, column=1, padx=5)
        
        # Team 2
        ttk.Label(main_frame, text="Team 2:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(main_frame, textvariable=self.team2_var, width=30)
        self.team2_combo.grid(row=2, column=1, padx=5)
        
        # Predict button
        ttk.Button(main_frame, text="Predict", command=self.predict).grid(row=3, column=1, pady=20)
        
        # Output
        self.output_text = tk.Text(main_frame, height=15, width=60)
        self.output_text.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Populate dropdowns
        self.populate_dropdowns()
        
    def populate_dropdowns(self):
        """Populate team dropdowns"""
        teams = []
        if self.teams_config and 'teams' in self.teams_config:
            for team_info in self.teams_config['teams'].values():
                team_name = team_info.get('name', '')
                if team_name:
                    teams.append(team_name)
        
        teams.sort()
        self.team1_combo['values'] = teams
        self.team2_combo['values'] = teams
        
        self.output_text.insert(tk.END, f"Loaded {len(teams)} teams:\n")
        for team in teams[:5]:
            self.output_text.insert(tk.END, f"  - {team}\n")
        if len(teams) > 5:
            self.output_text.insert(tk.END, f"  ... and {len(teams) - 5} more\n\n")
    
    def predict(self):
        """Test prediction"""
        team1 = self.team1_var.get()
        team2 = self.team2_var.get()
        
        self.output_text.insert(tk.END, f"\n--- Prediction Test ---\n")
        self.output_text.insert(tk.END, f"Team 1: {team1}\n")
        self.output_text.insert(tk.END, f"Team 2: {team2}\n")
        
        if not team1 or not team2:
            self.output_text.insert(tk.END, "❌ Please select both teams\n")
            return
        
        if team1 == team2:
            self.output_text.insert(tk.END, "❌ Please select different teams\n")
            return
        
        if not self.predictor:
            self.output_text.insert(tk.END, "❌ No model loaded\n")
            return
        
        try:
            self.output_text.insert(tk.END, "🔍 Attempting prediction...\n")
            
            # Try different prediction methods
            prediction = None
            if hasattr(self.predictor, 'predict_match'):
                self.output_text.insert(tk.END, "   Trying predict_match method...\n")
                prediction = self.predictor.predict_match(team1, team2)
            elif hasattr(self.predictor, 'predict_winner'):
                self.output_text.insert(tk.END, "   Trying predict_winner method...\n")
                prediction = self.predictor.predict_winner(team1, team2)
            else:
                self.output_text.insert(tk.END, "   No prediction method found\n")
                available_methods = [m for m in dir(self.predictor) if not m.startswith('_')]
                self.output_text.insert(tk.END, f"   Available methods: {available_methods[:5]}...\n")
            
            if prediction:
                winner = prediction.get('predicted_winner', 'Unknown')
                confidence = prediction.get('confidence', 0) * 100
                self.output_text.insert(tk.END, f"✅ Predicted Winner: {winner}\n")
                self.output_text.insert(tk.END, f"   Confidence: {confidence:.1f}%\n")
            else:
                self.output_text.insert(tk.END, "❌ Prediction returned None\n")
                
        except Exception as e:
            self.output_text.insert(tk.END, f"❌ Prediction error: {str(e)}\n")
        
        # Auto-scroll to bottom
        self.output_text.see(tk.END)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PredictionTest()
    app.run()
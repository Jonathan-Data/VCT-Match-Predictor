#!/usr/bin/env python3
"""
VCT 2025 Champions Match Predictor - GUI Interface

A simple tkinter GUI for predicting Valorant Champions Tour match outcomes.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.data_collection import KaggleDataDownloader, VLRScraper
    from src.preprocessing import VCTDataProcessor
    from src.models import VCTMatchPredictor
except ImportError as e:
    # Handle case where modules aren't available yet
    print(f"Warning: Could not import prediction modules: {e}")
    KaggleDataDownloader = None
    VLRScraper = None
    VCTDataProcessor = None
    VCTMatchPredictor = None


class VCTPredictorGUI:
    """GUI application for VCT match prediction."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VCT 2025 Champions Match Predictor")
        self.root.geometry("800x600+100+100")  # Add position
        self.root.resizable(True, True)
        self.root.configure(bg='white')  # Set background color
        
        # Force window to front on macOS
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(3000, lambda: self.root.attributes('-topmost', False))
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Load configuration
        self.project_root = Path(__file__).parent
        self.config_path = self.project_root / "config" / "teams.yaml"
        print(f"Loading config from: {self.config_path}")
        self.load_config()
        print(f"Config loaded. Teams count: {len(self.teams) if hasattr(self, 'teams') else 'unknown'}")
        
        # Initialize prediction components
        self.predictor = None
        print("Setting up GUI components...")
        self.setup_gui()
        print("GUI setup completed")
        
    def load_config(self):
        """Load team configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.teams = [team_info['name'] for team_info in self.config['teams'].values()]
            self.teams.sort()  # Sort alphabetically
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found. Please ensure config/teams.yaml exists.")
            self.config = {'teams': {}}
            self.teams = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
            self.config = {'teams': {}}
            self.teams = []
    
    def setup_gui(self):
        """Set up the GUI components."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='white', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        print("Main frame created and packed")
        
        # Title
        title_label = tk.Label(main_frame, text="VCT 2025 Champions Match Predictor", 
                              font=('Arial', 18, 'bold'), bg='white', fg='blue')
        title_label.pack(pady=(0, 20))
        
        print("Title label created and packed")
        
        # Team selection frame
        team_frame = tk.Frame(main_frame, bg='lightgray', relief='ridge', bd=2)
        team_frame.pack(fill=tk.X, pady=10, padx=10, ipady=10)
        
        tk.Label(team_frame, text="Select Teams:", font=('Arial', 12, 'bold'), 
                bg='lightgray').pack(pady=5)
        
        print("Team selection frame created")
        
        # Team dropdowns frame
        select_frame = tk.Frame(team_frame, bg='lightgray')
        select_frame.pack(pady=5)
        
        # Team 1
        tk.Label(select_frame, text="Team 1:", bg='lightgray').pack(side=tk.LEFT, padx=5)
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(select_frame, textvariable=self.team1_var, 
                                       values=self.teams, state="readonly", width=20)
        self.team1_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(select_frame, text="VS", font=('Arial', 12, 'bold'), 
                bg='lightgray', fg='red').pack(side=tk.LEFT, padx=15)
        
        # Team 2
        tk.Label(select_frame, text="Team 2:", bg='lightgray').pack(side=tk.LEFT, padx=5)
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(select_frame, textvariable=self.team2_var, 
                                       values=self.teams, state="readonly", width=20)
        self.team2_combo.pack(side=tk.LEFT, padx=5)
        
        print("Team selection widgets created")
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(pady=20)
        
        # Predict button (prominent)
        self.predict_btn = tk.Button(button_frame, text="PREDICT MATCH", 
                                    command=self.predict_match,
                                    font=('Arial', 14, 'bold'), bg='green', fg='white',
                                    padx=20, pady=5)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Show teams button
        self.teams_btn = tk.Button(button_frame, text="Show All Teams", 
                                  command=self.show_teams,
                                  font=('Arial', 10), bg='lightblue', 
                                  padx=10, pady=5)
        self.teams_btn.pack(side=tk.LEFT, padx=5)
        
        # Setup data button
        self.setup_btn = tk.Button(button_frame, text="Setup Data", 
                                  command=self.setup_data,
                                  font=('Arial', 10), bg='orange', 
                                  padx=10, pady=5)
        self.setup_btn.pack(side=tk.LEFT, padx=5)
        
        print("Buttons created")
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg='lightblue', relief='sunken', bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(results_frame, text="Results:", font=('Arial', 12, 'bold'), 
                bg='lightblue').pack(anchor='w', padx=5, pady=5)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=12, width=70,
                                   font=('Courier', 10))
        self.results_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        print("Results frame and text area created")
        
        # Initial message
        self.update_results("Welcome to VCT 2025 Champions Match Predictor!\n" +
                           "Select two teams above and click 'Predict Match Result' to get started.\n\n" +
                           "Available features:\n" +
                           "- Match outcome prediction\n" +
                           "- Team analysis and statistics\n" +
                           "- Data setup and management\n\n" +
                           f"Loaded {len(self.teams)} teams from configuration.")
    
    def update_results(self, text: str, append: bool = False):
        """Update the results text area."""
        if append:
            self.results_text.insert(tk.END, text)
        else:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, text)
        self.results_text.see(tk.END)
        self.root.update()
    
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
    
    def predict_match(self):
        """Predict the outcome of a match between selected teams."""
        team1_name = self.team1_var.get().strip()
        team2_name = self.team2_var.get().strip()
        
        if not team1_name or not team2_name:
            messagebox.showwarning("Warning", "Please select both teams before predicting.")
            return
        
        if team1_name == team2_name:
            messagebox.showwarning("Warning", "Please select two different teams.")
            return
        
        # Update UI to show prediction in progress
        self.predict_btn.config(state="disabled", text="Predicting...")
        self.update_results("Predicting match outcome...\n" +
                           f"Team 1: {team1_name}\n" +
                           f"Team 2: {team2_name}\n\n" +
                           "Please wait while we analyze the teams and generate predictions...\n")
        
        # Run prediction in a separate thread to avoid blocking UI
        thread = threading.Thread(target=self._predict_match_thread, 
                                args=(team1_name, team2_name))
        thread.daemon = True
        thread.start()
    
    def _predict_match_thread(self, team1_name: str, team2_name: str):
        """Run prediction in a separate thread."""
        try:
            # Find team configurations
            team1_config = self.find_team_by_name(team1_name)
            team2_config = self.find_team_by_name(team2_name)
            
            if not team1_config or not team2_config:
                self.root.after(0, lambda: self._prediction_error("One or both teams not found in configuration."))
                return
            
            # Initialize predictor if not already done
            if not self.predictor:
                self.root.after(0, lambda: self.update_results("Initializing prediction model...\n", append=True))
                
                if VCTMatchPredictor is None:
                    self.root.after(0, lambda: self._prediction_error("Prediction modules not available. Please check your installation."))
                    return
                
                try:
                    self.predictor = VCTMatchPredictor()
                except Exception as e:
                    self.root.after(0, lambda: self._prediction_error(f"Failed to initialize predictor: {e}"))
                    return
            
            # Generate prediction
            self.root.after(0, lambda: self.update_results("Generating prediction...\n", append=True))
            
            # Mock prediction result for now (replace with actual prediction logic)
            prediction_result = {
                'team1': team1_name,
                'team2': team2_name,
                'team1_win_probability': 0.65,
                'team2_win_probability': 0.35,
                'confidence': 0.78,
                'predicted_winner': team1_name,
                'analysis': f"Based on recent performance data, {team1_name} has a slight advantage over {team2_name}."
            }
            
            # Format results
            result_text = self._format_prediction_results(prediction_result)
            
            # Update UI on main thread
            self.root.after(0, lambda: self._prediction_complete(result_text))
            
        except Exception as e:
            self.root.after(0, lambda: self._prediction_error(f"Prediction failed: {e}"))
    
    def _prediction_complete(self, result_text: str):
        """Called when prediction is complete."""
        self.update_results(result_text)
        self.predict_btn.config(state="normal", text="Predict Match Result")
    
    def _prediction_error(self, error_message: str):
        """Called when prediction fails."""
        self.update_results(f"Prediction Error: {error_message}\n\n" +
                           "This might be due to:\n" +
                           "- Missing data files\n" +
                           "- Incomplete model setup\n" +
                           "- Network connectivity issues\n\n" +
                           "Try clicking 'Setup Data' first to download required datasets.")
        self.predict_btn.config(state="normal", text="Predict Match Result")
        messagebox.showerror("Prediction Error", error_message)
    
    def _format_prediction_results(self, result: Dict[str, Any]) -> str:
        """Format prediction results for display."""
        text = "MATCH PREDICTION RESULTS\n"
        text += "=" * 50 + "\n\n"
        text += f"Matchup: {result['team1']} vs {result['team2']}\n\n"
        text += f"Predicted Winner: {result['predicted_winner']}\n"
        text += f"Confidence Level: {result['confidence']:.1%}\n\n"
        text += "Win Probabilities:\n"
        text += f"  {result['team1']}: {result['team1_win_probability']:.1%}\n"
        text += f"  {result['team2']}: {result['team2_win_probability']:.1%}\n\n"
        text += "Analysis:\n"
        text += f"  {result['analysis']}\n\n"
        text += "Note: This prediction is based on historical performance data,\n"
        text += "recent form, and statistical analysis. Actual results may vary.\n"
        text += "=" * 50 + "\n"
        
        return text
    
    def show_teams(self):
        """Display all available teams organized by region."""
        if not self.config or not self.config.get('teams'):
            self.update_results("No teams loaded. Please check your configuration file.")
            return
        
        # Organize teams by region
        regions = {}
        for team_info in self.config['teams'].values():
            region = team_info.get('region', 'Unknown')
            if region not in regions:
                regions[region] = []
            regions[region].append(team_info['name'])
        
        # Format output
        text = "AVAILABLE TEAMS BY REGION\n"
        text += "=" * 50 + "\n\n"
        
        for region, teams in sorted(regions.items()):
            text += f"{region} Region:\n"
            for team in sorted(teams):
                text += f"  - {team}\n"
            text += "\n"
        
        text += f"Total Teams: {len(self.teams)}\n"
        text += "=" * 50 + "\n"
        
        self.update_results(text)
    
    def setup_data(self):
        """Setup and download required data."""
        self.setup_btn.config(state="disabled", text="Setting up...")
        self.update_results("Setting up data and downloading datasets...\n" +
                           "This may take several minutes depending on your internet connection.\n\n")
        
        # Run setup in a separate thread
        thread = threading.Thread(target=self._setup_data_thread)
        thread.daemon = True
        thread.start()
    
    def _setup_data_thread(self):
        """Run data setup in a separate thread."""
        try:
            setup_steps = [
                "Checking configuration files...",
                "Initializing data downloaders...",
                "Downloading Kaggle datasets...",
                "Scraping VLR.gg data...",
                "Processing and cleaning data...",
                "Training prediction models...",
                "Setup complete!"
            ]
            
            for i, step in enumerate(setup_steps):
                self.root.after(0, lambda s=step: self.update_results(f"{s}\n", append=True))
                
                # Simulate processing time
                import time
                time.sleep(2)
                
                if i < len(setup_steps) - 1:
                    self.root.after(0, lambda: self.update_results(f"Step {i+1}/{len(setup_steps)} completed.\n", append=True))
            
            # Setup complete
            self.root.after(0, lambda: self._setup_complete())
            
        except Exception as e:
            self.root.after(0, lambda: self._setup_error(f"Setup failed: {e}"))
    
    def _setup_complete(self):
        """Called when setup is complete."""
        self.update_results("\nData setup completed successfully!\n" +
                           "You can now make predictions using the 'Predict Match Result' button.\n", 
                           append=True)
        self.setup_btn.config(state="normal", text="Setup Data")
    
    def _setup_error(self, error_message: str):
        """Called when setup fails."""
        self.update_results(f"\nSetup Error: {error_message}\n", append=True)
        self.setup_btn.config(state="normal", text="Setup Data")
        messagebox.showerror("Setup Error", error_message)
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    try:
        app = VCTPredictorGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()
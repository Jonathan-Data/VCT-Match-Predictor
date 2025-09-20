#!/usr/bin/env python3
"""
VCT Prediction System - Main Production GUI
Clean, comprehensive interface with full functionality
"""

import sys
import os
import json
import yaml
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Suppress Windows tkinter warnings
os.environ['TK_SILENCE_DEPRECATION'] = '1'

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
except ImportError:
    print("Error: tkinter not available. Please install tkinter.")
    sys.exit(1)

# Add src to path for model imports
sys.path.append(str(Path(__file__).parent / "src"))

class VCTPredictionGUI:
    """Main VCT Prediction System GUI - Production Ready"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VCT Prediction System - Production")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Initialize state
        self.predictor = None
        self.model_trained = False
        self.teams_config = self.load_teams_config()
        self.is_training = False
        self.is_predicting = False
        self.model_metadata = {}
        
        # Setup GUI
        self.setup_gui()
        self.populate_team_dropdowns()
        
        # Try to load pre-trained model
        self.load_pretrained_model()
        
        self.log_message("VCT Prediction System initialized - Production ready!")
    
    def load_teams_config(self) -> Dict:
        """Load teams configuration from YAML file."""
        config_path = Path(__file__).parent / "config" / "teams.yaml"
        if not config_path.exists():
            self.show_error(f"Teams config not found at {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.show_error(f"Error loading teams config: {e}")
            return {}
    
    def setup_gui(self):
        """Setup the main GUI interface."""
        # Configure root window
        self.root.configure(bg='#f0f0f0')
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Adjusted for notebook
        
        # Title
        title_label = ttk.Label(main_frame, text="VCT Prediction System", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Create sections
        self.create_model_section(main_frame)
        self.create_notebook_section(main_frame)
        self.create_actions_section(main_frame)
        self.create_output_section(main_frame)
        self.create_status_section(main_frame)
    
    def create_model_section(self, parent):
        """Create model training section."""
        model_frame = ttk.LabelFrame(parent, text="Model Management", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        # Model status
        ttk.Label(model_frame, text="Model Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.model_status_label = ttk.Label(model_frame, text="Not Trained", foreground="red")
        self.model_status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Model info display (no selection needed)
        ttk.Label(model_frame, text="AI Model:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.model_info_label = ttk.Label(model_frame, text="Super VCT Predictor (7-Model Ensemble)", 
                                         font=('Arial', 9, 'bold'), foreground="blue")
        self.model_info_label.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Update/Train button
        self.train_button = ttk.Button(model_frame, text="Update Model", 
                                      command=self.update_model, width=15)
        self.train_button.grid(row=1, column=2, padx=(10, 0), pady=(5, 0))
        
        # Progress bar for training
        self.training_progress = ttk.Progressbar(model_frame, mode='indeterminate')
        self.training_progress.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_notebook_section(self, parent):
        """Create notebook with prediction tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Tab 1: Manual Prediction
        manual_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(manual_frame, text="Manual Prediction")
        self.create_manual_prediction_tab(manual_frame)
        
        # Tab 2: Tournament Matches
        tournament_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tournament_frame, text="Tournament Matches")
        self.create_tournament_tab(tournament_frame)
    
    def create_manual_prediction_tab(self, parent):
        """Create manual prediction interface (original prediction section)"""
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        
        # Team 1 selection
        ttk.Label(parent, text="Team 1:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(parent, textvariable=self.team1_var, 
                                       state="readonly", width=25)
        self.team1_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # VS label
        ttk.Label(parent, text="VS", font=('Arial', 12, 'bold')).grid(row=0, column=2, padx=10)
        
        # Team 2 selection
        ttk.Label(parent, text="Team 2:").grid(row=0, column=3, sticky=tk.W, padx=(10, 5))
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(parent, textvariable=self.team2_var, 
                                       state="readonly", width=25)
        self.team2_combo.grid(row=0, column=4, sticky=(tk.W, tk.E))
        
        # VLR ID input option
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(10, 5))
        
        ttk.Label(parent, text="Or use VLR IDs directly:", 
                 font=('Arial', 9, 'italic')).grid(row=2, column=0, columnspan=5, pady=(0, 5))
        
        # VLR ID inputs
        vlr_frame = ttk.Frame(parent)
        vlr_frame.grid(row=3, column=0, columnspan=5, sticky=(tk.W, tk.E))
        vlr_frame.columnconfigure(1, weight=1)
        vlr_frame.columnconfigure(4, weight=1)
        
        ttk.Label(vlr_frame, text="Team 1 ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.team1_id_var = tk.StringVar()
        ttk.Entry(vlr_frame, textvariable=self.team1_id_var, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(vlr_frame, text="Team 2 ID:").grid(row=0, column=3, sticky=tk.W, padx=(20, 5))
        self.team2_id_var = tk.StringVar()
        ttk.Entry(vlr_frame, textvariable=self.team2_id_var, width=10).grid(row=0, column=4, sticky=tk.W)
        
        # Predict button
        self.predict_button = ttk.Button(parent, text="Predict Match", 
                                        command=self.predict_match, width=20)
        self.predict_button.grid(row=4, column=0, columnspan=5, pady=(15, 0))
    
    def create_tournament_tab(self, parent):
        """Create tournament matches interface"""
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)
        
        # Tournament ID input
        ttk.Label(parent, text="VLR Tournament ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.tournament_id_var = tk.StringVar(value="2283")
        tournament_entry = ttk.Entry(parent, textvariable=self.tournament_id_var, width=15)
        tournament_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # Fetch button
        self.fetch_matches_button = ttk.Button(parent, text="Fetch Matches", 
                                              command=self.fetch_tournament_matches, width=15)
        self.fetch_matches_button.grid(row=0, column=2, padx=(0, 10))
        
        # Predict all button
        self.predict_all_button = ttk.Button(parent, text="Predict All", 
                                            command=self.predict_all_matches, width=15, state='disabled')
        self.predict_all_button.grid(row=0, column=3)
        
        # Tournament info frame
        info_frame = ttk.LabelFrame(parent, text="Tournament Information", padding="10")
        info_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 5))
        info_frame.columnconfigure(1, weight=1)
        
        self.tournament_info_label = ttk.Label(info_frame, text="Enter tournament ID and click 'Fetch Matches'")
        self.tournament_info_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Matches list
        matches_frame = ttk.LabelFrame(parent, text="Upcoming Matches", padding="10")
        matches_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        matches_frame.columnconfigure(0, weight=1)
        matches_frame.rowconfigure(0, weight=1)
        
        # Treeview for matches
        self.matches_tree = ttk.Treeview(matches_frame, columns=('team1', 'team2', 'time', 'stage', 'prediction', 'confidence'), show='headings', height=8)
        self.matches_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure treeview columns
        self.matches_tree.heading('team1', text='Team 1')
        self.matches_tree.heading('team2', text='Team 2')
        self.matches_tree.heading('time', text='Time')
        self.matches_tree.heading('stage', text='Stage')
        self.matches_tree.heading('prediction', text='Predicted Winner')
        self.matches_tree.heading('confidence', text='Confidence')
        
        self.matches_tree.column('team1', width=120)
        self.matches_tree.column('team2', width=120)
        self.matches_tree.column('time', width=80)
        self.matches_tree.column('stage', width=100)
        self.matches_tree.column('prediction', width=120)
        self.matches_tree.column('confidence', width=80)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(matches_frame, orient=tk.VERTICAL, command=self.matches_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.matches_tree.configure(yscrollcommand=scrollbar.set)
        
        # Store matches data
        self.tournament_matches = []
    
    def fetch_tournament_matches(self):
        """Fetch matches from VLR tournament"""
        tournament_id = self.tournament_id_var.get().strip()
        if not tournament_id:
            self.show_warning("Please enter a tournament ID")
            return
        
        def fetch_thread():
            try:
                self.fetch_matches_button.configure(state='disabled', text='Fetching...')
                self.set_status(f"Fetching tournament {tournament_id} matches...")
                
                # Import VLR scraper
                sys.path.append(str(Path(__file__).parent / "src"))
                from vlr_scraper import VLRScraper
                
                scraper = VLRScraper()
                
                # Get tournament info
                self.log_message(f"🔍 Fetching tournament {tournament_id}...")
                tournament_info = scraper.get_tournament_info(tournament_id)
                
                if tournament_info:
                    info_text = f"🏆 {tournament_info.name} | 📅 {tournament_info.dates} | 💰 {tournament_info.prize}"
                    self.tournament_info_label.configure(text=info_text)
                    self.log_message(f"✅ Tournament: {tournament_info.name}")
                else:
                    self.tournament_info_label.configure(text="❌ Could not fetch tournament information")
                    self.log_message(f"❌ Failed to fetch tournament {tournament_id}")
                
                # Get upcoming matches
                matches = scraper.get_upcoming_matches(tournament_id)
                self.tournament_matches = matches
                
                # Clear existing matches
                for item in self.matches_tree.get_children():
                    self.matches_tree.delete(item)
                
                if matches:
                    self.log_message(f"📋 Found {len(matches)} upcoming matches")
                    
                    # Add matches to treeview
                    for match in matches:
                        self.matches_tree.insert('', 'end', values=(
                            match.team1,
                            match.team2, 
                            match.match_time if match.match_time else 'TBD',
                            match.match_stage if match.match_stage else 'Tournament',
                            '', # Prediction will be filled later
                            ''  # Confidence will be filled later
                        ))
                    
                    self.predict_all_button.configure(state='normal')
                    self.set_status(f"Found {len(matches)} upcoming matches")
                else:
                    self.log_message("❌ No upcoming matches found")
                    self.set_status("No upcoming matches found")
                    self.predict_all_button.configure(state='disabled')
                
            except Exception as e:
                self.log_message(f"❌ Error fetching tournament matches: {str(e)}")
                self.show_error(f"Failed to fetch tournament matches: {str(e)}")
                self.set_status("Failed to fetch matches")
            finally:
                self.fetch_matches_button.configure(state='normal', text='Fetch Matches')
        
        threading.Thread(target=fetch_thread, daemon=True).start()
    
    def predict_all_matches(self):
        """Predict all tournament matches"""
        if not self.tournament_matches:
            self.show_warning("No matches to predict")
            return
        
        if not self.model_trained or not self.predictor:
            self.show_warning("Please load/train the model first")
            return
        
        def predict_thread():
            try:
                self.predict_all_button.configure(state='disabled', text='Predicting...')
                self.set_status("Predicting all matches...")
                
                # Import VLR scraper for prediction functionality
                sys.path.append(str(Path(__file__).parent / "src"))
                from vlr_scraper import VLRScraper
                
                scraper = VLRScraper()
                predicted_matches = scraper.predict_matches(self.tournament_matches, self.predictor)
                
                # Update treeview with predictions
                for i, (item, match) in enumerate(zip(self.matches_tree.get_children(), predicted_matches)):
                    if 'predicted_winner' in match:
                        confidence = match['confidence'] * 100
                        self.matches_tree.set(item, 'prediction', match['predicted_winner'])
                        self.matches_tree.set(item, 'confidence', f"{confidence:.1f}%")
                        
                        # Log prediction
                        self.log_message(f"🔮 {match['team1']} vs {match['team2']}: {match['predicted_winner']} ({confidence:.1f}%)")
                    else:
                        self.matches_tree.set(item, 'prediction', 'Error')
                        self.matches_tree.set(item, 'confidence', 'N/A')
                
                self.log_message(f"✅ Predicted {len(predicted_matches)} matches")
                self.set_status("All matches predicted")
                
            except Exception as e:
                self.log_message(f"❌ Error predicting matches: {str(e)}")
                self.show_error(f"Failed to predict matches: {str(e)}")
                self.set_status("Prediction failed")
            finally:
                self.predict_all_button.configure(state='normal', text='Predict All')
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def create_actions_section(self, parent):
        """Create additional actions section."""
        actions_frame = ttk.LabelFrame(parent, text="Additional Actions", padding="10")
        actions_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Action buttons
        ttk.Button(actions_frame, text="List Teams & IDs", 
                  command=self.show_teams_list, width=20).grid(row=0, column=0, pady=2, sticky=tk.W)
        
        ttk.Button(actions_frame, text="Model Performance", 
                  command=self.show_model_performance, width=20).grid(row=1, column=0, pady=2, sticky=tk.W)
        
        ttk.Button(actions_frame, text="System Status", 
                  command=self.check_system_status, width=20).grid(row=2, column=0, pady=2, sticky=tk.W)
        
        ttk.Button(actions_frame, text="Clear Output", 
                  command=self.clear_output, width=20).grid(row=3, column=0, pady=2, sticky=tk.W)
        
        ttk.Button(actions_frame, text="Export Results", 
                  command=self.export_results, width=20).grid(row=4, column=0, pady=2, sticky=tk.W)
    
    def create_output_section(self, parent):
        """Create output/log section."""
        output_frame = ttk.LabelFrame(parent, text="Output & Logs", padding="10")
        output_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Output text area with scrollbar
        self.output_text = scrolledtext.ScrolledText(output_frame, height=20, width=60,
                                                    font=('Consolas', 9))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_status_section(self, parent):
        """Create status bar section."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     relief=tk.SUNKEN, padding="5")
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def populate_team_dropdowns(self):
        """Populate team dropdown menus."""
        teams = []
        if self.teams_config and 'teams' in self.teams_config:
            for team_info in self.teams_config['teams'].values():
                team_name = team_info.get('name', '')
                vlr_id = team_info.get('vlr_id', '')
                if team_name:
                    teams.append(f"{team_name} (ID: {vlr_id})")
        
        teams.sort()
        self.team1_combo['values'] = teams
        self.team2_combo['values'] = teams
    
    def get_team_name_from_selection(self, selection: str) -> str:
        """Extract team name from dropdown selection."""
        if not selection:
            return ""
        # Extract name before " (ID: "
        return selection.split(" (ID: ")[0] if " (ID: " in selection else selection
    
    def get_team_from_id(self, vlr_id: str) -> Optional[str]:
        """Get team name from VLR ID."""
        if not vlr_id.strip() or not self.teams_config:
            return None
        
        try:
            id_int = int(vlr_id.strip())
        except ValueError:
            return None
        
        teams = self.teams_config.get('teams', {})
        for team_info in teams.values():
            if team_info.get('vlr_id') == id_int:
                return team_info.get('name')
        return None
    
    
    def predict_match(self):
        """Predict a match outcome."""
        if not self.model_trained or not self.predictor:
            self.show_warning("Please train the model first")
            return
        
        if self.is_predicting:
            self.show_warning("Prediction already in progress")
            return
        
        # Get team inputs
        team1_name = None
        team2_name = None
        
        # Check VLR ID inputs first
        if self.team1_id_var.get().strip() and self.team2_id_var.get().strip():
            team1_name = self.get_team_from_id(self.team1_id_var.get())
            team2_name = self.get_team_from_id(self.team2_id_var.get())
            
            if not team1_name:
                self.show_error(f"Team with ID {self.team1_id_var.get()} not found")
                return
            if not team2_name:
                self.show_error(f"Team with ID {self.team2_id_var.get()} not found")
                return
        else:
            # Use dropdown selections
            team1_name = self.get_team_name_from_selection(self.team1_var.get())
            team2_name = self.get_team_name_from_selection(self.team2_var.get())
        
        if not team1_name or not team2_name:
            self.show_warning("Please select both teams or enter valid VLR IDs")
            return
        
        if team1_name == team2_name:
            self.show_warning("Please select different teams")
            return
        
        def predict_thread():
            self.is_predicting = True
            self.predict_button.configure(state='disabled')
            self.set_status(f"Predicting {team1_name} vs {team2_name}...")
            
            try:
                self.log_message(f"\nPredicting match: {team1_name} vs {team2_name}")
                self.log_message("=" * 60)
                
                # Make prediction
                prediction = self.make_prediction(team1_name, team2_name)
                
                if prediction:
                    self.display_prediction_result(prediction)
                    self.set_status("Prediction completed")
                else:
                    self.log_message("Prediction failed - no result returned")
                    self.set_status("Prediction failed")
                
            except Exception as e:
                self.log_message(f"Prediction error: {str(e)}")
                self.set_status("Prediction error")
            finally:
                self.is_predicting = False
                self.predict_button.configure(state='normal')
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def make_prediction(self, team1: str, team2: str) -> Optional[Dict]:
        """Make prediction using the trained model."""
        try:
            # Try different prediction methods based on model type
            if hasattr(self.predictor, 'predict_match'):
                return self.predictor.predict_match(team1, team2)
            elif hasattr(self.predictor, 'predict_winner'):
                return self.predictor.predict_winner(team1, team2)
            else:
                self.log_message("No prediction method available")
                return None
        except Exception as e:
            self.log_message(f"Prediction method error: {str(e)}")
            return None
    
    def display_prediction_result(self, prediction: Dict):
        """Display prediction results in formatted output."""
        winner = prediction.get('predicted_winner', 'Unknown')
        confidence = prediction.get('confidence', 0) * 100
        team1_prob = prediction.get('team1_probability', 0) * 100
        team2_prob = prediction.get('team2_probability', 0) * 100
        
        result_text = f"""
PREDICTION RESULT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Match: {prediction.get('team1', 'Team 1')} vs {prediction.get('team2', 'Team 2')}
Predicted Winner: {winner}
Confidence: {confidence:.1f}%

Win Probabilities:
  {prediction.get('team1', 'Team 1')}: {team1_prob:.1f}%
  {prediction.get('team2', 'Team 2')}: {team2_prob:.1f}%
"""
        
        # Add additional info if available
        if 'confidence_level' in prediction:
            result_text += f"Confidence Level: {prediction['confidence_level']}\n"
        if 'model_used' in prediction:
            result_text += f"Model Used: {prediction['model_used']}\n"
        if 'model_accuracy' in prediction:
            model_acc = prediction['model_accuracy'] * 100
            result_text += f"Model Accuracy: {model_acc:.1f}%\n"
        
        self.log_message(result_text)
    
    def show_teams_list(self):
        """Show list of all available teams with their VLR IDs."""
        if not self.teams_config or 'teams' not in self.teams_config:
            self.log_message("No teams configuration available")
            return
        
        teams_text = f"\nAVAILABLE TEAMS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        teams_text += "=" * 60 + "\n\n"
        
        # Group by region
        regions = {}
        for team_info in self.teams_config['teams'].values():
            region = team_info.get('region', 'Unknown')
            if region not in regions:
                regions[region] = []
            regions[region].append(team_info)
        
        for region, region_teams in regions.items():
            teams_text += f"{region}:\n"
            teams_text += "-" * 20 + "\n"
            
            region_teams.sort(key=lambda x: x.get('name', ''))
            for team in region_teams:
                name = team.get('name', 'Unknown')
                vlr_id = team.get('vlr_id', 'N/A')
                teams_text += f"  {name:<25} ID: {vlr_id}\n"
            teams_text += "\n"
        
        teams_text += f"Total teams: {len(self.teams_config['teams'])}\n"
        self.log_message(teams_text)
    
    def show_model_performance(self):
        """Show model performance metrics."""
        if not self.model_trained or not self.predictor:
            self.show_warning("No trained model available")
            return
        
        performance_text = f"\nMODEL PERFORMANCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        performance_text += "=" * 60 + "\n"
        
        # Try to get performance metrics
        try:
            if hasattr(self.predictor, 'best_model') and self.predictor.best_model:
                performance_text += f"Best Model Type: {getattr(self.predictor, 'best_model_name', 'Unknown')}\n"
            
            if hasattr(self.predictor, 'model_performance'):
                for model_name, metrics in self.predictor.model_performance.items():
                    performance_text += f"\n{model_name.upper()}:\n"
                    performance_text += f"  Accuracy: {metrics.get('accuracy', 0):.3f}\n"
                    performance_text += f"  AUC: {metrics.get('auc', 0):.3f}\n"
            
            performance_text += f"\nModel Status: {'Trained' if self.model_trained else 'Not Trained'}\n"
            performance_text += f"Model Type: {self.model_type_var.get()}\n"
            
        except Exception as e:
            performance_text += f"Error retrieving performance metrics: {str(e)}\n"
        
        self.log_message(performance_text)
    
    def check_system_status(self):
        """Check system status."""
        status_text = f"\nSYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        status_text += "=" * 60 + "\n"
        
        # Model status
        status_text += f"Model Status: {'Trained' if self.model_trained else 'Not Trained'}\n"
        status_text += f"Model Type: {self.model_type_var.get()}\n"
        status_text += f"Teams Loaded: {len(self.teams_config.get('teams', {}))}\n"
        
        # System info
        status_text += f"Python Version: {sys.version.split()[0]}\n"
        status_text += f"Working Directory: {Path.cwd()}\n"
        
        # Check required modules
        modules_to_check = ['numpy', 'pandas', 'scikit-learn', 'tensorflow']
        status_text += "\nModule Status:\n"
        
        for module in modules_to_check:
            try:
                __import__(module)
                status_text += f"  ✓ {module}: Available\n"
            except ImportError:
                status_text += f"  ✗ {module}: Missing\n"
        
        self.log_message(status_text)
    
    def export_results(self):
        """Export results and logs."""
        try:
            output_content = self.output_text.get('1.0', tk.END)
            if not output_content.strip():
                self.show_warning("No output to export")
                return
            
            # Create exports directory
            exports_dir = Path.cwd() / "exports"
            exports_dir.mkdir(exist_ok=True)
            
            # Export with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = exports_dir / f"vct_predictions_{timestamp}.txt"
            
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(f"VCT Prediction System Export - {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(output_content)
            
            self.log_message(f"Results exported to: {export_file}")
            messagebox.showinfo("Export Complete", f"Results exported to:\n{export_file}")
            
        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")
    
    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete('1.0', tk.END)
        self.log_message("Output cleared")
    
    def log_message(self, message: str):
        """Add message to output area."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.update()
    
    def set_status(self, message: str):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)
        self.log_message(f"ERROR: {message}")
    
    def show_warning(self, message: str):
        """Show warning message."""
        messagebox.showwarning("Warning", message)
        self.log_message(f"WARNING: {message}")
    
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
            self.log_message("⚠️ No pre-trained model found. Use 'Update Model' to create one.")
            self.log_message("   Recommended: Run 'py pretrain_super_model.py' for best performance")
            self.model_status_label.configure(text="Pre-training Required", foreground="orange")
            self.train_button.configure(text="Pre-train Super Model")
            return False
    
    def _load_super_model(self, model_file, metadata_file):
        """Load the Super VCT Predictor model"""
        try:
            # Ensure models path is available for unpickling
            models_path = Path(__file__).parent / "src" / "models"
            if str(models_path) not in sys.path:
                sys.path.append(str(models_path))
            
            # Load model metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
            
            # Load the actual super model
            import pickle
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.predictor = model_data['predictor']
            self.model_trained = True
            
            # Update UI
            accuracy = self.model_metadata.get('model_accuracy', 0) * 100
            base_models = self.model_metadata.get('base_models', [])
            model_count = len(base_models)
            
            self.model_status_label.configure(text=f"Super Model Ready ({accuracy:.1f}%)", foreground="green")
            
            # Log success with Super Model details
            training_date = self.model_metadata.get('training_date', 'Unknown')
            tournaments = len(self.model_metadata.get('tournaments_included', []))
            matches = self.model_metadata.get('total_matches', 0)
            feature_count = self.model_metadata.get('feature_count', 0)
            
            self.log_message(f"🎉 Super VCT Predictor loaded successfully!")
            self.log_message(f"   🚀 Ensemble: {model_count} models ({', '.join(base_models)})")
            self.log_message(f"   🎯 Accuracy: {accuracy:.1f}%")
            self.log_message(f"   🔧 Features: {feature_count}")
            self.log_message(f"   🏆 Tournaments: {tournaments}")
            self.log_message(f"   📈 Matches: {matches}")
            self.log_message(f"   📅 Trained: {training_date[:10] if training_date else 'Unknown'}")
            
            # Log top features if available
            top_features = self.model_metadata.get('top_features', [])
            if top_features:
                self.log_message(f"   🔍 Top features: {', '.join([f[0] for f in top_features[:3]])}...")
            
            return True
            
        except Exception as e:
            self.show_error(f"Failed to load Super VCT model: {str(e)}")
            return False
    
    def _load_old_model(self, model_file, metadata_file):
        """Load the old robust model (fallback)"""
        try:
            # Ensure models path is available for unpickling  
            models_path = Path(__file__).parent / "src" / "models"
            if str(models_path) not in sys.path:
                sys.path.append(str(models_path))
            
            # Load model metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
            
            # Load the actual model
            import pickle
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.predictor = model_data['predictor']
            self.model_trained = True
            
            # Update UI
            accuracy = self.model_metadata.get('model_accuracy', 0) * 100
            self.model_status_label.configure(text=f"Ready ({accuracy:.1f}% accuracy)", foreground="green")
            
            # Log success
            training_date = self.model_metadata.get('training_date', 'Unknown')
            tournaments = len(self.model_metadata.get('tournaments_included', []))
            matches = self.model_metadata.get('total_matches', 0)
            
            self.log_message(f"✅ Legacy model loaded (Consider upgrading to Super Model)")
            self.log_message(f"   📊 Model: {self.model_metadata.get('best_model_name', 'robust').upper()}")
            self.log_message(f"   🎯 Accuracy: {accuracy:.1f}%")
            self.log_message(f"   🏆 Tournaments: {tournaments}")
            self.log_message(f"   📈 Matches: {matches}")
            self.log_message(f"   📅 Trained: {training_date[:10] if training_date else 'Unknown'}")
            self.log_message(f"   💡 Run 'py pretrain_super_model.py' for better accuracy")
            
            return True
            
        except Exception as e:
            self.show_error(f"Failed to load pre-trained model: {str(e)}")
            return False
    
    def update_model(self):
        """Update model with latest data or create from scratch"""
        if self.is_training:
            self.show_warning("Model update already in progress")
            return
        
        def update_thread():
            self.is_training = True
            self.training_progress.start()
            self.train_button.configure(state='disabled')
            
            try:
                if self.model_trained and self.predictor:
                    self.set_status("Updating model with latest data...")
                    self.log_message("🔄 Updating pre-trained model with latest data...")
                    success = self.incremental_update()
                else:
                    self.set_status("Pre-training model from scratch...")
                    self.log_message("🚀 Creating pre-trained model from scratch...")
                    success = self.full_pretrain()
                
                if success:
                    self.model_trained = True
                    accuracy = getattr(self.predictor, 'test_results', {}).get(
                        getattr(self.predictor, 'best_model_name', 'svm'), {}
                    ).get('accuracy', 0) * 100
                    
                    self.model_status_label.configure(text=f"Updated ({accuracy:.1f}% accuracy)", foreground="green")
                    self.log_message("✅ Model update completed successfully!")
                    self.set_status("Model updated and ready for predictions")
                else:
                    self.model_status_label.configure(text="Update Failed", foreground="red")
                    self.log_message("❌ Model update failed!")
                    self.set_status("Model update failed")
                
            except Exception as e:
                self.log_message(f"❌ Update error: {str(e)}")
                self.model_status_label.configure(text="Update Error", foreground="red")
                self.set_status("Update error occurred")
            finally:
                self.is_training = False
                self.training_progress.stop()
                self.train_button.configure(state='normal')
        
        threading.Thread(target=update_thread, daemon=True).start()
    
    def incremental_update(self) -> bool:
        """Perform incremental update with latest data"""
        try:
            # Collect any new data since last training
            self.log_message("🔍 Checking for new tournament data...")
            
            # Run data collection to get latest results
            import subprocess
            result = subprocess.run([
                sys.executable, 'collect_2025_data.py'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log_message("✅ Data collection completed")
                
                # Check if there's actually new data to retrain on
                current_matches = len(self.predictor.matches_df) if hasattr(self.predictor, 'matches_df') else 0
                
                # Reload data to see if there are new matches
                if hasattr(self.predictor, 'load_and_process_all_data'):
                    self.predictor.load_and_process_all_data()
                    new_matches = len(self.predictor.matches_df) if hasattr(self.predictor, 'matches_df') else 0
                    
                    if new_matches > current_matches:
                        self.log_message(f"📈 Found {new_matches - current_matches} new matches")
                        
                        # Quick retrain with new data - simplified approach
                        self.log_message("🔄 Performing incremental model update...")
                        
                        # For now, we'll do a full retrain but it's still faster than GUI training
                        # In a more advanced system, you could implement true incremental learning
                        features, labels = self.predictor.create_comprehensive_features()
                        if features is not None:
                            self.predictor.implement_temporal_splits(features, labels)
                            self.predictor.train_and_validate_models()
                            self.predictor.evaluate_final_performance()
                            
                            self.log_message(f"✅ Model updated with {new_matches} total matches")
                            return True
                    else:
                        self.log_message("ℹ️ No new matches found - model is up to date")
                        return True
                else:
                    self.log_message("⚠️ Cannot update - predictor doesn't support data reloading")
                    return False
            else:
                self.log_message(f"⚠️ Data collection had issues: {result.stderr}")
                # Still try to update with existing data
                return True
        
        except Exception as e:
            self.log_message(f"❌ Incremental update error: {str(e)}")
            return False
    
    def full_pretrain(self) -> bool:
        """Perform full pre-training from scratch using Super VCT Predictor"""
        try:
            # Import and initialize Super VCT Predictor
            sys.path.append(str(Path(__file__).parent / "src"))
            
            # Try to import Super VCT Predictor first
            try:
                from models.super_vct_predictor import SuperVCTPredictor
                self.log_message("🚀 Initializing Super VCT Predictor...")
                self.predictor = SuperVCTPredictor()
                using_super_model = True
            except ImportError:
                # Fallback to robust predictor
                from models.robust_ml_predictor import RobustVCTPredictor
                self.log_message("🤖 Initializing Robust VCT Predictor (fallback)...")
                self.predictor = RobustVCTPredictor()
                using_super_model = False
            
            # Load all data
            self.log_message("📊 Loading all tournament data...")
            if not self.predictor.load_and_process_all_data():
                return False
            
            # Create features
            self.log_message("🔧 Creating comprehensive features...")
            features, labels = self.predictor.create_comprehensive_features()
            if features is None:
                return False
            
            # Train model
            if using_super_model:
                self.log_message("🎯 Training Super Ensemble (7 models)...")
                self.log_message("   This may take several minutes...")
                success = self.predictor.train_super_ensemble(features, labels)
                if not success:
                    return False
                    
                # Log ensemble performance
                if hasattr(self.predictor, 'model_scores'):
                    self.log_message(f"   📊 Individual model scores:")
                    for name, scores in self.predictor.model_scores.items():
                        self.log_message(f"     {name.upper()}: {scores['test_score']:.3f}")
                
            else:
                self.log_message("🎯 Training robust model...")
                self.predictor.implement_temporal_splits(features, labels)
                self.predictor.train_and_validate_models()
                self.predictor.evaluate_final_performance()
            
            # Save the new model
            self.save_updated_model(using_super_model)
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Full pre-training error: {str(e)}")
            return False
    
    def save_updated_model(self, using_super_model=False):
        """Save the updated model"""
        try:
            model_dir = Path(__file__).parent / "models" / "pretrained"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine model type and file names
            if using_super_model or hasattr(self.predictor, 'super_ensemble'):
                model_file = model_dir / "super_vct_model.pkl"
                metadata_file = model_dir / "super_model_metadata.json"
                model_type = 'SuperVCTPredictor'
                model_name = 'Super VCT Ensemble'
            else:
                model_file = model_dir / "vct_model_pretrained.pkl"
                metadata_file = model_dir / "model_metadata.json"
                model_type = 'RobustVCTPredictor'
                model_name = 'Robust VCT Predictor'
            
            # Save model data
            model_data = {
                'predictor': self.predictor,
                'model_type': model_type,
                'training_date': datetime.now().isoformat(),
                'data_version': datetime.now().strftime('%Y-%m-%d'),
                'total_matches': len(self.predictor.matches_df) if hasattr(self.predictor, 'matches_df') else 0
            }
            
            import pickle
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Get model performance
            if using_super_model and hasattr(self.predictor, 'test_results'):
                accuracy = self.predictor.test_results.get('super_ensemble', {}).get('accuracy', 0)
                best_model_name = 'super_ensemble'
            else:
                accuracy = getattr(self.predictor, 'test_results', {}).get(
                    getattr(self.predictor, 'best_model_name', 'svm'), {}
                ).get('accuracy', 0)
                best_model_name = getattr(self.predictor, 'best_model_name', 'svm')
            
            # Update metadata
            if using_super_model:
                self.model_metadata = {
                    'model_type': model_type,
                    'model_name': model_name,
                    'best_model_name': best_model_name,
                    'model_accuracy': accuracy,
                    'training_date': datetime.now().isoformat(),
                    'total_matches': model_data['total_matches'],
                    'feature_count': len(getattr(self.predictor, 'feature_names', [])),
                    'base_models': list(getattr(self.predictor, 'base_models', {}).keys()),
                    'ensemble_weights': getattr(self.predictor, 'ensemble_weights', {}),
                    'model_scores': getattr(self.predictor, 'model_scores', {}),
                    'tournaments_included': self.predictor.matches_df['tournament'].unique().tolist() if hasattr(self.predictor, 'matches_df') and self.predictor.matches_df is not None else [],
                    'last_updated': datetime.now().isoformat()
                }
            else:
                self.model_metadata = {
                    'model_type': model_type,
                    'model_name': model_name,
                    'best_model_name': best_model_name,
                    'model_accuracy': accuracy,
                    'training_date': datetime.now().isoformat(),
                    'total_matches': model_data['total_matches'],
                    'tournaments_included': self.predictor.matches_df['tournament'].unique().tolist() if hasattr(self.predictor, 'matches_df') and self.predictor.matches_df is not None else [],
                    'last_updated': datetime.now().isoformat()
                }
            
            with open(metadata_file, 'w') as f:
                json.dump(self.model_metadata, f, indent=2, default=str)
            
            accuracy_pct = accuracy * 100
            self.log_message(f"💾 {model_name} saved successfully ({accuracy_pct:.1f}% accuracy)")
            
        except Exception as e:
            self.log_message(f"⚠️ Could not save model: {str(e)}")
    
    def run(self):
        """Start the GUI application."""
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start main loop
        self.root.mainloop()

def main():
    """Main entry point."""
    try:
        print("Starting VCT Prediction System - Production GUI...")
        app = VCTPredictionGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        if 'tkinter' in str(e).lower():
            print("Please ensure tkinter is installed and available.")
        sys.exit(1)

if __name__ == "__main__":
    main()
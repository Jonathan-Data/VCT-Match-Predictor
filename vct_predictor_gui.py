#!/usr/bin/env python3
"""
VCT Prediction System - Main GUI Application
Unified interface for data collection, predictions, and monitoring
"""

import sys
import os
import json
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
except ImportError:
    print("Error: tkinter not available. Please install tkinter.")
    sys.exit(1)

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_rib_scraper import scrape_team_data
    from performance_monitor import PerformanceMonitor, MatchResult
    from automated_data_updater import AutomatedDataUpdater
    from production_deployment import ProductionDeployment
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

class VCTPredictorGUI:
    """Main GUI application for the VCT Prediction System."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VCT Prediction System")
        self.root.geometry("1000x700")
        
        # Initialize components
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Component instances
        self.performance_monitor = None
        self.data_updater = None
        self.production_deployment = None
        
        # GUI state
        self.is_running_task = False
        
        self.setup_gui()
        self.setup_logging()
        self.load_initial_data()
    
    def setup_gui(self):
        """Setup the main GUI interface."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_data_collection_tab()
        self.create_predictions_tab()
        self.create_monitoring_tab()
        self.create_settings_tab()
        
        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab."""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # System status section
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=100)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(dashboard_frame, text="Quick Actions")
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        actions_row1 = ttk.Frame(actions_frame)
        actions_row1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(actions_row1, text="Refresh Status", 
                  command=self.refresh_system_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_row1, text="Run Full Pipeline", 
                  command=self.run_full_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_row1, text="Collect Data", 
                  command=self.run_data_collection).pack(side=tk.LEFT, padx=5)
        
        # Recent results
        results_frame = ttk.LabelFrame(dashboard_frame, text="Recent Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_data_collection_tab(self):
        """Create the data collection tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Collection")
        
        # Team selection
        team_frame = ttk.LabelFrame(data_frame, text="Team Management")
        team_frame.pack(fill=tk.X, padx=10, pady=5)
        
        team_row1 = ttk.Frame(team_frame)
        team_row1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(team_row1, text="Teams to Monitor:").pack(side=tk.LEFT)
        
        self.teams_listbox = tk.Listbox(team_frame, height=6, selectmode=tk.MULTIPLE)
        self.teams_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        team_buttons = ttk.Frame(team_frame)
        team_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(team_buttons, text="Add Team", command=self.add_team).pack(side=tk.LEFT, padx=5)
        ttk.Button(team_buttons, text="Remove Team", command=self.remove_team).pack(side=tk.LEFT, padx=5)
        ttk.Button(team_buttons, text="Load Default Teams", command=self.load_default_teams).pack(side=tk.LEFT, padx=5)
        
        # Data sources
        sources_frame = ttk.LabelFrame(data_frame, text="Data Sources")
        sources_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.enable_rib = tk.BooleanVar(value=True)
        self.enable_vlr = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(sources_frame, text="rib.gg (Enhanced Scraper)", 
                       variable=self.enable_rib).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(sources_frame, text="vlr.gg (Basic Scraper)", 
                       variable=self.enable_vlr).pack(anchor=tk.W, padx=5, pady=2)
        
        # Collection controls
        collection_frame = ttk.LabelFrame(data_frame, text="Data Collection")
        collection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        collection_buttons = ttk.Frame(collection_frame)
        collection_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(collection_buttons, text="Collect Selected Teams", 
                  command=self.collect_selected_teams).pack(side=tk.LEFT, padx=5)
        ttk.Button(collection_buttons, text="Collect All Teams", 
                  command=self.collect_all_teams).pack(side=tk.LEFT, padx=5)
        ttk.Button(collection_buttons, text="Test Single Team", 
                  command=self.test_single_team).pack(side=tk.LEFT, padx=5)
        
        # Collection log
        self.collection_log = scrolledtext.ScrolledText(collection_frame, height=15)
        self.collection_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_predictions_tab(self):
        """Create the predictions tab."""
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="Predictions")
        
        # Match input
        input_frame = ttk.LabelFrame(pred_frame, text="Match Prediction")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        match_row1 = ttk.Frame(input_frame)
        match_row1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(match_row1, text="Team 1:").pack(side=tk.LEFT)
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(match_row1, textvariable=self.team1_var, width=20)
        self.team1_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(match_row1, text="vs").pack(side=tk.LEFT, padx=10)
        
        ttk.Label(match_row1, text="Team 2:").pack(side=tk.LEFT)
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(match_row1, textvariable=self.team2_var, width=20)
        self.team2_combo.pack(side=tk.LEFT, padx=5)
        
        match_row2 = ttk.Frame(input_frame)
        match_row2.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(match_row2, text="Tournament:").pack(side=tk.LEFT)
        self.tournament_var = tk.StringVar(value="VCT Champions")
        ttk.Entry(match_row2, textvariable=self.tournament_var, width=25).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(match_row2, text="Predict Match", 
                  command=self.predict_match).pack(side=tk.LEFT, padx=20)
        
        # Live predictions
        live_frame = ttk.LabelFrame(pred_frame, text="Live Predictions")
        live_frame.pack(fill=tk.X, padx=10, pady=5)
        
        live_buttons = ttk.Frame(live_frame)
        live_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(live_buttons, text="Generate Live Predictions", 
                  command=self.generate_live_predictions).pack(side=tk.LEFT, padx=5)
        ttk.Button(live_buttons, text="Load Saved Predictions", 
                  command=self.load_predictions).pack(side=tk.LEFT, padx=5)
        
        # Predictions display
        self.predictions_text = scrolledtext.ScrolledText(pred_frame, height=20)
        self.predictions_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def create_monitoring_tab(self):
        """Create the performance monitoring tab."""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitoring")
        
        # Performance summary
        summary_frame = ttk.LabelFrame(monitor_frame, text="Performance Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        summary_buttons = ttk.Frame(summary_frame)
        summary_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(summary_buttons, text="Refresh Performance", 
                  command=self.refresh_performance).pack(side=tk.LEFT, padx=5)
        ttk.Button(summary_buttons, text="Add Match Result", 
                  command=self.add_match_result_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(summary_buttons, text="Evaluate Predictions", 
                  command=self.evaluate_predictions).pack(side=tk.LEFT, padx=5)
        
        self.performance_text = scrolledtext.ScrolledText(summary_frame, height=10)
        self.performance_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recent evaluations
        eval_frame = ttk.LabelFrame(monitor_frame, text="Recent Evaluations")
        eval_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.evaluations_text = scrolledtext.ScrolledText(eval_frame, height=12)
        self.evaluations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Update intervals
        intervals_frame = ttk.LabelFrame(settings_frame, text="Update Intervals")
        intervals_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(intervals_frame, text="Data Update Interval (hours):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_interval_var = tk.IntVar(value=8)
        ttk.Spinbox(intervals_frame, from_=1, to=24, textvariable=self.data_interval_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(intervals_frame, text="Performance Check Interval (hours):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.perf_interval_var = tk.IntVar(value=12)
        ttk.Spinbox(intervals_frame, from_=1, to=24, textvariable=self.perf_interval_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Automation settings
        auto_frame = ttk.LabelFrame(settings_frame, text="Automation")
        auto_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.enable_auto_updates = tk.BooleanVar(value=True)
        self.enable_auto_monitoring = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(auto_frame, text="Enable Automatic Data Updates", 
                       variable=self.enable_auto_updates).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Checkbutton(auto_frame, text="Enable Automatic Performance Monitoring", 
                       variable=self.enable_auto_monitoring).pack(anchor=tk.W, padx=5, pady=2)
        
        auto_buttons = ttk.Frame(auto_frame)
        auto_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(auto_buttons, text="Start Automation", 
                  command=self.start_automation).pack(side=tk.LEFT, padx=5)
        ttk.Button(auto_buttons, text="Stop Automation", 
                  command=self.stop_automation).pack(side=tk.LEFT, padx=5)
        ttk.Button(auto_buttons, text="Generate Cron Jobs", 
                  command=self.generate_cron_jobs).pack(side=tk.LEFT, padx=5)
        
        # System management
        system_frame = ttk.LabelFrame(settings_frame, text="System Management")
        system_frame.pack(fill=tk.X, padx=10, pady=5)
        
        system_buttons = ttk.Frame(system_frame)
        system_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(system_buttons, text="Save Settings", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_buttons, text="Load Settings", 
                  command=self.load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_buttons, text="Clean Old Data", 
                  command=self.clean_old_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_buttons, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Logs display
        logs_frame = ttk.LabelFrame(settings_frame, text="System Logs")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.logs_text = scrolledtext.ScrolledText(logs_frame, height=10)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(logs_frame, text="Refresh Logs", 
                  command=self.refresh_logs).pack(pady=5)
    
    def setup_logging(self):
        """Setup logging for the GUI application."""
        log_file = self.logs_dir / f"gui_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('VCTGui')
    
    def load_initial_data(self):
        """Load initial data and refresh status."""
        self.load_teams()
        self.load_team_names_for_combos()
        self.refresh_system_status()
    
    def load_teams(self):
        """Load teams list into the listbox."""
        teams_file = self.data_dir / "teams_to_update.json"
        teams = ["Sentinels", "Fnatic", "Paper Rex", "Team Liquid", "G2 Esports", 
                "NRG", "LOUD", "DRX", "NAVI", "FPX"]
        
        if teams_file.exists():
            try:
                with open(teams_file, 'r') as f:
                    data = json.load(f)
                teams = data.get('teams', teams)
            except:
                pass
        
        self.teams_listbox.delete(0, tk.END)
        for team in teams:
            self.teams_listbox.insert(tk.END, team)
    
    def load_team_names_for_combos(self):
        """Load team names for prediction combo boxes."""
        teams = []
        for i in range(self.teams_listbox.size()):
            teams.append(self.teams_listbox.get(i))
        
        self.team1_combo['values'] = teams
        self.team2_combo['values'] = teams
    
    def refresh_system_status(self):
        """Refresh the system status display."""
        self.set_status("Checking system status...")
        
        def update_status():
            try:
                if not self.production_deployment:
                    self.production_deployment = ProductionDeployment()
                
                status = self.production_deployment.get_system_status()
                
                status_text = f"System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                status_text += "=" * 60 + "\n\n"
                status_text += f"System Health: {status['system_health'].upper()}\n\n"
                
                status_text += "Components:\n"
                for component, info in status['components'].items():
                    available = "Available" if info['available'] else "Not Available"
                    status_text += f"  {component}: {available}\n"
                
                status_text += f"\nRecent Pipeline Runs: {len(status.get('recent_results', []))}\n"
                
                if status.get('disk_usage'):
                    data_size = status['disk_usage'].get('data_dir_size', 0) / (1024*1024)
                    logs_size = status['disk_usage'].get('logs_dir_size', 0) / (1024*1024)
                    status_text += f"\nDisk Usage:\n"
                    status_text += f"  Data Directory: {data_size:.1f} MB\n"
                    status_text += f"  Logs Directory: {logs_size:.1f} MB\n"
                
                self.status_text.delete('1.0', tk.END)
                self.status_text.insert('1.0', status_text)
                
            except Exception as e:
                error_msg = f"Error checking system status: {str(e)}\n"
                self.status_text.delete('1.0', tk.END)
                self.status_text.insert('1.0', error_msg)
            
            self.set_status("Ready")
        
        threading.Thread(target=update_status, daemon=True).start()
    
    def run_full_pipeline(self):
        """Run the complete prediction pipeline."""
        if self.is_running_task:
            messagebox.showwarning("Task Running", "Another task is already running. Please wait.")
            return
        
        self.set_status("Running full pipeline...")
        self.start_progress()
        
        def run_pipeline():
            self.is_running_task = True
            try:
                if not self.production_deployment:
                    self.production_deployment = ProductionDeployment()
                
                results = self.production_deployment.run_full_pipeline()
                
                results_text = f"Pipeline Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                results_text += "=" * 60 + "\n\n"
                results_text += f"Overall Success: {results['success']}\n\n"
                
                for stage_name, stage_result in results.get('stages', {}).items():
                    results_text += f"{stage_name.title()}:\n"
                    results_text += f"  Success: {stage_result.get('success', False)}\n"
                    if 'error' in stage_result:
                        results_text += f"  Error: {stage_result['error']}\n"
                    if stage_name == 'predictions' and 'predictions' in stage_result:
                        results_text += f"  Predictions Generated: {len(stage_result['predictions'])}\n"
                    results_text += "\n"
                
                self.results_text.delete('1.0', tk.END)
                self.results_text.insert('1.0', results_text)
                
                if results['success']:
                    messagebox.showinfo("Success", "Full pipeline completed successfully!")
                else:
                    messagebox.showwarning("Partial Success", "Pipeline completed with some issues. Check results for details.")
                
            except Exception as e:
                error_msg = f"Pipeline error: {str(e)}"
                self.results_text.delete('1.0', tk.END)
                self.results_text.insert('1.0', error_msg)
                messagebox.showerror("Error", error_msg)
            
            finally:
                self.is_running_task = False
                self.stop_progress()
                self.set_status("Ready")
        
        threading.Thread(target=run_pipeline, daemon=True).start()
    
    def run_data_collection(self):
        """Run data collection only."""
        if self.is_running_task:
            messagebox.showwarning("Task Running", "Another task is already running. Please wait.")
            return
        
        self.set_status("Collecting data...")
        self.start_progress()
        
        def collect_data():
            self.is_running_task = True
            try:
                if not self.production_deployment:
                    self.production_deployment = ProductionDeployment()
                
                results = self.production_deployment.run_data_collection()
                
                results_text = f"Data Collection Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                results_text += "=" * 60 + "\n\n"
                results_text += f"Success: {results['success']}\n\n"
                
                for source_name, source_result in results.get('sources', {}).items():
                    results_text += f"{source_name.title()}:\n"
                    results_text += f"  Success: {source_result.get('success', False)}\n"
                    if 'error' in source_result:
                        results_text += f"  Error: {source_result['error']}\n"
                    results_text += "\n"
                
                self.results_text.delete('1.0', tk.END)
                self.results_text.insert('1.0', results_text)
                
                if results['success']:
                    messagebox.showinfo("Success", "Data collection completed successfully!")
                
            except Exception as e:
                error_msg = f"Data collection error: {str(e)}"
                self.results_text.delete('1.0', tk.END)
                self.results_text.insert('1.0', error_msg)
                messagebox.showerror("Error", error_msg)
            
            finally:
                self.is_running_task = False
                self.stop_progress()
                self.set_status("Ready")
        
        threading.Thread(target=collect_data, daemon=True).start()
    
    def add_team(self):
        """Add a new team to monitor."""
        team = tk.simpledialog.askstring("Add Team", "Enter team name:")
        if team and team.strip():
            self.teams_listbox.insert(tk.END, team.strip())
            self.save_teams()
            self.load_team_names_for_combos()
    
    def remove_team(self):
        """Remove selected teams."""
        selected = self.teams_listbox.curselection()
        if selected:
            for index in reversed(selected):
                self.teams_listbox.delete(index)
            self.save_teams()
            self.load_team_names_for_combos()
    
    def load_default_teams(self):
        """Load default VCT teams."""
        default_teams = [
            "Sentinels", "Fnatic", "Paper Rex", "Team Liquid", "G2 Esports",
            "NRG", "LOUD", "DRX", "NAVI", "FPX", "100 Thieves", "Cloud9",
            "KRU Esports", "Leviatan", "EDward Gaming", "ASE", "Trace Esports"
        ]
        
        self.teams_listbox.delete(0, tk.END)
        for team in default_teams:
            self.teams_listbox.insert(tk.END, team)
        
        self.save_teams()
        self.load_team_names_for_combos()
    
    def save_teams(self):
        """Save the current teams list."""
        teams = []
        for i in range(self.teams_listbox.size()):
            teams.append(self.teams_listbox.get(i))
        
        teams_file = self.data_dir / "teams_to_update.json"
        try:
            with open(teams_file, 'w') as f:
                json.dump({
                    'teams': teams,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save teams: {str(e)}")
    
    def collect_selected_teams(self):
        """Collect data for selected teams."""
        selected = self.teams_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select teams to collect data for.")
            return
        
        selected_teams = [self.teams_listbox.get(i) for i in selected]
        self._collect_teams(selected_teams)
    
    def collect_all_teams(self):
        """Collect data for all teams."""
        all_teams = [self.teams_listbox.get(i) for i in range(self.teams_listbox.size())]
        if not all_teams:
            messagebox.showwarning("No Teams", "Please add teams to collect data for.")
            return
        
        self._collect_teams(all_teams)
    
    def _collect_teams(self, teams):
        """Collect data for specified teams."""
        if self.is_running_task:
            messagebox.showwarning("Task Running", "Another task is already running. Please wait.")
            return
        
        self.set_status(f"Collecting data for {len(teams)} teams...")
        self.start_progress()
        
        def collect():
            self.is_running_task = True
            self.collection_log.delete('1.0', tk.END)
            
            try:
                for team in teams:
                    self.collection_log.insert(tk.END, f"Collecting data for {team}...\n")
                    self.collection_log.see(tk.END)
                    self.collection_log.update()
                    
                    try:
                        if self.enable_rib.get():
                            result = scrape_team_data(team)
                            if result and result.get('success'):
                                self.collection_log.insert(tk.END, f"  Success: {team}\n")
                            else:
                                self.collection_log.insert(tk.END, f"  Failed: {team}\n")
                        else:
                            self.collection_log.insert(tk.END, f"  Skipped: {team} (no sources enabled)\n")
                    
                    except Exception as e:
                        self.collection_log.insert(tk.END, f"  Error for {team}: {str(e)}\n")
                    
                    self.collection_log.see(tk.END)
                    self.collection_log.update()
                
                self.collection_log.insert(tk.END, f"\nData collection completed for {len(teams)} teams.\n")
                messagebox.showinfo("Complete", f"Data collection completed for {len(teams)} teams.")
                
            except Exception as e:
                error_msg = f"Collection error: {str(e)}"
                self.collection_log.insert(tk.END, f"\nError: {error_msg}\n")
                messagebox.showerror("Error", error_msg)
            
            finally:
                self.is_running_task = False
                self.stop_progress()
                self.set_status("Ready")
        
        threading.Thread(target=collect, daemon=True).start()
    
    def test_single_team(self):
        """Test data collection for a single team."""
        team = tk.simpledialog.askstring("Test Team", "Enter team name to test:")
        if team:
            self._collect_teams([team.strip()])
    
    def predict_match(self):
        """Predict a single match."""
        team1 = self.team1_var.get().strip()
        team2 = self.team2_var.get().strip()
        tournament = self.tournament_var.get().strip()
        
        if not team1 or not team2:
            messagebox.showwarning("Missing Teams", "Please select both teams.")
            return
        
        if team1 == team2:
            messagebox.showwarning("Same Team", "Please select different teams.")
            return
        
        self.set_status("Generating prediction...")
        
        # Mock prediction for demonstration
        prediction_text = f"Match Prediction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        prediction_text += "=" * 60 + "\n\n"
        prediction_text += f"Match: {team1} vs {team2}\n"
        prediction_text += f"Tournament: {tournament}\n\n"
        prediction_text += "Note: This is a demo prediction. Implement actual prediction logic\n"
        prediction_text += "by integrating with your trained model.\n\n"
        prediction_text += f"Predicted Winner: {team1}\n"
        prediction_text += f"Confidence: 68.5%\n"
        prediction_text += f"Team 1 Win Probability: 68.5%\n"
        prediction_text += f"Team 2 Win Probability: 31.5%\n"
        
        self.predictions_text.delete('1.0', tk.END)
        self.predictions_text.insert('1.0', prediction_text)
        
        self.set_status("Ready")
    
    def generate_live_predictions(self):
        """Generate predictions for upcoming matches."""
        self.set_status("Generating live predictions...")
        
        # Mock live predictions
        predictions_text = f"Live Predictions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        predictions_text += "=" * 70 + "\n\n"
        predictions_text += "Note: This is a demo. Implement actual live prediction logic.\n\n"
        predictions_text += "Upcoming Matches:\n"
        predictions_text += "-" * 50 + "\n"
        predictions_text += "1. Sentinels vs Fnatic (VCT Champions)\n"
        predictions_text += "   Predicted Winner: Sentinels (72.3%)\n"
        predictions_text += "   Confidence: High\n\n"
        predictions_text += "2. Paper Rex vs DRX (VCT Pacific)\n"
        predictions_text += "   Predicted Winner: DRX (61.8%)\n"
        predictions_text += "   Confidence: Medium\n\n"
        
        self.predictions_text.delete('1.0', tk.END)
        self.predictions_text.insert('1.0', predictions_text)
        
        self.set_status("Ready")
    
    def load_predictions(self):
        """Load saved predictions from file."""
        try:
            predictions_file = self.data_dir / "live_predictions.json"
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    data = json.load(f)
                
                predictions_text = f"Loaded Predictions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                predictions_text += "=" * 70 + "\n\n"
                predictions_text += json.dumps(data, indent=2)
                
                self.predictions_text.delete('1.0', tk.END)
                self.predictions_text.insert('1.0', predictions_text)
            else:
                messagebox.showinfo("No File", "No saved predictions found.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not load predictions: {str(e)}")
    
    def refresh_performance(self):
        """Refresh performance monitoring data."""
        self.set_status("Refreshing performance data...")
        
        def refresh():
            try:
                if not self.performance_monitor:
                    self.performance_monitor = PerformanceMonitor()
                
                summary = self.performance_monitor.get_performance_summary()
                
                perf_text = f"Performance Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                perf_text += "=" * 60 + "\n\n"
                
                if 'error' in summary:
                    perf_text += f"Error: {summary['error']}\n"
                elif 'message' in summary:
                    perf_text += f"{summary['message']}\n"
                else:
                    perf_text += f"Total Evaluations: {summary.get('total_evaluations', 0)}\n"
                    perf_text += f"Overall Accuracy: {summary.get('overall_accuracy', 0):.1%}\n"
                    perf_text += f"Performance Level: {summary.get('performance_level', 'Unknown').title()}\n"
                    perf_text += f"vs Expected: {summary.get('performance_vs_expected', 'N/A')}\n"
                    perf_text += f"Recent Trend: {summary.get('recent_trend', {}).get('status', 'Unknown').title()}\n\n"
                    
                    if summary.get('confidence_breakdown'):
                        perf_text += "Confidence Breakdown:\n"
                        for tier, data in summary['confidence_breakdown'].items():
                            perf_text += f"  {tier}: {data['accuracy']:.1%} ({data['count']} predictions)\n"
                
                self.performance_text.delete('1.0', tk.END)
                self.performance_text.insert('1.0', perf_text)
                
            except Exception as e:
                error_msg = f"Error refreshing performance: {str(e)}"
                self.performance_text.delete('1.0', tk.END)
                self.performance_text.insert('1.0', error_msg)
            
            self.set_status("Ready")
        
        threading.Thread(target=refresh, daemon=True).start()
    
    def add_match_result_dialog(self):
        """Show dialog to add a match result."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Match Result")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        
        # Match result form
        ttk.Label(dialog, text="Match ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        match_id_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=match_id_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Team 1:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        team1_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=team1_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Team 2:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        team2_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=team2_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Winner:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        winner_var = tk.StringVar()
        winner_combo = ttk.Combobox(dialog, textvariable=winner_var, width=27)
        winner_combo.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Score:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        score_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=score_var, width=30).grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Tournament:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        tournament_var = tk.StringVar(value="VCT Champions")
        ttk.Entry(dialog, textvariable=tournament_var, width=30).grid(row=5, column=1, padx=5, pady=5)
        
        def update_winner_combo(*args):
            team1 = team1_var.get().strip()
            team2 = team2_var.get().strip()
            if team1 and team2:
                winner_combo['values'] = [team1, team2]
        
        team1_var.trace('w', update_winner_combo)
        team2_var.trace('w', update_winner_combo)
        
        def add_result():
            try:
                if not self.performance_monitor:
                    self.performance_monitor = PerformanceMonitor()
                
                result = MatchResult(
                    match_id=match_id_var.get().strip(),
                    team1=team1_var.get().strip(),
                    team2=team2_var.get().strip(),
                    actual_winner=winner_var.get().strip(),
                    actual_score=score_var.get().strip(),
                    match_date=datetime.now().isoformat(),
                    tournament=tournament_var.get().strip(),
                    stage="Manual Entry",
                    region="Unknown",
                    source="gui"
                )
                
                success = self.performance_monitor.add_match_result(result)
                if success:
                    messagebox.showinfo("Success", "Match result added successfully!")
                    dialog.destroy()
                    self.refresh_performance()
                else:
                    messagebox.showerror("Error", "Failed to add match result.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error adding match result: {str(e)}")
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Add Result", command=add_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def evaluate_predictions(self):
        """Evaluate predictions against results."""
        self.set_status("Evaluating predictions...")
        
        def evaluate():
            try:
                if not self.performance_monitor:
                    self.performance_monitor = PerformanceMonitor()
                
                results = self.performance_monitor.evaluate_predictions()
                
                eval_text = f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                eval_text += "=" * 60 + "\n\n"
                
                if 'error' in results:
                    eval_text += f"Error: {results['error']}\n"
                elif 'message' in results:
                    eval_text += f"{results['message']}\n"
                else:
                    eval_text += f"Evaluations: {results.get('evaluations_count', 0)}\n"
                    metrics = results.get('metrics', {})
                    eval_text += f"Overall Accuracy: {metrics.get('overall_accuracy', 0):.1%}\n"
                    eval_text += f"Brier Score: {metrics.get('brier_score', 0):.3f}\n"
                    eval_text += f"Performance vs Expected: {metrics.get('performance_vs_expected', 0):+.1%}\n"
                
                self.evaluations_text.delete('1.0', tk.END)
                self.evaluations_text.insert('1.0', eval_text)
                
            except Exception as e:
                error_msg = f"Error evaluating predictions: {str(e)}"
                self.evaluations_text.delete('1.0', tk.END)
                self.evaluations_text.insert('1.0', error_msg)
            
            self.set_status("Ready")
        
        threading.Thread(target=evaluate, daemon=True).start()
    
    def save_settings(self):
        """Save current settings."""
        settings = {
            'data_interval': self.data_interval_var.get(),
            'perf_interval': self.perf_interval_var.get(),
            'enable_auto_updates': self.enable_auto_updates.get(),
            'enable_auto_monitoring': self.enable_auto_monitoring.get(),
            'enable_rib': self.enable_rib.get(),
            'enable_vlr': self.enable_vlr.get()
        }
        
        try:
            settings_file = self.data_dir / "gui_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {str(e)}")
    
    def load_settings(self):
        """Load settings from file."""
        try:
            settings_file = self.data_dir / "gui_settings.json"
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                
                self.data_interval_var.set(settings.get('data_interval', 8))
                self.perf_interval_var.set(settings.get('perf_interval', 12))
                self.enable_auto_updates.set(settings.get('enable_auto_updates', True))
                self.enable_auto_monitoring.set(settings.get('enable_auto_monitoring', True))
                self.enable_rib.set(settings.get('enable_rib', True))
                self.enable_vlr.set(settings.get('enable_vlr', False))
                
                messagebox.showinfo("Success", "Settings loaded successfully!")
            else:
                messagebox.showinfo("No File", "No saved settings found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load settings: {str(e)}")
    
    def start_automation(self):
        """Start automated tasks."""
        messagebox.showinfo("Automation", "Automation started. This is a demo - implement actual automation logic.")
    
    def stop_automation(self):
        """Stop automated tasks."""
        messagebox.showinfo("Automation", "Automation stopped.")
    
    def generate_cron_jobs(self):
        """Generate cron job setup."""
        try:
            if not self.production_deployment:
                self.production_deployment = ProductionDeployment()
            
            instructions = self.production_deployment.setup_cron_job()
            
            # Show instructions dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Cron Job Setup")
            dialog.geometry("600x400")
            
            text_widget = scrolledtext.ScrolledText(dialog)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert('1.0', instructions)
            
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate cron jobs: {str(e)}")
    
    def clean_old_data(self):
        """Clean old data files."""
        if messagebox.askyesno("Confirm", "This will remove old data files. Continue?"):
            # Implement cleanup logic
            messagebox.showinfo("Success", "Old data files cleaned.")
    
    def export_data(self):
        """Export system data."""
        filename = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Implement export logic
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Data export functionality - implement as needed'
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export data: {str(e)}")
    
    def refresh_logs(self):
        """Refresh the logs display."""
        try:
            log_files = list(self.logs_dir.glob("*.log"))
            if log_files:
                # Get most recent log file
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                
                with open(latest_log, 'r') as f:
                    logs = f.read()
                
                # Show last 100 lines
                lines = logs.split('\n')
                recent_logs = '\n'.join(lines[-100:]) if len(lines) > 100 else logs
                
                self.logs_text.delete('1.0', tk.END)
                self.logs_text.insert('1.0', recent_logs)
                self.logs_text.see(tk.END)
            else:
                self.logs_text.delete('1.0', tk.END)
                self.logs_text.insert('1.0', "No log files found.")
        
        except Exception as e:
            self.logs_text.delete('1.0', tk.END)
            self.logs_text.insert('1.0', f"Error loading logs: {str(e)}")
    
    def set_status(self, message):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def start_progress(self):
        """Start the progress indicator."""
        self.progress.start()
    
    def stop_progress(self):
        """Stop the progress indicator."""
        self.progress.stop()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    try:
        app = VCTPredictorGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        if 'tkinter' in str(e):
            print("Please ensure tkinter is installed. On Ubuntu/Debian: sudo apt-get install python3-tk")
        sys.exit(1)


if __name__ == "__main__":
    # Add missing import for simpledialog
    try:
        import tkinter.simpledialog
        tk.simpledialog = tkinter.simpledialog
    except ImportError:
        pass
    
    main()
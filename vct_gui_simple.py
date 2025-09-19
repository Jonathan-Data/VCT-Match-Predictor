#!/usr/bin/env python3
"""
VCT Prediction System - Simple GUI
Simplified interface optimized for macOS
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

# Suppress the macOS deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
    print("tkinter imported successfully")
except ImportError:
    print("Error: tkinter not available. Please install tkinter.")
    sys.exit(1)

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

class VCTSimpleGUI:
    """Simple GUI application for the VCT Prediction System."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VCT Prediction System")
        self.root.geometry("900x600")
        
        # Initialize components
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # GUI state
        self.is_running_task = False
        
        self.setup_gui()
        self.load_initial_data()
    
    def setup_gui(self):
        """Setup the main GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="VCT Prediction System", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create sections
        self.create_team_management_section(main_frame)
        self.create_predictions_section(main_frame)
        self.create_actions_section(main_frame)
        self.create_output_section(main_frame)
        
        # Status bar
        self.create_status_bar()
    
    def create_team_management_section(self, parent):
        """Create team management section."""
        team_frame = ttk.LabelFrame(parent, text="Team Management")
        team_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Team list
        list_frame = ttk.Frame(team_frame)
        list_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(list_frame, text="Teams to Monitor:").pack(anchor=tk.W)
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.X, pady=5)
        
        self.teams_listbox = tk.Listbox(listbox_frame, height=4, selectmode=tk.MULTIPLE)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.teams_listbox.yview)
        self.teams_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.teams_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Team management buttons
        team_buttons = ttk.Frame(list_frame)
        team_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(team_buttons, text="Add Team", command=self.add_team).pack(side=tk.LEFT, padx=5)
        ttk.Button(team_buttons, text="Remove Team", command=self.remove_team).pack(side=tk.LEFT, padx=5)
        ttk.Button(team_buttons, text="Load Defaults", command=self.load_default_teams).pack(side=tk.LEFT, padx=5)
    
    def create_predictions_section(self, parent):
        """Create predictions section."""
        pred_frame = ttk.LabelFrame(parent, text="Match Prediction")
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Team selection
        teams_row = ttk.Frame(pred_frame)
        teams_row.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(teams_row, text="Team 1:").pack(side=tk.LEFT)
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(teams_row, textvariable=self.team1_var, width=20, state="readonly")
        self.team1_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(teams_row, text="vs").pack(side=tk.LEFT, padx=10)
        
        ttk.Label(teams_row, text="Team 2:").pack(side=tk.LEFT)
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(teams_row, textvariable=self.team2_var, width=20, state="readonly")
        self.team2_combo.pack(side=tk.LEFT, padx=5)
        
        # Tournament info
        tournament_row = ttk.Frame(pred_frame)
        tournament_row.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(tournament_row, text="Tournament:").pack(side=tk.LEFT)
        self.tournament_var = tk.StringVar(value="VCT Champions")
        ttk.Entry(tournament_row, textvariable=self.tournament_var, width=25).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(tournament_row, text="Predict Match", command=self.predict_match).pack(side=tk.LEFT, padx=20)
    
    def create_actions_section(self, parent):
        """Create actions section."""
        actions_frame = ttk.LabelFrame(parent, text="Actions")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Action buttons
        buttons_row1 = ttk.Frame(actions_frame)
        buttons_row1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_row1, text="Collect Team Data", command=self.collect_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_row1, text="System Status", command=self.check_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_row1, text="Performance Check", command=self.check_performance).pack(side=tk.LEFT, padx=5)
        
        buttons_row2 = ttk.Frame(actions_frame)
        buttons_row2.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_row2, text="Add Match Result", command=self.add_match_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_row2, text="Setup Automation", command=self.setup_automation).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_row2, text="View Logs", command=self.view_logs).pack(side=tk.LEFT, padx=5)
    
    def create_output_section(self, parent):
        """Create output section."""
        output_frame = ttk.LabelFrame(parent, text="Output")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=100)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create status bar."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
    
    def load_initial_data(self):
        """Load initial data."""
        self.load_teams()
        self.load_team_names_for_combos()
        self.log_message("VCT Prediction System initialized successfully")
    
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
    
    def add_team(self):
        """Add a new team to monitor."""
        team = simpledialog.askstring("Add Team", "Enter team name:")
        if team and team.strip():
            self.teams_listbox.insert(tk.END, team.strip())
            self.save_teams()
            self.load_team_names_for_combos()
            self.log_message(f"Added team: {team.strip()}")
    
    def remove_team(self):
        """Remove selected teams."""
        selected = self.teams_listbox.curselection()
        if selected:
            removed_teams = []
            for index in reversed(selected):
                removed_teams.append(self.teams_listbox.get(index))
                self.teams_listbox.delete(index)
            self.save_teams()
            self.load_team_names_for_combos()
            self.log_message(f"Removed teams: {', '.join(removed_teams)}")
        else:
            messagebox.showwarning("No Selection", "Please select teams to remove.")
    
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
        self.log_message(f"Loaded {len(default_teams)} default teams")
    
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
        
        # Simple mock prediction logic
        import random
        winner = team1 if random.random() > 0.5 else team2
        confidence = random.uniform(0.55, 0.85)
        team1_prob = confidence if winner == team1 else 1 - confidence
        team2_prob = 1 - team1_prob
        
        prediction_text += f"Predicted Winner: {winner}\n"
        prediction_text += f"Confidence: {confidence:.1%}\n"
        prediction_text += f"Team 1 ({team1}) Win Probability: {team1_prob:.1%}\n"
        prediction_text += f"Team 2 ({team2}) Win Probability: {team2_prob:.1%}\n\n"
        
        prediction_text += "Note: This is a demo prediction. For real predictions,\n"
        prediction_text += "ensure team data has been collected and models are trained.\n"
        
        self.log_message(prediction_text)
        self.set_status("Ready")
    
    def collect_data(self):
        """Collect data for selected teams."""
        selected = self.teams_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select teams to collect data for.")
            return
        
        selected_teams = [self.teams_listbox.get(i) for i in selected]
        
        if self.is_running_task:
            messagebox.showwarning("Task Running", "Another task is already running. Please wait.")
            return
        
        self.set_status(f"Collecting data for {len(selected_teams)} teams...")
        self.start_progress()
        
        def collect():
            self.is_running_task = True
            
            try:
                self.log_message(f"Starting data collection for {len(selected_teams)} teams...")
                
                for team in selected_teams:
                    self.log_message(f"Collecting data for {team}...")
                    
                    try:
                        # Try to use the enhanced scraper
                        from enhanced_rib_scraper import scrape_team_data
                        result = scrape_team_data(team)
                        
                        if result and result.get('success'):
                            confidence = result.get('confidence', 0.5)
                            method = result.get('scraping_method', 'unknown')
                            self.log_message(f"  Success: {team} (confidence: {confidence:.1%}, method: {method})")
                        else:
                            self.log_message(f"  Failed: {team}")
                    
                    except Exception as e:
                        self.log_message(f"  Error for {team}: {str(e)}")
                
                self.log_message(f"Data collection completed for {len(selected_teams)} teams.")
                messagebox.showinfo("Complete", f"Data collection completed for {len(selected_teams)} teams.")
                
            except Exception as e:
                error_msg = f"Collection error: {str(e)}"
                self.log_message(f"Error: {error_msg}")
                messagebox.showerror("Error", error_msg)
            
            finally:
                self.is_running_task = False
                self.stop_progress()
                self.set_status("Ready")
        
        threading.Thread(target=collect, daemon=True).start()
    
    def check_status(self):
        """Check system status."""
        self.set_status("Checking system status...")
        
        def check():
            try:
                # Run system status check
                result = subprocess.run([
                    sys.executable, 'production_deployment.py', '--status'
                ], capture_output=True, text=True, timeout=30)
                
                status_text = f"System Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                status_text += "=" * 60 + "\n\n"
                
                if result.returncode == 0:
                    status_text += result.stdout
                else:
                    status_text += f"Status check failed:\n{result.stderr}"
                
                self.log_message(status_text)
                
            except subprocess.TimeoutExpired:
                self.log_message("Status check timed out after 30 seconds.")
            except FileNotFoundError:
                self.log_message("production_deployment.py not found. Some features may be unavailable.")
            except Exception as e:
                self.log_message(f"Error checking status: {str(e)}")
            
            self.set_status("Ready")
        
        threading.Thread(target=check, daemon=True).start()
    
    def check_performance(self):
        """Check performance monitoring."""
        self.set_status("Checking performance...")
        
        def check():
            try:
                # Run performance check
                result = subprocess.run([
                    sys.executable, 'performance_monitor.py'
                ], capture_output=True, text=True, timeout=30)
                
                perf_text = f"Performance Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                perf_text += "=" * 60 + "\n\n"
                
                if result.returncode == 0:
                    # Parse the last part of stdout for summary
                    lines = result.stdout.split('\n')
                    summary_lines = lines[-20:] if len(lines) > 20 else lines
                    perf_text += '\n'.join(summary_lines)
                else:
                    perf_text += f"Performance check failed:\n{result.stderr}"
                
                self.log_message(perf_text)
                
            except subprocess.TimeoutExpired:
                self.log_message("Performance check timed out after 30 seconds.")
            except FileNotFoundError:
                self.log_message("performance_monitor.py not found. Performance monitoring unavailable.")
            except Exception as e:
                self.log_message(f"Error checking performance: {str(e)}")
            
            self.set_status("Ready")
        
        threading.Thread(target=check, daemon=True).start()
    
    def add_match_result(self):
        """Add a match result."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Match Result")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Form fields
        ttk.Label(dialog, text="Team 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        team1_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=team1_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Team 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        team2_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=team2_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Winner:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        winner_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=winner_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Score:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        score_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=score_var, width=30).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Tournament:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        tournament_var = tk.StringVar(value="VCT Champions")
        ttk.Entry(dialog, textvariable=tournament_var, width=30).grid(row=4, column=1, padx=5, pady=5)
        
        def add_result():
            # For now, just log the result
            result_text = f"Match Result Added - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            result_text += f"Match: {team1_var.get()} vs {team2_var.get()}\n"
            result_text += f"Winner: {winner_var.get()}\n"
            result_text += f"Score: {score_var.get()}\n"
            result_text += f"Tournament: {tournament_var.get()}\n"
            
            self.log_message(result_text)
            messagebox.showinfo("Success", "Match result recorded!")
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Add Result", command=add_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def setup_automation(self):
        """Setup automation."""
        try:
            result = subprocess.run([
                sys.executable, 'production_deployment.py', '--setup-cron'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Show cron setup instructions
                dialog = tk.Toplevel(self.root)
                dialog.title("Automation Setup")
                dialog.geometry("700x500")
                
                text_widget = scrolledtext.ScrolledText(dialog)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                text_widget.insert('1.0', result.stdout)
                
                ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
            else:
                messagebox.showerror("Error", f"Automation setup failed:\n{result.stderr}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not setup automation: {str(e)}")
    
    def view_logs(self):
        """View system logs."""
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
                
                log_text = f"System Logs - {latest_log.name}\n"
                log_text += "=" * 60 + "\n\n"
                log_text += recent_logs
                
                self.log_message(log_text)
            else:
                self.log_message("No log files found.")
        
        except Exception as e:
            self.log_message(f"Error loading logs: {str(e)}")
    
    def log_message(self, message):
        """Log a message to the output area."""
        self.output_text.insert(tk.END, message + "\n\n")
        self.output_text.see(tk.END)
        self.output_text.update()
    
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
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        self.root.mainloop()


def main():
    """Main entry point for the simple GUI application."""
    try:
        print("Starting VCT Prediction System (Simple GUI)...")
        app = VCTSimpleGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        if 'tkinter' in str(e):
            print("Please ensure tkinter is installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
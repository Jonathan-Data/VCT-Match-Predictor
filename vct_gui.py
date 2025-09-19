#!/usr/bin/env python3
"""
VCT Prediction System - Simple Working GUI
Minimal, functional GUI that works reliably on macOS
"""

import sys
import os
import json
import threading
from pathlib import Path
from datetime import datetime

# Suppress macOS tkinter deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
    from tkinter.scrolledtext import ScrolledText
except ImportError:
    print("tkinter is not available. Please install tkinter.")
    sys.exit(1)

class VCTGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VCT Prediction System")
        self.root.geometry("800x600")
        
        # Data directories
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # State
        self.teams = []
        
        self.setup_gui()
        self.load_teams()
    
    def setup_gui(self):
        """Setup the GUI interface."""
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="VCT Prediction System", 
                              font=('Arial', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Teams section
        teams_frame = tk.LabelFrame(main_frame, text="Team Management", padx=5, pady=5)
        teams_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Team listbox
        self.teams_listbox = tk.Listbox(teams_frame, height=6, selectmode=tk.MULTIPLE)
        self.teams_listbox.pack(fill=tk.X, pady=5)
        
        # Team buttons
        team_btn_frame = tk.Frame(teams_frame)
        team_btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(team_btn_frame, text="Add Team", command=self.add_team).pack(side=tk.LEFT, padx=5)
        tk.Button(team_btn_frame, text="Remove Team", command=self.remove_team).pack(side=tk.LEFT, padx=5)
        tk.Button(team_btn_frame, text="Load Default Teams", command=self.load_default_teams).pack(side=tk.LEFT, padx=5)
        
        # Prediction section
        pred_frame = tk.LabelFrame(main_frame, text="Match Prediction", padx=5, pady=5)
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Team selection
        team_select_frame = tk.Frame(pred_frame)
        team_select_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(team_select_frame, text="Team 1:").pack(side=tk.LEFT)
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(team_select_frame, textvariable=self.team1_var, 
                                       width=20, state="readonly")
        self.team1_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(team_select_frame, text="vs").pack(side=tk.LEFT, padx=10)
        
        tk.Label(team_select_frame, text="Team 2:").pack(side=tk.LEFT)
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(team_select_frame, textvariable=self.team2_var, 
                                       width=20, state="readonly")
        self.team2_combo.pack(side=tk.LEFT, padx=5)
        
        # Tournament and predict button
        tournament_frame = tk.Frame(pred_frame)
        tournament_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(tournament_frame, text="Tournament:").pack(side=tk.LEFT)
        self.tournament_var = tk.StringVar(value="VCT Champions")
        tk.Entry(tournament_frame, textvariable=self.tournament_var, width=25).pack(side=tk.LEFT, padx=5)
        
        tk.Button(tournament_frame, text="Predict Match", 
                 command=self.predict_match).pack(side=tk.LEFT, padx=20)
        
        # Actions section
        actions_frame = tk.LabelFrame(main_frame, text="Actions", padx=5, pady=5)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Action buttons
        btn_frame1 = tk.Frame(actions_frame)
        btn_frame1.pack(fill=tk.X, pady=2)
        
        tk.Button(btn_frame1, text="Collect Data", command=self.collect_data).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame1, text="System Status", command=self.system_status).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame1, text="View Logs", command=self.view_logs).pack(side=tk.LEFT, padx=5)
        
        btn_frame2 = tk.Frame(actions_frame)
        btn_frame2.pack(fill=tk.X, pady=2)
        
        tk.Button(btn_frame2, text="Add Match Result", command=self.add_match_result).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Performance Check", command=self.performance_check).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame2, text="Setup Automation", command=self.setup_automation).pack(side=tk.LEFT, padx=5)
        
        # Output section
        output_frame = tk.LabelFrame(main_frame, text="Output", padx=5, pady=5)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = ScrolledText(output_frame, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_teams(self):
        """Load teams from file or use defaults."""
        teams_file = self.data_dir / "teams_to_update.json"
        default_teams = [
            "Sentinels", "Fnatic", "Paper Rex", "Team Liquid", "G2 Esports",
            "NRG", "LOUD", "DRX", "NAVI", "FPX"
        ]
        
        if teams_file.exists():
            try:
                with open(teams_file, 'r') as f:
                    data = json.load(f)
                self.teams = data.get('teams', default_teams)
            except:
                self.teams = default_teams
        else:
            self.teams = default_teams
        
        self.update_team_displays()
        self.log_message("VCT Prediction System initialized")
    
    def update_team_displays(self):
        """Update team list and combo boxes."""
        # Update listbox
        self.teams_listbox.delete(0, tk.END)
        for team in self.teams:
            self.teams_listbox.insert(tk.END, team)
        
        # Update combo boxes
        self.team1_combo['values'] = self.teams
        self.team2_combo['values'] = self.teams
    
    def save_teams(self):
        """Save teams to file."""
        teams_file = self.data_dir / "teams_to_update.json"
        try:
            with open(teams_file, 'w') as f:
                json.dump({
                    'teams': self.teams,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save teams: {e}")
    
    def add_team(self):
        """Add a new team."""
        team = simpledialog.askstring("Add Team", "Enter team name:")
        if team and team.strip():
            self.teams.append(team.strip())
            self.update_team_displays()
            self.save_teams()
            self.log_message(f"Added team: {team.strip()}")
    
    def remove_team(self):
        """Remove selected teams."""
        selected = self.teams_listbox.curselection()
        if selected:
            # Remove in reverse order to maintain indices
            for index in reversed(selected):
                removed_team = self.teams.pop(index)
                self.log_message(f"Removed team: {removed_team}")
            
            self.update_team_displays()
            self.save_teams()
        else:
            messagebox.showwarning("No Selection", "Please select teams to remove.")
    
    def load_default_teams(self):
        """Load default VCT teams."""
        default_teams = [
            "Sentinels", "Fnatic", "Paper Rex", "Team Liquid", "G2 Esports",
            "NRG", "LOUD", "DRX", "NAVI", "FPX", "100 Thieves", "Cloud9",
            "KRU Esports", "Leviatan", "EDward Gaming", "ASE", "Trace Esports"
        ]
        
        self.teams = default_teams
        self.update_team_displays()
        self.save_teams()
        self.log_message(f"Loaded {len(default_teams)} default teams")
    
    def predict_match(self):
        """Predict a match."""
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
        
        # Simple mock prediction
        import random
        winner = team1 if random.random() > 0.5 else team2
        confidence = random.uniform(0.55, 0.85)
        team1_prob = confidence if winner == team1 else 1 - confidence
        team2_prob = 1 - team1_prob
        
        result = f"""Match Prediction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Match: {team1} vs {team2}
Tournament: {tournament}

Predicted Winner: {winner}
Confidence: {confidence:.1%}
Team 1 ({team1}) Win Probability: {team1_prob:.1%}
Team 2 ({team2}) Win Probability: {team2_prob:.1%}

Note: This is a demo prediction. For real predictions, ensure team data 
has been collected and models are trained.
"""
        
        self.log_message(result)
        self.set_status("Ready")
    
    def collect_data(self):
        """Collect data for selected teams."""
        selected = self.teams_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select teams to collect data for.")
            return
        
        selected_teams = [self.teams[i] for i in selected]
        
        def collect():
            self.set_status(f"Collecting data for {len(selected_teams)} teams...")
            self.log_message(f"Starting data collection for {len(selected_teams)} teams...")
            
            for team in selected_teams:
                self.log_message(f"Collecting data for {team}...")
                
                try:
                    # Try to use enhanced scraper
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
            
            self.log_message("Data collection completed.")
            self.set_status("Ready")
            messagebox.showinfo("Complete", f"Data collection completed for {len(selected_teams)} teams.")
        
        threading.Thread(target=collect, daemon=True).start()
    
    def system_status(self):
        """Check system status."""
        def check_status():
            self.set_status("Checking system status...")
            
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, 'production_deployment.py', '--status'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.log_message(f"System Status:\n{result.stdout}")
                else:
                    self.log_message(f"Status check failed:\n{result.stderr}")
            
            except FileNotFoundError:
                self.log_message("production_deployment.py not found. Basic system info:")
                self.log_message(f"Python: {sys.executable}")
                self.log_message(f"Working directory: {Path.cwd()}")
                self.log_message(f"Data directory: {self.data_dir.absolute()}")
                self.log_message(f"Teams loaded: {len(self.teams)}")
            
            except Exception as e:
                self.log_message(f"Error checking status: {e}")
            
            self.set_status("Ready")
        
        threading.Thread(target=check_status, daemon=True).start()
    
    def performance_check(self):
        """Run performance check."""
        def check_performance():
            self.set_status("Checking performance...")
            
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, 'performance_monitor.py'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Get last few lines for summary
                    lines = result.stdout.split('\n')
                    summary_lines = lines[-15:] if len(lines) > 15 else lines
                    self.log_message("Performance Check Results:\n" + '\n'.join(summary_lines))
                else:
                    self.log_message(f"Performance check failed:\n{result.stderr}")
            
            except FileNotFoundError:
                self.log_message("performance_monitor.py not found. Performance monitoring unavailable.")
            except Exception as e:
                self.log_message(f"Error checking performance: {e}")
            
            self.set_status("Ready")
        
        threading.Thread(target=check_performance, daemon=True).start()
    
    def add_match_result(self):
        """Add a match result via dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Match Result")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        
        # Input fields
        tk.Label(dialog, text="Team 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        team1_var = tk.StringVar()
        tk.Entry(dialog, textvariable=team1_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(dialog, text="Team 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        team2_var = tk.StringVar()
        tk.Entry(dialog, textvariable=team2_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(dialog, text="Winner:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        winner_var = tk.StringVar()
        tk.Entry(dialog, textvariable=winner_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        
        tk.Label(dialog, text="Score:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        score_var = tk.StringVar()
        tk.Entry(dialog, textvariable=score_var, width=30).grid(row=3, column=1, padx=5, pady=5)
        
        tk.Label(dialog, text="Tournament:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        tournament_var = tk.StringVar(value="VCT Champions")
        tk.Entry(dialog, textvariable=tournament_var, width=30).grid(row=4, column=1, padx=5, pady=5)
        
        def add_result():
            result_text = f"""Match Result Added - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Match: {team1_var.get()} vs {team2_var.get()}
Winner: {winner_var.get()}
Score: {score_var.get()}
Tournament: {tournament_var.get()}
"""
            self.log_message(result_text)
            messagebox.showinfo("Success", "Match result recorded!")
            dialog.destroy()
        
        # Buttons
        btn_frame = tk.Frame(dialog)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        tk.Button(btn_frame, text="Add Result", command=add_result).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def setup_automation(self):
        """Setup automation."""
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 'production_deployment.py', '--setup-cron'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Show results in new window
                dialog = tk.Toplevel(self.root)
                dialog.title("Automation Setup")
                dialog.geometry("700x500")
                
                text_widget = ScrolledText(dialog)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                text_widget.insert('1.0', result.stdout)
                
                tk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
            else:
                messagebox.showerror("Error", f"Automation setup failed:\n{result.stderr}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not setup automation: {e}")
    
    def view_logs(self):
        """View recent logs."""
        try:
            log_files = list(self.logs_dir.glob("*.log"))
            if log_files:
                # Get most recent log file
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                
                with open(latest_log, 'r') as f:
                    logs = f.read()
                
                # Show last 50 lines
                lines = logs.split('\n')
                recent_logs = '\n'.join(lines[-50:]) if len(lines) > 50 else logs
                
                self.log_message(f"Recent Logs from {latest_log.name}:\n{'='*50}\n{recent_logs}")
            else:
                self.log_message("No log files found.")
        
        except Exception as e:
            self.log_message(f"Error loading logs: {e}")
    
    def log_message(self, message):
        """Add a message to the output area."""
        self.output_text.insert(tk.END, f"{message}\n\n")
        self.output_text.see(tk.END)
    
    def set_status(self, message):
        """Update the status bar."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def run(self):
        """Start the GUI."""
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.root.mainloop()

def main():
    """Main entry point."""
    print("Starting VCT Prediction System GUI...")
    app = VCTGui()
    app.run()

if __name__ == "__main__":
    main()
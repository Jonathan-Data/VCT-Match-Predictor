#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import yaml
from pathlib import Path

def create_minimal_gui():
    print("Creating minimal VCT GUI...")
    
    # Create window
    root = tk.Tk()
    root.title("VCT Predictor - Minimal")
    root.geometry("600x500+200+100")
    root.configure(bg='white')
    
    # Force to front
    root.lift()
    root.attributes('-topmost', True)
    root.after(2000, lambda: root.attributes('-topmost', False))
    
    print("Window created")
    
    # Load teams
    try:
        config_path = Path(__file__).parent / "config" / "teams.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        teams = [team_info['name'] for team_info in config['teams'].values()]
        teams.sort()
        print(f"Loaded {len(teams)} teams")
    except Exception as e:
        print(f"Error loading teams: {e}")
        teams = ["Team A", "Team B", "Team C"]
    
    # Main container
    main_frame = tk.Frame(root, bg='white', padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title = tk.Label(main_frame, text="VCT 2025 Match Predictor", 
                    font=('Arial', 18, 'bold'), bg='white', fg='blue')
    title.pack(pady=(0, 20))
    
    # Team selection
    team_frame = tk.Frame(main_frame, bg='lightgray', relief='ridge', bd=2)
    team_frame.pack(fill=tk.X, pady=10, padx=10, ipady=10)
    
    tk.Label(team_frame, text="Select Teams:", font=('Arial', 12, 'bold'), 
             bg='lightgray').pack(pady=5)
    
    # Team dropdowns
    select_frame = tk.Frame(team_frame, bg='lightgray')
    select_frame.pack(pady=5)
    
    # Team 1
    tk.Label(select_frame, text="Team 1:", bg='lightgray').pack(side=tk.LEFT, padx=5)
    team1_var = tk.StringVar()
    team1_combo = ttk.Combobox(select_frame, textvariable=team1_var, values=teams[:10], 
                              state="readonly", width=20)
    team1_combo.pack(side=tk.LEFT, padx=5)
    
    tk.Label(select_frame, text="VS", font=('Arial', 12, 'bold'), 
             bg='lightgray', fg='red').pack(side=tk.LEFT, padx=15)
    
    # Team 2
    tk.Label(select_frame, text="Team 2:", bg='lightgray').pack(side=tk.LEFT, padx=5)
    team2_var = tk.StringVar()
    team2_combo = ttk.Combobox(select_frame, textvariable=team2_var, values=teams[:10], 
                              state="readonly", width=20)
    team2_combo.pack(side=tk.LEFT, padx=5)
    
    # Predict button
    def predict():
        t1 = team1_var.get()
        t2 = team2_var.get()
        if t1 and t2 and t1 != t2:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Prediction: {t1} vs {t2}\\n")
            result_text.insert(tk.END, f"Winner: {t1} (65% confidence)\\n")
            result_text.insert(tk.END, "This is a mock prediction.\\n")
        else:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Please select two different teams.\\n")
    
    predict_btn = tk.Button(main_frame, text="PREDICT MATCH", command=predict,
                           font=('Arial', 14, 'bold'), bg='green', fg='white',
                           padx=20, pady=5)
    predict_btn.pack(pady=20)
    
    # Results
    result_frame = tk.Frame(main_frame, bg='lightblue', relief='sunken', bd=2)
    result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    tk.Label(result_frame, text="Results:", font=('Arial', 12, 'bold'), 
             bg='lightblue').pack(anchor='w', padx=5, pady=5)
    
    result_text = tk.Text(result_frame, height=8, width=60, font=('Courier', 10))
    result_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
    
    result_text.insert(tk.END, f"Minimal VCT Predictor Ready!\\n")
    result_text.insert(tk.END, f"Loaded {len(teams)} teams from config.\\n")
    result_text.insert(tk.END, "Select two teams and click PREDICT MATCH.\\n")
    
    print("All components created, starting mainloop...")
    root.mainloop()
    print("GUI closed")

if __name__ == "__main__":
    create_minimal_gui()
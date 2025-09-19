#!/usr/bin/env python3
"""
Debug version of VCT GUI to identify issues
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import yaml
from pathlib import Path

def debug_gui():
    """Create a simple debug GUI to test basic functionality"""
    try:
        # Create main window
        root = tk.Tk()
        root.title("VCT Debug GUI")
        root.geometry("600x400")
        
        # Force window to front on macOS
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        print("Window created successfully")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        print("Main frame created")
        
        # Add title
        title_label = ttk.Label(main_frame, text="VCT Debug GUI", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        print("Title label created")
        
        # Test loading configuration
        try:
            config_path = Path(__file__).parent / "config" / "teams.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            teams = [team_info['name'] for team_info in config['teams'].values()]
            teams.sort()
            
            status_text = f"✅ Configuration loaded successfully\\n{len(teams)} teams found"
            print(f"Config loaded: {len(teams)} teams")
            
        except Exception as e:
            status_text = f"❌ Configuration loading failed:\\n{str(e)}"
            teams = []
            print(f"Config error: {e}")
        
        # Status label
        status_label = ttk.Label(main_frame, text=status_text, font=('Arial', 12))
        status_label.pack(pady=10)
        
        # Team selection
        if teams:
            team_frame = ttk.Frame(main_frame)
            team_frame.pack(pady=10)
            
            ttk.Label(team_frame, text="Select a team:").pack(side=tk.LEFT, padx=(0, 10))
            
            team_var = tk.StringVar()
            team_combo = ttk.Combobox(team_frame, textvariable=team_var, values=teams[:10], state="readonly")
            team_combo.pack(side=tk.LEFT)
            
            print("Team selection created")
        
        # Test button
        def test_button():
            messagebox.showinfo("Test", "Button works! GUI is functioning properly.")
        
        test_btn = ttk.Button(main_frame, text="Test Button", command=test_button)
        test_btn.pack(pady=10)
        
        print("Test button created")
        
        # Text area for output
        text_area = tk.Text(main_frame, height=10, width=50)
        text_area.pack(pady=10, fill=tk.BOTH, expand=True)
        
        text_area.insert(tk.END, "Debug GUI initialized successfully!\\n")
        text_area.insert(tk.END, f"Teams loaded: {len(teams) if 'teams' in locals() else 0}\\n")
        text_area.insert(tk.END, "If you can see this text, the GUI is working properly.\\n")
        
        print("Text area created")
        print("Starting mainloop...")
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        print(f"Error in debug GUI: {e}")
        print(traceback.format_exc())
        messagebox.showerror("Debug GUI Error", f"Failed to create debug GUI:\\n{e}")

if __name__ == "__main__":
    print("Starting debug GUI...")
    debug_gui()
    print("Debug GUI finished.")
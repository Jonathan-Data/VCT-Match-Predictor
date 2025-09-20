#!/usr/bin/env python3
"""
Debug version to check manual prediction tab layout
"""

import tkinter as tk
from tkinter import ttk

class DebugManualPredictionTab:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Debug: Manual Prediction Tab")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create manual prediction tab layout
        self.create_manual_prediction_tab(main_frame)
        
    def create_manual_prediction_tab(self, parent):
        """Create manual prediction interface (debug version)"""
        print("Creating manual prediction tab...")
        
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        
        # Team 1 selection
        label1 = ttk.Label(parent, text="Team 1:")
        label1.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        print(f"Team 1 label created at row=0, column=0")
        
        self.team1_var = tk.StringVar()
        self.team1_combo = ttk.Combobox(parent, textvariable=self.team1_var, 
                                       state="readonly", width=25)
        self.team1_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.team1_combo['values'] = ['Team A', 'Team B', 'Team C']  # Test values
        print(f"Team 1 combo created at row=0, column=1")
        
        # VS label
        vs_label = ttk.Label(parent, text="VS", font=('Arial', 12, 'bold'))
        vs_label.grid(row=0, column=2, padx=10)
        print(f"VS label created at row=0, column=2")
        
        # Team 2 selection
        label2 = ttk.Label(parent, text="Team 2:")
        label2.grid(row=0, column=3, sticky=tk.W, padx=(10, 5))
        print(f"Team 2 label created at row=0, column=3")
        
        self.team2_var = tk.StringVar()
        self.team2_combo = ttk.Combobox(parent, textvariable=self.team2_var, 
                                       state="readonly", width=25)
        self.team2_combo.grid(row=0, column=4, sticky=(tk.W, tk.E))
        self.team2_combo['values'] = ['Team X', 'Team Y', 'Team Z']  # Test values
        print(f"Team 2 combo created at row=0, column=4")
        
        # VLR ID input option
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(10, 5))
        print(f"Separator created at row=1")
        
        id_label = ttk.Label(parent, text="Or use VLR IDs directly:", 
                            font=('Arial', 9, 'italic'))
        id_label.grid(row=2, column=0, columnspan=5, pady=(0, 5))
        print(f"ID label created at row=2")
        
        # VLR ID inputs
        vlr_frame = ttk.Frame(parent)
        vlr_frame.grid(row=3, column=0, columnspan=5, sticky=(tk.W, tk.E))
        vlr_frame.columnconfigure(1, weight=1)
        vlr_frame.columnconfigure(4, weight=1)
        print(f"VLR frame created at row=3")
        
        ttk.Label(vlr_frame, text="Team 1 ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.team1_id_var = tk.StringVar()
        ttk.Entry(vlr_frame, textvariable=self.team1_id_var, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(vlr_frame, text="Team 2 ID:").grid(row=0, column=3, sticky=tk.W, padx=(20, 5))
        self.team2_id_var = tk.StringVar()
        ttk.Entry(vlr_frame, textvariable=self.team2_id_var, width=10).grid(row=0, column=4, sticky=tk.W)
        print(f"VLR ID entries created")
        
        # Predict button - THIS IS THE KEY PART
        print(f"Creating predict button at row=4...")
        self.predict_button = ttk.Button(parent, text="Predict Match", 
                                        command=self.predict_match, width=20)
        self.predict_button.grid(row=4, column=0, columnspan=5, pady=(15, 0))
        print(f"✅ Predict button created at row=4, columnspan=5, pady=(15, 0)")
        
        # Add visual confirmation
        self.predict_button.configure(style='Accent.TButton')  # Try to make it stand out
        
        # Add some debug info
        debug_label = ttk.Label(parent, text="DEBUG: If you can see this, the button should be visible too!", 
                               foreground="red", font=('Arial', 10, 'bold'))
        debug_label.grid(row=5, column=0, columnspan=5, pady=10)
        print(f"Debug label created at row=5")
    
    def predict_match(self):
        """Test prediction method"""
        print("🎉 PREDICT BUTTON CLICKED!")
        team1 = self.team1_var.get()
        team2 = self.team2_var.get()
        team1_id = self.team1_id_var.get()
        team2_id = self.team2_id_var.get()
        
        result = f"Team 1: {team1}\nTeam 2: {team2}\nTeam 1 ID: {team1_id}\nTeam 2 ID: {team2_id}"
        
        # Show result in a popup
        import tkinter.messagebox as msgbox
        msgbox.showinfo("Prediction Test", f"Button works!\n\n{result}")
    
    def run(self):
        print("Starting GUI...")
        self.root.mainloop()

if __name__ == "__main__":
    app = DebugManualPredictionTab()
    app.run()
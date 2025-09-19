#!/usr/bin/env python3
"""
VCT 2025 Champions Match Predictor - GUI Launcher

Simple launcher script for the graphical user interface.
"""

if __name__ == "__main__":
    try:
        from ml_gui import MLPredictionGUI
        print("Starting VCT 2025 Champions Match Predictor GUI (Enhanced ML Version)...")
        app = MLPredictionGUI()
        app.run()
    except ImportError as e:
        print(f"Error: GUI dependencies not available: {e}")
        print("Please ensure tkinter is installed and ml_gui.py is in the project directory.")
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()

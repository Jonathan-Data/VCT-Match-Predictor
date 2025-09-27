
"""
VCT Predictor - GUI Launcher
Simple launcher for the main VCT prediction system GUI
"""

import sys
import os
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

try:
    from vct_gui import main

    if __name__ == "__main__":
        print("Starting VCT Prediction System GUI...")
        main()

except ImportError as e:
    print(f"Error: Required modules not found: {e}")
    print("\nPlease install required dependencies:")
    print("pip3 install tkinter pandas numpy scikit-learn requests beautifulsoup4 selenium joblib schedule")
    sys.exit(1)
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1)
#!/usr/bin/env python3
"""
VCT 2025 Champions Match Predictor - Main Entry Point

This is the main entry point for the VCT 2025 Champions Match Predictor.
It provides both GUI and command-line interfaces.
"""

import sys
import argparse
from pathlib import Path

def launch_gui():
    """Launch the GUI application"""
    try:
        from ml_gui import MLPredictionGUI
        print("🎮 Starting VCT 2025 Champions Match Predictor GUI...")
        print("📊 Enhanced ML Model with 83%+ accuracy")
        print("🏆 Ready for Champions Paris 2025!")
        print()
        
        app = MLPredictionGUI()
        app.run()
    except ImportError as e:
        print(f"❌ Error: GUI dependencies not available: {e}")
        print("Please ensure tkinter is installed and ml_gui.py is available.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_info():
    """Show project information"""
    print("🎮 VCT 2025 Champions Match Predictor")
    print("=" * 50)
    print("📊 Enhanced ML-based prediction system")
    print("🤖 Training data: VCT 2024 comprehensive dataset")
    print("🎯 Model accuracy: 83%+ (ensemble model)")
    print("🏆 Ready for: VCT Champions 2025 Paris")
    print()
    print("📁 Project Structure:")
    print("├── ml_gui.py          - Main GUI application")
    print("├── launch_gui.py      - GUI launcher")
    print("├── src/               - Source code")
    print("│   ├── models/        - ML models and predictors")
    print("│   ├── data_collection/ - Data collection tools")
    print("│   └── preprocessing/ - Data processing")
    print("├── data/              - Tournament data")
    print("├── config/            - Configuration files")
    print("└── tests/             - Test suite")
    print()
    print("🚀 Usage:")
    print("  python run.py          - Launch GUI")
    print("  python run.py --info   - Show this information")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VCT 2025 Champions Match Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--info', 
        action='store_true', 
        help='Show project information'
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_info()
    else:
        launch_gui()

if __name__ == "__main__":
    main()
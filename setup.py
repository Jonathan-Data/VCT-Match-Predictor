#!/usr/bin/env python3
"""
Setup script for VCT 2025 Champions Match Predictor
Handles installation, data downloading, and initial project setup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"Python version check passed: {sys.version}")

def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling Python dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("Warning: requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_kaggle_credentials():
    """Check and setup Kaggle API credentials"""
    print("\nChecking Kaggle credentials...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_config = kaggle_dir / "kaggle.json"
    
    if not kaggle_config.exists():
        print("\nKaggle credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place the file at:", str(kaggle_config))
        print("4. Run this setup again")
        
        # Create directory if it doesn't exist
        kaggle_dir.mkdir(exist_ok=True)
        
        return False
    
    # Set proper permissions
    try:
        os.chmod(str(kaggle_config), 0o600)
        print("Kaggle credentials found and configured")
        return True
    except Exception as e:
        print(f"Error setting Kaggle credentials permissions: {e}")
        return False

def download_datasets():
    """Download all configured Kaggle datasets"""
    print("\nDownloading Kaggle datasets...")
    
    try:
        # Import the data downloader
        sys.path.append(str(Path(__file__).parent / "src"))
        from data_collection.kaggle_downloader import KaggleDataDownloader
        
        downloader = KaggleDataDownloader()
        datasets = downloader.download_all_datasets()
        
        print(f"Successfully downloaded {len(datasets)} datasets")
        for name, path in datasets.items():
            print(f"  - {name}: {path}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        print("You can download datasets manually later using:")
        print("python src/data_collection/kaggle_downloader.py")
        return False

def create_directories():
    """Create necessary project directories"""
    print("\nCreating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "src/models",
        "logs",
        "output"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  - Created: {directory}")

def verify_project_structure():
    """Verify that all necessary files and directories exist"""
    print("\nVerifying project structure...")
    
    required_files = [
        "requirements.txt",
        "config/teams.yaml",
        "src/data_collection/kaggle_downloader.py",
        "src/models/enhanced_ml_predictor.py",
        "ml_gui.py",
        "run.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (Path(__file__).parent / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("Project structure verification passed")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print("\nSetting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        env_content = """# VCT Predictor Environment Configuration
PYTHONPATH=./src
DATA_DIR=./data
MODELS_DIR=./src/models
LOGS_DIR=./logs
"""
        env_file.write_text(env_content)
        print("Created .env file")
    
    return True

def run_initial_tests():
    """Run basic tests to ensure everything is working"""
    print("\nRunning initial tests...")
    
    try:
        # Test imports
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data_collection.kaggle_downloader import KaggleDataDownloader
        from models.enhanced_ml_predictor import EnhancedVCTPredictor
        
        print("  - Import tests passed")
        
        # Test configuration loading
        config_path = Path(__file__).parent / "config" / "teams.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print(f"  - Configuration loaded: {len(config['teams'])} teams, {len(config['kaggle_datasets'])} datasets")
        
        return True
        
    except Exception as e:
        print(f"  - Tests failed: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎮 VCT 2025 Champions Predictor Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Launch the enhanced ML GUI:")
    print("   python run.py")
    print("\n2. Or use the command line interface:")
    print("   python main.py predict \"Sentinels\" \"Fnatic\"")
    print("\n3. To download additional data manually:")
    print("   python src/data_collection/kaggle_downloader.py")
    print("\n4. For help and documentation:")
    print("   python run.py --help")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup VCT 2025 Champions Predictor")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip dataset download (for faster setup)")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal setup without datasets")
    
    args = parser.parse_args()
    
    print("🎮 VCT 2025 Champions Match Predictor Setup")
    print("=" * 50)
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Verifying project structure", verify_project_structure),
        ("Setting up environment", setup_environment),
    ]
    
    if not args.minimal and not args.skip_download:
        steps.extend([
            ("Setting up Kaggle credentials", setup_kaggle_credentials),
            ("Downloading datasets", download_datasets),
        ])
    
    steps.append(("Running initial tests", run_initial_tests))
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n🔧 {step_name}...")
        try:
            success = step_function()
            if success:
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"⚠️  {step_name} completed with warnings")
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "="*50)
    if failed_steps:
        print(f"Setup completed with {len(failed_steps)} warnings/failures:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nYou may need to address these issues manually.")
    else:
        print("✅ All setup steps completed successfully!")
    
    print_next_steps()

if __name__ == "__main__":
    main()
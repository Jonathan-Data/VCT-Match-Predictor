#!/usr/bin/env python3
"""
VCT Automated Data Update System
Scheduled jobs to keep team data current from multiple sources
"""

import sys
import os
import json
import logging
import schedule
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from vlr_scraper import VLRScraper
    from enhanced_rib_scraper import scrape_team_data
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Some features may be limited")

class AutomatedDataUpdater:
    """
    Manages scheduled data updates from multiple sources.
    """
    
    def __init__(self, data_dir: str = "data", logs_dir: str = "logs"):
        """Initialize the automated data updater."""
        self.data_dir = Path(data_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Data sources
        self.vlr_scraper = None
        self.rib_available = True
        
        # Update tracking
        self.last_vlr_update = None
        self.last_rib_update = None
        self.update_history = []
        
        # Configuration
        self.config = {
            'vlr_update_interval': 6,  # hours
            'rib_update_interval': 8,  # hours
            'max_retries': 3,
            'retry_delay': 300,  # seconds (5 minutes)
            'data_retention_days': 30,
            'enable_vlr': True,
            'enable_rib': True,
            'enable_performance_monitoring': True
        }
        
        self._load_config()
        self._init_scrapers()
        
        self.logger.info("🤖 VCT Automated Data Updater initialized")
    
    def _setup_logging(self):
        """Set up logging for automated updates."""
        log_file = self.logs_dir / f"data_updater_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self):
        """Load configuration from file if it exists."""
        config_file = self.data_dir / "updater_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                self.config.update(saved_config)
                self.logger.info("📝 Loaded configuration from file")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load config: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        config_file = self.data_dir / "updater_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("💾 Configuration saved")
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {e}")
    
    def _init_scrapers(self):
        """Initialize data scrapers."""
        try:
            if self.config['enable_vlr']:
                self.vlr_scraper = VLRScraper()
                self.logger.info("✅ VLR scraper initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ VLR scraper unavailable: {e}")
        
        try:
            if self.config['enable_rib']:
                # Test rib scraper
                test_result = scrape_team_data("test")
                if test_result:
                    self.logger.info("✅ Rib scraper available")
                else:
                    self.rib_available = False
        except Exception as e:
            self.logger.warning(f"⚠️ Rib scraper unavailable: {e}")
            self.rib_available = False
    
    def update_vlr_data(self) -> Dict[str, Any]:
        """Update team data from VLR.gg."""
        self.logger.info("🔄 Starting VLR data update...")
        
        update_result = {
            'source': 'vlr.gg',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'teams_updated': 0,
            'error': None
        }
        
        try:
            if not self.vlr_scraper:
                raise Exception("VLR scraper not initialized")
            
            # Get list of teams to update
            teams_to_update = self._get_teams_list()
            
            updated_teams = []
            for team_name in teams_to_update:
                try:
                    self.logger.info(f"📊 Updating {team_name} from VLR...")
                    team_data = self.vlr_scraper.get_team_data(team_name)
                    
                    if team_data and 'error' not in team_data:
                        # Save team data
                        team_file = self.data_dir / f"vlr_{team_name.replace(' ', '_').lower()}.json"
                        with open(team_file, 'w') as f:
                            json.dump({
                                'team_name': team_name,
                                'data': team_data,
                                'updated_at': datetime.now().isoformat(),
                                'source': 'vlr.gg'
                            }, f, indent=2)
                        
                        updated_teams.append(team_name)
                        self.logger.info(f"✅ Updated {team_name}")
                    else:
                        self.logger.warning(f"⚠️ No data for {team_name}")
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error updating {team_name}: {e}")
            
            update_result['success'] = len(updated_teams) > 0
            update_result['teams_updated'] = len(updated_teams)
            
            if update_result['success']:
                self.last_vlr_update = datetime.now()
                self.logger.info(f"✅ VLR update completed: {len(updated_teams)} teams updated")
            
        except Exception as e:
            update_result['error'] = str(e)
            self.logger.error(f"❌ VLR update failed: {e}")
        
        self.update_history.append(update_result)
        return update_result
    
    def update_rib_data(self) -> Dict[str, Any]:
        """Update team data from rib.gg."""
        self.logger.info("🔄 Starting rib.gg data update...")
        
        update_result = {
            'source': 'rib.gg',
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'teams_updated': 0,
            'error': None
        }
        
        try:
            if not self.rib_available:
                raise Exception("Rib scraper not available")
            
            # Get list of teams to update
            teams_to_update = self._get_teams_list()
            
            updated_teams = []
            for team_name in teams_to_update:
                try:
                    self.logger.info(f"📊 Updating {team_name} from rib.gg...")
                    team_data = scrape_team_data(team_name)
                    
                    if team_data and team_data.get('success', False):
                        # Save team data
                        team_file = self.data_dir / f"rib_{team_name.replace(' ', '_').lower()}.json"
                        with open(team_file, 'w') as f:
                            json.dump({
                                'team_name': team_name,
                                'data': team_data,
                                'updated_at': datetime.now().isoformat(),
                                'source': 'rib.gg'
                            }, f, indent=2)
                        
                        updated_teams.append(team_name)
                        self.logger.info(f"✅ Updated {team_name}")
                    else:
                        self.logger.warning(f"⚠️ No data for {team_name}")
                    
                    # Rate limiting
                    time.sleep(3)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error updating {team_name}: {e}")
            
            update_result['success'] = len(updated_teams) > 0
            update_result['teams_updated'] = len(updated_teams)
            
            if update_result['success']:
                self.last_rib_update = datetime.now()
                self.logger.info(f"✅ Rib update completed: {len(updated_teams)} teams updated")
            
        except Exception as e:
            update_result['error'] = str(e)
            self.logger.error(f"❌ Rib update failed: {e}")
        
        self.update_history.append(update_result)
        return update_result
    
    def _get_teams_list(self) -> List[str]:
        """Get list of teams to update."""
        # Common VCT teams to track
        priority_teams = [
            "Sentinels", "Fnatic", "Paper Rex", "Team Liquid", 
            "G2 Esports", "NRG", "LOUD", "DRX", "NAVI", "FPX",
            "100 Thieves", "Cloud9", "KRU Esports", "Leviatán",
            "EDward Gaming", "ASE", "Trace Esports", "FUT Esports"
        ]
        
        # Check if we have a teams list file
        teams_file = self.data_dir / "teams_to_update.json"
        if teams_file.exists():
            try:
                with open(teams_file, 'r') as f:
                    data = json.load(f)
                return data.get('teams', priority_teams)
            except:
                pass
        
        return priority_teams
    
    def update_teams_list(self, teams: List[str]):
        """Update the list of teams to monitor."""
        teams_file = self.data_dir / "teams_to_update.json"
        try:
            with open(teams_file, 'w') as f:
                json.dump({
                    'teams': teams,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
            self.logger.info(f"📝 Updated teams list: {len(teams)} teams")
        except Exception as e:
            self.logger.error(f"❌ Error updating teams list: {e}")
    
    def run_performance_check(self) -> Dict[str, Any]:
        """Run performance monitoring check."""
        self.logger.info("📊 Running performance monitoring check...")
        
        try:
            if not self.config['enable_performance_monitoring']:
                return {'message': 'Performance monitoring disabled'}
            
            monitor = PerformanceMonitor()
            
            # Evaluate any new predictions
            evaluation_results = monitor.evaluate_predictions()
            
            # Get performance summary
            summary = monitor.get_performance_summary(days_back=7)
            
            self.logger.info("✅ Performance monitoring check completed")
            
            return {
                'evaluation_results': evaluation_results,
                'performance_summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self):
        """Clean up old data files."""
        self.logger.info("🧹 Cleaning up old data...")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['data_retention_days'])
            
            # Clean up data files
            for file_path in self.data_dir.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        self.logger.info(f"🗑️ Removed old file: {file_path.name}")
                except:
                    pass
            
            # Clean up log files
            for log_file in self.logs_dir.glob("*.log"):
                try:
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        self.logger.info(f"🗑️ Removed old log: {log_file.name}")
                except:
                    pass
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the updater."""
        return {
            'timestamp': datetime.now().isoformat(),
            'last_vlr_update': self.last_vlr_update.isoformat() if self.last_vlr_update else None,
            'last_rib_update': self.last_rib_update.isoformat() if self.last_rib_update else None,
            'vlr_scraper_available': self.vlr_scraper is not None,
            'rib_scraper_available': self.rib_available,
            'config': self.config,
            'recent_updates': self.update_history[-10:] if self.update_history else []
        }
    
    def setup_schedule(self):
        """Set up automated scheduling."""
        self.logger.info("⏰ Setting up automated schedules...")
        
        # VLR updates every 6 hours by default
        if self.config['enable_vlr'] and self.vlr_scraper:
            schedule.every(self.config['vlr_update_interval']).hours.do(self.update_vlr_data)
            self.logger.info(f"📅 VLR updates scheduled every {self.config['vlr_update_interval']} hours")
        
        # Rib updates every 8 hours by default  
        if self.config['enable_rib'] and self.rib_available:
            schedule.every(self.config['rib_update_interval']).hours.do(self.update_rib_data)
            self.logger.info(f"📅 Rib updates scheduled every {self.config['rib_update_interval']} hours")
        
        # Performance monitoring every 12 hours
        if self.config['enable_performance_monitoring']:
            schedule.every(12).hours.do(self.run_performance_check)
            self.logger.info("📅 Performance monitoring scheduled every 12 hours")
        
        # Daily cleanup at 2 AM
        schedule.every().day.at("02:00").do(self.cleanup_old_data)
        self.logger.info("📅 Daily cleanup scheduled at 2 AM")
        
        # Initial updates
        self.logger.info("🚀 Running initial data updates...")
        if self.config['enable_vlr']:
            self.update_vlr_data()
        
        if self.config['enable_rib']:
            self.update_rib_data()
    
    def run_scheduler(self):
        """Run the scheduler in a loop."""
        self.logger.info("🔄 Starting scheduler...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Scheduler error: {e}")
    
    def run_manual_update(self, sources: List[str] = None):
        """Run manual updates for specified sources."""
        if sources is None:
            sources = ['vlr', 'rib']
        
        results = {}
        
        if 'vlr' in sources and self.config['enable_vlr']:
            results['vlr'] = self.update_vlr_data()
        
        if 'rib' in sources and self.config['enable_rib']:
            results['rib'] = self.update_rib_data()
        
        if 'performance' in sources and self.config['enable_performance_monitoring']:
            results['performance'] = self.run_performance_check()
        
        return results


def main():
    """Test and demonstrate the automated data updater."""
    print("🤖 VCT Automated Data Updater")
    print("=" * 50)
    
    # Initialize updater
    updater = AutomatedDataUpdater()
    
    # Show status
    print("\n📊 Current Status:")
    status = updater.get_status()
    print(f"VLR Scraper: {'✅' if status['vlr_scraper_available'] else '❌'}")
    print(f"Rib Scraper: {'✅' if status['rib_scraper_available'] else '❌'}")
    print(f"Last VLR Update: {status['last_vlr_update'] or 'Never'}")
    print(f"Last Rib Update: {status['last_rib_update'] or 'Never'}")
    
    # Run manual update test
    print(f"\n🔄 Running manual update test...")
    results = updater.run_manual_update(['vlr'])  # Test with VLR only first
    
    for source, result in results.items():
        print(f"\n{source.upper()} Update Result:")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
        print(f"  Teams Updated: {result['teams_updated']}")
        if result.get('error'):
            print(f"  Error: {result['error']}")
    
    # Ask if user wants to start scheduler
    print(f"\n⏰ Scheduler Options:")
    print("1. Run scheduler (will run continuously)")
    print("2. Setup schedule only (for manual control)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("🚀 Starting continuous scheduler...")
        updater.setup_schedule()
        updater.run_scheduler()
    elif choice == "2":
        print("📅 Setting up schedules...")
        updater.setup_schedule()
        print("✅ Schedules configured. Use run_scheduler() to start.")
    else:
        print("👋 Exiting...")


if __name__ == "__main__":
    main()
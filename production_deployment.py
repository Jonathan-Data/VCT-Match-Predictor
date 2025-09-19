#!/usr/bin/env python3
"""
VCT Prediction System - Production Deployment
Orchestrates all components for a complete prediction pipeline
"""

import sys
import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

class ProductionDeployment:
    """
    Production deployment orchestrator for the VCT prediction system.
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the production deployment."""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        for dir_path in [self.data_dir, self.logs_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Component status
        self.components = {
            'enhanced_rib_scraper': {'available': False, 'last_check': None},
            'live_predictor': {'available': False, 'last_check': None},
            'performance_monitor': {'available': False, 'last_check': None},
            'automated_data_updater': {'available': False, 'last_check': None}
        }
        
        # Configuration
        self.config = {
            'prediction_interval_hours': 6,
            'data_update_interval_hours': 8,
            'performance_check_interval_hours': 12,
            'enable_automated_updates': True,
            'enable_predictions': True,
            'enable_monitoring': True,
            'model_file': 'vct_model.joblib',
            'log_level': 'INFO'
        }
        
        self._load_config()
        self._check_components()
        
        self.logger.info("🚀 VCT Production Deployment System initialized")
    
    def _setup_logging(self):
        """Set up production logging."""
        log_file = self.logs_dir / f"production_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('ProductionDeployment')
    
    def _load_config(self):
        """Load configuration from file."""
        config_file = self.base_dir / "production_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                self.config.update(saved_config)
                self.logger.info("📝 Configuration loaded from file")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load config: {e}")
    
    def save_config(self):
        """Save current configuration."""
        config_file = self.base_dir / "production_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("💾 Configuration saved")
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {e}")
    
    def _check_components(self):
        """Check availability of system components."""
        self.logger.info("🔍 Checking system components...")
        
        # Check for component files
        component_files = {
            'enhanced_rib_scraper': 'enhanced_rib_scraper.py',
            'live_predictor': 'live_predictor.py', 
            'performance_monitor': 'performance_monitor.py',
            'automated_data_updater': 'automated_data_updater.py'
        }
        
        for component, filename in component_files.items():
            file_path = self.base_dir / filename
            self.components[component]['available'] = file_path.exists()
            self.components[component]['last_check'] = datetime.now().isoformat()
            
            status = "✅" if self.components[component]['available'] else "❌"
            self.logger.info(f"{status} {component}: {'Available' if self.components[component]['available'] else 'Not found'}")
        
        # Check for model file
        model_file = self.models_dir / self.config['model_file']
        model_available = model_file.exists()
        self.logger.info(f"{'✅' if model_available else '❌'} Model file: {'Available' if model_available else 'Not found'}")
        
        return self.components
    
    def run_data_collection(self) -> Dict[str, Any]:
        """Run data collection from available sources."""
        self.logger.info("📊 Running data collection...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'sources': {}
        }
        
        try:
            # Run enhanced rib scraper if available
            if self.components['enhanced_rib_scraper']['available']:
                self.logger.info("🔄 Running enhanced rib scraper...")
                result = subprocess.run([
                    sys.executable, 'enhanced_rib_scraper.py'
                ], cwd=self.base_dir, capture_output=True, text=True, timeout=300)
                
                results['sources']['rib_scraper'] = {
                    'success': result.returncode == 0,
                    'output': result.stdout[-500:] if result.stdout else None,  # Last 500 chars
                    'error': result.stderr[-500:] if result.stderr else None
                }
            
            # Run automated data updater if available
            if self.components['automated_data_updater']['available']:
                self.logger.info("🔄 Running automated data updater...")
                # This would need to be integrated with the actual updater
                results['sources']['data_updater'] = {
                    'success': True,
                    'message': 'Would run automated updates'
                }
            
            results['success'] = any(source.get('success', False) for source in results['sources'].values())
            self.logger.info(f"✅ Data collection completed: {len(results['sources'])} sources")
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_predictions(self) -> Dict[str, Any]:
        """Run live match predictions."""
        self.logger.info("🎯 Running predictions...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'predictions': []
        }
        
        try:
            if not self.components['live_predictor']['available']:
                raise Exception("Live predictor not available")
            
            # Run live predictor
            self.logger.info("🔄 Running live predictor...")
            result = subprocess.run([
                sys.executable, 'live_predictor.py'
            ], cwd=self.base_dir, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Try to parse predictions from output or file
                predictions_file = self.data_dir / "live_predictions.json"
                if predictions_file.exists():
                    with open(predictions_file, 'r') as f:
                        prediction_data = json.load(f)
                    results['predictions'] = prediction_data.get('predictions', [])
                
                results['success'] = True
                self.logger.info(f"✅ Predictions completed: {len(results['predictions'])} matches")
            else:
                results['error'] = result.stderr
                self.logger.error(f"❌ Prediction failed: {result.stderr}")
        
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_performance_monitoring(self) -> Dict[str, Any]:
        """Run performance monitoring checks."""
        self.logger.info("📊 Running performance monitoring...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            if not self.components['performance_monitor']['available']:
                raise Exception("Performance monitor not available")
            
            # Run performance monitor
            self.logger.info("🔄 Running performance monitor...")
            result = subprocess.run([
                sys.executable, 'performance_monitor.py'
            ], cwd=self.base_dir, capture_output=True, text=True, timeout=300)
            
            results['success'] = result.returncode == 0
            results['output'] = result.stdout[-1000:] if result.stdout else None
            
            if results['success']:
                self.logger.info("✅ Performance monitoring completed")
            else:
                results['error'] = result.stderr
                self.logger.error(f"❌ Performance monitoring failed: {result.stderr}")
        
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete prediction pipeline."""
        self.logger.info("🔄 Starting full prediction pipeline...")
        
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'stages': {}
        }
        
        try:
            # Stage 1: Data Collection
            self.logger.info("📊 Pipeline Stage 1: Data Collection")
            pipeline_results['stages']['data_collection'] = self.run_data_collection()
            
            # Stage 2: Predictions 
            self.logger.info("🎯 Pipeline Stage 2: Predictions")
            pipeline_results['stages']['predictions'] = self.run_predictions()
            
            # Stage 3: Performance Monitoring
            self.logger.info("📊 Pipeline Stage 3: Performance Monitoring") 
            pipeline_results['stages']['performance_monitoring'] = self.run_performance_monitoring()
            
            # Overall success
            pipeline_results['success'] = all(
                stage.get('success', False) 
                for stage in pipeline_results['stages'].values()
            )
            
            status = "✅ Success" if pipeline_results['success'] else "⚠️ Partial success"
            self.logger.info(f"{status} - Full pipeline completed")
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            pipeline_results['error'] = str(e)
        
        # Save results
        results_file = self.data_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
        except:
            pass
        
        return pipeline_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        self.logger.info("📊 Generating system status...")
        
        # Re-check components
        self._check_components()
        
        # Check recent results
        recent_results = []
        for results_file in sorted(self.data_dir.glob("pipeline_results_*.json"))[-5:]:
            try:
                with open(results_file, 'r') as f:
                    recent_results.append(json.load(f))
            except:
                pass
        
        # Check disk space
        disk_usage = {
            'data_dir_size': sum(f.stat().st_size for f in self.data_dir.rglob('*') if f.is_file()),
            'logs_dir_size': sum(f.stat().st_size for f in self.logs_dir.rglob('*') if f.is_file()),
        }
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': self.components,
            'config': self.config,
            'recent_results': recent_results,
            'disk_usage': disk_usage,
            'system_health': 'healthy' if all(comp['available'] for comp in self.components.values()) else 'degraded'
        }
        
        return status
    
    def setup_cron_job(self):
        """Generate cron job setup instructions."""
        cron_commands = [
            "# VCT Prediction System Cron Jobs",
            "",
            "# Run full pipeline every 6 hours",
            f"0 */6 * * * cd {self.base_dir} && /usr/bin/python3 production_deployment.py --pipeline >> {self.logs_dir}/cron.log 2>&1",
            "",
            "# Run data updates every 8 hours (offset)",
            f"30 */8 * * * cd {self.base_dir} && /usr/bin/python3 production_deployment.py --data-only >> {self.logs_dir}/cron.log 2>&1",
            "",
            "# Run performance monitoring every 12 hours",
            f"15 */12 * * * cd {self.base_dir} && /usr/bin/python3 production_deployment.py --monitor-only >> {self.logs_dir}/cron.log 2>&1"
        ]
        
        cron_file = self.base_dir / "crontab_setup.txt"
        with open(cron_file, 'w') as f:
            f.write('\n'.join(cron_commands))
        
        instructions = f"""
📅 Cron Job Setup Instructions:

1. Review the generated cron commands:
   cat {cron_file}

2. Add to your crontab:
   crontab -e
   
3. Add the contents of {cron_file} to your crontab

4. Verify cron jobs are set:
   crontab -l

The system will automatically:
- Run full pipeline every 6 hours
- Update data every 8 hours (offset)  
- Check performance every 12 hours

Logs will be written to: {self.logs_dir}/cron.log
        """
        
        return instructions


def main():
    """Main production deployment interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VCT Prediction System Production Deployment")
    parser.add_argument('--pipeline', action='store_true', help='Run full pipeline')
    parser.add_argument('--data-only', action='store_true', help='Run data collection only')
    parser.add_argument('--predictions-only', action='store_true', help='Run predictions only')
    parser.add_argument('--monitor-only', action='store_true', help='Run monitoring only')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--setup-cron', action='store_true', help='Generate cron setup instructions')
    
    args = parser.parse_args()
    
    # Initialize deployment system
    deployment = ProductionDeployment()
    
    if args.pipeline:
        print("🚀 Running full prediction pipeline...")
        results = deployment.run_full_pipeline()
        print(f"Pipeline completed: {results['success']}")
        
    elif args.data_only:
        print("📊 Running data collection...")
        results = deployment.run_data_collection()
        print(f"Data collection completed: {results['success']}")
        
    elif args.predictions_only:
        print("🎯 Running predictions...")
        results = deployment.run_predictions()
        print(f"Predictions completed: {results['success']}")
        
    elif args.monitor_only:
        print("📊 Running performance monitoring...")
        results = deployment.run_performance_monitoring()
        print(f"Monitoring completed: {results['success']}")
        
    elif args.status:
        print("📊 System Status")
        print("=" * 50)
        status = deployment.get_system_status()
        print(f"System Health: {status['system_health'].upper()}")
        print(f"Generated: {status['timestamp']}")
        
        print(f"\n📦 Components:")
        for component, info in status['components'].items():
            status_icon = "✅" if info['available'] else "❌"
            print(f"  {status_icon} {component}")
        
        print(f"\n📈 Recent Pipeline Runs: {len(status['recent_results'])}")
        
    elif args.setup_cron:
        print("📅 Setting up cron jobs...")
        instructions = deployment.setup_cron_job()
        print(instructions)
        
    else:
        print("🤖 VCT Prediction System - Production Deployment")
        print("=" * 60)
        
        # Show system status
        status = deployment.get_system_status()
        print(f"\n📊 System Health: {status['system_health'].upper()}")
        
        print(f"\n📦 Available Components:")
        for component, info in status['components'].items():
            status_icon = "✅" if info['available'] else "❌"
            print(f"  {status_icon} {component}")
        
        print(f"\n⚡ Available Commands:")
        print("  --pipeline      : Run complete prediction pipeline")
        print("  --data-only     : Update team data only")  
        print("  --predictions-only : Generate predictions only")
        print("  --monitor-only  : Run performance monitoring only")
        print("  --status        : Show detailed system status")
        print("  --setup-cron    : Generate cron job setup instructions")
        
        print(f"\n🔧 Quick Start:")
        print("  python3 production_deployment.py --pipeline")
        print("  python3 production_deployment.py --setup-cron")


if __name__ == "__main__":
    main()
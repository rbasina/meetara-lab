import os
import sys
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass
import time
from datetime import datetime
import logging # Import logging module

# Configure root logger to display DEBUG messages
logging.basicConfig(level=logging.DEBUG) # Set root logger to DEBUG

# -- Set up the Python path --
# This ensures that all project modules can be imported correctly from this entry point.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model-factory"))
sys.path.insert(0, str(project_root / "trinity_core"))
sys.path.insert(0, str(project_root / "scripts" / "training"))
# Add the scripts/validation directory to the Python path (moved here)
sys.path.insert(0, str(project_root / 'scripts' / 'validation'))

from trinity_core.agents.system_integration.complete_agent_ecosystem import CompleteAgentEcosystem
from trinity_core.intelligent_logger import IntelligentLogger
from trinity_core.core_components.config_manager import SmartTrinityConfigManager # Added for config validation in main

@dataclass
class TrinitySession:
    session_id: str
    start_time: datetime
    total_domains_in_config: int
    is_valid: bool = True # Flag to indicate if session initialization was successful
    processed_domains: int = 0
    successful_domains: int = 0
    failed_domains: int = 0
    total_processing_time: float = 0.0
    overall_quality_score: float = 0.0
    optimization_applied: list = None
    recommendations: list = None
    domain_breakdown: dict = None
    training_history_log: list = None


async def main():
    parser = argparse.ArgumentParser(description='Launch the Trinity training pipeline for specified domains.')

    # Group for mutually exclusive domain selection (either --category, --domains, or --all)
    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument(
        '--category',
        type=str,
        help='Specify a category of domains to train (e.g., healthcare, daily_life)'
    )
    domain_group.add_argument(
        '--domains',
        nargs='*',
        help='Specify one or more domains to train (e.g., shopping mental_health)'
    )
    domain_group.add_argument(
        '--all',
        action='store_true',
        help="Train all 62 domains"
    )

    # Simulation flag (defined once)
    parser.add_argument(
        '--simulation',
        action='store_true',
        help="Run in simulation mode (generates artifacts but no real training)"
    )

    # Generate synthetic data flag (defined once)
    parser.add_argument(
        '--generate-synthetic',
        action='store_true',
        help='Generate synthetically realistic data for training instead of loading real data or using basic simulation. Data will still be saved to appropriate dev/production paths.'
    )
    
    args = parser.parse_args()

    # Initialize configuration manager
    config_manager = SmartTrinityConfigManager()
    
    # Get total domains from config for logger initialization
    all_configured_domains = config_manager.get_all_domains_flat()
    total_domains_in_config = len(all_configured_domains)

    # Initialize session logger
    session_id_prefix = args.domains[0] if args.domains else args.category if args.category else "trinity_session"
    trinity_session = TrinitySession(
        session_id=f"{session_id_prefix}_{int(time.time())}",
        start_time=datetime.now(),
        total_domains_in_config=total_domains_in_config, # Pass the actual total domains
        is_valid=True # Mark as valid for now, until config validation explicitly fails
    )

    # IntelligentLogger initialization
    # Ensure log_base_dir is correctly set to logs/dev or logs/production
    if args.simulation or args.generate_synthetic:
        log_base_dir = Path("logs") / "dev"
    else:
        log_base_dir = Path("logs") / "production"
    
    logger = IntelligentLogger(
        session_id=trinity_session.session_id, 
        domain=trinity_session.session_id, # Use session ID as domain for the main session log
        is_simulation=args.simulation or args.generate_synthetic, # Pass simulation status
        log_base_dir=str(log_base_dir), # Pass the determined log directory
        total_domains_in_config=total_domains_in_config # Pass total domains to logger for its internal session data
    )
    logger.log_system_initialized("Intelligent Logging System")

    # Log config loading details explicitly now that config_manager is initialized
    logger.log_config_loading(yaml_loaded=True, total_domains=total_domains_in_config)

    # Initialize the complete agent ecosystem
    ecosystem = CompleteAgentEcosystem()

    # Determine domains to process
    domains_to_process = []
    if args.all:
        domains_to_process = all_configured_domains # Use the already fetched list
    elif args.category:
        try:
            domains_to_process = [d['domain_name'] for d in config_manager.get_domains_by_category(args.category)]
        except ValueError as e:
            logger.main_logger.error(f"Error: {e}")
            sys.exit(1)
    elif args.domains:
        for domain in args.domains:
            try:
                # Validate if the specified domain exists in the configuration
                domain_details = config_manager.get_tara_proven_params(domain)
                domains_to_process.append(domain)
                # Log successful domain validation
                logger.log_domain_validation(
                    domain=domain,
                    is_valid=True,
                    category=domain_details.get('category', 'N/A')
                )
            except ValueError:
                logger.main_logger.error(f"Error: Domain '{domain}' not found in configuration. Please check trinity_config.yaml.")
                # Log failed domain validation
                logger.log_domain_validation(
                    domain=domain,
                    is_valid=False,
                    suggestions=["Domain not found in config. Please verify name."]
                )
                sys.exit(1)

    if not domains_to_process:
        logger.main_logger.error("No domains specified for training. Use --category, --domains, or --all.")
        sys.exit(1)

    logger.main_logger.info(f"🎯 Training specific domains: {domains_to_process}")

    # Orchestrate the training process
    logger.log_training_start(domains_to_process) # Pass domains_to_process here
    
    # This will now return the overall results including total_domains_processed
    overall_results = await ecosystem.trinity_conductor.orchestrate_intelligent_training(
        target_domains=domains_to_process, # Corrected argument name
        simulation=args.simulation, # Corrected argument name
        generate_synthetic=args.generate_synthetic # Pass the generate_synthetic flag
    )
    
    logger.log_training_completed(overall_results)
    logger.log_comprehensive_summary(overall_results)

if __name__ == '__main__':
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Quantitative Stock Analysis Agent

This agent provides comprehensive analysis for stocks using market data,
sentiment analysis, macroeconomic factors, and technical indicators to
develop trading strategies and risk assessments.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import shutil
import traceback

from logging_utils import setup_logging, get_component_logger
from trader import run_trader

def clean_directories():
    """
    Delete logs and output directories for a fresh start each run
    """
    logger = logging.getLogger("CleanUp")
    
    # List of directories to clean
    dirs_to_clean = ['logs', 'output']
    
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                # Delete the directory and all contents
                shutil.rmtree(dir_path)
                logger.info(f"Successfully deleted {dir_name} directory")
            except Exception as e:
                logger.warning(f"Error deleting {dir_name} directory: {e}")
        else:
            logger.info(f"{dir_name} directory does not exist, no cleanup needed")
            
    # Recreate empty directories
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created empty {dir_name} directory")

if __name__ == "__main__":
    # Clean logs and output directories first
    clean_directories()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stock analysis and recommendation tool")
    parser.add_argument("--ticker", type=str, default="TSLA", help="Specific ticker to analyze")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--disable-output-redirect", action="store_true", 
                        help="Disable redirecting stdout/stderr to the logger")
    args = parser.parse_args()
    
    # Set up logging based on arguments
    log_level = getattr(logging, args.log_level)
    setup_logging(
        log_file=f"quant_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        console_level=log_level,
        redirect_stdout=not args.disable_output_redirect
    )
    
    # Get a logger for the main module
    logger = get_component_logger("Main")
    logger.info("Starting stock analysis application")
    
    try:
        # Run the trader with optional ticker
        run_trader(args.ticker)
        logger.info("Analysis and trading completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())

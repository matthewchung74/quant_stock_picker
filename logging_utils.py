#!/usr/bin/env python3
"""
Logging utilities for the quant agent.
Provides setup function and helper classes for logging configuration.
"""

import logging
import sys
from pathlib import Path

def setup_logging(log_file='quant_agent.log', console_level=logging.WARNING, file_level=logging.DEBUG, redirect_stdout=False):
    """
    Set up logging to both file and console with different verbosity levels.
    
    Args:
        log_file: Path to the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
        redirect_stdout: Whether to redirect stdout/stderr to logger (can cause recursion!)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / log_file
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    logger.handlers = []
    
    # ANSI escape codes for colors in console output
    COLORS = {
        'INFO': '\033[92m',  # Green
        'DEBUG': '\033[94m',  # Blue
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m',   # Reset
        'BOLD': '\033[1m',    # Bold
        'UNDERLINE': '\033[4m'  # Underline
    }
    
    # Create formatters with enhanced format that includes the agent/component name
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter('%(levelname)s: [%(name)s] %(message)s', COLORS))
    logger.addHandler(console_handler)
    
    # Redirect stdout and stderr to logger - WARNING: Can cause recursion if not careful!
    if redirect_stdout:
        sys.stdout = LoggerWriter(logging.INFO, 'stdout')
        sys.stderr = LoggerWriter(logging.ERROR, 'stderr')
    
    logging.info(f"Logging initialized: console level={console_level}, file level={file_level}, log file={log_path}")
    return logger

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output for better readability."""
    
    def __init__(self, fmt, colors=None):
        super().__init__(fmt)
        self.colors = colors or {
            'INFO': '\033[92m',  # Green
            'DEBUG': '\033[94m',  # Blue
            'WARNING': '\033[93m',  # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[91m\033[1m',  # Bold Red
            'RESET': '\033[0m',   # Reset
            'BOLD': '\033[1m',    # Bold
            'UNDERLINE': '\033[4m'  # Underline
        }
    
    def format(self, record):
        levelname = record.levelname
        name = record.name
        # Default to INFO color if level not found in COLORS
        level_color = self.colors.get(levelname, self.colors['INFO'])
        # Format the record with colors
        formatted_msg = super().format(record)
        # Replace the level name and component name with colored versions
        formatted_msg = formatted_msg.replace(
            f"{levelname}:", 
            f"{level_color}{levelname}{self.colors['RESET']}:", 
            1
        )
        formatted_msg = formatted_msg.replace(
            f"[{name}]", 
            f"{self.colors['BOLD']}[{name}]{self.colors['RESET']}", 
            1
        )
        return formatted_msg

class LoggerWriter:
    """Helper class that can be used to redirect stdout/stderr to the logger."""
    
    def __init__(self, level, name='root'):
        self.level = level
        self.buffer = ''
        self.name = name
        self.logger = logging.getLogger(name)
        
    def write(self, message):
        self.buffer += message
        if '\n' in message:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.logger.log(self.level, line.rstrip())
            self.buffer = lines[-1]
            
    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer)
            self.buffer = ''

def get_component_logger(component_name):
    """Get a logger for a specific component."""
    return logging.getLogger(component_name) 
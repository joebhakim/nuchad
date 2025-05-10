"""Path and data access utilities for the nuchad package."""

import importlib.resources
from pathlib import Path
import os

def get_project_root():
    """
    Get the path to the project root directory.
    
    Returns:
        Path object for the project root directory
    """
    return Path(__file__).parent.parent.parent.parent

def ensure_dir(directory):
    """
    Ensure that a directory exists.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Path object for the directory
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_data_file(filename):
    """
    Get the path to a data file using importlib.resources.
    
    Args:
        filename: Name of the data file
        
    Returns:
        Path object for the data file
    """
    # Use importlib.resources to handle package data files
    try:
        with importlib.resources.path('nuchad.data_files', filename) as path:
            return path
    except ImportError:
        # Fallback to the data directory in the project root
        data_dir = get_project_root() / 'data'
        return data_dir / filename

def get_data_path():
    """
    Get the path to the data directory.
    
    Returns:
        Path object for the data directory
    """
    return get_project_root() / 'data'

def get_results_dir():
    """
    Get the path to the results directory.
    
    Returns:
        Path object for the results directory
    """
    results_dir = get_project_root() / 'results'
    
    # Create the directory if it doesn't exist
    return ensure_dir(results_dir) 
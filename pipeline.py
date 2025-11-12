
import timeit
import sys
import subprocess
from pathlib import Path
from config import *
import argparse
import logging

from datetime import datetime

def get_pipeline_steps():

    pipeline_steps = [
        #### EVENTS:

        { # --> RAW
            "name": "Download dataset",
            "script": "download_dataset.py",
            "description": "Downloads dataset",
        }
    ]
    
    return pipeline_steps

def run_script(script_path, args=None):    
    command = [sys.executable, str(script_path)]
    
    # if args:

    try:
        # Run the script using the same Python interpreter as the current process
        result = subprocess.run(
            command,
            check=True,
            text=True
        )
        logging.info(f"Finished: {script_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}: {e}")
        return False

def run_pipeline():
    """Run the complete data processing pipeline"""
    
    logging.info("Starting pipeline")

    # Get the directory where the pipeline script is located
    pipeline_dir = Path(__file__).parent.absolute()

    start_dt=datetime.now()
    print("Starting:", datetime.now())
    # Run each step in sequence
    for step in get_pipeline_steps():
        script_path = pipeline_dir / step["script"]
        logging.info(f"Step: {step['name']}")
        logging.info(f"Description: {step['description']}")
        if not script_path.exists():
            logging.warning(f"Script not found: {script_path}")
            continue
        run_script(script_path, step.get("args"))
    print("Ending:", datetime.now() )
    print("Ended in:", datetime.now()-start_dt)

if __name__ == "__main__":    
    run_pipeline()
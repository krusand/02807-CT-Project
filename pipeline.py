
import timeit
import sys
import subprocess
from pathlib import Path
from config import *
import argparse

from datetime import datetime


def get_pipeline_steps():

    pipeline_steps = [
        #### EVENTS:

        { # DOWNLOAD DATASET
            "name": "Download dataset",
            "script": "scripts/01_download_dataset.py",
            "description": "Downloads dataset",
        },

        { # DATA SPLITS
            "name": "Data splits",
            "script": "scripts/02_data_split.py",
            "description": "Splits the combined order_products into a train, validation, and test set"
        },

        { # CALCULATE RATINGS
            "name": "Calculate rating",
            "script": "scripts/03_calculate_rating.py",
            "description": "Calculate rating",
        },

        { # USER CLUSTERING
            "name": "User clustering",
            "script": "scripts/04_clustering.py",
            "description": "Clusters users into clusters s.t. it's faster to run recommender system"
        },

        { # BASELINE RECOMMENDER
            "name": "Baseline recommender",
            "script": "scripts/05_baseline_recommender.py",
            "description": "Perform and evaluate top n recommender as baseline"
        },

        { # COLLABORATIVE FILTERING
            "name": "Collaborative filtering",
            "script": "scripts/06_collaborative_filter.py",
            "description": "Runs collaborative filtering"
        },

        { # APRIORI RECOMMENDATIONS
            "name": "Apriori Recommendations",
            "script": "scripts/07_apriori.py",
            "description": "Runs apriori algorithm to generate recommendations"
        },

        { # PCY TRAIN
            "name": "PCY Train",
            "script": "scripts/08a_pcy_train.py",
            "description": "Performs two passes to train the pcy recommender"
        },

        { # PCY RECOMMENDER
            "name": "APRIORI Recommendations",
            "script": "scripts/08b_pcy_recommender.py",
            "description": "Retrieves the trained pcy algorithm to generate and evaluate recommendations"
        },
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
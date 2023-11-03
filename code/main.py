import sys
import time
import logging
import re
from config import config, create_folder
from utils import IOHelper
from benchmark import benchmark
from utils.tables_utils import print_table
import warnings
import os

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


"""
Main entry of the program
Creates the logging files, loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""


def main():
    # Setting up logging
    run = create_folder()
    logging.basicConfig(filename=config["info_log"], level=logging.INFO)
    logging.info("Started the Logging")
    logging.info(f"Using {config['framework']}")
    start_time = time.time()

    # add for reproducibility

    config_location = config["model_dir"]

    re_groups = re.match("^/(.*/[a-z].*)/(runs/.*)", config_location)

    new_path = "./" + re_groups[2]

    print("*" * 10)
    print(f"Logs and Results are in {new_path}")
    print("*" * 10)

    # For being able to see progress that some methods use verbose (for debugging purposes)
    f = open(config["model_dir"] + "/console.out", "w")
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    # Load the data
    trainX, trainY = IOHelper.get_npz_data(config["data_dir"], verbose=True)

    # Start benchmark
    benchmark(trainX, trainY)
    # directory = 'results/standardML'
    # print_table(directory, preprocessing='max')

    logging.info("--- Runtime: %s seconds ---" % (time.time() - start_time))
    print(f"Total Runtime: {time.time() - start_time} seconds")
    logging.info("Finished Logging")


if __name__ == "__main__":
    main()

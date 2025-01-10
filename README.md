# psych-bach
For Victor Kaplan Kjellerup's Bachelor Project, Predicting the Psychedelic Experience from Brain Imaging Data

For reproducibility of results, please use miniconda to install and activate a conda-environment according to environment.yml file in root directiory. You can install or update environement (named psych by default) by running

    conda env create --file=environment.yml --force

from the root directory. This might take several minutes. For a much quicker installation, remove pytest from environment.yml, and all sections except the test will run as intended.

To run the pipeline, a file data/data.csv must be present. It must adhere to requirements specified in data/metadata.json for relevant columns "sub", "ses", DCC feature columns and 5d-ASC label columns. The pipeline is executed running

    python main.py --pipeline

and subsequently, the analyses can be carried out, using the stored results, running

    python main.py --analysis [--plot]

Example use with dummy data generation might look like

    python main.py --pipeline --n 80 --gen_data

whose statistics are computed using the same analysis interface as above. Please run

    python main.py --help

for further interface options.

To run the test suite (which only includes a test of the split assignment), simply run

    pytest

from the root directory. To generate data necessary for tests, please run

    gen_test_data.py

from the root.
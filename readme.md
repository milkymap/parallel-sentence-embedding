## Description

This Python project provides a way to vectorize text documents using SentenceTransformer library. The goal is to encode the documents into a numerical representation that can be used for similarity search or other natural language processing tasks. This project uses ZeroMQ to distribute the workload across multiple processes, allowing for faster processing of large volumes of text.

## Dependencies

- `json`
- `click`
- `networkx`
- `torch`
- `zmq`
- `numpy`
- `multiprocessing`
- `os`
- `glob`
- `typing`
- `time`
- `loguru`
- `rich`
- `sentence_transformers`

## Usage

To use this project, follow these steps:

1. Clone the repository and navigate to the root directory.

2. Install the dependencies by running the commands: `python -m venv env`, `pip install -U pip`, `pip install -r requirements.txt`.

3. Ensure that your data is in text file format, and that the files are located in a directory. The path to this directory should be passed as an argument to the `--path2files` option when running the script.

4. Run the script by running the command `python entrypoint.py`. The script will automatically partition the data and distribute the workload across multiple processes. The number of processes can be set using the `--nb_workers` option.

5. The script will output a matrix of vectors that represent the documents. These vectors can be used for various natural language processing tasks.

## Command-line Options

- `--path2files`: The path to the directory containing the text files to be vectorized. Default value is None.

- `--model_name`: The name of the transformer model to be used. Default value is `Sahajtomar/french_semantic`.

- `--cache_folder`: The cache folder for the transformer model. Default value is None.

- `--nb_workers`: The number of processes to be used for vectorizing the documents. Default value is 2.

## Credits

This project uses the following libraries:

- `json`: for encoding and decoding JSON data.

- `click`: for command-line interface.

- `networkx`: for community detection algorithm.

- `torch`: for vectorization of text.

- `zmq`: for distributed processing.

- `numpy`: for numerical processing.

- `multiprocessing`: for parallel processing.

- `os`: for file path manipulation.

- `glob`: for searching for files in a directory.

- `typing`: for type annotations.

- `time`: for timing code execution.

- `loguru`: for logging.

- `rich`: for progress bars.

- `sentence_transformers`: for text vectorization.

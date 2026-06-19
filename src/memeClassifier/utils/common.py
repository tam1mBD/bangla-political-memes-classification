import os
from box.exceptions import BoxValueError
from memeClassifier import logger
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Raises:
        ValueError: If yaml file is empty or cannot be read
        e: empty yaml file

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates list of directories if they do not exist.

    Args:
        path_to_directories (list): List of paths to directories to be created.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
        path (Path): The path where the JSON file will be saved.
        data (dict): The dictionary to be saved as JSON.
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        ConfigBox: The contents of the JSON file as a ConfigBox object.
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(data)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data in binary format using joblib.

    Args:
        data (Any): The data to be saved.
        path (Path): The path where the binary file will be saved.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.

    Args:
        path (Path): The path to the binary file.
    Returns:
        Any: The data loaded from the binary file.
    """
    data = joblib.load(filename=path)
    logger.info(f"Binary file loaded successfully from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in a human-readable format.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The size of the file in a human-readable format.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} KB"

def decode_image(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encode_image_into_base64(cropped_image_path):
    with open(cropped_image_path, 'rb') as f:
        return base64.b64encode(f.read())
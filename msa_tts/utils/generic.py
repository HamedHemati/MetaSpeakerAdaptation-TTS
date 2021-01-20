import yaml


def load_params(yml_file_path):
    r"""Loads param file and returns a dictionary."""
    with open(yml_file_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    
    return params


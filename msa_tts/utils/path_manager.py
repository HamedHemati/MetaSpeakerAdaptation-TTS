import os


class PathManager():
    r"""Manages output folder by initializing them and saving 
        paths to each output sub-folder.
    """
    def __init__(self, output_path):
        # Set paths
        self.output_path = output_path
        self.checkpoints_path = os.path.join(self.output_path, "checkpoints")
        self.logs_path = os.path.join(self.output_path, "logs")
        self.examples_path = os.path.join(self.output_path, "examples")
        self.inference_path = os.path.join(self.output_path, "inference")
        
        # Creates directories in the output folder
        folders = [f for f in dir(self) if f.endswith("_path")]
        for folder in folders:
            os.makedirs(getattr(self, folder), exist_ok=True)

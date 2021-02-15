import json
import torch
from .models import Generator
from  .utils import load_checkpoint, AttrDict


class HiFiGAN():
    def __init__(self, config_path, checkpoint_path, device):
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        
        self.generator = Generator(h).to(device)
        state_dict_g = load_checkpoint(checkpoint_path, device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        
    def inference(self, mel):
        with torch.no_grad():
            y_g_hat = self.generator(mel)
            audio = y_g_hat.squeeze()
        return audio


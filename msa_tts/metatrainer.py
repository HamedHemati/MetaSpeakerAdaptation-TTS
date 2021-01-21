import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .utils.path_manager import PathManager
from .dataloaders.dataloader_meta import get_dataloader
from .models.tacotron2nv import Tacotron2NV
from .models.modules_tacotron2nv.tacotron2nv_loss import Tacotron2Loss
from .utils.helpers import get_optimizer
from .utils.g2p.char_list import char_list


class MetaTrainer():
    r"""Base class Trainer. All trainers should inherit from this class."""
    def __init__(self, **params):
        self.params = params
        # Create output folders
        output_path = os.path.join(self.params["output_path"], 
                                   self.params["method"], 
                                   self.params["experiment_name"])
        self.path_manager = PathManager(output_path)
        
        # Save params as YAML file in the output directory
        with open(os.path.join(self.path_manager.output_path, "params.yml"), 'w') as yml_file:
            yaml.dump(self.params, yml_file)

        # Summary writer
        now = datetime.now()
        dt_string = now.strftime("%d_%m-%H_%M")
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_manager.logs_path, dt_string))
        
        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Compute device: {self.device}")

        # Dataloaders
        self._init_dataloaders()

        # Set model
        self.params["model"]["num_speakers"] = len(self.dataloader_metatrain.dataset.speaker_to_id.keys())
        self.params["model"]["n_symbols"] = len(char_list)
        self.params["model"]["n_mel_channels"] = params["audio_params"]["n_mels"]

        self.model_name = self.params["model_name"]
        self.speaker_emb_type = self.params["model"]["speaker_emb_type"]
        if self.model_name  == "Tacotron2NV":
            self.model = Tacotron2NV(self.params["model"])
        else:
            raise NotImplementedError
        self.model.to(self.device)

        # Optimizer and criterion
        self._init_criterion_optimizer()

        # Finetuning
        self._load_checkpoint()

    def _init_dataloaders(self):
        # Load meta-train loaders
        print("\nInitializing meta-train loaders")
        self.dataloader_metatrain, logs_mtr = get_dataloader("metatrain", **self.params)

        # Load meta-test loaders
        print("\nInitializing meta-test loaders")
        self.dataloader_metatest, logs_mts= get_dataloader("metatest", **self.params)
        print("\n")

        # Write DS details to a text file
        log_ds = "Meta-Train:\n\n" + logs_mtr + "\n\n\n" + "Meta-Test:\n\n" + logs_mts
        with open(os.path.join(self.path_manager.output_path, "dataset_details.txt"), 'w') as ds_details:
            ds_details.write(log_ds)
        

    def _init_criterion_optimizer(self):
        # Criterion
        if self.params["criterion"]["criterion_type"] == "Tacotron2Loss":
            self.criterion = Tacotron2Loss(n_frames_per_step=self.params["model"]["n_frames_per_step"],
                                           reduction=self.params["criterion"]["reduction"],
                                           pos_weight=self.params["criterion"]["pos_weight"],
                                           device=self.device)
        else:
            raise RuntimeError(f"Criterion {self.params['criterion']} not defined.")

        # Init optimizer
        self.inner_optimizer = get_optimizer(self.model, **self.params["optim_inner"])
        self.outer_optimizer = get_optimizer(self.model, **self.params["optim_outer"])

    def _unpack_batch(self, batch_items):
        r"""Un-packs batch items and sends them to compute device"""
        item_ids, inp_chars, inp_lens, mels, mel_lens, speakers_ids, spk_embs, stop_labels = batch_items
        if self.model_name == "Tacotron2NV":
            # Transfer batch items to compute_device
            inp_chars, inp_lens  = inp_chars.to(self.device), inp_lens.to(self.device)
            mels, mel_lens =  mels.to(self.device), mel_lens.to(self.device)
            
            if self.speaker_emb_type  == "learnable_lookup":
                speaker_vecs = speakers_ids.to(self.device)
            elif self.speaker_emb_type  == "static":
                speaker_vecs = spk_embs.to(self.device)

            stop_labels = stop_labels.to(self.device)
            d = {"inputs": inp_chars,
                 "input_lengths": inp_lens,
                 "melspecs": mels,
                 "melspec_lengths": mel_lens,
                 "speaker_vecs":speaker_vecs}
            return d, stop_labels
            # return [inp_chars, inp_lens, mels, mel_lens, speaker_vecs], stop_labels
        else:
            raise NotImplementedError

    def _save_checkpoint(self):
        k = self.step_global // 100
        checkpoint_path = os.path.join(self.path_manager.checkpoints_path, f"checkpoint_{k}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def log_writer(self, logs):
        r"""Writes a dictionary of logs to the tensorboard.log
            Inputs:
                   logs: dictionary of the form {k: (a, b)}
        """
        for k, v in logs.items():
            self.writer.add_scalar(k, v[0], v[1])

    def _load_checkpoint(self):
         # Load checkpoint
        print(f"Loading checkpoint from  {self.params['finetune_checkpoint_path']}")  
        ckpt = torch.load(self.params["finetune_checkpoint_path"], map_location=self.device)
        for name, param in self.model.named_parameters():
            try:
                self.model.state_dict()[name].copy_(ckpt[name])
            except:
                print(f"Could not load weights for {name}")

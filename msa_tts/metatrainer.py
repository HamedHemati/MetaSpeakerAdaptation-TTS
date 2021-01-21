import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .utils.path_manager import PathManager
from .dataloaders.dataloader_meta import get_dataloaders
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
        self.params["model"]["num_speakers"] = len(self.metatrain_trainloader.dataset.speaker_to_id.keys())
        self.params["model"]["n_symbols"] = len(char_list)
        self.params["model"]["n_mel_channels"] = params["audio_params"]["n_mels"]
        if self.params["model_name"] == "Tacotron2NV":
            self.model = Tacotron2NV(self.params["model"])
        else:
            raise NotImplementedError
        self.model.to(self.device)

        # Optimizer and criterion
        self._init_criterion_optimizer()

    def _init_dataloaders(self):
        # Load meta-train loaders
        print("\nInitializing meta-train loaders")
        self.metatrain_trainloader, self.metatrain_testloader, logs_mtr = get_dataloaders("metatrain", **self.params)
        self.metatrain_testloader.dataset.speaker_to_id = self.metatrain_trainloader.dataset.speaker_to_id
        self.metatrain_testloader.dataset.id_to_speaker = self.metatrain_trainloader.dataset.id_to_speaker

        # Load meta-test loaders
        print("\nInitializing meta-test loaders")
        self.metatest_trainloader, self.metatest_testloader, logs_mts= get_dataloaders("metatest", **self.params)
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
        item_ids, inp_chars, inp_lens, mels, mel_lens, speakers_ids, spk_embs, stop_labels = batch_items
        # Transfer batch items to compute_device
        inp_chars, inp_lens  = inp_chars.to(self.device), inp_lens.to(self.device)
        mels, mel_lens =  mels.to(self.device), mel_lens.to(self.device)
        speakers_ids = speakers_ids.to(self.device)
        spk_embs = spk_embs.to(self.device)
        stop_labels = stop_labels.to(self.device)

        return item_ids, inp_chars, inp_lens, mels, mel_lens, speakers_ids, spk_embs, stop_labels 
    
    def _save_checkpoint(self):
        k = self.step_global // 100
        checkpoint_path = os.path.join(self.path_manager.checkpoints_path, f"checkpoint_{k}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def log_writer(self, d):
        self.writer.add_scalar(f"test/{ds_name}", outer_loss_test_ds, self.step_global)
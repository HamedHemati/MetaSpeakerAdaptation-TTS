from .utils.limit_threads import *
import argparse
import os
import yaml
import torch
import higher
import copy
from copy import deepcopy
import pickle
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .utils.generic import load_params
from datetime import datetime
from .utils.path_manager import PathManager
from .dataloaders.dataloader_default_buffer import get_dataloader as  get_dataloader_default
from .dataloaders.dataloader_default_buffer import Collator
from .models.tacotron2nv import Tacotron2NV
from .models.modules_tacotron2nv.tacotron2nv_loss import Tacotron2Loss
from .utils.helpers import get_optimizer
from .utils.g2p.char_list import char_list
from .utils.metrics import mcd_batch
from .utils.plot import plot_spec_attn_example
from torch.nn.utils import clip_grad_norm_



def _unpack_batch(batch_items, device, add_item_id=False):
        r"""Un-packs batch items and sends them to compute device"""
        item_ids, inp_chars, inp_lens, mels, mel_lens, speakers_ids, spk_embs, stop_labels = batch_items
        # Transfer batch items to compute_device
        inp_chars, inp_lens  = inp_chars.to(device), inp_lens.to(device)
        mels, mel_lens =  mels.to(device), mel_lens.to(device)
        
        speaker_vecs = spk_embs.to(device)

        stop_labels = stop_labels.to(device)
        d = {"inputs": inp_chars,
                "input_lengths": inp_lens,
                "melspecs": mels,
                "melspec_lengths": mel_lens,
                "speaker_vecs":speaker_vecs}
        if add_item_id:
            d["item_ids"] = item_ids
        
        return d, stop_labels


def create_buffer_dl(dataloader, params, model, device):
    buffer_dl = deepcopy(dataloader)
    # Collator
    collator = Collator(reduction_factor=params["model"]["n_frames_per_step"], 
                        audio_processor=params["audio_processor"],
                        audio_params=params["audio_params"])

    buffer_dl = DataLoader(buffer_dl.dataset,
                           collate_fn=collator,
                           batch_size=params["buffer_batch_size"],
                           sampler=None,
                           num_workers=params["num_workers"],
                           drop_last=False,
                           pin_memory=True,
                           shuffle=params["buffer_shuffle"])

    # Shuffle the buffer dataloader item list
    random.shuffle(buffer_dl.dataset.items)
    # Select the top num_samples items
    buffer_dl.dataset.items = buffer_dl.dataset.items[:params["buffer_sample_size"]]
    # Update metadata dict
    buffer_dl.dataset.metadata = {item: buffer_dl.dataset.metadata[item] for item in buffer_dl.dataset.items}
    # Add mel-spec key with soft values to the new buffer item
    with torch.no_grad():
        for itr, (batch) in enumerate(buffer_dl, 1):
            model_inputs, stop_labels_gt = _unpack_batch(batch, device, add_item_id=True)
            item_ids = model_inputs["item_ids"]
            print(item_ids)
            del model_inputs["item_ids"]
            out_post, out_inner, out_stop, out_attn = model(**model_inputs)
            for item_itr, item_id in enumerate(item_ids):
                soft_melspec = out_post[item_itr].detach().cpu()
                soft_melspec = soft_melspec[:, :model_inputs["melspec_lengths"][item_itr].item()]
                buffer_dl.dataset.metadata[item_id]["melspec"] = soft_melspec
    
    return buffer_dl


def add_to_buffer_dl(buffer_dl, new_dataloader, num_samples, model, device):
    new_dl = deepcopy(new_dataloader)
    # Shuffle new_dl item list
    random.shuffle(new_dl.dataset.items)
    
    # Select the top num_samples items
    new_dl.dataset.items = new_dl.dataset.items[:num_samples]
    # New metadata dict
    new_metadata = {item: new_dl.dataset.metadata[item] for item in new_dl.dataset.items}
    new_dl.dataset.metadata = new_metadata
    
    # Update buffer dl
    buffer_dl.dataset.items = buffer_dl.dataset.items + new_dl.dataset.items
    buffer_dl.dataset.metadata.update(new_dl.dataset.metadata )
    buffer_dl.dataset.speaker_to_id.update(new_dl.dataset.speaker_to_id)
    buffer_dl.dataset.id_to_speaker.update(new_dl.dataset.id_to_speaker)
    
    with torch.no_grad():
        for itr, (batch) in enumerate(new_dl, 1):
            model_inputs, stop_labels_gt = _unpack_batch(batch, device, add_item_id=True)
            item_ids = model_inputs["item_ids"]
            del model_inputs["item_ids"]
            
            out_post, out_inner, out_stop, out_attn = model(**model_inputs)
            for item_itr, item_id in enumerate(item_ids):
                soft_melspec = out_post[item_itr].detach().cpu()
                soft_melspec = soft_melspec[:, :model_inputs["melspec_lengths"][item_itr].item()]
                buffer_dl.dataset.metadata[item_id]["melspec"] = soft_melspec
    
    return buffer_dl


def combine_dataloaders(dl1, dl2):
    # Update buffer dl
    dl1.dataset.items = dl1.dataset.items + dl2.dataset.items
    dl1.dataset.metadata.update(dl2.dataset.metadata)
    dl1.dataset.speaker_to_id.update(dl2.dataset.speaker_to_id)
    dl1.dataset.id_to_speaker.update(dl2.dataset.id_to_speaker)
    
    return dl1


class ExperienceReplayKnowledgeDistillTrainer():
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

        # Save all spakers
        self.all_speakers = self.params["dataset_train"]["speakers_list"]
        random.Random(self.params["speaker_seed"]).shuffle(self.all_speakers)
        
        # Set model
        self.params["model"]["num_speakers"] = 1 #len(self.dataloader_train.dataset.speaker_to_id.keys())
        self.params["model"]["n_symbols"] = len(char_list)
        self.params["model"]["n_mel_channels"] = params["audio_params"]["n_mels"]

        # Set freezing options
        self.params["model"]["freeze_charemb"] = params["freeze_charemb"]
        self.params["model"]["freeze_encoder"] = params["freeze_encoder"]
        self.params["model"]["freeze_decoder"] = params["freeze_decoder"]
        
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
        if self.params["finetune"]:
            self._load_checkpoint()

    def _init_dataloaders(self, speaker):
        # Load meta-train loaders
        print(f"\nInitializing train/test loaders for {speaker}")
        log_ds = ""

        self.params["dataset_train"]["speakers_list"] = speaker
        self.dataloader_train, self.dataloader_test, logs_tr = get_dataloader_default(**self.params)
        log_ds += "Train:\n\n" + logs_tr + "\n\n\n"

        # Write DS details to a text file
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
        self.optim = get_optimizer(self.model, **self.params["optim"])

    def _unpack_batch(self, batch_items, add_item_id=False):
        r"""Un-packs batch items and sends them to compute device"""
        item_ids, inp_chars, inp_lens, mels, mel_lens, speakers_ids, spk_embs, stop_labels = batch_items
        if self.model_name == "Tacotron2NV":
            # Transfer batch items to compute_device
            inp_chars, inp_lens  = inp_chars.to(self.device), inp_lens.to(self.device)
            mels, mel_lens =  mels.to(self.device), mel_lens.to(self.device)

            if self.speaker_emb_type  == "learnable_lookup":
                speaker_vecs = speakers_ids.to(self.device)
            elif self.speaker_emb_type in ["static", "static+linear"]:
                speaker_vecs = spk_embs.to(self.device)

            stop_labels = stop_labels.to(self.device)
            d = {"inputs": inp_chars,
                 "input_lengths": inp_lens,
                 "melspecs": mels,
                 "melspec_lengths": mel_lens,
                 "speaker_vecs":speaker_vecs,
                 }
            if add_item_id:
                d["item_ids"]: item_ids

            return d, stop_labels
            # return [inp_chars, inp_lens, mels, mel_lens, speaker_vecs], stop_labels
        else:
            raise NotImplementedError

    def _save_checkpoint(self, speaker, itr):
        checkpoint_path = os.path.join(self.path_manager.checkpoints_path, f"best_{itr}_{speaker}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def log_writer(self, logs, type="scalar"):
        r"""Writes a dictionary of logs to the tensorboard.log
            Inputs:
                   logs: dictionary of the form {k: (a, b)}
        """
        if type == "scalar":
            for k, v in logs.items():
                self.writer.add_scalar(k, v[0], v[1])
        elif type == "hist":
            for k, v in logs.items():
                self.writer.add_histogram(k, v[0], v[1])
        else:
            raise NotImplementedError

    def _load_checkpoint(self):
         # Load checkpoint
        print(f"Loading checkpoint from  {self.params['finetune_checkpoint_path']}")  
        ckpt = torch.load(self.params["finetune_checkpoint_path"], map_location=self.device)
        for name, param in self.model.named_parameters():
            try:
                self.model.state_dict()[name].copy_(ckpt[name])
            except:
                print(f"Could not load weights for {name}")

    def get_module_grads_flattened(self, step):
        r"""Retrieves gradients of each module and falttens them.
            Returns: dictionary of module grads {mod: ([grad_list], step)}
        """
        named_params = list(self.model.named_parameters())
        param_names = [n for (n, p) in named_params]
        model_parts = list(set([name.split(".")[0] for name in param_names]))

        module_grads ={}
        for model_part in model_parts:
            module_grad = [p.grad.flatten() for (n, p) in named_params 
                           if n.startswith(model_part) if p.grad != None]
            if len(module_grad) > 0:
                module_grad = torch.cat(module_grad, dim=0)
                module_grads["grad_" + model_part] = (module_grad, step)
        return module_grads

    def run(self):
        self.step_global = 0
        self.speakers_so_far = []
        self.cumutest_dict = {}
        
        # Initial finetuning
        num_initial_speakers = self.params["num_initial_speakers"]
        if num_initial_speakers > 0:
            initial_speakers = self.all_speakers[:num_initial_speakers]
            self._init_dataloaders(initial_speakers)
            
            speaker = initial_speakers[0]
            spk_itr = 0
            self._train(speaker, spk_itr)
            self._save_checkpoint(speaker, spk_itr)

        for spk_itr, speaker in enumerate(self.all_speakers, num_initial_speakers):
            self.speakers_so_far.append(speaker)
            # ========== For each task
            # Init dataloader
            self._init_dataloaders([speaker])
            # Initi optimizer
            self._init_criterion_optimizer()
            # Train task for one epoch
            self._train(speaker, spk_itr)
            self._save_checkpoint(speaker, spk_itr)  
            self._test_cumulative(speaker, spk_itr)

    def _train(self, speaker, spk_itr):
        # Init buffer in in the first speaker iteration anf update afterwards
        if spk_itr > 0:
            # Combine data loaders
            print("\n\nCombining train loader and the buffer ...")
            self.dataloader_train = combine_dataloaders(self.dataloader_train, self.buffer_dl)

        speaker_losses = []
        for epoch in range(1, self.params["n_max_epochs"] + 1):
            self.model.train()
            for itr, (batch) in enumerate(self.dataloader_train, 1):
                model_inputs, stop_labels_gt = self._unpack_batch(batch)
                mels_gt = model_inputs["melspecs"]
                mel_lens_gt = model_inputs["melspec_lengths"]
                if mels_gt.shape[0] == 1:
                    continue
                out_post, out_inner, out_stop, out_attn = self.model(**model_inputs)
                y_pred = (out_post, out_inner, out_stop, out_attn)
                y_gt = (mels_gt, stop_labels_gt)
                
                loss = self.criterion(y_pred, y_gt, mel_lens_gt)
                
                if self.params["clip_grad_norm"]:
                    grad_norm = clip_grad_norm_(self.model.parameters(), 
                                            self.params["grad_clip_thresh"])
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

                # ===== Logs
                # MCD and loss
                mcd_batch_value = mcd_batch(out_post.detach().cpu().transpose(1, 2).numpy(),
                                            mels_gt.cpu().transpose(1, 2).numpy(),
                                            mel_lens_gt.cpu().numpy())
                
                msg = f'|Speaker {spk_itr}/{len(self.all_speakers)}: Epoch {epoch} - {self.step_global}, itr {itr}/{len(self.dataloader_train)} ' + \
                      f'::  step loss: {loss.item():#.4} | mcd: {mcd_batch_value:#.4}'
                print(msg)

                self.step_global += 1

            if epoch % self.params["test_interval"] == 0:
                loss_test = self._test(epoch, speaker)
                speaker_losses.append(loss_test)
                if self.params["early_stopping"]:
                    if len(speaker_losses) > self.params["early_stopping_steps"] and \
                            speaker_losses[-self.params["early_stopping_steps"]-1] < min(speaker_losses[-self.params["early_stopping_steps"]:]):
                        print("Early stopping")
                        break

        # Plot example after each epoch
        idx = -1
        step_temp = self.step_global // 1000
        example_attn = out_attn[idx][:, :].detach().cpu().numpy()
        example_mel = out_post[idx].detach().cpu().numpy()
        example_mel_gt = mels_gt[idx].detach().cpu().numpy()
        plot_save_path = os.path.join(self.path_manager.examples_path, 
                                        f'{spk_itr}_train-spk{speaker}')
        plot_spec_attn_example(example_mel, 
                            example_mel_gt,  
                            example_attn,
                            plot_save_path,
                            length_mel=mel_lens_gt[idx].item(),
                            length_attn=model_inputs["input_lengths"][idx].item())

        # Init buffer or update buffer
        print("Updating buffer ...")
        if spk_itr == 0:
            self.buffer_dl = create_buffer_dl(self.dataloader_train, self.params, self.model, self.device)
        else:
            trainloader_temp = deepcopy(self.dataloader_train)
            # Update buffer
            self.buffer_dl = add_to_buffer_dl(self.buffer_dl, trainloader_temp, self.params["buffer_sample_size"], self.model, self.device)
        
        
    def _test(self, epoch, speaker):
        print(f"===== Testing epoch {epoch}")
        self.model.train()
        loss_total = 0.0
        mcd_batch_value_total = 0.0

        with torch.no_grad():
            for itr, (batch) in enumerate(self.dataloader_test, 1):
                model_inputs, stop_labels_gt = self._unpack_batch(batch)
                mels_gt = model_inputs["melspecs"]
                mel_lens_gt = model_inputs["melspec_lengths"]
                
                out_post, out_inner, out_stop, out_attn = self.model(**model_inputs)
                y_pred = (out_post, out_inner, out_stop, out_attn)
                y_gt = (mels_gt, stop_labels_gt)
                
                loss = self.criterion(y_pred, y_gt, mel_lens_gt)

                # ===== Logs
                # MCD and loss
                mcd_batch_value = mcd_batch(out_post.cpu().transpose(1, 2).numpy(),
                                            mels_gt.cpu().transpose(1, 2).numpy(),
                                            mel_lens_gt.cpu().numpy())

                loss_total += loss.item()
                mcd_batch_value_total += mcd_batch_value
        
        loss_total = loss_total / float(len(self.dataloader_test))
        mcd_batch_value_total = mcd_batch_value_total / float(len(self.dataloader_test))

        log_dict = {f"test/loss_{speaker}": (loss_total, self.step_global),
                    f"test/mcd_{speaker}": (mcd_batch_value_total, self.step_global)
                    }
        self.log_writer(log_dict)
    
        msg = f'| Epoch: {epoch}, itr: {self.step_global} ::  loss_total:' +\
                f' {loss_total:#.4} | mcd_total: {mcd_batch_value_total:#.4} '
        print(msg)
        return loss_total

    def _test_cumulative(self, speaker, spk_itr):
        # Load dataloaders
        print("-"*20, "Cumulative Testing")
        params = copy.deepcopy(self.params)

        self.cumutest_dict[spk_itr] = {"speaker": speaker, "losses":{}}

        for itr, test_speaker in enumerate(self.speakers_so_far):
            print(f"\nInitializing train/test loaders for {test_speaker}")
            params["dataset_train"]["speakers_list"] = [test_speaker]
            _, dataloader_test, logs = get_dataloader_default(**params)
            print(logs)

            # Test speaker
            self.model.train()
            loss_total = 0.0
            mcd_batch_value_total = 0.0

            with torch.no_grad():
                for itr, (batch) in enumerate(dataloader_test, 1):
                    model_inputs, stop_labels_gt = self._unpack_batch(batch)
                    mels_gt = model_inputs["melspecs"]
                    mel_lens_gt = model_inputs["melspec_lengths"]
                    
                    out_post, out_inner, out_stop, out_attn = self.model(**model_inputs)
                    y_pred = (out_post, out_inner, out_stop, out_attn)
                    y_gt = (mels_gt, stop_labels_gt)
                    
                    loss = self.criterion(y_pred, y_gt, mel_lens_gt)

                    # ===== Logs
                    # MCD and loss
                    mcd_batch_value = mcd_batch(out_post.cpu().transpose(1, 2).numpy(),
                                                mels_gt.cpu().transpose(1, 2).numpy(),
                                                mel_lens_gt.cpu().numpy())

                    loss_total += loss.item()
                    mcd_batch_value_total += mcd_batch_value
            
            loss_total = loss_total / float(len(dataloader_test))
            mcd_batch_value_total = mcd_batch_value_total / float(len(dataloader_test))
        
            msg = f'| Speaker: {test_speaker}, itr: {self.step_global} ::  loss_total:' +\
                    f' {loss_total:#.4} | mcd_total: {mcd_batch_value_total:#.4} '
            print(msg)
            
            # Set cumulative test loss for test speaker 
            self.cumutest_dict[spk_itr]["losses"][test_speaker] = loss_total

            # Plot example mel-spec
            idx = -1
            step_temp = self.step_global // 1000
            example_attn = out_attn[idx][:, :].detach().cpu().numpy()
            example_mel = out_post[idx].detach().cpu().numpy()
            example_mel_gt = mels_gt[idx].detach().cpu().numpy()
            plot_save_path = os.path.join(self.path_manager.examples_path, 
                                          f'cumTest_{spk_itr}_spk-{speaker}_to_spk-{test_speaker}')
            plot_spec_attn_example(example_mel, 
                                example_mel_gt,  
                                example_attn,
                                plot_save_path,
                                length_mel=mel_lens_gt[idx].item(),
                                length_attn=model_inputs["input_lengths"][idx].item())
        
        # Save cumulative test loss to a pickle file
        cumutest_dict_path = os.path.join(self.path_manager.examples_path, 
                                          f'cumutest.pkl')
        with open(cumutest_dict_path, "wb") as pkl_file:
            pickle.dump(self.cumutest_dict, pkl_file)
        
        print("-"*30 + "\n")

def main(args):
    params = load_params(os.path.join(args.params_path, "params.yml"))
    erkd = ExperienceReplayKnowledgeDistillTrainer(**params)
    erkd.run()


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str)
    args = parser.parse_args()
    main(args)

from .utils.limit_threads import *
import argparse
import os
import yaml
import torch
import higher
from torch.utils.tensorboard import SummaryWriter
from .utils.generic import load_params
from datetime import datetime
from .utils.path_manager import PathManager
from .dataloaders.dataloader_default import get_dataloader as  get_dataloader_default
from .dataloaders.dataloader_meta import get_dataloader as get_dataloader_meta
from .models.tacotron2nv import Tacotron2NV
from .models.modules_tacotron2nv.tacotron2nv_loss import Tacotron2Loss
from .utils.helpers import get_optimizer
from .utils.g2p.char_list import char_list
from .utils.metrics import mcd_batch
from .utils.plot import plot_spec_attn_example
from torch.nn.utils import clip_grad_norm_


class JointTrainer():
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
        self.params["model"]["num_speakers"] = len(self.dataloader_train.dataset.speaker_to_id.keys())
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
        if self.params["finetune"]:
            self._load_checkpoint()

    def _init_dataloaders(self):
        # Load meta-train loaders
        print("\nInitializing train/test loaders")
        log_ds = ""

        self.dataloader_train, self.dataloader_test, logs_tr = get_dataloader_default(**self.params)
        log_ds += "Train:\n\n" + logs_tr + "\n\n\n"

        # Load meta-test loaders
        if self.params["do_metatest"]:
            print("\nInitializing meta-test loaders")
            self.dataloader_metatest, logs_mts = get_dataloader_meta("metatest", **self.params)
            print("\n")
            log_ds += "Meta-Test:\n\n" + logs_mts

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
        self.inner_optimizer = get_optimizer(self.model, **self.params["optim_inner"])

    def _unpack_batch(self, batch_items):
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
                 "speaker_vecs":speaker_vecs}
            return d, stop_labels
            # return [inp_chars, inp_lens, mels, mel_lens, speaker_vecs], stop_labels
        else:
            raise NotImplementedError

    def _save_checkpoint(self):
        k = self.step_global // 100
        checkpoint_path = os.path.join(self.path_manager.checkpoints_path, f"checkpoint_{k}.pt")
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
        for epoch in range(1, self.params["n_epochs"] + 1):
            # Train task for one epoch
            self._train(epoch)

            self._test(epoch)

            if epoch % self.params["ckpt_save_epoch_interval"] == 0:
                self._save_checkpoint()
            
            if self.params["do_metatest"]:
                if epoch % self.params["metatest_epoch_interval"] == 0:
                    print("Meta-test phase ...")
                    self._metatest(epoch)
                    print("\n")

    def _train(self, epoch):
        print(f"===== Training epoch {epoch}")
        self.model.train()
        for itr, (batch) in enumerate(self.dataloader_train, 1):
            model_inputs, stop_labels_gt = self._unpack_batch(batch)
            mels_gt = model_inputs["melspecs"]
            mel_lens_gt = model_inputs["melspec_lengths"]
            
            out_post, out_inner, out_stop, out_attn = self.model(**model_inputs)
            y_pred = (out_post, out_inner, out_stop, out_attn)
            y_gt = (mels_gt, stop_labels_gt)
            
            loss = self.criterion(y_pred, y_gt, mel_lens_gt)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # ===== Logs
            # MCD and loss
            mcd_batch_value = mcd_batch(out_post.detach().cpu().transpose(1, 2).numpy(),
                                        mels_gt.cpu().transpose(1, 2).numpy(),
                                        mel_lens_gt.cpu().numpy())

            if self.step_global % self.params["tb_log_interval"] == 0:
                # Gardient histograms
                module_grads = self.get_module_grads_flattened(self.step_global)
                self.log_writer(module_grads, type="hist")
                
                
                log_dict = {f"train/loss": (loss, self.step_global),
                            f"train/mcd": (mcd_batch_value, self.step_global)
                            }
                self.log_writer(log_dict)
            
            msg = f'| Epoch: {epoch} - {self.step_global}, itr: {itr}/{len(self.dataloader_train)} ' + \
                  f'::  step loss: {loss.item():#.4} | mcd: {mcd_batch_value:#.4} '
            print(msg)

            self.step_global += 1
        
        # Plot example after each epoch
        idx = -1
        step_temp = self.step_global // 1000
        example_attn = out_attn[idx][:, :].detach().cpu().numpy()
        example_mel = out_post[idx].detach().cpu().numpy()
        example_mel_gt = mels_gt[idx].detach().cpu().numpy()
        plot_save_path = os.path.join(self.path_manager.examples_path, 
                                        f'train-{step_temp}K')
        plot_spec_attn_example(example_mel, 
                               example_mel_gt,  
                               example_attn,
                               plot_save_path,
                               length_mel=mel_lens_gt[idx].item(),
                               length_attn=model_inputs["input_lengths"][idx].item())

    def _test(self, epoch):
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

        log_dict = {f"test/loss": (loss_total, self.step_global),
                    f"test/mcd": (mcd_batch_value_total, self.step_global)
                    }
        self.log_writer(log_dict)
    
        msg = f'| Epoch: {epoch}, itr: {self.step_global} ::  loss_total:' +\
                f' {loss_total:#.4} | mcd_total: {mcd_batch_value_total:#.4} '
        print(msg)

    def _metatest(self, epoch):
        self.model.train()
        
        for itr_b, items_b in enumerate(self.dataloader_metatest):
            for spk in items_b.keys():
                # Run model in stateless mode
                grad_list = []
                with higher.innerloop_ctx(self.model, self.inner_optimizer, track_higher_grads =\
                                          self.params["track_higher_grads"]) as (fmodel, diffopt):                    
                    # ===== Train set
                    batch = items_b[spk]["train"]
                    model_inputs, stop_labels_gt = self._unpack_batch(batch)
                    mels_gt = model_inputs["melspecs"]
                    mel_lens_gt = model_inputs["melspec_lengths"]

                    # Iterate
                    for inner_iter in range(self.params["n_inner_test"]):
                        out_post, out_inner, out_stop, out_attn = fmodel(**model_inputs)
                        y_pred = (out_post, out_inner, out_stop, out_attn)
                        y_gt = (mels_gt, stop_labels_gt)
                        loss_train = self.criterion(y_pred, y_gt, mel_lens_gt)
                        diffopt.step(loss_train)
                    
                    # ===== Test set
                    # Load test data
                    batch = items_b[spk]["test"]
                    model_inputs, stop_labels_gt = self._unpack_batch(batch)
                    mels_gt = model_inputs["melspecs"]
                    mel_lens_gt = model_inputs["melspec_lengths"]

                    # Evaluate test loss without gradient
                    with torch.no_grad():
                        out_post, out_inner, out_stop, out_attn = fmodel(**model_inputs)
                        y_pred = (out_post, out_inner, out_stop, out_attn)
                        y_gt = (mels_gt, stop_labels_gt)
                        loss_test = self.criterion(y_pred, y_gt, mel_lens_gt)

                # ===== Logs
                # Example mel-spec plot
                idx = -1
                example_attn = out_attn[idx][:, :].detach().cpu().numpy()
                example_mel = out_post[idx].detach().cpu().numpy()
                example_mel_gt = mels_gt[idx].detach().cpu().numpy()
                plot_save_path = os.path.join(self.path_manager.examples_path, 
                                              f'metatest_epoch-{epoch}_{spk}')
                plot_spec_attn_example(example_mel, 
                                       example_mel_gt,  
                                       example_attn,
                                       plot_save_path,
                                       length_mel=mel_lens_gt[idx].item(),
                                       length_attn=model_inputs["input_lengths"][idx].item())
            
                mcd_batch_value = mcd_batch(out_post.cpu().transpose(1, 2).numpy(),
                                            mels_gt.cpu().transpose(1, 2).numpy(),
                                            mel_lens_gt.cpu().numpy())

                log_dict = {f"test/loss_{spk}": (loss_test, self.step_global),
                            f"test/mcd_{spk}": (mcd_batch_value, self.step_global)
                            }
                self.log_writer(log_dict)
                msg = f'| Epoch: {epoch}, itr: {self.step_global}, spk:{spk} ::  step loss:' +\
                      f' {loss_test.item():#.4} | mcd: {mcd_batch_value:#.4} '
                print(msg)


def main(args):
    params = load_params(os.path.join(args.params_path, "params.yml"))
    jt = JointTrainer(**params)
    jt.run()


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str)
    args = parser.parse_args()
    main(args)

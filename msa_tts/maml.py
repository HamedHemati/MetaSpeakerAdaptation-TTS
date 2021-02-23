from .utils.limit_threads import *
import argparse
import os
import higher
import torch
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from .metatrainer import MetaTrainer
from .utils.generic import load_params
from .utils.grad_utils import apply_grad, mix_grad
from .utils.metrics import mcd_batch
from .utils.plot import plot_spec_attn_example


class MAML(MetaTrainer):
    def __init__(self, **params):
        super(MAML, self).__init__(**params)

    def run(self):
        self.step_global = 0
        for epoch in range(1, self.params["n_epochs"] + 1):
            # Train task for one epoch
            self._metatrain(epoch)

            if epoch % self.params["ckpt_save_epoch_interval"] == 0:
                self._save_checkpoint()
            
            if epoch % self.params["metatest_epoch_interval"] == 0:
                print("Meta-test phase ...")
                self._metatest(epoch)
                print("\n")

    def _metatrain(self, epoch):
        self.model.train()
        
        for itr_b, items_b in enumerate(self.dataloader_metatrain):
            grad_list = []
            for spk in items_b.keys():
                # Run model in stateless mode
                with higher.innerloop_ctx(self.model, self.inner_optimizer, track_higher_grads =\
                                          self.params["track_higher_grads"]) as (fmodel, diffopt):                    
                    # ===== Train set
                    batch = items_b[spk]["train"]
                    model_inputs, stop_labels_gt = self._unpack_batch(batch)
                    mels_gt = model_inputs["melspecs"]
                    mel_lens_gt = model_inputs["melspec_lengths"]

                    # Iterate
                    for inner_iter in range(self.params["n_inner_train"]):
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

                    # Evaluate test loss 
                    out_post, out_inner, out_stop, out_attn = fmodel(**model_inputs)
                    y_pred = (out_post, out_inner, out_stop, out_attn)
                    y_gt = (mels_gt, stop_labels_gt)
                    loss_test = self.criterion(y_pred, y_gt, mel_lens_gt)
                    
                    # Compute gradients for outer update
                    # If FOMAML compute grad w.r.t. param_t-1 otherwise w.r.t. param_t0
                    if self.params["track_higher_grads"]:
                        task_grads = torch.autograd.grad(loss_test, fmodel.parameters(time=0))
                    else:
                        task_grads = torch.autograd.grad(loss_test, fmodel.parameters(time=-1))
                        
                    grad_list.append(task_grads)

                    # ===== Logs
                    # MCD and loss
                    mcd_batch_value = mcd_batch(out_post.detach().cpu().transpose(1, 2).numpy(),
                                                mels_gt.cpu().transpose(1, 2).numpy(),
                                                mel_lens_gt.cpu().numpy())

                    log_dict = {"train/mcd": (mcd_batch_value, self.step_global),
                                "train/loss": (loss_test, self.step_global),
                                f"train/loss_{spk}": (loss_test, self.step_global)
                                }
                    self.log_writer(log_dict)
                    msg = f'| Epoch: {epoch}, itr: {self.step_global}, spk:{spk} ::  step loss:' +\
                            f' {loss_test.item():#.4} | mcd: {mcd_batch_value:#.4} '
                    print(msg)

            # ===== Outer loop
            self.model.zero_grad()

            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad_list = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.model, grad_list)

            if self.params["clip_grad_norm"]:
                grad_norm = clip_grad_norm_(self.model.parameters(), 
                                            self.params["grad_clip_thresh"])
            
            self.outer_optimizer.step()

            # ===== Logs
            self.step_global += 1
            
            # # Gardient histograms
            # module_grads = self.get_module_grads_flattened(self.step_global)
            # self.log_writer(module_grads, type="hist")
            

    def _metatest(self, epoch):
        self.model.train()
        
        for itr_b, items_b in enumerate(self.dataloader_metatest):
            for spk in items_b.keys():
                # Run model in stateless mode
                grad_list = []
                with higher.innerloop_ctx(self.model, self.inner_optimizer, track_higher_grads =\
                                          self.params["track_higher_grads"]) as (fmodel, diffopt):
                    initial_params = deepcopy(fmodel.parameters())
                    
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
    maml = MAML(**params)
    maml.run()


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str)
    args = parser.parse_args()
    main(args)

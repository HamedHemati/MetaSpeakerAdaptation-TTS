from .utils.limit_threads import *
import argparse
import os
import sys
import yaml
import torch
import higher
import copy
import pickle
import soundfile
import random
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from .utils.generic import load_params
from datetime import datetime
from .utils.path_manager import PathManager
from .dataloaders.dataloader_default import get_dataloader as  get_dataloader_default
from .models.tacotron2nv import Tacotron2NV
from .models.modules_tacotron2nv.tacotron2nv_loss import Tacotron2Loss
from .utils.helpers import get_optimizer
from .utils.g2p.char_list import char_list
from .utils.metrics import mcd_batch
from .utils.plot import plot_spec_attn_example
from .utils.plot import plot_attention, plot_spectrogram
from .utils.g2p.grapheme2phoneme import Grapheme2Phoneme
from .utils.helpers import get_wavernn
from .utils.wavernn.audio_denoiser import AudioDenoiser


class InferCumulative():
    r"""Base class Trainer. All trainers should inherit from this class."""
    def __init__(self, **params):
        self.params = params
        # Create output folders
        output_path = os.path.join(self.params["output_path"], 
                                   self.params["method"], 
                                   self.params["experiment_name"])
        self.path_manager = PathManager(output_path)

        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Compute device: {self.device}")

        # Save all spakers
        self.all_speakers = self.params["dataset_train"]["speakers_list"]
        # random.Random(self.params["speaker_seed"]).shuffle(self.all_speakers)
        print(self.all_speakers)
        
        self._load_model()

    def _load_model(self):
        # Set model
        self.params["model"]["num_speakers"] = 1 #len(self.dataloader_train.dataset.speaker_to_id.keys())
        self.params["model"]["n_symbols"] = len(char_list)
        self.params["model"]["n_mel_channels"] = self.params["audio_params"]["n_mels"]

        # Set freezing options
        self.params["model"]["freeze_charemb"] = self.params["freeze_charemb"]
        self.params["model"]["freeze_encoder"] = self.params["freeze_encoder"]
        self.params["model"]["freeze_decoder"] = self.params["freeze_decoder"]
        
        self.model_name = self.params["model_name"]
        self.speaker_emb_type = self.params["model"]["speaker_emb_type"]
        if self.model_name  == "Tacotron2NV":
            self.model = Tacotron2NV(self.params["model"])
        else:
            raise NotImplementedError
        self.model.to(self.device)

    def _init_dataloaders(self, speaker):
        # Load meta-train loaders
        print(f"\nInitializing train/test loaders for {speaker}")
        log_ds = ""

        self.params["dataset_train"]["speakers_list"] = [speaker]
        _, self.dataloader_test, logs_tr = get_dataloader_default(**self.params)
        log_ds += "Train:\n\n" + logs_tr + "\n\n\n"

        # Write DS details to a text file
        with open(os.path.join(self.path_manager.output_path, "dataset_details.txt"), 'w') as ds_details:
            ds_details.write(log_ds)
        
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

    def _load_checkpoint(self, checkpoint_path):
         # Load checkpoint
        print(f"Loading checkpoint from  {checkpoint_path}")  
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
           
    def _init_wavernn(self):
        self.params_wavernn = load_params(self.params["vocoder_params_path"])
        self.wavernn = get_wavernn(self.device, **self.params_wavernn)
        noise_profile_path="experiments/files/noise_profiles/noise_prof1.wav"
        self.audio_denoiser = AudioDenoiser(noise_profile_path)

    def run(self):
        self.speakers_so_far = []
        # Speaker embedding
        with open(self.params["spk_emb_path"], "rb") as pkl_file:
            self.speaker_embeddings = pickle.load(pkl_file)
        # Set G2P
        self.g2p = Grapheme2Phoneme()
        # Init WaveRNN
        self._init_wavernn()
        
        for spk_itr, speaker in enumerate(self.all_speakers):
            self.speakers_so_far.append(speaker)
            if str(spk_itr) != self.params["checkpoint_id"]:
                continue
            print(f"Inferring for speaker {speaker}.")
            checkpoint_path = os.path.join(self.path_manager.checkpoints_path, 
                                        f"best_{spk_itr+self.params['num_initial_speakers']}_{speaker}.pt")
            self._load_checkpoint(checkpoint_path)
            for target_speaker in self.speakers_so_far:
                self._infer_for_speaker(spk_itr, speaker, target_speaker)

    def _infer_for_speaker(self, step, ref_speaker, speaker):
        """Generates mel-spec."""
        print(f"Inferring from {ref_speaker} to {speaker}.")
        self.model.eval()
        # Input char list tensor
        inp_chars, _ = self.g2p.convert(inp=self.params["input_text"], 
                                        language=self.params["language"], 
                                        convert_mode=self.params["convert_mode"])
        inp_chars = torch.tensor(inp_chars).long().to(self.device)
        inp_len = torch.tensor([len(inp_chars)]).to(self.device)

        # Speaker embedding
        spk_vec = torch.tensor(self.speaker_embeddings[speaker]["mean"]).unsqueeze(0).to(self.device)
        
        # Feed inputs to the models
        postnet_outputs, mel_lengths, attn_weights = self.model.infer(inp_chars.unsqueeze(0), 
                                                                 inp_len, 
                                                                 spk_vec)

        postnet_outputs = postnet_outputs.squeeze(0).detach().cpu().numpy().T
        attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()
        
        print(f"postnet_outputs: {postnet_outputs.shape}")
        print(f"attn_weights: {attn_weights.shape}")

        melspec = postnet_outputs.T        


        # Save mel-spec and wav
        file_name = f"{step}_{ref_speaker}_{speaker}"

        melspec_path = os.path.join(self.path_manager.inference_path, file_name + "_mel")
        plot_spectrogram(melspec, melspec_path)
        
        # Generate wav
        wav_path = os.path.join(self.path_manager.inference_path, file_name + ".wav")
        wav = self.wavernn.generate(torch.tensor(melspec).unsqueeze(0), True, 
                                    self.params_wavernn["target"], 
                                    self.params_wavernn["overlap"])
        wav = self.audio_denoiser.denoise(wav)
        soundfile.write(wav_path, wav, self.params["audio_params"]["sample_rate"])

        self.model.train()

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


def main(params):

    r"""Main function that sets and runs the trainer."""
    # Get experiment path
    experiment_path = os.environ["EXPERIMENT_PATH"]
    # Load YAML params file
    print(f"Experiment path: {experiment_path}")
    params = load_params(os.path.join(experiment_path, "params.yml"))
    
    # Add CMD params to the inference params
    params.update(cmd_params)

    # Add audio params
    audio_params = get_audio_params(params)
    if audio_params:
        params["audio"] = audio_params

    # Make inference
    ic = InferCumulative(**params)
    ic.run()


def get_audio_params(params):
    r"""Returns dictionary of audio_params."""
    # If audio_params_path given as CMD parameter, load it
    if "audio_params_path" in params:
        audio_params = load_params(params["audio_params_path"])
        return audio_params
    # Otherwise load params.yml from one of the datasets used for training the model
    else:
        return None


def get_cmd_params():
    r"""Retrieves list of parameters from command line and returns them as a dict."""
    args = sys.argv[1:]
    
    # Make sure number of arguments is even (must be key,value pair)
    assert len(args) % 2 ==0
    
    # Create CMD params
    cmd_params = {}
    for i in range(1,len(args), 2):
        # Remove -- from the beginning of keys
        key_name = args[i-1][2:]
        value = args[i]
        cmd_params[key_name] = value

    return cmd_params


if __name__ == "__main__":
    # Get CMD params
    cmd_params = get_cmd_params()

    main(cmd_params)

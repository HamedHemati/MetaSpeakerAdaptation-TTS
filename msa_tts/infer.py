from .utils.limit_threads import *
import torch
import argparse
import os
import sys
import importlib
import numpy as np
import librosa
import argparse
import os
import importlib
import soundfile
import yaml
import higher
import pickle
from .models.tacotron2nv import Tacotron2NV
from .models.modules_tacotron2nv.tacotron2nv_loss import Tacotron2Loss
from .utils.wavernn.audio_denoiser import AudioDenoiser
from .utils.generic import load_params
from .utils.ap import AudioProcessor
from .utils.path_manager import PathManager
from .utils.plot import plot_attention, plot_spectrogram
from .utils.g2p.grapheme2phoneme import Grapheme2Phoneme
from .utils.g2p.char_list import char_list
from .utils.helpers import get_wavernn
from .utils.helpers import get_optimizer
from .dataloaders.dataloader_meta import get_dataloader


class Inference():
    def __init__(self, **params):
        self.params = params

        r"""Makes inference with the model given the parameters."""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # Audio processor
        self.ap = AudioProcessor(self.params["audio_params"], device=self.device)

        # Set path manager
        output_path = os.path.join(self.params["output_path"], 
                                   self.params["method"], 
                                   self.params["experiment_name"])
        self.path_manager = PathManager(output_path)
        
        # Set G2P
        self.g2p = Grapheme2Phoneme()
        
        # Set n_mel and n_symbols in the model
        params["model"]["n_mel_channels"] = params["audio_params"]["n_mels"]
        params["model"]["n_symbols"] = len(char_list)
        
        # Set num_speakers = 1
        params["model"]["num_speakers"] = 1

        # Replace parameters from inference shell
        params["n_inner_test"] = int(params["n_inner_test"])

        # Model
        self._init_model()
        self._freeze_modules()

        # Outer optimizer
        self.inner_optimizer = get_optimizer(self.model, **params["optim_inner"])
        
        # Dataloader
        self.params["dataset_metatest"]["batch_size"] = int(params["batch_size"])
        self.dataloader_metatest, log = get_dataloader("metatest", **self.params)
        print(log)
        print(len(self.dataloader_metatest))
        # exit()

    def _init_model(self):
        r"""Initializes Tacotron model."""
        # Init model
        self.model_name = self.params["model_name"] 
        if self.model_name == "Tacotron2NV":
            self.model = Tacotron2NV(self.params["model"]).to(self.device)
        else:
            raise NotImplementedError

        self.speaker_emb_type = self.params["model"]["speaker_emb_type"]

        # Load checkpoint
        checkpoint_path = os.path.join(self.path_manager.checkpoints_path, 
                                       f"checkpoint_{self.params['checkpoint_id']}.pt")
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        print("Loaded model checkpoint.")

        # Criterion
        self.criterion = Tacotron2Loss(n_frames_per_step=self.params["model"]["n_frames_per_step"],
                                       reduction=self.params["criterion"]["reduction"],
                                       pos_weight=self.params["criterion"]["pos_weight"],
                                       device=self.device)

    def _freeze_modules(self):
        # Freeze layers
        try:
            # - Freeze char embedder
            if params["freeze_charemb"]:
                print("Freezing Char Embedder:")
                for name, param in self.model.named_parameters():
                    if name.startswith("embedding."):
                        print(f"Freezing {name}")
                        param.requires_grad = False

            # - Freeze encoder
            if params["freeze_encoder"]:
                print("Freezing Encoder:")
                for name, param in self.model.named_parameters():
                    if name.startswith("encoder."):
                        print(f"Freezing {name}")
                        param.requires_grad = False

            # - Freeze decoder
            if params["freeze_decoder"]:
                print("Freezing Decoder")
                for name, param in self.model.named_parameters():
                    if name.startswith("decoder."):
                        print(f"Freezing {name}")
                        param.requires_grad = False
        except:
            print("Freezing options missing ...")

    def _unpack_batch(self, batch_items):
        r"""Un-packs batch items and sends them to compute device"""
        item_ids, inp_chars, inp_lens, mels, mel_lens,\
        speakers_ids, spk_embs, stop_labels = batch_items
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

    def generate_melspec(self, model):
        """Generates mel-spec."""
        # Input char list tensor
        inp_chars, _ = self.g2p.convert(inp=self.params["input_text"], 
                                        language=self.params["language"], 
                                        convert_mode=self.params["convert_mode"])
        inp_chars = torch.tensor(inp_chars).long().to(self.device)
        inp_len = torch.tensor([len(inp_chars)]).to(self.device)

        # Speaker embedding
        with open(self.params["spk_emb_path"], "rb") as pkl_file:
            tmp = pickle.load(pkl_file)
        spk_vec = torch.tensor(tmp[self.params["speaker"]]).unsqueeze(0).to(self.device)

        # Feed inputs to the models
        postnet_outputs, mel_lengths, attn_weights = model.infer(inp_chars.unsqueeze(0), 
                                                                 inp_len, 
                                                                 spk_vec)

        postnet_outputs = postnet_outputs.squeeze(0).detach().cpu().numpy().T
        attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()
        
        print(f"postnet_outputs: {postnet_outputs.shape}")
        print(f"attn_weights: {attn_weights.shape}")

        mel = postnet_outputs.T
        
        return mel, attn_weights
    
    def make_inference(self):
        # ============ Meta adaptation
        for itr_b, items_b in enumerate(self.dataloader_metatest):
            for spk in items_b.keys():
                if spk == self.params["speaker"]:
                    self.model.train()
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
                            print(f"{inner_iter}/{self.params['n_inner_test']}, loss: {loss_train.item()}")

                        fmodel.eval()
                        print("Generating melspec ...")
                        melspec, attn_weights = self.generate_melspec(fmodel)
                        break

        # Set filename
        filename = self.params["input_text"][:10].lower().replace(" ", "_")
        
        # Save attention
        attn_path = os.path.join(self.path_manager.inference_path, filename + "_attn")
        plot_attention(attn_weights, attn_path)

        # Save melspec
        melspec_path = os.path.join(self.path_manager.inference_path, filename + "_mel")
        plot_spectrogram(melspec, melspec_path)

        # Get vocoder and generate wav
        print("Generating wav ...")
        if self.params["vocoder"] == "griffinlim":
            wav = self.ap.griffinlim_logmelspec(torch.tensor(melspec).unsqueeze(0))
            wav = wav.squeeze(0).cpu().numpy()

        elif self.params["vocoder"].startswith("wavernn"):
            params_wavernn = load_params(self.params["vocoder_params_path"])
            wavernn = get_wavernn(self.device, **params_wavernn)
            wav = wavernn.generate(torch.tensor(melspec).unsqueeze(0), True, 
                                   params_wavernn["target"], 
                                   params_wavernn["overlap"])
            noise_profile_path="experiments/files/noise_profiles/noise_prof1.wav"
            audio_denoiser = AudioDenoiser(noise_profile_path)
            wav = audio_denoiser.denoise(wav)

        # Save wav
        wav_path = os.path.join(self.path_manager.inference_path, filename + ".wav")
        soundfile.write(wav_path, wav, self.params["audio_params"]["sample_rate"])
        
        # Save melspec as npy file
        mel_npy_path = os.path.join(self.path_manager.inference_path, filename + ".npy")
        np.save(mel_npy_path, melspec)



##############################
#           Main
##############################
def main(cmd_params):
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
    inference = Inference(**params)
    inference.make_inference()


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

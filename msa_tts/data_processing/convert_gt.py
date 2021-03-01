from msa_tts.utils.limit_threads import * 
import torch
import argparse
import os
import sys
import soundfile as sf
import glob
import soundfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from msa_tts.utils.helpers import get_wavernn
from msa_tts.utils.wavernn.audio_denoiser import AudioDenoiser
from msa_tts.utils.generic import load_params
from msa_tts.utils.ap import AudioProcessor


class GTConvertor():
    def __init__(self, 
                 params):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params_wavernn = load_params(self.params["vocoder_params_path"])
        self.wavernn = get_wavernn(self.device, **self.params_wavernn)
        noise_profile_path = "experiments/files/noise_profiles/noise_prof1.wav"
        self.audio_denoiser = AudioDenoiser(noise_profile_path)
        self.audio_processor = AudioProcessor(self.params["audio_params"], self.device)

    def convert_file(self, source_wav_path, target_wav_path, log):
        # Generate wav
        print(log)
        waveform = self.audio_processor.load_audio(source_wav_path)[0].to(self.device)
        melspec = self.audio_processor.get_melspec(waveform)[2]
        wav_reconst = self.wavernn.generate(torch.tensor(melspec).unsqueeze(0), True, 
                                            self.params_wavernn["target"], 
                                            self.params_wavernn["overlap"])
        wav_reconst = self.audio_denoiser.denoise(wav_reconst)
        soundfile.write(target_wav_path, wav_reconst, self.params["audio_params"]["sample_rate"])

    def run(self):
        source_path = os.path.join(self.params["ds_path"], self.params["source_folder"])
        target_path = os.path.join(self.params["ds_path"], self.params["target_folder"])
        speakers = os.listdir(source_path)    
        speakers = [speaker for speaker in speakers if os.path.isdir(os.path.join(source_path, speaker))]

        for spk_itr, speaker in enumerate(speakers):
            # Create folder for individual speakers in the target path
            os.makedirs(os.path.join(target_path, speaker), exist_ok=True)
            # Get list of wav files from the source path
            wav_paths = glob.glob(os.path.join(source_path, speaker, "*.wav"))
            for wav_itr, source_wav_path in enumerate(wav_paths):
                target_wav_path = os.path.join(target_path, speaker, source_wav_path.split("/")[-1])
                log = f"Converting speaker {spk_itr}/{len(speakers)}: {wav_itr}/{len(wav_paths)}"
                self.convert_file(source_wav_path, target_wav_path, log)
                        
def main(params):

    r"""Main function that sets and runs the trainer."""

    # Add audio params
    audio_params = get_audio_params(params)
    params["audio_params"] = audio_params

    # Make inference
    gc = GTConvertor(params)
    gc.run()


def get_audio_params(params):
    r"""Returns dictionary of audio_params."""
    params = load_params(params["vocoder_params_path"])
    return params["audio_params"]


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

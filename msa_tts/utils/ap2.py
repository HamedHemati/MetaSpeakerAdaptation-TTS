import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf


class AudioProcessor2():
    def __init__(self, params, device=torch.device("cpu")):
        self.params = params
        self.device = device
        self.mel_basis = {}
        self.hann_window = {}
        mel = librosa_mel_fn(self.params["sample_rate"], 
                                 self.params["n_fft"], 
                                 self.params["n_mels"], 
                                 self.params["fmin"], 
                                 self.params["fmax"])
        self.mel_basis[str(self.params["fmax"])+'_'+str(self.device)] = torch.from_numpy(mel).float().to(self.device)
        self.hann_window[str(self.device)] = torch.hann_window(self.params["win_size"]).to(self.device)

    def load_audio(self, audio_path):
        wav, sr = sf.read(audio_path)
        wav = torch.FloatTensor(wav).unsqueeze(0).to(self.device)
        return wav
    
    def get_melspec(self, y):
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))
            
        y = torch.nn.functional.pad(y.unsqueeze(1), 
                                    (int((self.params["n_fft"]-self.params["hop_size"])/2),
                                     int((self.params["n_fft"]-self.params["hop_size"])/2)), 
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, 
                          self.params["n_fft"], 
                          hop_length=self.params["hop_size"], 
                          win_length=self.params["win_size"], 
                          window=self.hann_window[str(self.device)],
                          center=self.params["center"], 
                          pad_mode='reflect', 
                          normalized=False, 
                          onesided=True)

        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(self.mel_basis[str(self.params["fmax"])+'_'+str(self.device)], spec)
        log_melspec = self.spectral_normalize_torch(spec)

        return None, None, log_melspec
    
    def dynamic_range_compression(self,x, C=1, clip_val=1e-5):
        return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


    def dynamic_range_decompression(self, x, C=1):
        return np.exp(x) / C


    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)


    def dynamic_range_decompression_torch(self, x, C=1):
        return torch.exp(x) / C


    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output


    def spectral_de_normalize_torch(self, magnitudes):
        output = self.dynamic_range_decompression_torch(magnitudes)
        return output
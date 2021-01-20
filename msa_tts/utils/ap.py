import torch
import torchaudio
import librosa
import numpy as np
torchaudio.set_audio_backend("sox")


class AudioProcessor():
    def __init__(self, audio_params, device=torch.device("cpu")):
        self.audio_params = audio_params
        self.device = device
        # Set transformations
        # STFT
        self.transform_stft = torchaudio.transforms.Spectrogram(n_fft=audio_params["n_fft"],
                                                                win_length=audio_params["win_length"],
                                                                hop_length=audio_params["hop_length"],
                                                                window_fn=torch.hann_window).to(device)
        
        # Mel-scale transform
        self.transform_melscale = torchaudio.transforms.MelScale(n_mels=audio_params["n_mels"],
                                                                 sample_rate=audio_params["sample_rate"],
                                                                 f_min=audio_params["f_min"],
                                                                 f_max=audio_params["f_max"],
                                                                 n_stft=audio_params["n_fft"]//2+1).to(device)
        
        # MFCC
        self.transform_mfcc = torchaudio.transforms.MFCC(sample_rate=audio_params["sample_rate"],
                                                         n_mfcc=audio_params["n_mfcc"],
                                                         log_mels=True,
                                                         melkwargs={"n_fft": audio_params["n_fft"],
                                                                    "win_length": audio_params["win_length"],
                                                                    "hop_length": audio_params["hop_length"],
                                                                    "n_mels": audio_params["n_mels"],
                                                                    "f_min": audio_params["f_min"],
                                                                    "f_max": audio_params["f_max"],}).to(device)
        
        # GriffinLim
        self.griffinlim = torchaudio.transforms.GriffinLim(n_fft=audio_params["n_fft"],
                                                           n_iter=audio_params["griffinlim_iters"],
                                                           win_length=audio_params["win_length"],
                                                           hop_length=audio_params["hop_length"],
                                                           power=2,
                                                           rand_init=True,
                                                           momentum=0.99).to(device)
        

    def load_audio(self, audio_path):
        r"""Loads audio file and resamples it if necessary.
        
        Parameters:
        audio_path (torch.tensor): Path to the audio file
        
        Returns:
        torch.tensor: loaded waveform
        """
        x, sr = torchaudio.load(audio_path, normalization=lambda x: torch.abs(x).max())
        # Resample if sample rate of the input is different
        if sr != self.audio_params["sample_rate"]:
            x = torchaudio.transforms.Resample(orig_freq=sr, 
                                               new_freq=self.audio_params["sample_rate"])(x)
        return x
    
    def get_melspec(self, x):
        r"""Computes STFT and MelSpectrogram respectively.
        
        Parameters:
        x (torch.tensor): input waveform
        
        Returns:
        torch.tensor: STFT of the input waveform
        torch.tensor: Mel-scale spectrogram of the computed STFT
        torch.tensor: Log of the computed Mel-scale spectrogram
        """ 
        stft = self.transform_stft(x)
        log_stft = torch.log10(torch.clamp(stft, min=1e-10))
        
        melspec = self.transform_melscale(stft)
        log_melspec = torch.log10(torch.clamp(melspec, min=1e-10))

        return (stft, log_stft), melspec, log_melspec
    
    def get_mfcc(self, x):
        r"""Computes MFCC of a waveform.
        
        Parameters:
        x (torch.tensor): input waveform
        
        Returns:
        torch.tensor: MFCC transformation of the input waveform.
        """
        mfcc = self.transform_mfcc(x)
        
        return mfcc

    @staticmethod
    def trim_margin_silence(x, ref_level_db=26):
        r"""Trims margin silence of a waveform.
        
        Parameters:
        x (torch.tensor): input waveform
        ref_level_db: reference level in decibel
        
        Returns:
        torch.tensor: trimmed waveform
        """
        trimmed_x = librosa.effects.trim(x.numpy(), 
                                         top_db=ref_level_db, 
                                         frame_length=1024, 
                                         hop_length=256)[0]
        x = torch.FloatTensor(trimmed_x)
        
        return x
    
    def griffinlim_logmelspec(self, log_melspec):
        r"""Converts Mel-scaled spec to spectrogram and then 
            runs GriffinLim on a spectrogram to get the waveform.
        
        Parameters:
        spec (torch.tensor): input spectrogram
        
        Returns:
        torch.tensor: reconstrcuted waveform
        """
        EPS = 1e-10
        melspec = torch.pow(torch.tensor(10.0), log_melspec)
        fbank = torchaudio.functional.create_fb_matrix(self.audio_params["n_fft"]//2+1, 
                                                       self.audio_params["f_min"],
                                                       self.audio_params["f_max"], 
                                                       self.audio_params["n_mels"],
                                                       self.audio_params["sample_rate"])
        inv_mel_basis = torch.tensor(np.linalg.pinv(fbank.numpy()))
        spec = torch.matmul(inv_mel_basis.T, melspec.squeeze(0))
        
        spec = torch.clamp(spec, min=EPS)
        spec = torch.abs(spec)
        
        waveform = self.griffinlim(spec.unsqueeze(0).to(self.device))

        return waveform
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import os
import numpy as np
from msa_tts.utils.ap import AudioProcessor


# ====================
# ==================== TTS Dataset
# ====================

class TTSDataset(Dataset):
    def __init__(self, g2p, ds_data, **params):
        self.params = params
        self.text_processor = text_processor 
        self.ds_data = ds_data
        self.audio_processor = AudioProcessor(params["audio_params"])
        self._load_metadata()
        
    def _load_metadata(self):
        all_lines = self.ds_data["item_list"]
        
        # Create metadata dict
        self.metadata = {l[0]:{"ds_root": self.ds_data["dataset_path"],
                         "transcript": l[2], "duration": float(l[3]),
                         "trim_margin_silence": self.ds_data["trim_margin_silence"]} for l in all_lines}
        
        # Extend global list of speakers
        self.speakers_list =self.ds_data["speakers_list"]

        # Speaker ID
        self.speaker_to_id = {s:i for (i,s) in enumerate(self.speakers_list)}
        self.id_to_speaker = {b:a for (a,b) in self.speaker_to_id.items()}
        
        # Items' list
        self.items = list(self.metadata.keys())

    def __getitem__(self, index):
        item_id = self.items[index]
        
        # Get input chars (sequence of indices)
        transcript =  self.metadata[item_id]["transcript"]

        # ================ NEWLY ADDED
        # TODO evaluate the correctness of the pre-processing
        # Convert middle dots to commas
        transcript = transcript.replace(".", ",")
        # Replace last char with .
        if transcript[-1] != ",":
            transcript = transcript + "."
        else:
            transcript = transcript[:-1] + "."
        
        # Convert transcript with the text processor
        transcript, _ = self.text_processor.convert(transcript)
                            
        # Get speaker ID
        speaker_id = 0 # Always 0!
            
        # Load waveform
        waveform_path = os.path.join(self.metadata[item_id]["ds_root"],
                                     f'{item_id}')

        waveform = self.audio_processor.load_audio(waveform_path)[0].unsqueeze(0)       
        if self.metadata[item_id]["trim_margin_silence"] == True:
            waveform = self.audio_processor.trim_margin_silence(waveform)       

        transcript = torch.LongTensor(transcript)

        return item_id, transcript, speaker_id, waveform

    def __len__(self):
        return len(self.items)

    def get_audio_durations(self):
        trans_lens = [self.metadata[k]["duration"] for k in self.items]
        return trans_lens


# ====================
# ==================== Collator
# ====================

class TTSColator():
    r"""TTS Collator Class.""" 
    def __init__(self, reduction_factor, audio_params):
        self.reduction_factor = reduction_factor
        self.audio_processor = AudioProcessor(audio_params)

    def __call__(self, batch):
        r"""Prepares batch.
        batch: item_id, inp_chars, speaker_id, wavform
        """
        # Compute text lengths and 
        trans_lengths = np.array([len(t[1]) for t in batch])
        
        # Sort items w.r.t. the transcript length for RNN efficiency
        trans_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(trans_lengths), dim=0, descending=True)

        # Create list of batch items sorted by text length
        item_ids = [batch[idx][0] for idx in ids_sorted_decreasing]
        transcripts = [batch[idx][1] for idx in ids_sorted_decreasing]
        speaker_ids = [batch[idx][2] for idx in ids_sorted_decreasing]
        waveforms = [batch[idx][3] for idx in ids_sorted_decreasing]
        
        # Compute Mel-spectrograms
        melspecs = [self.audio_processor.get_melspec(waveform)[2] for waveform in waveforms]
        melspec_lengths = [mel.shape[-1] for mel in melspecs]

        # Create stop labels
        stop_targets = [torch.FloatTensor(np.array([0.] * (mel_len - 1) + [1.])) for mel_len in melspec_lengths]

        # Pad and preprate tensors
        transcripts = self.pad_and_prepare_transcripts(transcripts)
        melspecs = self.pad_and_prepare_spectrograms(melspecs)
        stop_targets = self.pad_and_prepare_stoptargets(stop_targets)

        # Convert numpy arrays to PyTorch tensors
        melspec_lengths = torch.LongTensor(melspec_lengths)
        speaker_ids = torch.LongTensor(speaker_ids)

        batch = (item_ids, transcripts, trans_lengths, melspecs, melspec_lengths, speaker_ids, stop_targets)
        
        return batch

    def pad_and_prepare_transcripts(self, transcripts):
        r"""Pads and prepares transcript tensors.
        Parameters:
        transcripts (list of LongTensor): list of 1-D tensors

        Returns:
        torch.LongTensor: padded and concated tensor of size B x max_len
        """
        # Find max len
        max_len = max([len(x) for x in transcripts])

        # Pad transcripts in the batch
        def pad_transcript(x):
            return F.pad(x, 
                         (0, max_len - x.shape[0]), 
                         mode='constant', 
                         value=0)

        padded_transcripts = [pad_transcript(x).unsqueeze(0) for x in transcripts]

        return torch.cat(padded_transcripts, dim=0)

    def pad_and_prepare_spectrograms(self, inputs):
        r"""Pads and prepares spectrograms.
        
        Parameters:
        inputs (list of FloatTensor): list of 3-D spectrogram tensors of shape 1 x C x L
                                      where C is number of energy channels

        Returns:
        tensor.FloatTensor: Padded and concatenated tensor of shape B x C x max_len
        """
        max_len = max([x.shape[-1] for x in inputs])
        remainder = max_len % self.reduction_factor
        max_len_red = max_len + (self.reduction_factor - remainder) if remainder > 0 else max_len

        def pad_spectrogram(x):
            return F.pad(x,
                         (0, max_len_red - x.shape[-1], 0, 0, 0, 0), 
                         mode='constant', 
                         value=0.0)
        padded_spectrograms = [pad_spectrogram(x) for x in inputs]

        return torch.cat(padded_spectrograms, dim=0)

    def pad_and_prepare_stoptargets(self, inputs):
        r"""Pads and prepares stop targets.
        
        Parameters:
        inputs (list of FloatTensor): list of 1-D tensors of shape L

        Returns:
        tensor.FloatTensor: Padded and concatenated tensor of shape B x max_len
        """
        max_len = max([x.shape[-1] for x in inputs])
        remainder = max_len % self.reduction_factor
        max_len_red = max_len + (self.reduction_factor - remainder) if remainder > 0 else max_len
        
        def pad_stop_label(x):
            return F.pad(x,
                         (0, max_len_red - x.shape[-1]),
                         mode='constant', 
                         value=1.0)

        padded_stops = [pad_stop_label(x).unsqueeze(0) for x in inputs]

        return torch.cat(padded_stops, dim=0)


# ====================
# ==================== Binned Sampler
# ==================== 

class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths))
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)



# ====================
# ==================== TTS Dataloader
# ====================

def get_dataloader(text_processor,
                   **params):
    dataloaders_train = {}
    dataloaders_test = {}
    ds_split_key = "datasets"

    for ds in params[ds_split_key].keys():
        print(f"Loading data for {ds}")
        ds_data = params[ds_split_key][ds]
        metafile_path = os.path.join(ds_data["dataset_path"], ds_data["meta_file"])
        with open(metafile_path) as metadata:
            all_lines = metadata.readlines()
        # Read metafile lines
        all_lines = [l.strip() for l in all_lines]
        all_lines = [l.split("|") for l in all_lines]
        
        # Keep transcripts larget than 5 
        all_lines = [l for l in all_lines if len(l[2]) > 5]

        # Shuffle the lines
        random.seed(params["dataset_random_seed"])
        random.shuffle(all_lines)
        
        # ====================== Train-set item list
        # Compute cumulative sum of durations
        cum_sum_duration = list(np.cumsum([float(l[3]) for l in all_lines]))

        # Find first index where cumsum of duration is larger than total_duration_train
        trainset_duration_sec = ds_data["total_duration_train_min"] * 60.0
        try:
            first_idx = list(map(lambda i: i> trainset_duration_sec, cum_sum_duration)).index(True) 
        except:
            print("Less data than expected.")
            first_idx = len(cum_sum_duration)
        trainset_item_list = all_lines[:first_idx]
        
        
        ds_data["item_list"] = trainset_item_list
        dataset_train = TTSDataset(text_processor, ds_data, **params)

        # Get transcript lengths for the sampler
        audio_durations_train = dataset_train.get_audio_durations()

        # Define sampler
        sampler_train = BinnedLengthSampler(audio_durations_train, params["batch_size"], params["batch_size"])
        collator = TTSColator(reduction_factor=params["model"]["n_frames_per_step"], audio_params=params["audio_params"])

        # Dataloader
        dataloader_train = DataLoader(dataset_train,
                                      collate_fn=collator,
                                      batch_size=params["batch_size"],
                                      sampler=sampler_train,
                                      num_workers=params["num_workers"],
                                      drop_last=False,
                                      pin_memory=True,
                                      shuffle=False)
        dataloaders_train[ds] = dataloader_train

        # ====================== Test-set item list
        # Update all_lines
        all_lines = all_lines[first_idx:]
        assert len(all_lines) != 0
        # Compute cumulative sum of durations
        cum_sum_duration = list(np.cumsum([float(l[3]) for l in all_lines]))

        # Find first index where cumsum of duration is larger than total_duration_train
        testset_duration_sec = ds_data["total_duration_test_min"] * 60.0
        try:
            first_idx = list(map(lambda i: i> testset_duration_sec, cum_sum_duration)).index(True) 
        except:
            print("Less data than expected.")
            first_idx = len(cum_sum_duration)

        testset_item_list = all_lines[:first_idx]

        ds_data["item_list"] = testset_item_list
        dataset_test = TTSDataset(text_processor, ds_data, **params)

        # Get transcript lengths for the sampler
        audio_durations_test = dataset_test.get_audio_durations()

        # Define sampler
        sampler_test = BinnedLengthSampler(audio_durations_test, params["batch_size"], params["batch_size"])
        collator = TTSColator(reduction_factor=params["model"]["n_frames_per_step"], audio_params=params["audio_params"])

        # Dataloader
        dataloader_test = DataLoader(dataset_test,
                                      collate_fn=collator,
                                      batch_size=params["batch_size"],
                                      sampler=sampler_test,
                                      num_workers=params["num_workers"],
                                      drop_last=False,
                                      pin_memory=True,
                                      shuffle=False)
        dataloaders_test[ds] =  dataloader_test                         

    loaded_ds = list(params[ds_split_key].keys())
    print("Loaded datasets: ", ds)
    assert len(dataloaders_train) == 1
    
    return dataloaders_train[loaded_ds[0]], dataloaders_test[loaded_ds[0]]



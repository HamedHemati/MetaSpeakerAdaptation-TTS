import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import os
import numpy as np
import pickle
from msa_tts.utils.g2p.grapheme2phoneme import Grapheme2Phoneme
from msa_tts.utils.ap import AudioProcessor

# ====================
# ==================== TTS Dataset
# ====================

class TTSDataset(Dataset):
    def __init__(self, ds_data, **params):
        self.ds_data = ds_data
        self.params = params
        self.g2p = Grapheme2Phoneme()
        self.audio_processor = AudioProcessor(params["audio_params"])
        self._load_metadata()
        
    def _load_metadata(self):
        self.metadata = {}
        speakers_list = []
        for speaker in self.ds_data["item_list"].keys():
            self.metadata[speaker] = {}

            # ===== Train items
            all_lines_train = self.ds_data["item_list"][speaker]["train"]
            self.metadata[speaker]["train"] = {l[1]:{"ds_root": self.ds_data["dataset_path"], 
                                                     "speaker": l[0], 
                                                     "transcript_phonemized":l[3], 
                                                     "duration": float(l[4]),
                                                     "audio_folder": self.ds_data["audio_folder"],
                                                     "trim_margin_silence": self.ds_data["trim_margin_silence"]} 
                                                     for l in all_lines_train}
            # ===== Test items
            all_lines_test = self.ds_data["item_list"][speaker]["test"]
            self.metadata[speaker]["test"] = {l[1]:{"ds_root": self.ds_data["dataset_path"], 
                                                    "speaker": l[0], 
                                                    "transcript_phonemized":l[3], 
                                                    "duration": float(l[4]),
                                                    "audio_folder": self.ds_data["audio_folder"],
                                                    "trim_margin_silence": self.ds_data["trim_margin_silence"]} 
                                                    for l in all_lines_test}
            speakers_list.append(speaker)

        # Speaker ID
        self.speaker_to_id = {s:i for (i,s) in enumerate(speakers_list)}
        self.id_to_speaker = {b:a for (a,b) in self.speaker_to_id.items()}
        

        # Items' list
        self.items = list(self.metadata.keys())
        
        # Load speaker emebddings
        with open(os.path.join(self.ds_data["dataset_path"], "spk_emb.pkl"), "rb") as pkl_file:
            self.spk_emb_dict = pickle.load(pkl_file)
        

    def __getitem__(self, index):
        speaker = self.items[index]
        modes = ["train", "test"]
        batch_items = {}
        for mode in modes:
            batch_items[mode] = []
            speaker_itemlist = list(self.metadata[speaker][mode].keys())
            random.shuffle(speaker_itemlist)
            for itr in range(min(len(speaker_itemlist), self.ds_data["batch_size"])):
                item_id = speaker_itemlist[itr]
                item = self.metadata[speaker][mode][item_id]
                # Get input chars (sequence of indices)
                transcript_phonemized = item["transcript_phonemized"]
                transcript, _ = self.g2p.convert(transcript_phonemized, 
                                                convert_mode="phone_to_idx")
                transcript = torch.LongTensor(transcript)
                                    
                # Get speaker ID
                speaker_name = item["speaker"]
                speaker_id = self.speaker_to_id[speaker_name]
                    
                # Load waveform
                if item["audio_folder"] == "" and len(self.speaker_to_id.keys()) == 1:
                    waveform_path = os.path.join(item["ds_root"], 
                                                item_id)
                else:
                    waveform_path = os.path.join(item["ds_root"], 
                                                item["audio_folder"],
                                                item["speaker"],
                                                item_id)
                
                waveform = self.audio_processor.load_audio(waveform_path)[0].unsqueeze(0)       
                if item["trim_margin_silence"] == True:
                    waveform = self.audio_processor.trim_margin_silence(waveform)       
            
                # Speaker embedding
                spk_emb = torch.FloatTensor(self.spk_emb_dict[speaker])

                batch_items[mode].append((item_id, transcript, speaker_id, waveform, spk_emb))
        return (speaker, batch_items)

    def __len__(self):
        return len(self.items)

    def get_audio_durations(self):
        trans_lens = [self.metadata[k]["duration"] for k in self.items]
        return trans_lens


# ====================
# ==================== Collator
# ====================

class MetaCollator():
    def __init__(self, reduction_factor, audio_params):
        self.reduction_factor = reduction_factor
        self.audio_processor = AudioProcessor(audio_params)

    def __call__(self, batch_tuples):
        batch_dict = {}
        for batch_tuple in batch_tuples:
            speaker, batch_items =  batch_tuple
            batch_dict[speaker] = {}

            for k in batch_items.keys():
                batch = batch_items[k]
                # Compute text lengths and 
                trans_lengths = np.array([len(t[1]) for t in batch])
                
                # Sort items w.r.t. the transcript length for RNN efficiency
                trans_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(trans_lengths), 
                                                                dim=0, 
                                                                descending=True)

                # Create list of batch items sorted by text length
                item_ids = [batch[idx][0] for idx in ids_sorted_decreasing]
                transcripts = [batch[idx][1] for idx in ids_sorted_decreasing]
                speaker_ids = [batch[idx][2] for idx in ids_sorted_decreasing]
                waveforms = [batch[idx][3] for idx in ids_sorted_decreasing]
                spk_embs = [batch[idx][4] for idx in ids_sorted_decreasing]
                
                # Compute Mel-spectrograms
                melspecs = [self.audio_processor.get_melspec(waveform)[2] for waveform in waveforms]
                melspec_lengths = [mel.shape[-1] for mel in melspecs]

                # Create stop labels
                stop_targets = [torch.FloatTensor(np.array([0.] * (mel_len - 1) + [1.])) 
                                for mel_len in melspec_lengths]

                # Pad and preprate tensors
                transcripts = self.pad_and_prepare_transcripts(transcripts)
                melspecs = self.pad_and_prepare_spectrograms(melspecs)
                stop_targets = self.pad_and_prepare_stoptargets(stop_targets)

                # Convert numpy arrays to PyTorch tensors
                melspec_lengths = torch.LongTensor(melspec_lengths)
                speaker_ids = torch.LongTensor(speaker_ids)
                spk_embs = torch.stack(spk_embs, dim=0)

                batch = (item_ids, transcripts, trans_lengths, melspecs, melspec_lengths, 
                        speaker_ids, spk_embs, stop_targets)

                batch_dict[speaker][k] = batch
        
        return batch_dict

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
# ==================== TTS Dataloader
# ====================

def get_dataloader(phase_name,
                    **params):
    # Read metafile lines
    ds_data = params[f"dataset_{phase_name}"]
    metafile_path = os.path.join(ds_data["dataset_path"], ds_data["meta_file"])
    with open(metafile_path) as metadata:
        all_lines_init = metadata.readlines()
        all_lines_init = [l.strip() for l in all_lines_init]
        all_lines_init = [l.split("|") for l in all_lines_init]

    ds_data["item_list"] = {s:{} for s in ds_data["speakers_list"]}
    logs = ""
    # Iterate over dataset speakers and set their corresponding item lists
    for speaker in ds_data["speakers_list"]:
        print(f"Loading data for {speaker}")
        # Select lines with the current speaker name
        all_lines = [l for l in all_lines_init if l[0] == speaker]
        
        # Shuffle the lines
        random.seed(params["dataset_random_seed"])
        random.shuffle(all_lines)
        
        # ====================== Train-set item list
        # Compute cumulative sum of durations
        cum_sum_duration = list(np.cumsum([float(l[4]) for l in all_lines]))

        # Find first index where cumsum of duration is larger than total_duration_per_spk
        total_duration_sec = ds_data["total_duration_per_spk"] * 60.0
        try:
            first_idx = list(map(lambda i: i> total_duration_sec, cum_sum_duration)).index(True) 
        except:
            first_idx = len(cum_sum_duration)
        
        # Split item list to train and test lists
        item_list = all_lines[:first_idx]
        train_split_idx = round(float(ds_data["perc_train"]) * len(item_list))
        assert train_split_idx < len(item_list)

        trainset_item_list = item_list[:train_split_idx]
        ds_data["item_list"][speaker]["train"] = trainset_item_list
        
        testset_item_list = item_list[train_split_idx:]
        ds_data["item_list"][speaker]["test"] = testset_item_list
        logs += f"Speaker {speaker}, trainset:{len(trainset_item_list)} utt,"+\
                f"testset:{len(testset_item_list)} utt \n"

    # Collator
    collator = MetaCollator(reduction_factor=params["model"]["n_frames_per_step"], 
                            audio_params=params["audio_params"])
    
    # Dataloader - Train
    dataset = TTSDataset(ds_data, **params)
    dataloader = DataLoader(dataset,
                            collate_fn=collator,
                            batch_size=params["meta_batch_size"],
                            sampler=None,
                            num_workers=params["num_workers"],
                            drop_last=False,
                            pin_memory=True,
                            shuffle=True)

    return dataloader, logs


if __name__ == "__main__":
    from msa_tts.utils.generic import load_params
    params_path = "./experiments/reptile_comvoiceDE/params.yml"
    params = load_params(params_path)   

    train_loader, logs = get_dataloader("metatrain", **params)
    print(logs)

    for itr, batch in enumerate(train_loader):
        print(batch)
        exit()

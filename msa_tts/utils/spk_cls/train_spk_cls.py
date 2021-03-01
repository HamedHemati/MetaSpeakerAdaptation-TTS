from msa_tts.utils.limit_threads import *

import sys
import os
import pickle
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from msa_tts.utils.generic import load_params
from.spk_cls_model import SpkCLSModel


class SpkEmbDataset(Dataset):
    def __init__(self, ds_spk_embs, spk_to_id):
        self.spk_embs = ds_spk_embs
        self.spk_to_id = spk_to_id
        self._load_items()
        
    def _load_items(self):
        self.items = []
        for spk in self.spk_embs.keys():
            for element in self.spk_embs[spk].keys():
                label = self.spk_to_id[spk]
                emb = torch.tensor(self.spk_embs[spk][element]).float()
                self.items.append((emb, label))
                
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        emb, label = self.items[idx]
        return emb, label


def train_spk_cls(params):
    spk_emb_path = params["spk_emb_path"]
    
    "/raid/hhemati/Datasets/Speech/TTS/CommonVoice/de/spk_emb.pkl"
    with open(spk_emb_path, "rb") as pkl_file:
        spk_embs = pickle.load(pkl_file)

    # ========== Target speakers
    target_speakers = params["dataset_train"]["speakers_list"]

    spk_to_id = {spk:itr for (itr, spk) in enumerate(target_speakers)}
    # ========== Dict for target speakers
    ds_spk_embs = {}
    for key in spk_embs:
        if key in target_speakers:
            ds_spk_embs[key] = spk_embs[key]

    # ========== Train and test splits
    perc_train = 0.9
    ds_spk_embs_train = {}
    ds_spk_embs_test = {}

    for speaker in ds_spk_embs:
        spk_elements = list(ds_spk_embs[speaker].keys())
        random.shuffle(spk_elements)
        train_last_idx = int(perc_train * len(spk_elements))
        elements_train = spk_elements[:train_last_idx]
        elements_test = spk_elements[train_last_idx:]
        
        ds_spk_embs_train[speaker] = {}
        for element in elements_train:
            ds_spk_embs_train[speaker].update({element: ds_spk_embs[speaker][element]})
        
        ds_spk_embs_test[speaker] = {}
        for element in elements_test:
            ds_spk_embs_test[speaker].update({element: ds_spk_embs[speaker][element]})
    
    # ========== Dataloaders 
    dataset_train = SpkEmbDataset(ds_spk_embs_train, spk_to_id)
    dataset_test = SpkEmbDataset(ds_spk_embs_test, spk_to_id)

    dataloader_train = DataLoader(dataset_train,batch_size=16, shuffle=True)
    dataloader_test = DataLoader(dataset_test,batch_size=16, shuffle=True)

    # ========== Define model and optimizer
    emb_size = 256
    hidden_size = 128
    num_cls = len(spk_to_id.keys())

    model = SpkCLSModel(emb_size, hidden_size, num_cls)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 20

    # ========== Train
    def train():
        model.train()
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}/{n_epochs}")
            loss_epoch = 0
            for itr, batch in enumerate(dataloader_train):
                inps, labels = batch
                out = model(inps)
                loss  = criterion(out, labels)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            loss_epoch = loss_epoch / len(dataloader_train)
            print(f"Epoch loss: {loss_epoch}")
            test()
            
    def test():
        model.eval()
        correct_total = 0
        items_total = 0
        with torch.no_grad():
            for itr, batch in enumerate(dataloader_test):
                inps, labels = batch
                out = model(inps)
                out_argmx = torch.argmax(out, dim=1)
                correct_total += torch.sum(out_argmx == labels)
                items_total += labels.shape[0]
        accuracy = correct_total / float(items_total)
        print(f"Accuracy: {accuracy}")
        model.train()

    train()

    # ========== Save the model
    os.makedirs(params["out_path"], exist_ok=True)
    checkpoint_path = os.path.join(params["out_path"], "ckpt.pt")
    
    torch.save(model.state_dict(), checkpoint_path)

    spk2id_path = os.path.join(params["out_path"], "spk2id.yml")
    with open(spk2id_path, 'w') as outfile:
        yaml.dump(spk_to_id, outfile)


def main(params):

    r"""Main function that sets and runs the trainer."""

    # DS Params
    ds_params = get_ds_params(params)
    params["dataset_train"] = ds_params


    train_spk_cls(params)


def get_ds_params(params):
    r"""Returns dictionary of audio_params."""
    params = load_params(params["params_path"])
    return params["dataset_train"]


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


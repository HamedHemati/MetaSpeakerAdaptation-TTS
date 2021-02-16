import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from .modules_tacotron2nv.encoder import Encoder
from .modules_tacotron2nv.decoder import Decoder, Postnet
from .modules_tacotron2nv.modules import get_mask_from_lengths


class Tacotron2NV(nn.Module):
    def __init__(self, params):
        super(Tacotron2NV, self).__init__()
        self.params = params
        self.mask_padding = params["mask_padding"]
        self.n_mel_channels = params["n_mel_channels"]
        self.n_frames_per_step = params["n_frames_per_step"]
        
        # ----- Char embedder
        self.embedding = nn.Embedding(params["n_symbols"], params["symbols_embedding_dim"])
        std = sqrt(2.0 / (params["n_symbols"] + params["symbols_embedding_dim"]))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # ----- Encoder
        self.encoder = Encoder(params["encoder_n_convolutions"],
                               params["encoder_embedding_dim"],
                               params["encoder_kernel_size"])
        encoder_embedding_dim = params["encoder_embedding_dim"]

        # ----- Speaker embedding
        if params["speaker_emb_type"] == "learnable_lookup":
            self.speaker_embedder = nn.Embedding(params["num_speakers"], 
                                                 params["speaker_embedding_dim"])
            encoder_embedding_dim += params["speaker_embedding_dim"]
        
        elif params["speaker_emb_type"] == "static":
            print("Using static speaker embeddings.")
            encoder_embedding_dim += params["speaker_embedding_dim"]
        
        elif params["speaker_emb_type"] == "static+linear":
            self.speaker_lin = nn.Linear(params["speaker_embedding_dim"],
                                         params["speaker_embedding_dim_lin"])
            encoder_embedding_dim += params["speaker_embedding_dim_lin"]
        
        else:
            raise NotImplementedError

        # ----- Decoder
        self.decoder = Decoder(params["n_mel_channels"], 
                               params["n_frames_per_step"],
                               encoder_embedding_dim, 
                               params["attention_params"],
                               params["decoder_rnn_dim"],
                               params["attention_rnn_dim"],
                               params["prenet_dim"], 
                               params["max_decoder_steps"],
                               params["gate_threshold"], 
                               params["p_attention_dropout"],
                               params["p_decoder_dropout"],
                               not params["decoder_no_early_stopping"])

        # ----- Postnet
        self.postnet = Postnet(params["n_mel_channels"], 
                               params["postnet_embedding_dim"],
                               params["postnet_kernel_size"],
                               params["postnet_n_convolutions"])

    def parse_output(self, outputs, output_lengths):
        # type: (List[Tensor], Tensor) -> List[Tensor]
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].masked_fill_(mask, 0.0)
            outputs[1].masked_fill_(mask, 0.0)
            outputs[2].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, 
                inputs, 
                input_lengths, 
                melspecs, 
                melspec_lengths,
                speaker_vecs):
        # Char embeddings       
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        # Freeze?
        if self.params["freeze_charemb"]:
            embedded_inputs = embedded_inputs.detach()
            
        # Encoder
        if self.params["use_residual_encoder"]:
            encoder_outputs = self.encoder(embedded_inputs, input_lengths) + \
                              embedded_inputs.transpose(1, 2)
        else:
            encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        # Freeze?
        if self.params["freeze_encoder"]:
            encoder_outputs = encoder_outputs.detach()

        # Speaker embedding
        if self.params["speaker_emb_type"] == "learnable_lookup":
            spk_emb_vec = self.speaker_embedder(speaker_vecs).unsqueeze(1)
        elif self.params["speaker_emb_type"] == "static":
            spk_emb_vec = speaker_vecs.unsqueeze(1)
        elif self.params["speaker_emb_type"] == "static+linear":
            spk_emb_vec = self.speaker_lin(speaker_vecs).unsqueeze(1)
        spk_emb_vec = spk_emb_vec.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat([encoder_outputs, spk_emb_vec], dim=-1)
        
        # Decoder
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, 
                                                             melspecs, 
                                                             input_lengths=input_lengths)
        # Freeze?
        if self.params["freeze_decoder"]:
            mel_outputs = mel_outputs.detach()
            gate_outputs = gate_outputs.detach()
            alignments = alignments.detach()

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
                                 melspec_lengths)


    def infer(self, 
              inputs, 
              input_lengths,
              speaker_vecs):
        # Char embeddings
        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        # Encoder
        if self.params["use_residual_encoder"]:
            encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths) + \
                              embedded_inputs.transpose(1, 2)
        else:
            encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)

        # Speaker embedding
        if self.params["speaker_emb_type"] == "learnable_lookup":
            spk_emb_vec = self.speaker_embedder(speaker_vecs).unsqueeze(1)
        elif self.params["speaker_emb_type"] == "static":
            spk_emb_vec = speaker_vecs.unsqueeze(1)
        elif self.params["speaker_emb_type"] == "static+linear":
            spk_emb_vec = self.speaker_lin(speaker_vecs).unsqueeze(1)
        spk_emb_vec = spk_emb_vec.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat([encoder_outputs, spk_emb_vec], dim=-1)

        # Decoder
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(encoder_outputs, 
                                                                                input_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0,2)

        return mel_outputs_postnet, mel_lengths, alignments

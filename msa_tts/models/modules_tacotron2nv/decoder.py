import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules import LinearNorm, ConvNorm, get_mask_from_lengths
from .lsa import Attention as LSA
from .forward_attn import ForwardAttention


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x


class Decoder(nn.Module):
    def __init__(self, 
                 n_mel_channels, 
                 n_frames_per_step,
                 encoder_embedding_dim, 
                 attention_params,
                 attention_rnn_dim,
                 decoder_rnn_dim,
                 prenet_dim, 
                 max_decoder_steps, 
                 gate_threshold,
                 p_attention_dropout, 
                 p_decoder_dropout,
                 early_stopping):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        # ----- Prenet
        self.prenet = Prenet(n_mel_channels * n_frames_per_step,
                             [prenet_dim, prenet_dim])

        # ----- Attention RNN
        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_embedding_dim,
                                         attention_rnn_dim)

        # ----- Attention
        self.attn_type = attention_params["attention_type"]
        if self.attn_type == "LSA":
            self.attention_layer = LSA(attention_rnn_dim, 
                                       encoder_embedding_dim,
                                       attention_params["attention_dim"], 
                                       attention_params["attention_location_n_filters"],
                                       attention_params["attention_location_kernel_size"])

        elif self.attn_type == "ForwardAttention":
            self.attention_layer = ForwardAttention(query_dim=attention_rnn_dim, 
                                                    embedding_dim=encoder_embedding_dim, 
                                                    attention_dim=attention_params["attention_dim"],
                                                    location_attention=True, 
                                                    attention_location_n_filters=attention_params["attention_location_n_filters"],
                                                    attention_location_kernel_size=attention_params["attention_location_kernel_size"], 
                                                    windowing=attention_params["windowing"], 
                                                    norm=attention_params["norm"], 
                                                    forward_attn=attention_params["forward_attn"],
                                                    trans_agent=attention_params["trans_agent"], 
                                                    forward_attn_mask=attention_params["forward_attn_mask"])
        else:
            raise RuntimeError(f"Attention type {self.attn_type} not defined.")
        
        # ----- Decoder RNN
        self.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_embedding_dim,
                                       decoder_rnn_dim, 
                                       1)
        
        # ----- Linear Projection
        self.linear_projection = LinearNorm(decoder_rnn_dim + encoder_embedding_dim,
                                            n_mel_channels * n_frames_per_step)

        self.gate_layer = LinearNorm(decoder_rnn_dim + encoder_embedding_dim, 
                                     1,
                                     bias=True, 
                                     w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B, self.n_mel_channels*self.n_frames_per_step,
            dtype=dtype, device=device)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(B, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(B, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(B, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(B, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_context = torch.zeros(B, self.encoder_embedding_dim, dtype=dtype, device=device)

        return (attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_context)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, 
               decoder_input, 
               attention_hidden, 
               attention_cell,
               decoder_hidden, 
               decoder_cell, 
               attention_context, 
               encoder_outputs, 
               mask):
        """ Decoder step using stored states, attention and encoder_outputs
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.p_attention_dropout, self.training)

        attention_context, attention_weights = self.attention_layer(attention_hidden, 
                                                                    encoder_outputs, 
                                                                    mask)

        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights, 
                attention_context)

    @torch.jit.ignore
    def forward(self, encoder_outputs, decoder_inputs, input_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        encoder_outputs: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        input_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """       
        decoder_input = self.get_go_frame(encoder_outputs).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(input_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_context) = self.initialize_decoder_states(encoder_outputs)
        
        self.attention_layer.init_states(encoder_outputs)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_context,
                                              encoder_outputs,
                                              mask)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(torch.stack(mel_outputs),
                                                                           torch.stack(gate_outputs),
                                                                           torch.stack(alignments))

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, 
              encoder_outputs, 
              input_lengths):
        """ Decoder inference
        PARAMS
        ------
        encoder_outputs: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(encoder_outputs)

        mask = get_mask_from_lengths(input_lengths)
        
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell, 
         attention_context) = self.initialize_decoder_states(encoder_outputs)
        
        mel_lengths = torch.zeros([encoder_outputs.size(0)], dtype=torch.int32, device=encoder_outputs.device)
        not_finished = torch.ones([encoder_outputs.size(0)], dtype=torch.int32, device=encoder_outputs.device)

        mel_outputs, gate_outputs, alignments = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        first_iter = True
        
        self.attention_layer.init_states(encoder_outputs)

        while True:
            decoder_input = self.prenet(decoder_input)
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_context,
                                              encoder_outputs,
                                              mask)

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = torch.le(torch.sigmoid(gate_output),
                           self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished*dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments, mel_lengths


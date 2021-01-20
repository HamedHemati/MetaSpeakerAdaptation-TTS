import torch
import torch.nn as nn
import numpy as np


# ========== Tacotron2Loss
class Tacotron2Loss():
    def __init__(self, n_frames_per_step, reduction, pos_weight, device):
        self.n_frames_per_step = n_frames_per_step
        self.device = device
        self.reduction = reduction
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction=reduction, 
                                                  pos_weight=torch.tensor(pos_weight))

    def __call__(self,
                 model_output,
                 targets,
                 mel_len):
        
        outputs, postnet_outputs, stop_values, _ = model_output
        mel, stop_labels = targets[0], targets[1]

        mel=mel.transpose(1,2)
        outputs = outputs.transpose(1,2)
        postnet_outputs = postnet_outputs.transpose(1,2)

        # Mel-spec loss
        l1_loss = self.l1_criterion(postnet_outputs, mel) + self.l1_criterion(outputs, mel)
        mse_loss = self.mse_criterion(postnet_outputs, mel) + self.mse_criterion(outputs, mel)
       
        # Stop loss
        bce_loss = self.bce_criterion(stop_values, stop_labels)
        
        if self.reduction == "none":
            # Compute weight masks and apply reduction
            mel_len_ = mel_len.cpu().numpy()
            masks = _pad_mask(mel_len_, self.n_frames_per_step).unsqueeze(-1).to(self.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(mel.size(0) * mel.size(2))
            logit_weights = weights.div(mel.size(0))
            
            # Apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = bce_loss.mul(logit_weights.squeeze(-1)).masked_select(masks.squeeze(-1)).sum()
            
        # Compute total loss
        loss = l1_loss + mse_loss + bce_loss

        return loss


def _pad_mask(mel_lens, r):
    max_len = max(mel_lens)
    remainder = max_len % r
    pad_len = max_len + (r - remainder) if remainder > 0 else max_len
    mask = [np.ones(( mel_lens[i]), dtype=bool) for i in range(len(mel_lens))]
    mask = np.stack([_pad_array(x, pad_len) for x in mask])
    return torch.tensor(mask)


def _pad_array(x, length):
    _pad = 0
    x = np.pad(
        x, [[0, length - x.shape[0]]],
        mode='constant',
        constant_values=False)
    return x
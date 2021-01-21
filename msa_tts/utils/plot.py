import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_attention(attn, path):
    """Plot attention."""
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(str(f'{path}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_spectrogram(M, path, length=None):
    """Plot spectrogram."""
    M = np.flip(M, axis=0)
    if length: 
        M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    fig.savefig(str(f'{path}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_spec_attn_example(M1, M2, attn, path, length_mel=None, length_attn=None):
    """Plot double spectrograms with attention."""
    M1 = np.flip(M1, axis=0)
    M2 = np.flip(M2, axis=0)
    attn = attn.T
    if length_mel: 
        M1 = M1[:, :length_mel]
        M2 = M2[:, :length_mel]

    if length_attn:
        attn = attn[:length_attn, :length_mel]
        
    fig, ax = plt.subplots(3)
    fig.set_figheight(10)
    fig.set_figwidth(12)

    ax[0].imshow(attn, interpolation='nearest', aspect='auto')
    ax[1].imshow(M1, interpolation='nearest', aspect='auto')
    ax[2].imshow(M2, interpolation='nearest', aspect='auto')

    fig.savefig(str(f'{path}.png'), bbox_inches='tight')
    plt.close(fig)


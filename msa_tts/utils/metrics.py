import numpy as np


def mcd(C, C_hat):
    """C and C_hat are NumPy arrays of shape (T, D),
    representing mel-cepstral coefficients.

    """
    K = 10 / np.log(10) * np.sqrt(2)
    mcd = K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))

    return mcd


def mcd_batch(output, mel, mel_len):
    K = 10 / np.log(10) * np.sqrt(2)
    diff = [np.array(mel[i, :mel_len[i], :] - output[i, :mel_len[i], :]) for i in range(output.shape[0])]
    mean_sep = [np.mean(np.sqrt(np.sum(diff[i] ** 2, axis=1))) for i in range(len(diff))]
    mean_all = sum(mean_sep) / len(mean_sep)  
    mcd = K * mean_all
        
    return mcd


import random
import numpy as np
import torch
from scipy import signal

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# def preprocess(x, resample_rate=128, lowcut=1, highcut=50):
#     # リサンプリングとバンドパスフィルタ
#     x = signal.resample(x, resample_rate)
#     x = signal.butter(4, [lowcut, highcut], 'bandpass', fs=resample_rate, output='sos')
    
#     # チャンネル次元を追加 (trial, seq_len) -> (trial, 1, seq_len)
#     x = np.expand_dims(x, axis=1)
    
#     return x

def preprocess(x, resample_rate=128, lowcut=1, highcut=50, baseline_correction=True):
    # リサンプリング
    num_samples = int(len(x) * resample_rate / 1000)  # Assuming original rate is 1000 Hz
    x = signal.resample(x, num_samples)

    # バンドパスフィルタ
    sos = signal.butter(4, [lowcut, highcut], 'bandpass', fs=resample_rate, output='sos')
    x = signal.sosfilt(sos, x)

    # スケーリング（標準化）
    x = (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)

    # ベースライン補正
    if baseline_correction:
        baseline = np.mean(x[:, :resample_rate], axis=-1, keepdims=True)  # 例えば最初の1秒をベースラインとする
        x = x - baseline

    # チャンネル次元を追加 (trial, seq_len) -> (trial, 1, seq_len)
    x = np.expand_dims(x, axis=1)

    return x
import random
import numpy as np
import torch
from scipy import signal

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def preprocess(x, resample_rate=128, lowcut=1, highcut=50):
    # リサンプリングとバンドパスフィルタ
    x = signal.resample(x, resample_rate)
    x = signal.butter(4, [lowcut, highcut], 'bandpass', fs=resample_rate, output='sos')
    
    # チャンネル次元を追加 (trial, seq_len) -> (trial, 1, seq_len)
    x = np.expand_dims(x, axis=1)
    
    return x
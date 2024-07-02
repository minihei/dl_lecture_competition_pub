import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torchvision import transforms

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess_func=None):
        super().__init__()

        self.preprocess_func = preprocess_func
       
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        x, subject_idx = self.X[i], self.subject_idxs[i]
        if self.preprocess_func is not None:
            x = self.preprocess_func(x.numpy())  # TensorからNumpy配列に変換して前処理を適用
            x = torch.tensor(x)  # 再度Tensorに変換
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

        # if hasattr(self, "y"):
        #     return x, self.y[i], subject_idx
        # else:
        #     return x, subject_idx

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
        

# from scipy import signal

# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data", preprocess_func=None):
#         super().__init__()

#         self.preprocess_func = preprocess_func
       
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
        
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         x, subject_idx = self.X[i], self.subject_idxs[i]
#         if self.preprocess_func is not None:
#             x = self.preprocess_func(x)
        
#         if hasattr(self, "y"):
#             return x, self.y[i], subject_idx
#         else:
#             return x, subject_idx
       
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]

# def preprocess(x, resample_rate=128, lowcut=1, highcut=50):
#     # リサンプリングとバンドパスフィルタ
#     x = signal.resample(x, resample_rate)
#     sos = signal.butter(4, [lowcut, highcut], 'bandpass', fs=resample_rate, output='sos')
#     x = signal.sosfilt(sos, x)
    
#     # チャンネル次元を追加 (trial, seq_len) -> (trial, 1, seq_len)
#     x = np.expand_dims(x, axis=0)
    
#     return x
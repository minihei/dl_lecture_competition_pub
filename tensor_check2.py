import torch
import os

# ファイルのパスを指定
file_path = "P:\\DL2024\\data\\train_X.pt"
file_path2 = "P:\\DL2024\\data\\test_subject_idxs.pt"

# ファイルを読み込む
#data_train = torch.load(file_path)
data_index = torch.load(file_path2)

# データの型と形状を表示
#print("データの型:", type(data_train))
#print("データの形状:", data_train.shape)
#print("ユニークデータ(train_subject_idxs.pt):", torch.unique(data_train))
#print("ユニークデータ(train_subject_idxs.pt):", torch.unique(data_train))
#print("データの長さ", len(data_train.shape))


# データの型と形状を表示
print("データの型:", type(data_index))
print("データの形状:", data_index.shape)
print("ユニークデータ(train_subject_idxs.pt):", torch.unique(data_index))
print("データの長さ", len(data_index.shape))

for i in range(4):
    print(f"被験者 {i} のサンプル数:", (data_index == i).sum().item())


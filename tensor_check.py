import torch

# Load the subject index files
train_subject_idxs = torch.load('P:/DL2024/data/train_subject_idxs.pt')
val_subject_idxs = torch.load('P:/DL2024/data/val_subject_idxs.pt')
test_subject_idxs = torch.load('P:/DL2024/data/test_subject_idxs.pt')

# Print the contents to understand the structure
print("Train Subject Indexes:", train_subject_idxs)
print("Validation Subject Indexes:", val_subject_idxs)
print("Test Subject Indexes:", test_subject_idxs)
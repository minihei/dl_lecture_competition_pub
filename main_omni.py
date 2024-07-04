import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets_omni import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
from src.utils import preprocess

@hydra.main(version_base=None, config_path="configs", config_name="config_omni")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDataset(split='train', data_dir=args.data_dir, preprocess_func=preprocess)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    for X, y, subject_idxs in train_loader:
        print(f"Train Input shape: {X.shape}")
        break

    val_set = ThingsMEGDataset(split='val', data_dir=args.data_dir, preprocess_func=preprocess)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    for X, y, subject_idxs in val_loader:
        print(f"Val Input shape: {X.shape}")
        break

    test_set = ThingsMEGDataset(split='test', data_dir=args.data_dir, preprocess_func=preprocess)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    for X, subject_idxs in test_loader:
        print(f"Test Input shape: {X.shape}")
        break

    # ------------------
    #       Model
    # ------------------

    model = BasicConvClassifier(
        train_set.num_classes, 
        train_set.seq_len, 
        train_set.num_channels,
        weight_decay=args.weight_decay
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            #X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            # y_pred = model(X, subject_idxs)  # subject_idxsをモデルに渡す

            loss = F.cross_entropy(y_pred, y)
            regularization_loss = model.regularization_loss()
            total_loss = loss + regularization_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_losses.append(total_loss.item())  # 損失をリストに追加
            train_accs.append(acc.item())  # 精度をリストに追加

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            #X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
                #y_pred = model(X, subject_idxs)  # 検証時もsubject_idxsを使用
            
            val_losses.append(F.cross_entropy(y_pred, y).item())
            val_accs.append(accuracy(y_pred, y).item())

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accs)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accs)

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_losses):.3f} | train acc: {np.mean(train_accs):.3f} | val loss: {np.mean(val_losses):.3f} | val acc: {np.mean(val_accs):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_losses), "train_acc": np.mean(train_accs), "val_loss": np.mean(val_losses), "val_acc": np.mean(val_accs)})
        
        if np.mean(val_accs) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_accs)
            
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()

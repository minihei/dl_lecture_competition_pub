import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import scipy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
from src.utils import preprocess

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": 0}

    train_set = ThingsMEGDataset(split='train', data_dir=args.data_dir, preprocess_func=preprocess)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)

    val_set = ThingsMEGDataset(split='val', data_dir=args.data_dir, preprocess_func=preprocess)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, **loader_args)

    test_set = ThingsMEGDataset(split='test', data_dir=args.data_dir, preprocess_func=preprocess)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, **loader_args)

    # ------------------
    #       Model
    # ------------------

    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels, weight_decay=args.weight_decay
    ).to("cpu")

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
    ).to("cpu")
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to("cpu"), y.to("cpu")

            y_pred = model(X)
            
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
            X, y = X.to("cpu"), y.to("cpu")
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_losses.append(F.cross_entropy(y_pred, y).item())
            val_accs.append(accuracy(y_pred, y).item())

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accs)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accs)

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {epoch_train_loss:.3f} | train acc: {epoch_train_acc:.3f} | val loss: {epoch_val_loss:.3f} | val acc: {epoch_val_acc:.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": epoch_train_loss, "train_acc": epoch_train_acc, "val_loss": epoch_val_loss, "val_acc": epoch_val_acc})
        
        if epoch_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = epoch_val_acc
            
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location="cpu"))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to("cpu")).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()

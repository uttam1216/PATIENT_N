## k-fold EfficientNet-B5 model
# to fine-tunes EfficientNet-B5 (ImageNet pretrained) on each fold's train+val set on binary task of seizure-onset vs non-seizure classification and learn EEG-aware image embeddings
import sys, os, json, time, math, random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from sklearn.metrics import precision_recall_fscore_support
# !{sys.executable} -m pip install timm    # to uncomment in case timm is not installed
import timm

# function for seeding
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_safe_device(prefer_cuda: bool = True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:2")    # to use a particular GPU
    return torch.device("cpu")
    
# fucntion for metrics
def compute_seizure_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)

    p1, r1, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pmi, rmi, fmi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

    return {
        "seizure_precision": float(p1[0]),
        "seizure_recall": float(r1[0]),
        "seizure_f1": float(f1[0]),
        "macro_precision": float(pm),
        "macro_recall": float(rm),
        "macro_f1": float(fm),
        "micro_precision": float(pmi),
        "micro_recall": float(rmi),
        "micro_f1": float(fmi),
    }


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    for xb, yb, _p in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float().view(-1)
        logits = model(xb).view(-1)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(int)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    y_pred = (probs >= threshold).astype(int)
    return compute_seizure_metrics(y_true, y_pred)
    

# defining class for dataset
class ScalogramCSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, png_root: str, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.png_root = png_root
        self.transform = transform
        for c in ["pstrst", "label"]:
            if c not in self.df.columns:
                raise ValueError(f"df must contain '{c}'")

    def __len__(self):
        return len(self.df)

    def _path(self, pstrst: str, label: int) -> str:
        sub = "sz" if int(label) == 1 else "ns"
        return os.path.join(self.png_root, sub, f"{pstrst}.png")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pstrst = row["pstrst"]
        label = int(row["label"])
        path = self._path(pstrst, label)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing PNG: {path}")

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(float(label), dtype=torch.float32), pstrst
        
def build_efficientnet_b5(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    return timm.create_model("efficientnet_b5", pretrained=pretrained, num_classes=num_classes)
    
class EfficientNetB5Embedder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        feats = self.model.forward_features(x)
        pooled = self.model.global_pool(feats)
        if pooled.ndim == 4:
            pooled = pooled.flatten(1)
        return pooled
        
# config 
@dataclass
class EffNetFoldConfig:
    seed: int = 42
    prefer_cuda: bool = True
    img_size: int = 456
    batch_size: int = 64
    num_epochs: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 3
    num_workers: int = 8
    use_weighted_sampler: bool = True
    threshold: float = 0.5
    
# function to fine-tune per fold on train set only and then validate on validation fold
def finetune_efficientnet_b5_fold_train_val(fold_name: str,train_df: pd.DataFrame,val_df: pd.DataFrame,
    png_root: str,out_dir: str,cfg: EffNetFoldConfig) -> Dict:
    seed_everything(cfg.seed)
    device = get_safe_device(cfg.prefer_cuda)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    train_df = train_df[["pstrst", "label"]].copy()
    val_df   = val_df[["pstrst", "label"]].copy()
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)
    train_tf = T.Compose([T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std)])
    eval_tf = T.Compose([T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std)])

    ds_tr = ScalogramCSVDataset(train_df, png_root=png_root, transform=train_tf)
    ds_va = ScalogramCSVDataset(val_df,   png_root=png_root, transform=eval_tf)

    # imbalance handling (pos_weight + optional sampler)
    y_tr = train_df["label"].values.astype(int)
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    pos_weight = (n_neg / max(1, n_pos))

    sampler = None
    if cfg.use_weighted_sampler:
        w_pos = 0.5 / max(1, n_pos)
        w_neg = 0.5 / max(1, n_neg)
        weights = np.where(y_tr == 1, w_pos, w_neg).astype(np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr,batch_size=cfg.batch_size,shuffle=(sampler is None),
        sampler=sampler,num_workers=cfg.num_workers,pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),drop_last=False)
    dl_va = DataLoader(ds_va,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),persistent_workers=(cfg.num_workers > 0))

    model = build_efficientnet_b5(num_classes=1, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))

    meta = asdict(cfg)
    meta.update({"fold": fold_name,"pos_weight": float(pos_weight),"n_train": int(len(train_df)),"n_val": int(len(val_df))})
    with open(outp / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    best_f1, best_epoch, bad, log_rows = -1.0, -1, 0, []

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb, _p in tqdm(dl_tr, desc=f"[{fold_name}] train ep{epoch:02d}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        va_metrics = eval_epoch(model, dl_va, device, threshold=cfg.threshold)
        va_f1 = va_metrics["seizure_f1"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_seizure_f1": float(va_f1),"val_seizure_precision": float(va_metrics["seizure_precision"]),
            "val_seizure_recall": float(va_metrics["seizure_recall"]), "time_sec": float(time.time() - t0)}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(outp / "train_log.csv", index=False)
        if va_f1 > best_f1:
            best_f1 = float(va_f1)
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "meta": meta, "epoch": epoch}, outp / "best_model.pt")
        else:
            bad += 1
        torch.save({"model_state": model.state_dict(), "meta": meta, "epoch": epoch}, outp / "last_model.pt")
        print(f"[{fold_name}] epoch={epoch:02d} train_loss={row['train_loss']:.4f} val_sz_f1={va_f1:.4f} best={best_f1:.4f}")
        if bad >= cfg.patience:
            print(f"[{fold_name}] Early stopping at epoch={epoch} (best={best_f1:.4f} @ {best_epoch})")
            break
    return {
        "fold": fold_name,
        "device": str(device),
        "best_val_sz_f1": best_f1,
        "best_epoch": best_epoch,
        "out_dir": str(outp),
        "best_model_path": str(outp / "best_model.pt"),
    }
    

# function to export embeddings using fold-specific best model
@torch.no_grad()
def export_scalo_embeddings_for_split(df: pd.DataFrame,split_name: str,png_root: str,model_ckpt_path: str,out_dir: str,cfg: EffNetFoldConfig):
    device = get_safe_device(cfg.prefer_cuda)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    model = build_efficientnet_b5(num_classes=1, pretrained=False).to(device)
    ckpt = torch.load(model_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    embedder = EfficientNetB5Embedder(model).to(device)
    embedder.eval()
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)
    tf = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])
    ds = ScalogramCSVDataset(df[["pstrst", "label"]].copy(), png_root=png_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    split_root = outp / split_name
    (split_root / "sz").mkdir(parents=True, exist_ok=True)
    (split_root / "ns").mkdir(parents=True, exist_ok=True)

    for xb, yb, pstrst_list in tqdm(dl, desc=f"Export {split_name}", leave=True):
        xb = xb.to(device, non_blocking=True)
        feats = embedder(xb).detach().cpu().numpy().astype(np.float32)  # [B, 2048]
        yb_np = yb.detach().cpu().numpy().astype(int)

        for i, pstrst in enumerate(pstrst_list):
            sub = "sz" if int(yb_np[i]) == 1 else "ns"
            np.save(split_root / sub / f"{pstrst}.npy", feats[i])
            
# function to run on all folds and fine-tune on TRAIN only, alidate on VAL and export embeddings for train/val/test
def run_effnetb5_iter2_per_fold_train_val_clean(src: str,folds: List[int] = [5],cfg: Optional[EffNetFoldConfig] = None):
    if cfg is None:
        cfg = EffNetFoldConfig()
    srcp = Path(src)
    png_root = str(srcp / "scalograms")
    model_root = srcp / "efficientNetB5"
    emb_root   = srcp / "iter2_emb"
    model_root.mkdir(parents=True, exist_ok=True)
    emb_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for k in folds:
        fold_name = f"fold_{k}"
        fold_dir = srcp / fold_name
        train_df = pd.read_pickle(fold_dir / "train_pat.pkl")
        val_df   = pd.read_pickle(fold_dir / "val_pat.pkl")
        test_df  = pd.read_pickle(fold_dir / "test_pat.pkl")
        out_dir_fold = model_root / fold_name
        emb_dir_fold = emb_root / fold_name / "scalo_emb"
        emb_dir_fold.mkdir(parents=True, exist_ok=True)
        info = finetune_efficientnet_b5_fold_train_val(
            fold_name=fold_name,
            train_df=train_df,
            val_df=val_df,
            png_root=png_root,
            out_dir=str(out_dir_fold),
            cfg=cfg,
        )
        best_ckpt = info["best_model_path"]
        export_scalo_embeddings_for_split(train_df, "train", png_root, best_ckpt, str(emb_dir_fold), cfg)
        export_scalo_embeddings_for_split(val_df,   "val",   png_root, best_ckpt, str(emb_dir_fold), cfg)
        export_scalo_embeddings_for_split(test_df,  "test",  png_root, best_ckpt, str(emb_dir_fold), cfg)
        summary.append(info)
    return summary          

# function to create embeddings for each fold sets
def gen_emb_with_best_effnetb5_iter2_per_fold(src: str, folds: List[int] = [4], cfg: Optional[EffNetFoldConfig] = None):
    """
    for each fold_k:
      - fine-tune EfficientNet-B5 on train+val scalograms
      - export embeddings for train/val/test of that fold
    saves:
      - models/logs: src/efficientNetB5/fold_k/
      - embeddings : src/iter2_emb/fold_k/scalo_emb/{train,val,test}/{sz,ns}/
    """
    if cfg is None:
        cfg = EffNetFoldConfig()
    srcp = Path(src)
    png_root = str(srcp / "scalograms")
    model_root = srcp / "efficientNetB5"
    emb_root   = srcp / "iter2_emb"
    model_root.mkdir(parents=True, exist_ok=True)
    emb_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for k in folds:
        fold_name = f"fold_{k}"
        fold_dir = srcp / fold_name
        train_df = pd.read_pickle(fold_dir / "train_pat.pkl")
        val_df   = pd.read_pickle(fold_dir / "val_pat.pkl")
        test_df  = pd.read_pickle(fold_dir / "test_pat.pkl")
        out_dir_fold = model_root / fold_name
        emb_dir_fold = emb_root / fold_name / "scalo_emb"
        emb_dir_fold.mkdir(parents=True, exist_ok=True)
        
        # 2) Export embeddings for train/val/test using best_model.pt
        best_ckpt = '/media/data/ukumar/iBehave/data_files/feb25/efficientNetB5/' +fold_name+ '/best_model.pt'   #finetune_info["best_model_path"]
        export_scalo_embeddings_for_split(df=train_df,split_name="train",png_root=png_root,model_ckpt_path=best_ckpt,
            out_dir=str(emb_dir_fold),cfg=cfg)
        export_scalo_embeddings_for_split(df=val_df,split_name="val",png_root=png_root,model_ckpt_path=best_ckpt,
            out_dir=str(emb_dir_fold),cfg=cfg)
        export_scalo_embeddings_for_split(df=test_df,split_name="test",png_root=png_root,model_ckpt_path=best_ckpt,
            out_dir=str(emb_dir_fold),cfg=cfg)
            

def main(src):
    for i in range(5):
        fold_num = i+1
        cfg = EffNetFoldConfig(img_size=456, batch_size=32,          # can increase if GPU allows
                               num_epochs=5, lr=2e-4, weight_decay=1e-4, patience=3, num_workers=8,
                               use_weighted_sampler=True, threshold=0.5, prefer_cuda=True)
        # finetune the model
        summary = run_effnetb5_iter2_per_fold_train_val_clean(src=src, folds=[fold_num], cfg=cfg)
        model_root = src + "efficientNetB5/"
        with open(model_root + 'iter2_effnet_summary_fold_'+str(fold_num)+'.json", "w") as f:
             json.dump(summary, f, indent=2)
        # generate the embeddings and save them
        gen_emb_with_best_effnetb5_iter2_per_fold(src=src, folds=[fold_num], cfg=cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    args = parser.parse_args()
    main(args.src)

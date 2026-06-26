# PATIENT+N: Profiling feATure Importance for focal Epileptic seizure oNset detecTion in New Patients
# feature category-based importance profiling
# end-to-end multi-task prediction
# patient-level EMA routing 
# learning multimodal-fused embeddings with modality dropout
# combined loss optimization

import numpy as np
import pandas as pd
import os, json, time, pickle, glob
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt

# fuction to check if a dir is present
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

# function to load a pkl to df
def load_pkl_to_df(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)

# to get beta value to penalizes seizure-onset misclassifications more to deal with class imbalance
def compute_pos_weight_from_train_labels(y: np.ndarray) -> float:
    y = y.astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / max(pos, 1))
    
def window_level_report(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)
    rep = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["ns", "sz"],
        output_dict=True,
        zero_division=0
    )
    rep["accuracy"] = float(accuracy_score(y_true, y_pred))
    return rep
    
def patient_level_metrics_from_preds(
    df_preds: pd.DataFrame,
    agg: str = "max",
    min_windows_per_patient: int = 2,
    thr: float = 0.5
) -> Dict[str, float]:
    g = df_preds.groupby("patient_id")
    rows = []
    for pid, sub in g:
        if len(sub) < min_windows_per_patient:
            continue
        y_true = int(sub["y_true"].max())
        if agg == "max":
            y_prob = float(sub["y_prob"].max())
        elif agg == "mean":
            y_prob = float(sub["y_prob"].mean())
        else:
            raise ValueError("agg must be 'max' or 'mean'")

        rows.append((y_true, y_prob))
    if len(rows) == 0:
        return {
            "n_patients_used": 0,
            "patient_auroc": float("nan"),
            "patient_f1_sz": float("nan"),
            "patient_precision_sz": float("nan"),
            "patient_recall_sz": float("nan"),
        }
    y_true = np.array([r[0] for r in rows], dtype=int)
    y_prob = np.array([r[1] for r in rows], dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    rep = classification_report(y_true, y_pred,labels=[0, 1],target_names=["ns", "sz"],output_dict=True,zero_division=0)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
    return {
        "n_patients_used": int(len(y_true)),
        "patient_auroc": float(auroc),
        "patient_f1_sz": float(rep["sz"]["f1-score"]),
        "patient_precision_sz": float(rep["sz"]["precision"]),
        "patient_recall_sz": float(rep["sz"]["recall"]),
    }
    

# function for category parsing
def feature_category(col: str) -> str:
    s = str(col).lower()
    if "energy" in s:
        return "energy"
    if "kurt" in s:
        return "kurt"
    if s.startswith("corr_") or "corr_" in s:
        return "CORR"
    if s.startswith("plv_") or "plv_" in s:
        return "PLV"
    if s.startswith("hj_") or "hj_" in s:
        return "HJ"
    if s.startswith("sp_") or "sp_" in s:
        return "SP"
    return "other"
    

def build_category_feature_map(df: pd.DataFrame, categories: List[str], exclude_cols: List[str]) -> Dict[str, List[str]]:
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    cat_map = {cat: [] for cat in categories}
    for c in feat_cols:
        cat = feature_category(c)
        if cat in cat_map:
            cat_map[cat].append(c)
    empty = [cat for cat in categories if len(cat_map[cat]) == 0]
    if empty:
        print("[ERROR] Some categories have 0 columns:", empty)
        for cat in empty:
            hits = [c for c in feat_cols if cat.lower() in str(c).lower() or "dwt" in str(c).lower()]
            print(f"  candidates for {cat}: {hits[:12]}")
        raise ValueError(f"Category mapping failed for: {empty}")
    return cat_map
    
# fold-specific normalized dict loader
def load_fold_normalized_cluster_dicts(src: str, fold_num: int):
    path = os.path.join(
        src, "XGBoost_SHAP", f"fold_{fold_num}",
        f"cluster_category_weights_normalized_fold{fold_num}.json"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fold-specific normalized dict file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    cat_dict = {int(k): v for k, v in data["cluster_feat_catg_dict"].items()}
    imp_dict = {int(k): v for k, v in data["cluster_feat_catg_importance_dict"].items()}
    K = int(max(cat_dict.keys()))
    return path, K, cat_dict, imp_dict
    
def build_cluster_category_weight_matrix(
    n_clusters: int,
    categories: List[str],
    cluster_cat_dict: Dict[int, List[str]],
    cluster_imp_dict: Dict[int, List[float]],
) -> np.ndarray:
    C = len(categories)
    cat_to_j = {c: j for j, c in enumerate(categories)}
    W = np.zeros((n_clusters, C), dtype=np.float32)
    for cid in range(1, n_clusters + 1):
        cats = cluster_cat_dict.get(cid, [])
        ws   = cluster_imp_dict.get(cid, [])
        for c, w in zip(cats, ws):
            if c in cat_to_j:
                W[cid - 1, cat_to_j[c]] += float(w)
        s = W[cid - 1].sum()
        if s > 1e-12:
            W[cid - 1] /= s
        else:
            W[cid - 1] = 1.0 / C
    return W
    
# embedding resolver
class ScalogramEmbeddingResolver:
    def __init__(self, src: str, fold_num: int, split: str):
        self.base = os.path.join(src, "iter2_emb", f"fold_{fold_num}", "scalo_emb", split)

    def load(self, key: str, label: int) -> np.ndarray:
        sub = "sz" if int(label) == 1 else "ns"
        p = os.path.join(self.base, sub, f"{key}.npy")
        return np.load(p).astype(np.float32)
        

# dataset + collate
class SeizureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, cat_map: Dict[str, List[str]], emb_resolver: ScalogramEmbeddingResolver):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.cat_map = cat_map
        self.emb = emb_resolver
        self.keys = self.df[cfg.key_col].astype(str).tolist()
        self.pids = self.df[cfg.patient_col].astype(str).tolist()
        self.y = self.df[cfg.label_col].astype(np.int64).to_numpy()
        self.c = self.df[cfg.cluster_col].astype(np.int64).to_numpy()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx: int):
        key = self.keys[idx]
        pid = self.pids[idx]
        y = int(self.y[idx])
        c = int(self.c[idx])
        x_cats = {}
        for cat, cols in self.cat_map.items():
            x_cats[cat] = np.asarray(self.df.loc[idx, cols].values, dtype=np.float32).reshape(-1)
        x_emb = self.emb.load(key, y)
        return x_cats, x_emb, y, c, pid, key
        
        
def collate_fn(batch):
    cats = batch[0][0].keys()
    x_cats = {}
    for cat in cats:
        arr = np.stack([np.asarray(b[0][cat], dtype=np.float32).reshape(-1) for b in batch], axis=0)
        t = torch.from_numpy(arr)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_cats[cat] = t
    x_emb = torch.from_numpy(np.stack([np.asarray(b[1], dtype=np.float32).reshape(-1) for b in batch], axis=0))
    y     = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    c     = torch.tensor([b[3] for b in batch], dtype=torch.long)
    pids  = [b[4] for b in batch]
    keys  = [b[5] for b in batch]
    return x_cats, x_emb, y, c, pids, keys
    
# fucntion to apply patient-level EMA smoothing
class GateEMA:
    def __init__(self, K: int, momentum: float = 0.9):
        self.K = int(K)
        self.momentum = float(momentum)
        self.store: Dict[str, np.ndarray] = {}
    def reset(self):
        self.store.clear()
    @torch.no_grad()
    def update_from_batch(self, pids: List[str], probs: torch.Tensor):
        probs_cpu = probs.detach().float().cpu().numpy()
        for pid, p in zip(pids, probs_cpu):
            if pid in self.store:
                self.store[pid] = self.momentum * self.store[pid] + (1.0 - self.momentum) * p
            else:
                self.store[pid] = p
    def get_probs_for_batch(self, pids: List[str], device: str, fallback: torch.Tensor) -> torch.Tensor:
        fb = fallback.detach().float().cpu().numpy()
        out = []
        for i, pid in enumerate(pids):
            out.append(self.store.get(pid, fb[i]))
        out = torch.tensor(np.stack(out, axis=0), dtype=torch.float32, device=device)
        out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return out
        
        

# MLP model
class CategoryMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)
        
   
# final end-to-end model
class FinalClusterGatedModel(nn.Module):
    def __init__(self,cat_in_dims: Dict[str,int],categories:List[str],emb_dim_in:int,emb_dim_proj:int,tab_hidden:int,
        tab_proj_dim:int,n_clusters:int,W_init:torch.Tensor,dropout:float,cluster_temp:float=0.8,gate_dropout:float=0.05,
        p_drop_scalo: float = 0.20,p_drop_tab: float = 0.10):
        super().__init__()
        self.categories = list(categories)
        self.C = len(self.categories)
        self.K = int(n_clusters)
        self.cluster_temp = float(cluster_temp)
        self.gate_dropout = float(gate_dropout)
        self.p_drop_scalo = float(p_drop_scalo)
        self.p_drop_tab = float(p_drop_tab)
        self.cat_enc = nn.ModuleDict({
            cat: CategoryMLP(cat_in_dims[cat], tab_hidden, tab_proj_dim, dropout) for cat in self.categories })
        self.s_proj = nn.Sequential(nn.LayerNorm(emb_dim_in),nn.Linear(emb_dim_in, emb_dim_proj),nn.GELU(),nn.Dropout(dropout))
        # scalogram + raw tabular summary go into cluster head
        self.cluster_head = nn.Sequential(
            nn.LayerNorm(emb_dim_proj + tab_proj_dim),
            nn.Linear(emb_dim_proj + tab_proj_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.K))
        self.seiz_head = nn.Sequential(
            nn.LayerNorm(emb_dim_proj + tab_proj_dim),
            nn.Linear(emb_dim_proj + tab_proj_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1))
        self.register_buffer("W", W_init.clone().detach().float())
        
    def forward(self, x_cats: Dict[str, torch.Tensor], x_emb: torch.Tensor, pids: List[str], gate_ema: Optional[GateEMA], 
                use_patient_gate: bool = True):
        # category embeddings [B,C,D]
        E_list = []
        for cat in self.categories:
            E_list.append(self.cat_enc[cat](x_cats[cat]))
        E = torch.stack(E_list, dim=1)
        # scalogram embedding
        z_s = self.s_proj(x_emb)
        # raw tabular summary before gating
        z_tab_raw = E.mean(dim=1)
        # cluster head uses BOTH scalogram and raw tabular summary
        z_mm = torch.cat([z_s, z_tab_raw], dim=-1)
        cluster_logits = self.cluster_head(z_mm)
        P_win = F.softmax(cluster_logits / max(self.cluster_temp, 1e-6), dim=-1)
        # patient-level EMA routing
        P_use = P_win
        if use_patient_gate and gate_ema is not None:
            if self.training:
                gate_ema.update_from_batch(pids, P_win)
            P_use = gate_ema.get_probs_for_batch(pids, device=P_win.device, fallback=P_win)
        # cluster-informed gating
        gate = P_use @ self.W
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.training and self.gate_dropout > 0:
            mask = (torch.rand_like(gate) > self.gate_dropout).float()
            gate = gate * mask
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        z_tab = (E * gate.unsqueeze(-1)).sum(dim=1)
        # modality dropout before final fusion
        z_s_used = z_s
        z_tab_used = z_tab
        if self.training:
            if self.p_drop_scalo > 0:
                drop_scalo_mask = (torch.rand((z_s.shape[0], 1), device=z_s.device) > self.p_drop_scalo).float()
                z_s_used = z_s_used * drop_scalo_mask
            if self.p_drop_tab > 0:
                drop_tab_mask = (torch.rand((z_tab.shape[0], 1), device=z_tab.device) > self.p_drop_tab).float()
                z_tab_used = z_tab_used * drop_tab_mask
        z_fuse = torch.cat([z_s_used, z_tab_used], dim=-1)
        logits_main = self.seiz_head(z_fuse).squeeze(-1)
        return logits_main, cluster_logits, P_win, gate
        

# prediction collection
@torch.no_grad()
def predict_collect_final(model: FinalClusterGatedModel,loader: DataLoader,device: str,gate_ema: GateEMA,use_patient_gate: bool = True):
    model.eval()
    all_y,all_p,rows = [],[],[]
    for x_cats, x_emb, y, c, pids, keys in loader:
        x_cats = {k: v.to(device) for k, v in x_cats.items()}
        x_emb  = x_emb.to(device)
        y_t    = y.to(device)
        c_t    = c.to(device)
        logits_main, cluster_logits, P_win, gate = model(
            x_cats, x_emb,
            pids=pids,
            gate_ema=gate_ema,
            use_patient_gate=use_patient_gate,
        )
        prob = torch.sigmoid(logits_main).detach().cpu().numpy()
        y_np = y_t.detach().cpu().numpy().astype(int)
        c_np = c_t.detach().cpu().numpy().astype(int)
        all_y.append(y_np)
        all_p.append(prob)
        for i in range(len(keys)):
            rows.append({
                "pstrst": keys[i],
                "patient_id": pids[i],
                "y_true": int(y_np[i]),
                "y_prob": float(prob[i]),
                "cluster_id": int(c_np[i]),
            })
    y_true = np.concatenate(all_y, axis=0)
    y_prob = np.concatenate(all_p, axis=0)
    df = pd.DataFrame(rows)
    return y_true, y_prob, df
    
 
# config
@dataclass
class FinalConfig:
    src: str = "/media/data/ukumar/iBehave/data_files/feb25/"
    train_pkl: str = "train_pat.pkl"
    val_pkl: str = "val_pat.pkl"
    test_pkl: str = "test_pat.pkl"
    key_col: str = "pstrst"
    patient_col: str = "patient_id"
    label_col: str = "label"
    cluster_col: str = "cluster_id"
    categories: Tuple[str, ...] = ("CORR", "SP", "PLV", "HJ", "energy", "kurt")
    # initializing with best params of previous models on same sets
    emb_dim_in: int = 2048
    emb_dim_proj: int = 128
    tab_hidden: int = 128
    tab_proj_dim: int = 160
    dropout: float = 0.30
    lr: float = 1.1292485225691712e-4
    weight_decay: float = 1.6664253836896376e-6
    epochs: int = 20
    early_stop_patience: int = 5
    batch_size: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    lambda_cluster: float = 0.10
    grad_clip: float = 1.0
    gate_ema_momentum: float = 0.90
    cluster_temp: float = 0.80
    gate_dropout: float = 0.05
    p_drop_scalo: float = 0.20
    p_drop_tab: float = 0.10
    min_windows_per_patient: int = 10
    device: str = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    
# train one fold
def train_one_fold_final(cfg: FinalConfig, run_dir: str, fold_num: int) -> Dict[str, Any]:
    fold_dir = os.path.join(cfg.src, f"fold_{fold_num}")
    train_df = load_pkl_to_df(os.path.join(fold_dir, cfg.train_pkl))
    val_df   = load_pkl_to_df(os.path.join(fold_dir, cfg.val_pkl))
    test_df  = load_pkl_to_df(os.path.join(fold_dir, cfg.test_pkl))
    # fold-specific priors
    dict_path, K_fold, cat_dict, imp_dict = load_fold_normalized_cluster_dicts(cfg.src, fold_num)
    W_np = build_cluster_category_weight_matrix(n_clusters=K_fold,categories=list(cfg.categories),
        cluster_cat_dict=cat_dict,cluster_imp_dict=imp_dict)
    W = torch.tensor(W_np, dtype=torch.float32, device=cfg.device)
    exclude = [cfg.key_col, cfg.patient_col, cfg.label_col, cfg.cluster_col]
    cat_map = build_category_feature_map(train_df, list(cfg.categories), exclude_cols=exclude)
    cat_in_dims = {cat: len(cols) for cat, cols in cat_map.items()}
    emb_train = ScalogramEmbeddingResolver(cfg.src, fold_num, "train")
    emb_val   = ScalogramEmbeddingResolver(cfg.src, fold_num, "val")
    emb_test  = ScalogramEmbeddingResolver(cfg.src, fold_num, "test")
    train_ds = SeizureDataset(train_df, cfg, cat_map, emb_train)
    val_ds   = SeizureDataset(val_df,   cfg, cat_map, emb_val)
    test_ds  = SeizureDataset(test_df,  cfg, cat_map, emb_test)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0))
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0))
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0))
    model = FinalClusterGatedModel(cat_in_dims=cat_in_dims,categories=list(cfg.categories),emb_dim_in=cfg.emb_dim_in,
        emb_dim_proj=cfg.emb_dim_proj,tab_hidden=cfg.tab_hidden,tab_proj_dim=cfg.tab_proj_dim,n_clusters=K_fold,
        W_init=W,dropout=cfg.dropout,cluster_temp=cfg.cluster_temp,gate_dropout=cfg.gate_dropout,
        p_drop_scalo=cfg.p_drop_scalo,p_drop_tab=cfg.p_drop_tab).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_train_np = train_df[cfg.label_col].to_numpy(dtype=np.int64)
    pos_weight_val = compute_pos_weight_from_train_labels(y_train_np)
    pos_weight = torch.tensor([pos_weight_val], device=cfg.device, dtype=torch.float32)
    bce_main = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # L_sz
    ce = nn.CrossEntropyLoss() # L_ca
    fold_out = ensure_dir(os.path.join(run_dir, "folds", f"fold_{fold_num}"))
    ckpt_dir = ensure_dir(os.path.join(fold_out, "checkpoints"))
    met_dir  = ensure_dir(os.path.join(fold_out, "metrics"))
    pred_dir = ensure_dir(os.path.join(fold_out, "preds"))
    with open(os.path.join(met_dir, "fold_info.json"), "w") as f:
        json.dump({"fold": fold_num, "dict_path": dict_path, "K_fold": int(K_fold), "W_shape": list(W.shape), 
                   "pos_weight": float(pos_weight_val), "cat_in_dims": cat_in_dims, "cfg": asdict(cfg),}, f, indent=2)
    gate_ema_train = GateEMA(K=K_fold, momentum=cfg.gate_ema_momentum)
    best_val, best_state, bad_epochs, history = -1e9, None, 0, []
    def c_to_targets(c: torch.Tensor) -> torch.Tensor:
        c_adj = c.clone()
        if c_adj.min().item() >= 1:
            c_adj = c_adj - 1
        return c_adj.clamp(0, K_fold - 1)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x_cats, x_emb, y, c, pids, keys in train_ld:
            x_cats = {k: v.to(cfg.device) for k, v in x_cats.items()}
            x_emb  = x_emb.to(cfg.device)
            y_t    = y.to(cfg.device)
            c_t    = c.to(cfg.device)
            logits_main, cluster_logits, P_win, gate = model(x_cats, x_emb,pids=pids,gate_ema=gate_ema_train,use_patient_gate=True)
            loss_main = bce_main(logits_main, y_t)
            loss_cl   = ce(cluster_logits, c_to_targets(c_t))
            loss = loss_main + cfg.lambda_cluster * loss_cl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * y_t.size(0)
            n += y_t.size(0)
        gate_ema_val = GateEMA(K=K_fold, momentum=cfg.gate_ema_momentum)
        yv_true, yv_prob, _ = predict_collect_final(model, val_ld, cfg.device, gate_ema=gate_ema_val, use_patient_gate=True)
        val_rep = window_level_report(yv_true, yv_prob, thr=0.5)
        val_score = float(val_rep["sz"]["f1-score"])
        history.append({"epoch": epoch,"train_loss": float(running / max(1, n)),"val_accuracy": float(val_rep["accuracy"]),
            "val_sz_f1": float(val_rep["sz"]["f1-score"]),"val_sz_precision": float(val_rep["sz"]["precision"]),
            "val_sz_recall": float(val_rep["sz"]["recall"]),"val_macro_f1": float(val_rep["macro avg"]["f1-score"])})
        if val_score > best_val + 1e-4:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                break
    pd.DataFrame(history).to_csv(os.path.join(met_dir, "training_history.csv"), index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "cat_in_dims": cat_in_dims, "categories": list(cfg.categories), "K_fold": int(K_fold),
            "cfg": asdict(cfg), "fold": fold_num, "pos_weight": float(pos_weight_val), "dict_path": dict_path},
        os.path.join(ckpt_dir, "best_model.pt"))
    gate_ema_val = GateEMA(K=K_fold, momentum=cfg.gate_ema_momentum)
    yv_true, yv_prob, val_preds = predict_collect_final(model, val_ld, cfg.device, gate_ema=gate_ema_val, use_patient_gate=True)
    gate_ema_test = GateEMA(K=K_fold, momentum=cfg.gate_ema_momentum)
    yt_true, yt_prob, test_preds = predict_collect_final(model, test_ld, cfg.device,gate_ema=gate_ema_test,use_patient_gate=True)
    val_preds_path = os.path.join(pred_dir, "val_preds.csv")
    test_preds_path = os.path.join(pred_dir, "test_preds.csv")
    val_preds.to_csv(val_preds_path, index=False)
    test_preds.to_csv(test_preds_path, index=False)
    val_report = window_level_report(yv_true, yv_prob, thr=0.5)
    test_report = window_level_report(yt_true, yt_prob, thr=0.5)
    test_patient_max = patient_level_metrics_from_preds(test_preds, agg="max", min_windows_per_patient=cfg.min_windows_per_patient)
    test_patient_mean = patient_level_metrics_from_preds(test_preds, agg="mean", min_windows_per_patient=cfg.min_windows_per_patient)
    with open(os.path.join(met_dir, "report_val_window.json"), "w") as f:
        json.dump(val_report, f, indent=2)
    with open(os.path.join(met_dir, "report_test_window.json"), "w") as f:
        json.dump(test_report, f, indent=2)
    with open(os.path.join(met_dir, "metrics_test_patient_max.json"), "w") as f:
        json.dump(test_patient_max, f, indent=2)
    with open(os.path.join(met_dir, "metrics_test_patient_mean.json"), "w") as f:
        json.dump(test_patient_mean, f, indent=2)
    paper_row = {"fold": fold_num,"K_fold": int(K_fold),"pos_weight": float(pos_weight_val),
        "test_ns_P": float(test_report["ns"]["precision"]),"test_ns_R": float(test_report["ns"]["recall"]),
        "test_ns_F1": float(test_report["ns"]["f1-score"]),"test_ns_support": int(test_report["ns"]["support"]),
        "test_sz_P": float(test_report["sz"]["precision"]),"test_sz_R": float(test_report["sz"]["recall"]),
        "test_sz_F1": float(test_report["sz"]["f1-score"]),"test_sz_support": int(test_report["sz"]["support"]),
        "test_macro_P": float(test_report["macro avg"]["precision"]),"test_macro_R": float(test_report["macro avg"]["recall"]),
        "test_macro_F1": float(test_report["macro avg"]["f1-score"]),"test_accuracy": float(test_report["accuracy"]),
        "patient_max_F1_sz": float(test_patient_max["patient_f1_sz"]),"patient_max_AUROC": float(test_patient_max["patient_auroc"]),
        "patient_mean_F1_sz": float(test_patient_mean["patient_f1_sz"]),"patient_mean_AUROC": float(test_patient_mean["patient_auroc"])}
    with open(os.path.join(met_dir, "paper_row_test.json"), "w") as f:
        json.dump(paper_row, f, indent=2)
    return {"paper_row": paper_row,
        "val_report_window": val_report,
        "test_report_window": test_report,
        "test_patient_max": test_patient_max,
        "test_patient_mean": test_patient_mean,
        "val_preds_path": val_preds_path,
        "test_preds_path": test_preds_path}
        
# CV runner
def run_cv_final(cfg: FinalConfig, folds=(1,2,3,4,5)):
    run_dir = ensure_dir(os.path.join(cfg.src, "runs_cluster_gated",
        f"final_cluster_gated_noAux_lambda0p1__{time.strftime('%Y%m%d_%H%M%S')}"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    rows = []
    for fold in folds:
        out = train_one_fold_final(cfg, run_dir, fold)
        rows.append(out["paper_row"])
        print(f"[fold {fold}] test_sz_F1={out['paper_row']['test_sz_F1']:.4f}  K={out['paper_row']['K_fold']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "cv_paper_rows.csv"), index=False)
    summary = {}
    for col in ["test_sz_F1","test_sz_P","test_sz_R","test_macro_P","test_macro_R","test_macro_F1","test_accuracy"]:
        summary[col] = {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))}
    with open(os.path.join(run_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nCV TEST (mean±std):")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("\nSaved run_dir:", run_dir)
    return run_dir, df, summary
    
def main(src):
    cfg_final = FinalConfig()
    cfg_final.src = src
    run_dir_final, df_final, summary_final = run_cv_final(cfg_final, folds=(1,2,3,4,5))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    args = parser.parse_args()
    main(args.src)

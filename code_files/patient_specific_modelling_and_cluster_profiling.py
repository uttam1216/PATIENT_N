# k_FOLD patient-specific XGBoost model for creation of cluster profiles
import os, json, math, pickle, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import shap

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# function to check if a directory exists at a given path, ignore if present, else make it
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_pkl(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(df, f)
        
def load_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)
        
# function to parse unique session from pstrst
def parse_session_id_from_pstrst(pstrst: str) -> str:
    """
    pstrst format example:
      patientID__session__fnsz__windowStartTime
    session = the second token (index 1) split by '__'
    """
    parts = str(pstrst).split("__")
    if len(parts) < 2:
        return "unknown_session"
    return parts[1]
    
def plot_2d_scatter(points_2d: np.ndarray, labels: np.ndarray, title: str, out_png: str):
    """
    points_2d: [N,2]
    labels:    [N] ints (cluster ids)
    """
    plt.figure(figsize=(7, 6))
    for cid in sorted(np.unique(labels).tolist()):
        m = labels == cid
        plt.scatter(points_2d[m, 0], points_2d[m, 1], s=12, alpha=0.75, label=f"c{cid}")
    plt.title(title)
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.legend(markerscale=1.5, fontsize=8, frameon=True)
    plt.grid(True, alpha=0.25)
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    
# function to visualize patient vectors clustered by kmeans labels
def patient_level_cluster_visualizations(Xs: np.ndarray,  # standardized patient vectors used for kmeans
    labels_0based: np.ndarray, out_dir: str, fold_name: str, do_tsne: bool = False, seed: int = 42):
    labels_1based = labels_0based + 1

    # PCA 2D
    pca = PCA(n_components=2, random_state=seed)
    pts = pca.fit_transform(Xs)
    plot_2d_scatter(pts, labels_1based,
        title=f"{fold_name} — Patient vectors (PCA2D) colored by cluster",
        out_png=os.path.join(out_dir, "patients_pca2d.png"))

    # Optional TSNE for nicer separation (slower)
    if do_tsne and Xs.shape[0] >= 5:
        perplexity = min(30, max(2, Xs.shape[0] // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=seed)
        pts_t = tsne.fit_transform(Xs)
        plot_2d_scatter(pts_t, labels_1based, title=f"{fold_name} — Patient vectors (t-SNE2D) colored by cluster",
            out_png=os.path.join(out_dir, "patients_tsne2d.png"))
            
# function to convert each record to a compact category vector by summing ABS(feature) per category, then normalize each vector to sum=1.
def record_category_vector(df: pd.DataFrame, feat_cols: list, categories: list) -> np.ndarray:
    # we map feature -> category index
    cat_to_i = {c: i for i, c in enumerate(categories)}
    feat_cat_idx = []
    for f in feat_cols:
        c = feature_category(f)
        feat_cat_idx.append(cat_to_i.get(c, None))
    Xf = df[feat_cols].to_numpy(dtype=np.float32)
    Xabs = np.abs(Xf)
    N = Xabs.shape[0]
    C = len(categories)
    Xc = np.zeros((N, C), dtype=np.float32)
    for j, ci in enumerate(feat_cat_idx):
        if ci is None:
            continue
        Xc[:, ci] += Xabs[:, j]

    s = Xc.sum(axis=1, keepdims=True)
    s[s < 1e-8] = 1.0
    Xc = Xc / s
    return Xc
    
# function to create 2D PCA scatter plots of RECORDS colored by assigned cluster_id for seizure-only records (label=1) & all
def record_level_cluster_visualizations(df: pd.DataFrame, feat_cols: list, categories: list, out_dir: str, fold_name: str,
    max_points: int = 12000, seed: int = 42):
    rng = np.random.RandomState(seed)
    def _plot_subset(sub_df: pd.DataFrame, tag: str):
        if len(sub_df) == 0:
            return
        # subsample for speed/visibility
        if len(sub_df) > max_points:
            sub_df = sub_df.sample(n=max_points, random_state=seed).reset_index(drop=True)
        y_cluster = sub_df[CLUSTER_COL].to_numpy().astype(int)
        Xc = record_category_vector(sub_df, feat_cols, categories)  # [N,C]
        Xs = StandardScaler().fit_transform(Xc)
        pca = PCA(n_components=2, random_state=seed)
        pts = pca.fit_transform(Xs)
        plot_2d_scatter(pts, y_cluster,
            title=f"{fold_name} — Records {tag} (PCA2D on category-aggregated |features|)",
            out_png=os.path.join(out_dir, f"records_{tag}_pca2d.png"))
    _plot_subset(df[df[Y_COL] == 1].copy(), tag="seizure_only")
    _plot_subset(df.copy(), tag="all")
    
    
# to split feature col by '_' and then find if a category string exists
# Feature columns look like:  CORR_FP2-F8_F8-T4, SP_alpha_FP2-F8, energy_.., HJ_..., category is prefix before first '_'.
def feature_category(col: str) -> str:
    return str(col).split("_", 1)[0]
    
# function to detect feature category from column name; 
def feature_category(feat_name: str) -> str:
    f = feat_name.lower()
    # order matters to avoid DWT catching energy/kurt first
    if "corr_" in f:
        return "CORR"
    if "plv_" in f:
        return "PLV"
    if "hj_" in f:
        return "HJ"
    if "sp_" in f:
        return "SP"
    # IMPORTANT: detect energy/kurt BEFORE generic DWT
    if "energy" in f:
        return "energy"
    if "kurt" in f:
        return "kurt"
    if "dwt" in f:
        return "DWT"
    return "other"
    
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]
    
# we split our train set into two Set A (all patients records who have atleast two seizure sessions
# and Set B where all patients in this have only one seizure session.
# next we train our patient specific model only on Split A
# Set A: session-wise holdout per patient by creating a temporary test as a split within this Set A of train set 
# as this temporary test will be evaluted after training on remaining records in train
def split_sessions_per_patient(dfA: pd.DataFrame, test_ratio: float = 0.20, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    for each patient in dfA, split their sessions so that:
      - at least 1 full session goes into test_A
      - remaining sessions go into train_A
      - approximate 4:1 (train:test) at session level
    """
    rng = random.Random(seed)

    dfA = dfA.copy()
    dfA["session_id"] = dfA[KEY_COL].apply(parse_session_id_from_pstrst)

    train_parts = []
    test_parts = []

    for pid, g in dfA.groupby(PAT_COL):
        sess = sorted(g["session_id"].unique().tolist())
        n = len(sess)
        if n < 2:
            # should not happen in A, but just in case
            train_parts.append(g)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        # keep test sessions random but stable
        sess_shuf = sess[:]
        rng.shuffle(sess_shuf)
        test_sess = set(sess_shuf[:n_test])
        train_sess = set(sess_shuf[n_test:])
        if len(train_sess) == 0:
            # force at least one train session
            train_sess = set([sess_shuf[-1]])
            test_sess = set(sess_shuf[:-1])

        test_parts.append(g[g["session_id"].isin(test_sess)])
        train_parts.append(g[g["session_id"].isin(train_sess)])

    train_A = pd.concat(train_parts, axis=0).drop(columns=["session_id"])
    test_A = pd.concat(test_parts, axis=0).drop(columns=["session_id"])

    return train_A, test_A
    
# XGBoost helpers
def compute_scale_pos_weight(y: np.ndarray) -> float:
    y = y.astype(int)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg / max(pos, 1))
    
# class as minimal wrapper around xgboost Booster to provide predict_proba like sklearn
class XGBBoosterWrapper:
    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(X)
        p = self.booster.predict(d)  # probability for class 1
        p = np.asarray(p).reshape(-1)
        # return [P(class0), P(class1)]
        return np.stack([1.0 - p, p], axis=1)
        
def xgb_train_params(scale_pos_weight: float, seed: int = 42) -> dict:
    return dict(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,      # <-- use reg_lambda (safe alias)
        reg_alpha=0.0,
        gamma=0.0,
        seed=seed,
        scale_pos_weight=float(scale_pos_weight),
        nthread=16,
        tree_method="hist",
        device="cpu",
    )
    
def train_xgb_patient_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int = 42,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    force_gpu: bool = True,
) -> XGBBoosterWrapper:

    Xtr = train_df[feature_cols].to_numpy()
    ytr = train_df[Y_COL].to_numpy().astype(int)
    Xva = val_df[feature_cols].to_numpy()
    yva = val_df[Y_COL].to_numpy().astype(int)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    spw = compute_scale_pos_weight(ytr)
    params = xgb_train_params(spw, seed=seed)
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    # Verify device used (see function below)
    verify_xgb_used_cuda(booster)
    return XGBBoosterWrapper(booster)
    
def eval_patient_seizure_f1(model, df: pd.DataFrame, feature_cols: List[str], thr: float = 0.5) -> float:
    y = df[Y_COL].to_numpy().astype(int)
    X = df[feature_cols].to_numpy()
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= thr).astype(int)
    # seizure F1 (positive class=1)
    return float(f1_score(y, pred, pos_label=1, zero_division=0))
    
    
import json
def verify_xgb_used_cuda(booster: xgb.Booster):
    """
    Best-effort check: booster.save_config() includes the 'device' and/or GPU settings.
    Raises if CUDA not detected.
    """
    cfg = json.loads(booster.save_config())
    # Different versions store this differently; we check multiple places.
    cfg_str = json.dumps(cfg).lower()

    used_cuda = ("cuda" in cfg_str) or ("gpu" in cfg_str)
    if not used_cuda:
        # Print small hint for debugging
        print("XGBoost config snippet:", cfg_str[:500])
        raise RuntimeError(
            "CUDA does not appear to be enabled for this booster. "
            "XGBoost may not be built with CUDA or GPU is not visible."
        )
    else:
        None
        #print(" Verified: XGBoost booster config indicates CUDA/GPU usage.")
        
# function to compute mean(|SHAP|) per feature; we use a row cap for speed
def shap_importance_abs_mean(model, df: pd.DataFrame, feature_cols: List[str], max_rows: int = 20000) -> pd.Series:
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_SEED)
    X = df[feature_cols].to_numpy()
    explainer = shap.TreeExplainer(model.booster)
    sv = explainer.shap_values(X)  # [N, D]
    sv = np.asarray(sv)
    imp = np.mean(np.abs(sv), axis=0)
    return pd.Series(imp, index=feature_cols).sort_values(ascending=False)
    
# function to cluster by category-aggregated importances
def patient_category_vector(feat_imp: pd.Series, categories: List[str]) -> np.ndarray:
    """
    we aggregate feature importances to category-level vector of length len(categories).
    then normalize to sum=1.
    """
    cat_vals = {c: 0.0 for c in categories}
    for feat, v in feat_imp.items():
        cat = feature_category(feat)
        if cat in cat_vals:
            cat_vals[cat] += float(v)

    vec = np.array([cat_vals[c] for c in categories], dtype=np.float32)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec
    
# function to compute inertia for k in [2..k_max], plot elbow, pick k by a simple knee heuristic, then enforce min_cluster_size
def choose_k_elbow_with_constraints(X: np.ndarray,k_min: int = 3,k_max: int = 6,min_cluster_size: int = 2,
                                    seed: int = 42,out_png: str = None) -> int:
    ks = list(range(2, k_max + 1))
    inertias, models = [],[]
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        km.fit(X)
        inertias.append(km.inertia_)
        models.append(km)
    # plot elbow
    if out_png is not None:
        plt.figure()
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("k (clusters)")
        plt.ylabel("inertia")
        plt.title("Elbow curve (patient category-importance vectors)")
        plt.grid(True)
        ensure_dir(os.path.dirname(out_png))
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.show()
    # heuristic: choose k where relative improvement drops below 10%
    rel_impr = []
    for i in range(1, len(inertias)):
        prev = inertias[i - 1]
        cur = inertias[i]
        rel_impr.append((prev - cur) / max(prev, 1e-12))
    # map improvement index to k: rel_impr[i-1] corresponds to k=ks[i]
    k_choice = k_max
    for i in range(1, len(ks)):
        k = ks[i]
        if k < k_min:
            continue
        if rel_impr[i - 1] < 0.10:
            k_choice = k
            break
    # enforce constraints: min cluster size >=2, otherwise reduce k
    for k in range(k_choice, k_min - 1, -1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(X)
        sizes = np.bincount(km.labels_, minlength=k)
        if sizes.min() >= min_cluster_size:
            return k
    return k_min
    
# we infer each patient all records of Set B by running a model trained on each of the clusters formed by clustering set A
# then the cluster for which a patient's seizure F1 is the best, we assign all records of that patient to that cluster
# we should remember here that set B of the train has only one seizure and so we excluded it earlier from clustering
# but now we need to have a cluster id for all patients and all records of teh training (set A and set B)
def train_cluster_models(df_train: pd.DataFrame,patient_to_cluster: Dict[str, int],feature_cols: List[str],seed: int = 42) -> Dict[int, xgb.XGBClassifier]:
    """
    Train one XGB model per cluster on all records belonging to patients in that cluster.
    Uses an internal random split for early stopping.
    """
    models = {}
    df_train = df_train.copy()

    # patient->cluster is 1..K
    for cid in sorted(set(patient_to_cluster.values())):
        pids = [p for p, c in patient_to_cluster.items() if c == cid]
        sub = df_train[df_train[PAT_COL].isin(pids)]
        if len(sub) < 200:
            # still train, but note very small cluster
            pass

        # internal split
        sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(sub)
        n_val = max(1, int(0.2 * n))
        val = sub.iloc[:n_val]
        trn = sub.iloc[n_val:]

        model = train_xgb_patient_model(trn, val, feature_cols, seed=seed + cid)
        models[cid] = model

    return models
    
def assign_patients_by_best_cluster_model(
    df_patients: pd.DataFrame,
    cluster_models: Dict[int, xgb.XGBClassifier],
    feature_cols: List[str],
    thr: float = 0.5
) -> Dict[str, int]:
    """
    for each patient in df_patients, evaluate seizure F1 under each cluster model,
    assign cluster with best seizure F1.
    """
    assignments = {}
    for pid, g in df_patients.groupby(PAT_COL):
        best_c = None
        best_f = -1.0
        for cid, model in cluster_models.items():
            f = eval_patient_seizure_f1(model, g, feature_cols, thr=thr)
            if f > best_f:
                best_f = f
                best_c = cid
        if best_c is None:
            best_c = sorted(cluster_models.keys())[0]
        assignments[pid] = int(best_c)
    return assignments
    
# main fold processor
def process_fold_groundtruth_clusters(fold_num:int, src:str = SRC, src_orig:str=SRC_ORIG, thr:float=0.5, k_min:int=3, k_max:int=6):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    assert 1 <= fold_num <= 5, "Min fold num is 1 and max 5, can be changed inside function process_fold_groundtruth_clusters if there are more folds."
    fold_name = f"fold_{fold_num}"
    # ---- load original fold data ----
    in_dir = os.path.join(src_orig, fold_name)
    train_path = os.path.join(in_dir, "train_pat.pkl")
    val_path   = os.path.join(in_dir, "val_pat.pkl")
    test_path  = os.path.join(in_dir, "test_pat.pkl")
    train_df = load_pkl(train_path)
    val_df   = load_pkl(val_path)
    test_df  = load_pkl(test_path)
    # merged for A/B split decisions
    merged_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    # feature columns (numeric)
    feat_cols = get_feature_cols(merged_df)
    # sanity: keep only numeric feature columns
    # (if any non-numeric slipped in)
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(merged_df[c])]
    feat_cols = numeric_cols

    # ---- Step 1-3: Build set A and set B ----
    merged_df = merged_df.copy()
    merged_df["session_id"] = merged_df[KEY_COL].apply(parse_session_id_from_pstrst)

    pat_sessions = merged_df.groupby(PAT_COL)["session_id"].nunique()
    setA = sorted(pat_sessions[pat_sessions >= 2].index.tolist())
    setB = sorted(pat_sessions[pat_sessions < 2].index.tolist())

    setA_df = merged_df[merged_df[PAT_COL].isin(setA)].drop(columns=["session_id"]).reset_index(drop=True)
    setB_df = merged_df[merged_df[PAT_COL].isin(setB)].drop(columns=["session_id"]).reset_index(drop=True)

    print(f"[{fold_name}] Patients: total={merged_df[PAT_COL].nunique()}  setA={len(setA)}  setB={len(setB)}")

    # ---- Step 4: Session-wise split for set A ----
    train_A_df, test_A_df = split_sessions_per_patient(setA_df, test_ratio=0.20, seed=RANDOM_SEED + fold_num)

    # ---- Step 5 + 7: Patient-specific XGB on set A, SHAP importances saved ----
    out_root = ensure_dir(os.path.join(src, "XGBoost_SHAP", fold_name))
    pat_root = ensure_dir(os.path.join(out_root, "pat_id_wise_data"))

    categories = sorted({feature_category(c) for c in feat_cols})
    print(f"[{fold_name}] Detected categories: {categories}")

    patient_feat_imp = {}   # pid -> pd.Series feature_importance
    patient_f1 = {}         # pid -> session-heldout F1

    for pid in setA:
        trn = train_A_df[train_A_df[PAT_COL] == pid]
        tst = test_A_df[test_A_df[PAT_COL] == pid]

        # If a patient ended up with empty test due to edge case, skip to keep stable.
        if len(tst) == 0 or len(trn) == 0:
            continue

        model = train_xgb_patient_model(trn, tst, feat_cols, seed=RANDOM_SEED + fold_num)
        f1 = eval_patient_seizure_f1(model, tst, feat_cols, thr=thr)
        patient_f1[pid] = f1

        imp = shap_importance_abs_mean(model, trn, feat_cols, max_rows=20000)
        patient_feat_imp[pid] = imp

        # save importance csv
        pid_dir = ensure_dir(os.path.join(pat_root, pid))
        imp_df = imp.reset_index()
        imp_df.columns = ["feature", "mean_abs_shap"]
        imp_df.to_csv(os.path.join(pid_dir, f"{pid}_feature_importance.csv"), index=False)

    # save patient model summary
    pat_summary = pd.DataFrame({
        "patient_id": list(patient_f1.keys()),
        "session_holdout_sz_f1": list(patient_f1.values())
    }).sort_values("session_holdout_sz_f1", ascending=False)
    pat_summary.to_csv(os.path.join(out_root, "setA_patient_xgb_summary.csv"), index=False)

    # ---- Step 8: KMeans on category-aggregated importance vectors ----
    # Build matrix: [n_patients, n_categories]
    pids_used = sorted(patient_feat_imp.keys())
    X = np.stack([patient_category_vector(patient_feat_imp[pid], categories) for pid in pids_used], axis=0)

    # scale (optional but often helps KMeans)
    Xs = StandardScaler().fit_transform(X)

    elbow_png = os.path.join(out_root, "elbow_curve_kmeans.png")
    k = choose_k_elbow_with_constraints(
        Xs, k_min=k_min, k_max=k_max, min_cluster_size=2,
        seed=RANDOM_SEED + fold_num,
        out_png=elbow_png
    )
    print(f"[{fold_name}] Chosen k={k}")

    km = KMeans(n_clusters=k, random_state=RANDOM_SEED + fold_num, n_init=20).fit(Xs)
    labels = km.labels_  # 0..k-1
    
    # --- Silhouette score (patient-level) ---
    sil = float("nan")
    try:
        if k >= 2 and Xs.shape[0] > k:
            sil = float(silhouette_score(Xs, labels, metric="euclidean"))
    except Exception as e:
        print(f"[{fold_name}] silhouette_score failed: {e}")
    
    patient_to_cluster_A = {pid: int(lbl + 1) for pid, lbl in zip(pids_used, labels)}  # 1..k
    
    with open(os.path.join(out_root, "patient_to_cluster_setA.json"), "w") as f:
        json.dump(patient_to_cluster_A, f, indent=2)
    
    # save centroid table (interpretable)
    centroids = pd.DataFrame(km.cluster_centers_, columns=categories)
    centroids.insert(0, "cluster_id", np.arange(1, k + 1))
    centroids.to_csv(os.path.join(out_root, "cluster_centroids_category_space.csv"), index=False)
    
    # save clustering meta
    clust_meta = {
        "k": int(k),
        "kmeans_n_init": 20,
        "kmeans_random_state": int(RANDOM_SEED + fold_num),
        "silhouette_patient_vectors": sil,
        "patient_vector_dim": int(Xs.shape[1]),
        "n_patients_used_in_kmeans": int(Xs.shape[0]),
        "standardization": "StandardScaler on category-importance vectors",
    }
    with open(os.path.join(out_root, "clustering_meta.json"), "w") as f:
        json.dump(clust_meta, f, indent=2)

    print(f"[{fold_name}] silhouette (patient vectors) = {sil:.4f}")
    print(f"[{fold_name}] cluster sizes:", pd.Series(labels + 1).value_counts().sort_index().to_dict())
    
    # --- Visualizations: patient-level ---
    viz_dir = ensure_dir(os.path.join(out_root, "cluster_viz"))
    patient_level_cluster_visualizations(
        Xs=Xs,
        labels_0based=labels,
        out_dir=viz_dir,
        fold_name=fold_name,
        do_tsne=False,  # set True if TSNE is needed
        seed=RANDOM_SEED + fold_num
    )
    
    with open(os.path.join(out_root, "patient_to_cluster_setA.json"), "w") as f:
        json.dump(patient_to_cluster_A, f, indent=2)

    # also save cluster centroids in category space (interpretable)
    centroids = pd.DataFrame(km.cluster_centers_, columns=categories)
    centroids.insert(0, "cluster_id", np.arange(1, k + 1))
    centroids.to_csv(os.path.join(out_root, "cluster_centroids_category_space.csv"), index=False)

    # ---- Step 9: Train cluster XGB models on set A (all setA_df records), assign set B patients ----
    cluster_models = train_cluster_models(setA_df, patient_to_cluster_A, feat_cols, seed=RANDOM_SEED + fold_num)

    setB_assign = {}
    if len(setB_df) > 0:
        setB_assign = assign_patients_by_best_cluster_model(setB_df, cluster_models, feat_cols, thr=thr)

    with open(os.path.join(out_root, "patient_to_cluster_setB.json"), "w") as f:
        json.dump(setB_assign, f, indent=2)

    # combined mapping for train+val patients
    patient_to_cluster_trainval = {}
    patient_to_cluster_trainval.update(patient_to_cluster_A)
    patient_to_cluster_trainval.update(setB_assign)

    # ---- Step 10: Update train/val cluster_id and save into src/fold_k/ ----
    def apply_cluster_map(df: pd.DataFrame, mapping: Dict[str, int]) -> pd.DataFrame:
        df = df.copy()
        df[CLUSTER_COL] = df[PAT_COL].map(mapping).astype("float")
        # if any patient wasn't assigned (shouldn't happen), fill with 1
        df[CLUSTER_COL] = df[CLUSTER_COL].fillna(1).astype(int)
        return df

    train_new = apply_cluster_map(train_df, patient_to_cluster_trainval)
    val_new   = apply_cluster_map(val_df, patient_to_cluster_trainval)

    # ---- Step 11: assign cluster_id for test patients too (best cluster model), save ----
    # We re-train cluster models on train+val (A+B) records for best stability, then assign test patients.
    trainval_new = pd.concat([train_new, val_new], axis=0).reset_index(drop=True)

    # Build cluster membership from train+val assignments (patients)
    trainval_pat_to_cluster = {pid: int(cid) for pid, cid in patient_to_cluster_trainval.items()}

    cluster_models_trainval = train_cluster_models(trainval_new, trainval_pat_to_cluster, feat_cols, seed=RANDOM_SEED + 999 + fold_num)
    test_assign = assign_patients_by_best_cluster_model(test_df, cluster_models_trainval, feat_cols, thr=thr)

    with open(os.path.join(out_root, "patient_to_cluster_test.json"), "w") as f:
        json.dump(test_assign, f, indent=2)

    test_new = apply_cluster_map(test_df, test_assign)

    # --- Record-level visualization of separability (colored by cluster_id) ---
    # Use train+val merged for "records look" in training distribution
    trainval_new = pd.concat([train_new, val_new], axis=0).reset_index(drop=True)
    
    record_level_cluster_visualizations(df=trainval_new,feat_cols=feat_cols,categories=categories,
        out_dir=viz_dir,fold_name=f"{fold_name} — Train+Val",max_points=12000,seed=RANDOM_SEED + fold_num)
    
    # Optional: visualize test too (useful to show shift)
    record_level_cluster_visualizations(df=test_new,feat_cols=feat_cols,categories=categories,out_dir=viz_dir,
        fold_name=f"{fold_name} — Test",max_points=12000,seed=RANDOM_SEED + 100 + fold_num)

    # ---- Save revised PKLs into the *main* src fold directory ----
    out_fold_dir = ensure_dir(os.path.join(src, fold_name))
    save_pkl(train_new, os.path.join(out_fold_dir, "train_pat.pkl"))
    save_pkl(val_new,   os.path.join(out_fold_dir, "val_pat.pkl"))
    save_pkl(test_new,  os.path.join(out_fold_dir, "test_pat.pkl"))

    # also log a compact summary
    summary = {"fold": fold_num, "k": k, "n_patients_trainval": int(pd.concat([train_df, val_df])[PAT_COL].nunique()),
        "n_setA": len(setA), "n_setB": len(setB), "n_patients_A_used_in_kmeans": len(pids_used),
        "out_root": out_root, "saved_train_pkl": os.path.join(out_fold_dir, "train_pat.pkl"),
        "saved_val_pkl": os.path.join(out_fold_dir, "val_pat.pkl"), 
        "saved_test_pkl": os.path.join(out_fold_dir, "test_pat.pkl"),
        "silhouette_patient_vectors": sil, 
        "cluster_sizes_setA_kmeans": {str(i): int(c) for i, c in pd.Series(labels+1).value_counts().sort_index().items()}}
    with open(os.path.join(out_root, "fold_cluster_generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{fold_name}] DONE. Revised PKLs saved to: {out_fold_dir}")
    print(f"[{fold_name}] Artifacts in: {out_root}")

    return summary
    
# function to normalize the feature category importances, drop anything below 0.05 of importance
def normalize_cluster_cat_dicts(cluster_feat_catg_dict, cluster_feat_catg_importance_dict, drop_below=0.05, min_keep=2):
    """
    - drops weights < drop_below
    - drops weights <= 0
    - renormalizes to sum=1
    - ensures at least min_keep categories remain (uses top categories)
    returns: (new_cat_dict, new_w_dict)
    """
    new_cat,new_w = {},{}

    for cid in sorted(cluster_feat_catg_dict.keys()):
        cats = list(cluster_feat_catg_dict[cid])
        ws = list(cluster_feat_catg_importance_dict[cid])

        # pair, remove invalid
        pairs = [(c, float(w)) for c, w in zip(cats, ws) if float(w) > 0]
        if len(pairs) == 0:
            # fallback: uniform over original cats
            cats0 = cats[:min_keep] if len(cats) >= min_keep else cats
            if len(cats0) == 0:
                continue
            ws0 = [1.0 / len(cats0)] * len(cats0)
            new_cat[cid] = cats0
            new_w[cid] = ws0
            continue

        # sort desc
        pairs.sort(key=lambda x: x[1], reverse=True)

        # drop small
        kept = [(c, w) for c, w in pairs if w >= drop_below]

        # ensure min_keep
        if len(kept) < min_keep:
            kept = pairs[:min_keep]

        # renormalize
        s = sum(w for _, w in kept)
        if s <= 0:
            # fallback uniform
            kept_c = [c for c, _ in kept]
            kept_w = [1.0 / len(kept_c)] * len(kept_c)
        else:
            kept_c = [c for c, _ in kept]
            kept_w = [w / s for _, w in kept]

        new_cat[cid] = kept_c
        new_w[cid] = kept_w

    return new_cat, new_w
    
def load_fold_centroids_as_dicts(fold: int, src=SRC):
    """
    Reads src/XGBoost_SHAP/fold_k/cluster_centroids_category_space.csv
    Produces raw (possibly negative) dicts then clips negatives to 0 and returns.
    """
    fold_dir = os.path.join(src, f"XGBoost_SHAP/fold_{fold}")
    cent_csv = os.path.join(fold_dir, "cluster_centroids_category_space.csv")
    if not os.path.exists(cent_csv):
        raise FileNotFoundError(f"Missing centroids file: {cent_csv}")

    cent = pd.read_csv(cent_csv)
    cat_cols = [c for c in cent.columns if c != "cluster_id"]

    raw_cat = {}
    raw_w = {}

    for _, row in cent.iterrows():
        cid = int(row["cluster_id"])
        w = row[cat_cols].astype(float)

        # IMPORTANT: these were in standardized space; clip negatives to 0 for gating
        w[w < 0] = 0

        # sort desc
        w = w.sort_values(ascending=False)
        raw_cat[cid] = w.index.tolist()
        raw_w[cid] = w.values.tolist()

    return raw_cat, raw_w, cat_cols
    
def save_cluster_dicts(fold: int, cat_dict: dict, w_dict: dict, src=SRC, tag="normalized"):
    out_dir = ensure_dir(os.path.join(src, f"XGBoost_SHAP/fold_{fold}"))
    out_json = os.path.join(out_dir, f"cluster_category_weights_{tag}_fold{fold}.json")
    payload = {
        "fold": fold,
        "cluster_feat_catg_dict": {int(k): v for k, v in cat_dict.items()},
        "cluster_feat_catg_importance_dict": {int(k): v for k, v in w_dict.items()},
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved:", out_json)
    return out_json
    
def dicts_to_matrix(cat_dict, w_dict, categories_order):
    """
    Returns DataFrame: rows=clusters, cols=categories_order, values=weights (0 if absent)
    """
    clusters = sorted(cat_dict.keys())
    mat = np.zeros((len(clusters), len(categories_order)), dtype=np.float32)
    for i, cid in enumerate(clusters):
        cats = cat_dict[cid]
        ws = w_dict[cid]
        for c, w in zip(cats, ws):
            if c in categories_order:
                j = categories_order.index(c)
                mat[i, j] = float(w)
    df = pd.DataFrame(mat, index=[f"c{c}" for c in clusters], columns=categories_order)
    return df
    
def plot_heatmap(df: pd.DataFrame, title: str, out_png: str):
    plt.figure(figsize=(1.2 + 1.2*df.shape[1], 1.0 + 0.6*df.shape[0]))
    im = plt.imshow(df.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(range(df.shape[0]), df.index)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="category weight (normalized)")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    

    
# function to run all functions 
# it prints the elbow curve, silhouette score of clusters made in each of the five folds and the numbe rof clusters formed i.e. K
def main(SRC):
    # Config
    SRC_ORIG = SRC # if needed, can have another separate copy of all folds where all pkls have no cluster as original files else all pkls will get replaced with ones with cluster_id assignment
    META_COLS = ["patient_id", "pstrst", "label", "cluster_id"]
    PAT_COL = "patient_id"
    KEY_COL = "pstrst"
    Y_COL = "label"
    CLUSTER_COL = "cluster_id"
    RANDOM_SEED = 42
    for i in range(5):
        summary_f1 = process_fold_groundtruth_clusters(fold_num=i+1)
        print(summary_f1)
    ALL_FOLD_MATS, ALL_CATS = [], None
    for fold in [1, 2, 3, 4, 5]:
        raw_cat, raw_w, cat_cols = load_fold_centroids_as_dicts(fold)
        # The category order we want in plots (stable)
        if ALL_CATS is None:
            ALL_CATS = sorted(list(set(cat_cols + ["CORR","SP","PLV","HJ","energy","kurt"])))
            # to prefer canonical order if present
            canonical = ["CORR","SP","PLV","HJ","energy","kurt"]
            ALL_CATS = [c for c in canonical if c in ALL_CATS] + [c for c in ALL_CATS if c not in canonical]

        norm_cat, norm_w = normalize_cluster_cat_dicts(raw_cat, raw_w, drop_below=0.05, min_keep=2)
        save_cluster_dicts(fold, norm_cat, norm_w, tag="normalized")

        df_mat = dicts_to_matrix(norm_cat, norm_w, ALL_CATS)
        ALL_FOLD_MATS.append((fold, df_mat))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    args = parser.parse_args()
    main(args.src)


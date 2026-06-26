import os
import math
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pywt
import pickle

from sklearn.preprocessing import StandardScaler

from scipy.signal import welch
from scipy.stats import kurtosis
from scipy.integrate import simpson
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler

import warnings
from scipy.stats import kurtosis

# all 18 bipolar channels in the strict same order as the saved eeg segments
LST_CHANNELS = [
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ', 'CZ-PZ'
]


# function to create fold folders in the parent folder where train_pat.pkl, val_pat.pkl and test_pat.pkl will be saved
def create_output_dirs(output_root):
    os.makedirs(output_root, exist_ok=True)
    for fold_num in range(1, 6):
        os.makedirs(os.path.join(output_root, f"fold_{fold_num}"), exist_ok=True)
        
# function to measure how close a split is to desired seizure-onset to non-seizure ratio 1:7
def compute_ratio_distance(df, target_pos_frac=1.0 / 8.0):
    if len(df) == 0:
        return float("inf")

    pos_frac = df["label"].mean()
    return abs(pos_frac - target_pos_frac)
    
# function to generate candidate outer folds and sort them by how close test label ratio is to 1:7
def get_ranked_outer_splits(df, n_splits=5, random_state=42, target_pos_frac=1.0 / 8.0):
    X = df.drop(columns=[])
    y = df["label"].values
    groups = df["patient_id"].values

    outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    ranked_splits = []

    for split_id, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        ratio_dist = compute_ratio_distance(test_df, target_pos_frac=target_pos_frac)

        ranked_splits.append({
            "split_id": split_id,
            "trainval_idx": trainval_idx,
            "test_idx": test_idx,
            "test_ratio_dist": ratio_dist,
            "test_pos_frac": test_df["label"].mean() if len(test_df) > 0 else np.nan,
            "test_size": len(test_df),
            "test_patients": test_df["patient_id"].nunique(),
        })

    ranked_splits = sorted(ranked_splits, key=lambda x: (x["test_ratio_dist"], -x["test_size"]))
    return ranked_splits
    
# function to split trainval into train and val while keeping unique patients and roughly preserving class balance
def split_trainval_into_train_val(trainval_df, random_state=42, target_pos_frac=1.0 / 8.0):
    X = trainval_df.drop(columns=[])
    y = trainval_df["label"].values
    groups = trainval_df["patient_id"].values

    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)

    best_obj = None

    for inner_split_id, (train_idx, val_idx) in enumerate(inner_cv.split(X, y, groups), start=1):
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        train_ratio_dist = compute_ratio_distance(train_df, target_pos_frac=target_pos_frac)
        val_ratio_dist = compute_ratio_distance(val_df, target_pos_frac=target_pos_frac)

        # preferring better val ratio first, then better train ratio
        score = val_ratio_dist + 0.5 * train_ratio_dist

        curr_obj = {
            "score": score,
            "train_df": train_df,
            "val_df": val_df,
            "train_ratio_dist": train_ratio_dist,
            "val_ratio_dist": val_ratio_dist,
            "train_pos_frac": train_df["label"].mean() if len(train_df) > 0 else np.nan,
            "val_pos_frac": val_df["label"].mean() if len(val_df) > 0 else np.nan,
        }

        if best_obj is None or curr_obj["score"] < best_obj["score"]:
            best_obj = curr_obj

    return best_obj["train_df"], best_obj["val_df"]


# function to collect all .npy eeg segment files from normalized_eeg_segments/{train|val|test}/{sz|ns}
def collect_all_npy_files(input_root):
    all_rows = []

    for split_name in ["train", "val", "test"]:
        for cls_name in ["sz", "ns"]:
            curr_dir = os.path.join(input_root, split_name, cls_name)
            if not os.path.exists(curr_dir):
                continue

            for fname in os.listdir(curr_dir):
                if not fname.endswith(".npy"):
                    continue

                pstrst = fname.replace(".npy", "")
                patient_id = pstrst.split("__")[0]
                label = 1 if cls_name == "sz" else 0

                all_rows.append({
                    "pstrst": pstrst,
                    "patient_id": patient_id,
                    "label": label,
                    "orig_split": split_name,
                    "orig_class": cls_name,
                    "file_path": os.path.join(curr_dir, fname),
                })

    df = pd.DataFrame(all_rows)
    return df


# function to load one npy eeg segment and check if it has expected shape
def load_npy_eeg(npy_path, expected_channels=18, expected_samples=2048):
    eeg = np.load(npy_path)

    if eeg.ndim != 2:
        raise ValueError(f"EEG array is not 2D for file: {npy_path}")

    if eeg.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {eeg.shape[0]} for file: {npy_path}")

    if eeg.shape[1] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {eeg.shape[1]} for file: {npy_path}")

    return eeg.astype(np.float32)
    
# function to safely compute Pearson correlation
# if either signal has near-zero variance, return 0.0 instead of nan
def safe_corrcoef(signal1, signal2, eps=1e-12):
    signal1 = np.asarray(signal1, dtype=np.float64)
    signal2 = np.asarray(signal2, dtype=np.float64)

    if signal1.ndim != 1 or signal2.ndim != 1 or len(signal1) != len(signal2):
        return 0.0

    std1 = np.std(signal1)
    std2 = np.std(signal2)

    if (not np.isfinite(std1)) or (not np.isfinite(std2)):
        return 0.0

    if std1 < eps or std2 < eps:
        return 0.0

    c = np.corrcoef(signal1, signal2)[0, 1]

    if not np.isfinite(c):
        return 0.0

    return float(c)


# function to safely compute kurtosis
# if coeffs are too short or nearly constant, return 0.0 instead of nan
def safe_kurtosis(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        x = x.ravel()

    # scipy kurtosis with bias=False needs enough points to behave well
    if len(x) < 4:
        return 0.0

    if not np.all(np.isfinite(x)):
        x = x[np.isfinite(x)]

    if len(x) < 4:
        return 0.0

    if np.std(x) < eps:
        return 0.0

    try:
        k = kurtosis(x, fisher=True, bias=False, nan_policy='omit')
        if not np.isfinite(k):
            return 0.0
        return float(k)
    except Exception:
        return 0.0


# simple hilbert transform implementation so that PLV can be computed without extra dependency
def hilbert_transform(x):
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.fft.ifft(X * h)


# function to calculate phase locking value between two signals
def calculate_plv(signal1, signal2):
    analytic_signal1 = hilbert_transform(signal1)
    analytic_signal2 = hilbert_transform(signal2)

    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return float(plv)


# function to calculate Hjorth complexity from one signal
def compute_hjorth_complexity(signal):
    activity = np.var(signal)
    if activity <= 1e-12:
        return 0.0

    diff1 = np.diff(signal)
    if np.var(diff1) <= 1e-12:
        return 0.0

    diff2 = np.diff(diff1)

    mobility = np.sqrt(np.var(diff1) / activity)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / (mobility + 1e-12)

    return float(complexity)


# function to get wavelet coefficients mapped approximately to EEG bands
def get_band_coeffs_from_dwt(signal, wavelet='db4', level=5, fs=256):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # approximate mapping for fs=256
    # A5 ~ 0-4 Hz
    # D5 ~ 4-8 Hz
    # D4 ~ 8-16 Hz
    # D3 ~ 16-32 Hz
    # D2 ~ 32-64 Hz
    band_coeffs = {
        "delta": coeffs[0],   # A5
        "theta": coeffs[1],   # D5
        "alpha": coeffs[2],   # D4
        "beta": coeffs[3],    # D3
        "gamma": coeffs[4],   # D2
    }

    return band_coeffs


# function to extract all features from one eeg segment and return them as a dict
def extract_features_from_one_segment(eeg, lst_channels, fs=256):
    feat = {}

    channel_indices = {ch: idx for idx, ch in enumerate(lst_channels)}

    # adding CORR and PLV features between all unique channel pairs
    channel_pairs = list(combinations(lst_channels, 2))
    for ch1, ch2 in channel_pairs:
        idx1 = channel_indices[ch1]
        idx2 = channel_indices[ch2]

        s1 = eeg[idx1]
        s2 = eeg[idx2]

        # safe Pearson correlation coefficient
        c = safe_corrcoef(s1, s2)
        feat[f"CORR_{ch1}_{ch2}"] = round(float(c), 6)

        # PLV
        try:
            plv = calculate_plv(s1, s2)
            if not np.isfinite(plv):
                plv = 0.0
        except Exception:
            plv = 0.0

        feat[f"PLV_{ch1}_{ch2}"] = round(float(plv), 6)

    # adding spectral power, Hjorth complexity and DWT log energy per whole 8 sec
    freq_bands = {
        'delta': (0.1, 4),
        'theta': (4, 8),
        'alpha': (8, 16),
        'beta': (16, 32),
        'gamma': (32, 64)
    }

    for ch_idx, ch_name in enumerate(lst_channels):
        signal = np.asarray(eeg[ch_idx], dtype=np.float64)

        # Hjorth complexity
        try:
            hj = compute_hjorth_complexity(signal)
            if not np.isfinite(hj):
                hj = 0.0
        except Exception:
            hj = 0.0
        feat[f"HJ_Comp_{ch_name}"] = round(float(hj), 6)

        # spectral power features
        try:
            freqs, psd = welch(signal, fs=fs, nperseg=min(512, len(signal)))
            dfreq = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

            for band, (low, high) in freq_bands.items():
                idx_band = (freqs >= low) & (freqs <= high)

                if np.any(idx_band):
                    band_power = simpson(psd[idx_band], dx=dfreq)
                else:
                    band_power = 0.0

                val = np.log(band_power + 1e-12)
                if not np.isfinite(val):
                    val = 0.0

                feat[f"SP_{band}_{ch_name}"] = round(float(val), 6)

        except Exception:
            for band in freq_bands:
                feat[f"SP_{band}_{ch_name}"] = 0.0

        # DWT whole 8 sec log energy features
        try:
            band_coeffs = get_band_coeffs_from_dwt(signal, wavelet='db4', level=5, fs=fs)
            for band, coeff in band_coeffs.items():
                energy = np.sum(np.square(coeff))
                val = np.log(energy + 1e-12)
                if not np.isfinite(val):
                    val = 0.0
                feat[f"DWT_{band}_log_energy_{ch_name}"] = round(float(val), 6)
        except Exception:
            for band in freq_bands:
                feat[f"DWT_{band}_log_energy_{ch_name}"] = 0.0

    # adding DWT kurtosis per 1 second for all 8 seconds
    samples_per_sec = fs
    total_secs = eeg.shape[1] // samples_per_sec

    for ch_idx, ch_name in enumerate(lst_channels):
        signal = np.asarray(eeg[ch_idx], dtype=np.float64)

        for sec_idx in range(total_secs):
            seg = signal[sec_idx * samples_per_sec:(sec_idx + 1) * samples_per_sec]

            try:
                band_coeffs_1s = get_band_coeffs_from_dwt(seg, wavelet='db4', level=5, fs=fs)
                for band, coeff in band_coeffs_1s.items():
                    k = safe_kurtosis(coeff)
                    feat[f"DWT_{band}_kurt_{ch_name}_{sec_idx + 1}"] = round(float(k), 6)

            except Exception:
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    feat[f"DWT_{band}_kurt_{ch_name}_{sec_idx + 1}"] = 0.0

    return feat


# function to build base dataframe and then add all feature categories into it from normalized npy eeg segments
def build_full_feature_dataframe(base_df, fs=256):
    results = []

    total = len(base_df)
    for ctr, row in enumerate(base_df.itertuples(index=False), start=1):
        if ctr % 200 == 0 or ctr == total:
            print(f"processing eeg segments: {ctr}/{total}")

        try:
            eeg = load_npy_eeg(row.file_path, expected_channels=18, expected_samples=2048)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Precision loss occurred in moment calculation due to catastrophic cancellation.*",
                    category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in divide",
                    category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in divide",
                    category=RuntimeWarning
                )

                feat_dict = extract_features_from_one_segment(eeg, lst_channels=LST_CHANNELS, fs=fs)

        except Exception as ex:
            print(f"skipping {row.file_path} due to error: {ex}")
            feat_dict = {}

        row_dict = {
            "pstrst": row.pstrst,
            "patient_id": row.patient_id,
            "label": row.label,
        }
        row_dict.update(feat_dict)
        results.append(row_dict)

    feat_df = pd.DataFrame(results)
    return feat_df
    
# function to clean feature dataframe by handling NaNs and inf values
# strategy:
# 1) we replace inf with nan
# 2) we fill nan using column median
# 3) we drop rows if still nan remains
def clean_feature_dataframe(df, meta_cols):
    df = df.copy()

    feature_cols = [c for c in df.columns if c not in meta_cols]

    print("\nCleaning feature dataframe...")

    # Step 1: replace inf with nan
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Step 2: count NaNs before filling
    nan_before = df[feature_cols].isna().sum().sum()
    print(f"total NaN values before filling: {nan_before}")

    # Step 3: fill NaN using median (robust choice)
    medians = df[feature_cols].median()
    medians = medians.fillna(0.0)

    df[feature_cols] = df[feature_cols].fillna(medians)

    # Step 4: check if any NaN still remains
    nan_after_fill = df[feature_cols].isna().sum().sum()
    print(f"total NaN values after filling: {nan_after_fill}")

    # Step 5: drop rows ONLY if still NaN exists
    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)

    print(f"rows before cleaning: {before_rows}")
    print(f"rows after cleaning: {after_rows}")
    print(f"rows dropped: {before_rows - after_rows}")

    return df


# function to remove highly correlated features using Spearman correlation threshold
def spearman_correlation_filter(df, meta_cols, threshold=0.97):
    work_df = df.copy()

    feature_cols = [c for c in work_df.columns if c not in meta_cols]
    feature_df = work_df[feature_cols].copy()

    # replacing inf values if any
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # filling nan values column-wise with median so correlation matrix can be computed
    for col in feature_df.columns:
        med = feature_df[col].median()
        if pd.isna(med):
            med = 0.0
        feature_df[col] = feature_df[col].fillna(med)

    print("computing Spearman correlation matrix for feature filtering...")
    corr_matrix = feature_df.corr(method="spearman").abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]

    filtered_df = pd.concat([work_df[meta_cols].copy(), feature_df.drop(columns=cols_to_drop)], axis=1)

    return filtered_df, cols_to_drop
    
# function to group feature columns by their feature category prefix
def get_feature_category_columns(feature_cols):
    category_cols = {
        "CORR": [],
        "PLV": [],
        "SP": [],
        "HJ": [],
        "DWT_ENERGY": [],
        "DWT_KURT": [],
        "OTHER": [],
    }

    for col in feature_cols:
        if col.startswith("CORR_"):
            category_cols["CORR"].append(col)
        elif col.startswith("PLV_"):
            category_cols["PLV"].append(col)
        elif col.startswith("SP_"):
            category_cols["SP"].append(col)
        elif col.startswith("HJ_"):
            category_cols["HJ"].append(col)
        elif col.startswith("DWT_") and "_log_energy_" in col:
            category_cols["DWT_ENERGY"].append(col)
        elif col.startswith("DWT_") and "_kurt_" in col:
            category_cols["DWT_KURT"].append(col)
        else:
            category_cols["OTHER"].append(col)

    return category_cols
    
# function to scale numeric feature columns category-wise using only train set statistics
# metadata columns are kept unchanged
# each feature category gets its own scaler so that one modality does not dominate another
def scale_fold_datasets_by_category(train_df, val_df, test_df, meta_cols, fold_dir):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    # keeping same column order
    train_df = train_df[meta_cols + feature_cols].copy()
    val_df = val_df[meta_cols + feature_cols].copy()
    test_df = test_df[meta_cols + feature_cols].copy()

    # replacing inf with nan
    for df_ in [train_df, val_df, test_df]:
        df_[feature_cols] = df_[feature_cols].replace([np.inf, -np.inf], np.nan)

    # filling missing values using train medians only
    train_medians = train_df[feature_cols].median()
    train_medians = train_medians.fillna(0.0)

    train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)
    val_df[feature_cols] = val_df[feature_cols].fillna(train_medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_medians)

    category_cols = get_feature_category_columns(feature_cols)

    scaler_store = {}
    train_scaled_df = train_df[meta_cols].copy()
    val_scaled_df = val_df[meta_cols].copy()
    test_scaled_df = test_df[meta_cols].copy()

    for cat_name, cols in category_cols.items():
        if len(cols) == 0:
            continue

        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(train_df[cols].values)
        val_scaled = scaler.transform(val_df[cols].values)
        test_scaled = scaler.transform(test_df[cols].values)

        train_scaled_df = pd.concat(
            [train_scaled_df, pd.DataFrame(train_scaled, columns=cols, index=train_df.index)],
            axis=1
        )
        val_scaled_df = pd.concat(
            [val_scaled_df, pd.DataFrame(val_scaled, columns=cols, index=val_df.index)],
            axis=1
        )
        test_scaled_df = pd.concat(
            [test_scaled_df, pd.DataFrame(test_scaled, columns=cols, index=test_df.index)],
            axis=1
        )

        scaler_store[cat_name] = {
            "scaler": scaler,
            "columns": cols,
        }

    scaler_obj = {
        "meta_cols": meta_cols,
        "feature_cols": feature_cols,
        "category_scalers": scaler_store,
        "train_medians": train_medians.to_dict(),
    }

    with open(os.path.join(fold_dir, "feature_category_scalers.pkl"), "wb") as f:
        pickle.dump(scaler_obj, f)

    # restoring original column order: metadata first, then all feature cols
    train_scaled_df = train_scaled_df[meta_cols + feature_cols].copy()
    val_scaled_df = val_scaled_df[meta_cols + feature_cols].copy()
    test_scaled_df = test_scaled_df[meta_cols + feature_cols].copy()

    return train_scaled_df, val_scaled_df, test_scaled_df


# function to scale numeric feature columns fold-wise using only train set statistics
# metadata columns are kept unchanged and only feature columns are standardized
def scale_fold_datasets(train_df, val_df, test_df, meta_cols, fold_dir):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    # making sure column order is same across all three sets
    val_df = val_df[meta_cols + feature_cols].copy()
    test_df = test_df[meta_cols + feature_cols].copy()
    train_df = train_df[meta_cols + feature_cols].copy()

    # handling inf/nan before scaling
    for df_ in [train_df, val_df, test_df]:
        df_[feature_cols] = df_[feature_cols].replace([np.inf, -np.inf], np.nan)

    # fill using train medians only so that no information leaks from val/test
    train_medians = train_df[feature_cols].median()
    train_medians = train_medians.fillna(0.0)

    train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)
    val_df[feature_cols] = val_df[feature_cols].fillna(train_medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_medians)

    scaler = RobustScaler()

    train_scaled = scaler.fit_transform(train_df[feature_cols].values)
    val_scaled = scaler.transform(val_df[feature_cols].values)
    test_scaled = scaler.transform(test_df[feature_cols].values)

    train_scaled_df = pd.concat(
        [
            train_df[meta_cols].reset_index(drop=True),
            pd.DataFrame(train_scaled, columns=feature_cols)
        ],
        axis=1
    )

    val_scaled_df = pd.concat(
        [
            val_df[meta_cols].reset_index(drop=True),
            pd.DataFrame(val_scaled, columns=feature_cols)
        ],
        axis=1
    )

    test_scaled_df = pd.concat(
        [
            test_df[meta_cols].reset_index(drop=True),
            pd.DataFrame(test_scaled, columns=feature_cols)
        ],
        axis=1
    )

    # saving scaler object for this fold
    scaler_obj = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "meta_cols": meta_cols,
        "train_medians": train_medians.to_dict()
    }

    with open(os.path.join(fold_dir, "feature_category_scalers.pkl"), "wb") as f:
        pickle.dump(scaler_obj, f)

    return train_scaled_df, val_scaled_df, test_scaled_df
    
# function to summarize one test split with support counts and ratio information
def get_test_split_stats(test_df, target_pos_frac=1.0 / 8.0):
    pos_count = int((test_df["label"] == 1).sum())
    neg_count = int((test_df["label"] == 0).sum())
    total_count = len(test_df)

    if total_count == 0:
        pos_frac = 0.0
    else:
        pos_frac = pos_count / total_count

    ratio_dist = abs(pos_frac - target_pos_frac)

    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "total_count": total_count,
        "pos_frac": pos_frac,
        "ratio_dist": ratio_dist,
    }
    
# function to score a complete 5-fold outer split run
# lower score means:
# 1) test ratios are closer to 1:7
# 2) seizure support in test is more even across folds
# 3) non-seizure support in test is more even across folds
# 4) total test size is more even across folds
def score_outer_cv_run(test_stats_list, w_ratio=1.0, w_pos=1.0, w_neg=0.5, w_total=0.5):
    pos_counts = np.array([x["pos_count"] for x in test_stats_list], dtype=float)
    neg_counts = np.array([x["neg_count"] for x in test_stats_list], dtype=float)
    total_counts = np.array([x["total_count"] for x in test_stats_list], dtype=float)
    ratio_dists = np.array([x["ratio_dist"] for x in test_stats_list], dtype=float)

    # coefficient of variation style normalization so scales remain comparable
    pos_cv = np.std(pos_counts) / (np.mean(pos_counts) + 1e-12)
    neg_cv = np.std(neg_counts) / (np.mean(neg_counts) + 1e-12)
    total_cv = np.std(total_counts) / (np.mean(total_counts) + 1e-12)
    mean_ratio_dist = np.mean(ratio_dists)

    score = (
        w_ratio * mean_ratio_dist +
        w_pos * pos_cv +
        w_neg * neg_cv +
        w_total * total_cv
    )

    return {
        "score": float(score),
        "mean_ratio_dist": float(mean_ratio_dist),
        "pos_cv": float(pos_cv),
        "neg_cv": float(neg_cv),
        "total_cv": float(total_cv),
    }
    
# function to search over multiple grouped-stratified 5-fold runs and choose the one
# whose test folds jointly have:
# - ratio closest to 1:7
# - seizure support as even as possible
# - non-seizure support as even as possible
# - total support as even as possible
def select_best_outer_cv_run(
    df,
    n_splits=5,
    base_random_state=42,
    n_trials=50,
    target_pos_frac=1.0 / 8.0
):
    X = df.drop(columns=[])
    y = df["label"].values
    groups = df["patient_id"].values

    best_run = None

    for trial_idx in range(n_trials):
        curr_seed = base_random_state + trial_idx

        outer_cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=curr_seed
        )

        splits = []
        test_stats_list = []

        for split_id, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
            trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            stats = get_test_split_stats(test_df, target_pos_frac=target_pos_frac)

            splits.append({
                "split_id": split_id,
                "trainval_idx": trainval_idx,
                "test_idx": test_idx,
                "test_df": test_df,
                "trainval_df": trainval_df,
                "stats": stats,
            })
            test_stats_list.append(stats)

        run_score = score_outer_cv_run(test_stats_list)

        curr_run = {
            "seed": curr_seed,
            "splits": splits,
            "test_stats_list": test_stats_list,
            "run_score": run_score,
        }

        if best_run is None or curr_run["run_score"]["score"] < best_run["run_score"]["score"]:
            best_run = curr_run

    return best_run
    

# function to print summary of the selected outer 5-fold run
def print_outer_cv_run_summary(best_run):
    print("\nSelected outer CV run summary")
    print(f"chosen random seed: {best_run['seed']}")
    print(f"run score: {best_run['run_score']['score']:.6f}")
    print(f"mean test ratio distance from 1:7: {best_run['run_score']['mean_ratio_dist']:.6f}")
    print(f"test seizure support CV: {best_run['run_score']['pos_cv']:.6f}")
    print(f"test non-seizure support CV: {best_run['run_score']['neg_cv']:.6f}")
    print(f"test total support CV: {best_run['run_score']['total_cv']:.6f}")

    for fold_num, split_obj in enumerate(best_run["splits"], start=1):
        stats = split_obj["stats"]
        print(
            f"fold_{fold_num}: "
            f"test_sz={stats['pos_count']}, "
            f"test_ns={stats['neg_count']}, "
            f"test_total={stats['total_count']}, "
            f"test_pos_frac={stats['pos_frac']:.4f}"
        )
    

# function to create 5 folds with unique patients across train/val/test
# while trying to keep:
# 1) test sz:ns ratio as close to 1:7 as possible
# 2) test seizure support as even as possible across folds
# 3) test non-seizure support as even as possible across folds
# 4) test total support as even as possible across folds
# and then scale numeric features using only train fold statistics
def make_group_stratified_5fold_splits(df, output_root, random_state=42, outer_search_trials=50):
    target_pos_frac = 1.0 / 8.0   # sz:ns = 1:7 means positive fraction should be 1/8
    meta_cols = ["pstrst", "patient_id", "label"]

    best_outer_run = select_best_outer_cv_run(
        df=df,
        n_splits=5,
        base_random_state=random_state,
        n_trials=outer_search_trials,
        target_pos_frac=target_pos_frac
    )

    print_outer_cv_run_summary(best_outer_run)

    for fold_num, split_obj in enumerate(best_outer_run["splits"], start=1):
        fold_dir = os.path.join(output_root, f"fold_{fold_num}")

        trainval_idx = split_obj["trainval_idx"]
        test_idx = split_obj["test_idx"]

        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # split trainval further into train and val
        train_df, val_df = split_trainval_into_train_val(
            trainval_df=trainval_df,
            random_state=random_state + fold_num,
            target_pos_frac=target_pos_frac
        )

        # sanity check that patient overlap is absent
        tr_pats = set(train_df["patient_id"].unique())
        va_pats = set(val_df["patient_id"].unique())
        te_pats = set(test_df["patient_id"].unique())

        assert len(tr_pats.intersection(va_pats)) == 0, f"patient overlap found between train and val in fold {fold_num}"
        assert len(tr_pats.intersection(te_pats)) == 0, f"patient overlap found between train and test in fold {fold_num}"
        assert len(va_pats.intersection(te_pats)) == 0, f"patient overlap found between val and test in fold {fold_num}"

        # saving unscaled pickle files
        train_df.to_pickle(os.path.join(fold_dir, "train_pat.pkl"))
        val_df.to_pickle(os.path.join(fold_dir, "val_pat.pkl"))
        test_df.to_pickle(os.path.join(fold_dir, "test_pat.pkl"))

        # optional csv copies for checking
        train_df.to_csv(os.path.join(fold_dir, "train_pat.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val_pat.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test_pat.csv"), index=False)

        train_scaled_df, val_scaled_df, test_scaled_df = scale_fold_datasets_by_category(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        meta_cols=meta_cols,
        fold_dir=fold_dir)

        # saving scaled versions too
        train_scaled_df.to_pickle(os.path.join(fold_dir, "train_pat_scaled.pkl"))
        val_scaled_df.to_pickle(os.path.join(fold_dir, "val_pat_scaled.pkl"))
        test_scaled_df.to_pickle(os.path.join(fold_dir, "test_pat_scaled.pkl"))

        train_scaled_df.to_csv(os.path.join(fold_dir, "train_pat_scaled.csv"), index=False)
        val_scaled_df.to_csv(os.path.join(fold_dir, "val_pat_scaled.csv"), index=False)
        test_scaled_df.to_csv(os.path.join(fold_dir, "test_pat_scaled.csv"), index=False)

        # fold summary
        test_stats = get_test_split_stats(test_df, target_pos_frac=target_pos_frac)

        print(f"\nfold_{fold_num} summary")
        print(f"train shape: {train_df.shape}, patients: {train_df['patient_id'].nunique()}, pos_frac: {train_df['label'].mean():.4f}")
        print(f"val   shape: {val_df.shape}, patients: {val_df['patient_id'].nunique()}, pos_frac: {val_df['label'].mean():.4f}")
        print(f"test  shape: {test_df.shape}, patients: {test_df['patient_id'].nunique()}, pos_frac: {test_df['label'].mean():.4f}")
        print(f"test_sz={test_stats['pos_count']}, test_ns={test_stats['neg_count']}, test_total={test_stats['total_count']}")
        print(f"test ratio distance from 1:7 target: {test_stats['ratio_dist']:.6f}")
        print(f"scaled files saved in: {fold_dir}")
        




# function to save all major output tables
def save_main_tables(base_df, full_feat_df, filtered_df, dropped_cols, output_root):
    base_df[["pstrst", "patient_id", "label"]].to_csv(
        os.path.join(output_root, "segments_metadata.csv"), index=False
    )

    full_feat_df.to_csv(
        os.path.join(output_root, "all_features_before_corr_filter.csv"), index=False
    )

    filtered_df.to_csv(
        os.path.join(output_root, "all_features_after_corr_filter.csv"), index=False
    )

    pd.DataFrame({"dropped_feature": dropped_cols}).to_csv(
        os.path.join(output_root, "dropped_high_corr_features.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract EEG features from normalized npy segments, apply Spearman correlation filtering, and create 5-fold patient-wise stratified splits."
    )
    parser.add_argument(
        "input_root",
        type=str,
        help="Path to normalized_eeg_segments, e.g. /media/data/ukumar/iBehave/data_files/feb25/normalized_eeg_segments"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=256,
        help="Sampling frequency of normalized eeg segments"
    )
    parser.add_argument(
        "--corr_thresh",
        type=float,
        default=0.97,
        help="Spearman correlation threshold for removing highly correlated features"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for fold creation"
    )
    parser.add_argument(
    "--outer_search_trials",
    type=int,
    default=50,
    help="Number of candidate grouped-stratified outer 5-fold runs to try for balancing test support and ratio"
    )

    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    parent_dir = os.path.dirname(input_root)

    # feature tables to go inside parent_folder/feature_tables/
    feature_output_root = os.path.join(parent_dir, "feature_tables")

    # folds should go directly in parent folder 
    fold_output_root = parent_dir

    os.makedirs(feature_output_root, exist_ok=True)
    create_output_dirs(fold_output_root)

    base_df = collect_all_npy_files(input_root=input_root)
    if base_df.empty:
        raise RuntimeError("No .npy files found under the given input_root.")

    print(f"total eeg segments found: {len(base_df)}")
    print(f"unique patients found: {base_df['patient_id'].nunique()}")
    print(f"seizure-onset segments: {(base_df['label'] == 1).sum()}")
    print(f"non-seizure segments: {(base_df['label'] == 0).sum()}")
    
    # extracting features
    full_feat_df = build_full_feature_dataframe(base_df=base_df, fs=args.fs)

    # clean dataframe
    meta_cols = ["pstrst", "patient_id", "label"]
    full_feat_df = clean_feature_dataframe(full_feat_df, meta_cols) 
    
    print("\nSanity check after cleaning:")
    print(full_feat_df.isna().sum().sum(), "total NaNs remaining")
    print("label distribution:")
    print(full_feat_df["label"].value_counts(normalize=True))   

    filtered_df, dropped_cols = spearman_correlation_filter(
        df=full_feat_df,
        meta_cols=meta_cols,
        threshold=args.corr_thresh
    )

    print(f"total columns before filtering: {full_feat_df.shape[1]}")
    print(f"total dropped highly correlated feature columns: {len(dropped_cols)}")
    print(f"total columns after filtering: {filtered_df.shape[1]}")

    save_main_tables(
        base_df=base_df,
        full_feat_df=full_feat_df,
        filtered_df=filtered_df,
        dropped_cols=dropped_cols,
        output_root=feature_output_root
    )
    
    make_group_stratified_5fold_splits(
    df=filtered_df,
    output_root=fold_output_root,
    random_state=args.random_state,
    outer_search_trials=args.outer_search_trials
)

    print(f"\nFeature tables saved under: {feature_output_root}")
    print(f"Fold pickle files saved under: {fold_output_root}/fold_1 ... fold_5")

if __name__ == "__main__":
    main()

import os
import math
import argparse
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text
import pandas as pd
import mne


# function to load data from postgresql table into pandas data frames
def load_df(table_name):
    try:
        engine = create_engine('postgresql+psycopg2://tuheeg_user:tuheeg@localhost:5432/tuheeg')

        # loading table data in a dataframe
        try:
            query = text(f"SELECT * FROM public.{table_name};")
        except Exception as ex:
            print(ex)
            return None

        with engine.connect() as conn:
            df = pd.read_sql_query(query, con=conn)

        return df

    except Exception as ex:
        print(ex)
        return None


# function to normalize channel names so matching is stable across edf files
def normalize_channel_name(ch_name):
    if ch_name is None:
        return None
    ch = str(ch_name).strip().upper()
    ch = ch.replace("-REF", "-REF")
    ch = ch.replace("-REF.", "-REF")
    ch = ch.replace("EEG ", "EEG ")
    ch = " ".join(ch.split())
    return ch


# function to create required output folders
def create_output_dirs(output_root):
    for split in ["train", "val", "test"]:
        for sub in ["sz", "ns"]:
            os.makedirs(os.path.join(output_root, split, sub), exist_ok=True)


# function to map TUH dir_type names to output split folder names
def map_dir_type_to_split(dir_type):
    if dir_type == "train":
        return "train"
    elif dir_type == "dev":
        return "val"
    elif dir_type == "eval":
        return "test"
    return None


# function to create full_session_id and pstr key in channel dataframe
def add_pstr_column(channel_df):
    channel_df = channel_df.copy()

    channel_df["patient_id"] = channel_df["patient_id"].astype(str)
    channel_df["session_id"] = channel_df["session_id"].astype(str)
    channel_df["t_id"] = channel_df["t_id"].astype(str)
    channel_df["tcp_ref"] = channel_df["tcp_ref"].astype(str)
    channel_df["label"] = channel_df["label"].astype(str)
    channel_df["session_date"] = channel_df["session_date"].astype(str)

    # session string example 
    channel_df["full_session_id"] = (
        channel_df["session_id"] + "_" +
        channel_df["session_date"].str.replace("-", "_", regex=False)
    )

    # unique file-level key for later use
    channel_df["pstr"] = (
        channel_df["patient_id"] + "__" +
        channel_df["full_session_id"] + "__" +
        channel_df["t_id"] + "__" +
        channel_df["tcp_ref"] + "__" +
        channel_df["label"]
    )

    return channel_df


# function to keep only fnsz records with allowed tcp_ref montage and seizure duration > 3 sec
def get_filtered_fnsz_df(channel_df):
    df = channel_df.copy()

    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df["seizure_duration"] = df["stop_time"] - df["start_time"]

    # only focal seizures
    df = df[df["label"].str.lower() == "fnsz"].copy()

    # keep only average reference style montages 
    # allowing *_tcp_ar and *_tcp_ar_a, rejecting *_tcp_le
    df = df[
        (
            df["tcp_ref"].str.contains("_tcp_ar", case=False, na=False)
            | df["tcp_ref"].str.contains("_tcp_ar_a", case=False, na=False)
        )
        & (~df["tcp_ref"].str.contains("_tcp_le", case=False, na=False))
    ].copy()

    # dropping focal seizure rows where stop_time - start_time <= 3 sec
    df = df[df["seizure_duration"] > 3].copy()

    return df


# function to get edf file lookup from patient table
def build_edf_lookup(patient_df):
    df = patient_df.copy()

    needed_cols = ["dir_type", "patient_id", "session_id", "session_date", "tcp_ref", "t_id", "file_path", "file_type"]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"patient table missing columns: {missing_cols}")

    edf_df = df[df["file_type"].astype(str).str.lower() == "edf"].copy()
    edf_df["session_date"] = edf_df["session_date"].astype(str)
    edf_df["full_session_id"] = (
        edf_df["session_id"].astype(str) + "_" +
        edf_df["session_date"].str.replace("-", "_", regex=False)
    )

    lookup = {}
    for row in edf_df.itertuples(index=False):
        key = (
            str(row.patient_id),
            str(row.full_session_id),
            str(row.t_id),
            str(row.tcp_ref),
        )
        lookup[key] = {
            "edf_path": str(row.file_path),
            "dir_type": str(row.dir_type),
        }

    return lookup


# function to define the 18 bipolar pairs in strict order
def get_channel_pairs():
    channel_pairs = [
        ("EEG FP2-REF", "EEG F8-REF"),
        ("EEG F8-REF", "EEG T4-REF"),
        ("EEG T4-REF", "EEG T6-REF"),
        ("EEG T6-REF", "EEG O2-REF"),
        ("EEG FP1-REF", "EEG F7-REF"),
        ("EEG F7-REF", "EEG T3-REF"),
        ("EEG T3-REF", "EEG T5-REF"),
        ("EEG T5-REF", "EEG O1-REF"),
        ("EEG FP2-REF", "EEG F4-REF"),
        ("EEG F4-REF", "EEG C4-REF"),
        ("EEG C4-REF", "EEG P4-REF"),
        ("EEG P4-REF", "EEG O2-REF"),
        ("EEG FP1-REF", "EEG F3-REF"),
        ("EEG F3-REF", "EEG C3-REF"),
        ("EEG C3-REF", "EEG P3-REF"),
        ("EEG P3-REF", "EEG O1-REF"),
        ("EEG FZ-REF", "EEG CZ-REF"),
        ("EEG CZ-REF", "EEG PZ-REF")
    ]
    return [(normalize_channel_name(a), normalize_channel_name(b)) for a, b in channel_pairs]


# function to load an edf file, filter it, notch it, resample it and convert it to 18 bipolar channels
def preprocess_edf_to_bipolar(edf_path, sfreq=256, l_freq=0.1, h_freq=64.0, notch_freq=60.0):
    channel_pairs = get_channel_pairs()

    # reading edf with preload so that filters and resampling can be applied
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # making channel names more stable for matching
    rename_map = {}
    for ch in raw.ch_names:
        rename_map[ch] = normalize_channel_name(ch)
    raw.rename_channels(rename_map)

    # butterworth bandpass filtering to 0.1 to 64 Hz
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose="ERROR"
    )

    # notch filter to remove line noise at 60 Hz
    raw.notch_filter(freqs=[notch_freq], method="iir", verbose="ERROR")

    # resampling all files to a fixed sfreq
    raw.resample(sfreq=sfreq, verbose="ERROR")

    ch_to_idx = {normalize_channel_name(ch): idx for idx, ch in enumerate(raw.ch_names)}
    full_data = raw.get_data()
    n_times = full_data.shape[1]

    # building the 18 bipolar channels, if a channel is missing then that bipolar row stays zero
    bipolar_data = np.zeros((18, n_times), dtype=np.float32)

    for i, (ch_a, ch_b) in enumerate(channel_pairs):
        idx_a = ch_to_idx.get(ch_a, None)
        idx_b = ch_to_idx.get(ch_b, None)

        if idx_a is not None and idx_b is not None:
            bipolar_data[i, :] = full_data[idx_a, :] - full_data[idx_b, :]
        else:
            bipolar_data[i, :] = 0.0

    duration_sec = n_times / float(sfreq)

    return bipolar_data, duration_sec


# function to extract one fixed-length window and zero pad at the end if file ends early
def extract_window(bipolar_data, start_sec, window_sec=8, sfreq=256):
    start_idx = int(round(start_sec * sfreq))
    end_idx = start_idx + int(window_sec * sfreq)

    out = np.zeros((bipolar_data.shape[0], int(window_sec * sfreq)), dtype=np.float32)

    if start_idx >= bipolar_data.shape[1]:
        return out

    clipped_end = min(end_idx, bipolar_data.shape[1])
    take = bipolar_data[:, start_idx:clipped_end]
    out[:, :take.shape[1]] = take

    return out


# function to check if two intervals overlap
def intervals_overlap(a_start, a_stop, b_start, b_stop):
    return max(a_start, b_start) < min(a_stop, b_stop)


# function to check whether a candidate window overlaps any forbidden interval
def overlaps_any_interval(st, en, intervals):
    for a, b in intervals:
        if intervals_overlap(st, en, a, b):
            return True
    return False


# function to group fnsz annotations into session/file-level seizure events
# constraint applied here:
# if more than one seizures are present in a session, next section is considered only when
# next start is more than 600 seconds away from the first start of current section
def group_fnsz_events(file_fnsz_df, max_gap_from_first_start=600.0):
    if file_fnsz_df.empty:
        return []

    df = file_fnsz_df.sort_values(["start_time", "stop_time"]).reset_index(drop=True)

    events = []
    current_rows = []
    current_first_start = None

    for row in df.itertuples(index=False):
        st = float(row.start_time)
        sp = float(row.stop_time)

        if current_first_start is None:
            current_rows = [(st, sp)]
            current_first_start = st
        else:
            # keeping annotations in same seizure section as long as next start is within 600 sec
            if st <= (current_first_start + max_gap_from_first_start):
                current_rows.append((st, sp))
            else:
                event_start = min(x[0] for x in current_rows)
                event_stop = max(x[1] for x in current_rows)
                events.append((event_start, event_stop))

                current_rows = [(st, sp)]
                current_first_start = st

    if len(current_rows) > 0:
        event_start = min(x[0] for x in current_rows)
        event_stop = max(x[1] for x in current_rows)
        events.append((event_start, event_stop))

    return events


# function to create seizure-onset window start times for one seizure event
# constraints applied here:
# 1) first st starts at start_time - (n-d)
# 2) st is clipped to 0 if it becomes negative
# 3) last st stops at min(start_time + 20, stop_time - 1)
# 4) if stop_time - start_time < n, only one onset window is made
def get_sz_window_starts(event_start, event_stop, n=8, d=1):
    duration = event_stop - event_start
    first_st = max(0, int(math.floor(event_start - (n - d))))

    if duration < n:
        return [first_st]

    max_st = int(math.floor(min(event_start + 20, event_stop - 1)))

    if max_st < first_st:
        return [first_st]

    return list(range(first_st, max_st + 1, d))

# function to check whether a seizure-onset window contains
# at least one channel onset time
def window_contains_channel_onset(window_start, window_len, event_channel_df):
    window_stop = window_start + window_len

    for row in event_channel_df.itertuples(index=False):
        onset_time = float(row.start_time)

        if window_start <= onset_time < window_stop:
            return True

    return False

# function to build all seizure-onset window metadata and the onset intervals for sanity checks later
def build_sz_window_metadata(file_fnsz_df, file_info, n=8, d=1):
    events = group_fnsz_events(file_fnsz_df, max_gap_from_first_start=600.0)

    if len(events) == 0:
        return [], []

    first_row = file_fnsz_df.iloc[0]
    pstr = first_row["pstr"]
    patient_id = first_row["patient_id"]
    dir_type = file_info["dir_type"]

    manifest_rows = []
    onset_intervals = []

    seen_pstrst = set()

    for event_start, event_stop in events:

    # all channel annotations belonging to this seizure event
    event_channel_df = file_fnsz_df[
        (file_fnsz_df["start_time"] >= event_start) &
        (file_fnsz_df["start_time"] <= event_stop)
    ]

    starts = get_sz_window_starts(event_start, event_stop, n=n, d=d)

    for st in starts:
        # keep only windows containing at least one channel onset
        if not window_contains_channel_onset(
            window_start=st,
            window_len=n,
            event_channel_df=event_channel_df
        ):
            continue

        pstrst = f"{pstr}__{int(st)}"

        if pstrst in seen_pstrst:
            continue

        seen_pstrst.add(pstrst)

        manifest_rows.append({
                "patient_id": patient_id,
                "pstr": pstr,
                "pstrst": pstrst,
                "label": "fnsz",
                "dir_type": dir_type,
                "sz_label": "sz",
                "window_start_sec": int(st),
            })

            # storing onset window interval for later sanity check against non-seizure windows
            onset_intervals.append((int(st), int(st) + n))

    return manifest_rows, onset_intervals


# function to collect all annotation intervals from all labels for one file
# this is used so that non-seizure windows do not overlap any annotated seizure/background label region
def get_all_annotation_intervals(file_all_labels_df):
    intervals = []
    if file_all_labels_df.empty:
        return intervals

    for row in file_all_labels_df.itertuples(index=False):
        try:
            st = float(row.start_time)
            sp = float(row.stop_time)
            if pd.notna(st) and pd.notna(sp) and sp > st:
                intervals.append((st, sp))
        except Exception:
            pass

    return intervals


# function to build candidate non-seizure windows from one file
# constraints applied here:
# 1) same fnsz sessions only
# 2) window must not overlap any annotation interval from any label
# 3) window must not overlap any seizure-onset window
# 4) before a focal seizure onset, non-seizure start must be at least 2*n seconds earlier
def build_ns_window_metadata(file_fnsz_df, file_all_labels_df, file_info, duration_sec, onset_intervals, n=8, d=1):
    events = group_fnsz_events(file_fnsz_df, max_gap_from_first_start=600.0)

    if len(events) == 0:
        return []

    first_row = file_fnsz_df.iloc[0]
    pstr = first_row["pstr"]
    patient_id = first_row["patient_id"]
    dir_type = file_info["dir_type"]
    split_name = map_dir_type_to_split(dir_type)

    all_ann_intervals = get_all_annotation_intervals(file_all_labels_df)

    max_st = int(math.floor(duration_sec - n))
    if max_st < 0:
        return []

    ns_rows = []
    seen_pstrst = set()

    for st in range(0, max_st + 1, d):
        en = st + n

        # must not overlap any annotated interval from any label
        if overlaps_any_interval(st, en, all_ann_intervals):
            continue

        # sanity check that it never overlaps already defined seizure-onset windows
        if overlaps_any_interval(st, en, onset_intervals):
            continue

        # keeping a stricter gap before each focal seizure onset
        too_close_to_any_fnsz_onset = False
        for event_start, event_stop in events:
            if st < event_start and st > (event_start - 2 * n):
                too_close_to_any_fnsz_onset = True
                break

        if too_close_to_any_fnsz_onset:
            continue

        pstrst = f"{pstr}__{int(st)}"

        if pstrst in seen_pstrst:
            continue

        seen_pstrst.add(pstrst)
        ns_rows.append({
            "patient_id": patient_id,
            "pstr": pstr,
            "pstrst": pstrst,
            "label": "fnsz",
            "dir_type": dir_type,
            "split_name": split_name,
            "sz_label": "ns",
            "window_start_sec": int(st),
        })

    return ns_rows
    
# function to count saved seizure-onset rows split-wise
def get_splitwise_sz_counts(saved_manifest_rows):
    split_sz_counts = {"train": 0, "val": 0, "test": 0}

    for row in saved_manifest_rows:
        if row["sz_label"] == "sz":
            split_name = map_dir_type_to_split(row["dir_type"])
            if split_name in split_sz_counts:
                split_sz_counts[split_name] += 1

    return split_sz_counts
    
# function to collect all candidate non-seizure rows split-wise from all processed fnsz files
def collect_all_ns_candidates(all_file_infos, n=8, d=1):
    split_candidates = {"train": [], "val": [], "test": []}

    for file_obj in all_file_infos:
        file_fnsz_df = file_obj["file_fnsz_df"]
        file_all_labels_df = file_obj["file_all_labels_df"]
        file_info = file_obj["file_info"]
        duration_sec = file_obj["duration_sec"]
        onset_intervals = file_obj["onset_intervals"]

        ns_rows = build_ns_window_metadata(
            file_fnsz_df=file_fnsz_df,
            file_all_labels_df=file_all_labels_df,
            file_info=file_info,
            duration_sec=duration_sec,
            onset_intervals=onset_intervals,
            n=n,
            d=d
        )

        if len(ns_rows) == 0:
            continue

        split_name = ns_rows[0]["split_name"]
        if split_name in split_candidates:
            split_candidates[split_name].extend(ns_rows)

    return split_candidates
    
# function to allocate non-seizure rows such that each split first tries to reach 1:7
# if a split cannot reach 1:7 from its own valid candidates, deficit can be filled from other splits having extra candidates
def allocate_ns_rows_with_split_priority(split_candidates, split_sz_counts, target_ratio=7):
    split_targets = {
        split_name: split_sz_counts[split_name] * target_ratio
        for split_name in ["train", "val", "test"]
    }

    allocated = {"train": [], "val": [], "test": []}
    deficits = {"train": 0, "val": 0, "test": 0}
    leftovers = {"train": [], "val": [], "test": []}

    # first trying to satisfy each split from its own candidate pool
    for split_name in ["train", "val", "test"]:
        target_ns = split_targets[split_name]
        candidates = split_candidates.get(split_name, [])

        if len(candidates) >= target_ns:
            allocated[split_name] = candidates[:target_ns]
            leftovers[split_name] = candidates[target_ns:]
            deficits[split_name] = 0
        else:
            allocated[split_name] = candidates
            leftovers[split_name] = []
            deficits[split_name] = target_ns - len(candidates)

    # then trying to fill deficits from other splits that have extra candidates
    # this is especially useful if train cannot reach 1:7 but val/test have more than enough
    for needy_split in ["train", "val", "test"]:
        if deficits[needy_split] <= 0:
            continue

        need = deficits[needy_split]

        for donor_split in ["train", "val", "test"]:
            if donor_split == needy_split:
                continue

            donor_left = leftovers[donor_split]
            if len(donor_left) == 0:
                continue

            take = min(need, len(donor_left))
            allocated[needy_split].extend(donor_left[:take])
            leftovers[donor_split] = donor_left[take:]
            need -= take

            if need == 0:
                break

        deficits[needy_split] = need

    return allocated, split_targets, deficits


# function to save one .npy eeg segment
def save_npy_segment(arr, output_root, dir_type, sz_label, pstrst):
    split = map_dir_type_to_split(dir_type)
    if split is None:
        return None

    save_path = os.path.join(output_root, split, sz_label, f"{pstrst}.npy")
    np.save(save_path, arr.astype(np.float32))
    return save_path


# function to save all seizure-onset windows for one edf file
def save_sz_segments_for_file(bipolar_data, sz_rows, output_root, n=8, sfreq=256):
    saved_rows = []

    for row in sz_rows:
        st = int(row["window_start_sec"])
        pstrst = row["pstrst"]
        arr = extract_window(bipolar_data, st, window_sec=n, sfreq=sfreq)
        save_npy_segment(arr, output_root, row["dir_type"], row["sz_label"], pstrst)
        saved_rows.append({
            "patient_id": row["patient_id"],
            "pstr": row["pstr"],
            "pstrst": row["pstrst"],
            "label": row["label"],
            "dir_type": row["dir_type"],
            "sz_label": row["sz_label"],
        })

    return saved_rows


# function to save selected non-seizure windows for one edf file
def save_ns_segments_for_file(bipolar_data, ns_rows, output_root, n=8, sfreq=256):
    saved_rows = []

    for row in ns_rows:
        st = int(row["window_start_sec"])
        pstrst = row["pstrst"]
        arr = extract_window(bipolar_data, st, window_sec=n, sfreq=sfreq)
        save_npy_segment(arr, output_root, row["dir_type"], row["sz_label"], pstrst)
        saved_rows.append({
            "patient_id": row["patient_id"],
            "pstr": row["pstr"],
            "pstrst": row["pstrst"],
            "label": row["label"],
            "dir_type": row["dir_type"],
            "sz_label": row["sz_label"],
        })

    return saved_rows


# function to run the whole preprocessing pipeline and save segments + manifest dataframe
def generate_segments(output_root, n=8, d=1, sfreq=256):
    create_output_dirs(output_root)

    # loading tables from postgresql
    patient_df = load_df("patient")
    channel_df = load_df("channel_annotations")

    if patient_df is None or channel_df is None:
        raise RuntimeError("could not load patient or channel_annotations table")

    # adding pstr key
    channel_df = add_pstr_column(channel_df)

    # filtered focal seizure records only
    fnsz_df = get_filtered_fnsz_df(channel_df)

    if fnsz_df.empty:
        print("No focal seizure records found after filtering.")
        return

    # edf lookup from patient table
    edf_lookup = build_edf_lookup(patient_df)

    # file-level grouping key
    key_cols = ["patient_id", "full_session_id", "t_id", "tcp_ref", "pstr"]

    # to build all manifest rows here
    saved_manifest_rows = []

    # first pass: build and save all seizure-onset windows
    all_file_infos = []
    total_sz_count = 0

    grouped = fnsz_df.groupby(key_cols, dropna=False)

    for key, file_fnsz_df in grouped:
        patient_id, full_session_id, t_id, tcp_ref, pstr = key

        edf_key = (str(patient_id), str(full_session_id), str(t_id), str(tcp_ref))
        if edf_key not in edf_lookup:
            print(f"EDF path not found in patient table for {edf_key}")
            continue

        file_info = edf_lookup[edf_key]
        edf_path = file_info["edf_path"]

        if not os.path.exists(edf_path):
            print(f"EDF file missing on disk: {edf_path}")
            continue

        try:
            bipolar_data, duration_sec = preprocess_edf_to_bipolar(
                edf_path=edf_path,
                sfreq=sfreq,
                l_freq=0.1,
                h_freq=64.0,
                notch_freq=60.0
            )
        except Exception as ex:
            print(f"Could not process EDF file {edf_path}")
            print(ex)
            continue

        # all labels from same file for overlap checks later
        file_all_labels_df = channel_df[
            (channel_df["patient_id"].astype(str) == str(patient_id)) &
            (channel_df["full_session_id"].astype(str) == str(full_session_id)) &
            (channel_df["t_id"].astype(str) == str(t_id)) &
            (channel_df["tcp_ref"].astype(str) == str(tcp_ref))
        ].copy()

        sz_rows, onset_intervals = build_sz_window_metadata(
            file_fnsz_df=file_fnsz_df,
            file_info=file_info,
            n=n,
            d=d
        )

        if len(sz_rows) == 0:
            continue

        saved_sz_rows = save_sz_segments_for_file(
            bipolar_data=bipolar_data,
            sz_rows=sz_rows,
            output_root=output_root,
            n=n,
            sfreq=sfreq
        )

        saved_manifest_rows.extend(saved_sz_rows)
        total_sz_count += len(saved_sz_rows)

        all_file_infos.append({
            "file_fnsz_df": file_fnsz_df,
            "file_all_labels_df": file_all_labels_df,
            "file_info": file_info,
            "edf_path": edf_path,
            "duration_sec": duration_sec,
            "onset_intervals": onset_intervals,
        })

        print(f"Saved {len(saved_sz_rows)} seizure-onset windows from {edf_path}")

    print(f"Total seizure-onset windows saved: {total_sz_count}")

    # computing split-wise seizure counts and required non-seizure targets
    split_sz_counts = get_splitwise_sz_counts(saved_manifest_rows)
    print("Split-wise seizure counts:", split_sz_counts)

    split_candidates = collect_all_ns_candidates(
        all_file_infos=all_file_infos,
        n=n,
        d=d
    )

    print("Candidate non-seizure windows before allocation:")
    print({
        "train": len(split_candidates["train"]),
        "val": len(split_candidates["val"]),
        "test": len(split_candidates["test"]),
    })

    allocated_ns_rows, split_targets, deficits = allocate_ns_rows_with_split_priority(
        split_candidates=split_candidates,
        split_sz_counts=split_sz_counts,
        target_ratio=7
    )

    print("Split-wise non-seizure targets:", split_targets)
    print("Remaining deficits after borrowing extras:", deficits)
    print("Final allocated non-seizure counts:")
    print({
        "train": len(allocated_ns_rows["train"]),
        "val": len(allocated_ns_rows["val"]),
        "test": len(allocated_ns_rows["test"]),
    })

    # now saving allocated non-seizure rows
    total_ns_count = 0

    # to avoid reprocessing same edf too many times, grouping allocations by edf identity
    alloc_map = {}
    for split_name in ["train", "val", "test"]:
        for row in allocated_ns_rows[split_name]:
            pstr = row["pstr"]
            alloc_map.setdefault(pstr, []).append(row)

    for file_obj in all_file_infos:
        file_fnsz_df = file_obj["file_fnsz_df"]
        file_info = file_obj["file_info"]
        edf_path = file_obj["edf_path"]

        first_row = file_fnsz_df.iloc[0]
        pstr = first_row["pstr"]

        if pstr not in alloc_map:
            continue

        ns_rows_to_save = alloc_map[pstr]

        try:
            bipolar_data, _ = preprocess_edf_to_bipolar(
                edf_path=edf_path,
                sfreq=sfreq,
                l_freq=0.1,
                h_freq=64.0,
                notch_freq=60.0
            )
        except Exception as ex:
            print(f"Could not reprocess EDF file for non-seizure windows {edf_path}")
            print(ex)
            continue

        saved_ns_rows = save_ns_segments_for_file(
            bipolar_data=bipolar_data,
            ns_rows=ns_rows_to_save,
            output_root=output_root,
            n=n,
            sfreq=sfreq
        )

        saved_manifest_rows.extend(saved_ns_rows)
        total_ns_count += len(saved_ns_rows)

        print(f"Saved {len(saved_ns_rows)} non-seizure windows from {edf_path}")

    print(f"Total non-seizure windows saved: {total_ns_count}")
    if total_sz_count > 0:
        print(f"Overall ratio obtained: 1:{(total_ns_count / total_sz_count):.4f}")

    # saving final manifest dataframe
    manifest_df = pd.DataFrame(saved_manifest_rows)
    if not manifest_df.empty:
        manifest_df = manifest_df[["patient_id", "pstr", "pstrst", "label", "dir_type", "sz_label"]].copy()
        manifest_path = os.path.join(output_root, "segment_manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Saved manifest csv at: {manifest_path}")

        for split_dir_type, split_name in [("train", "train"), ("dev", "val"), ("eval", "test")]:
            split_df = manifest_df[manifest_df["dir_type"] == split_dir_type].copy()
            if not split_df.empty:
                split_manifest_path = os.path.join(output_root, f"segment_manifest_{split_name}.csv")
                split_df.to_csv(split_manifest_path, index=False)
                print(f"Saved split manifest csv at: {split_manifest_path}")
    else:
        print("No manifest rows created.")


def main():
    parser = argparse.ArgumentParser(description="Generate 8-second focal seizure-onset and non-seizure EEG segments from TUH tables.")
    parser.add_argument(
        "output_root",
        type=str,
        help="Root folder where segments will be saved, for example /media/data/ukumar/iBehave/data_files/feb25/eeg_segments"
    )
    parser.add_argument("--n", type=int, default=8, help="window length in seconds")
    parser.add_argument("--d", type=int, default=1, help="stride in seconds")
    parser.add_argument("--sfreq", type=int, default=256, help="target sampling frequency")

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    generate_segments(
        output_root=str(output_root),
        n=args.n,
        d=args.d,
        sfreq=args.sfreq
    )


if __name__ == "__main__":
    main()

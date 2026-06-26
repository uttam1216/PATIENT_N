import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


# function to create output folder structure same as eeg_segments
def create_output_dirs(output_root):
    for split in ["train", "val", "test"]:
        for cls in ["sz", "ns"]:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)


# function to collect all .npy files from eeg_segments preserving split and class structure
def collect_all_segment_files(input_root):
    all_files = []

    for split in ["train", "val", "test"]:
        for cls in ["sz", "ns"]:
            curr_dir = os.path.join(input_root, split, cls)
            if not os.path.exists(curr_dir):
                continue

            for fname in os.listdir(curr_dir):
                if fname.endswith(".npy"):
                    fpath = os.path.join(curr_dir, fname)
                    all_files.append({
                        "file_path": fpath,
                        "file_name": fname,
                        "split": split,
                        "class_type": cls,
                    })

    return all_files


# function to get patient_id from pstrst file name
# example:
# patID__session__tid__tcpRef__szType__windowStartTime.npy
# patient_id here is patID
def get_patient_id_from_filename(file_name):
    base_name = file_name.replace(".npy", "")
    parts = base_name.split("__")
    if len(parts) < 2:
        raise ValueError(f"unexpected file name format: {file_name}")
    return parts[0]


# function to group segment file paths patient-wise
def group_files_by_patient(all_files):
    patient_files = defaultdict(list)

    for item in all_files:
        patient_id = get_patient_id_from_filename(item["file_name"])
        patient_files[patient_id].append(item)

    return patient_files


# function to compute per-patient per-channel mean and std taking all segments together
# each segment is expected to have shape (18, 2048)
def compute_patient_channel_stats(file_items):
    channel_sum = None
    channel_sq_sum = None
    total_points_per_channel = 0

    for item in file_items:
        arr = np.load(item["file_path"])

        if arr.ndim != 2:
            raise ValueError(f"segment is not 2D: {item['file_path']} with shape {arr.shape}")

        if channel_sum is None:
            n_channels = arr.shape[0]
            channel_sum = np.zeros(n_channels, dtype=np.float64)
            channel_sq_sum = np.zeros(n_channels, dtype=np.float64)

        # summing across time axis for each channel
        channel_sum += arr.sum(axis=1)
        channel_sq_sum += (arr ** 2).sum(axis=1)
        total_points_per_channel += arr.shape[1]

    if channel_sum is None or total_points_per_channel == 0:
        raise ValueError("no valid segment data found while computing patient statistics")

    channel_mean = channel_sum / total_points_per_channel
    channel_var = (channel_sq_sum / total_points_per_channel) - (channel_mean ** 2)
    channel_var = np.maximum(channel_var, 0.0)
    channel_std = np.sqrt(channel_var)

    # avoiding division by zero for channels with constant or all-zero signal
    channel_std[channel_std == 0] = 1.0

    return channel_mean.astype(np.float32), channel_std.astype(np.float32)


# function to normalize one segment using per-patient per-channel mean and std
def normalize_segment(arr, channel_mean, channel_std):
    # reshaping mean and std to (channels, 1) so broadcasting works correctly
    mean_2d = channel_mean[:, np.newaxis]
    std_2d = channel_std[:, np.newaxis]
    norm_arr = (arr - mean_2d) / std_2d
    return norm_arr.astype(np.float32)


# function to save normalized segment in same relative folder structure
def save_normalized_segment(norm_arr, input_root, output_root, original_file_path):
    rel_path = os.path.relpath(original_file_path, input_root)
    out_path = os.path.join(output_root, rel_path)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    np.save(out_path, norm_arr)
    return out_path


# function to save patient-wise normalization statistics for later reuse or checking
def save_patient_stats(stats_dict, output_root):
    stats_dir = os.path.join(output_root, "patient_stats")
    os.makedirs(stats_dir, exist_ok=True)

    for patient_id, stat_obj in stats_dict.items():
        save_path = os.path.join(stats_dir, f"{patient_id}_stats.npz")
        np.savez(
            save_path,
            mean=stat_obj["mean"],
            std=stat_obj["std"]
        )


# function to run complete normalization pipeline
def normalize_all_segments(input_root, output_root):
    create_output_dirs(output_root)

    all_files = collect_all_segment_files(input_root)
    if len(all_files) == 0:
        print("No .npy segment files found in input folder.")
        return

    patient_files = group_files_by_patient(all_files)

    print(f"Total segment files found: {len(all_files)}")
    print(f"Total patients found: {len(patient_files)}")

    patient_stats = {}
    total_saved = 0

    # first computing patient-wise stats
    for patient_id, file_items in patient_files.items():
        try:
            channel_mean, channel_std = compute_patient_channel_stats(file_items)
            patient_stats[patient_id] = {
                "mean": channel_mean,
                "std": channel_std,
            }
            print(f"Computed normalization stats for patient {patient_id} using {len(file_items)} segments")
        except Exception as ex:
            print(f"Could not compute stats for patient {patient_id}")
            print(ex)

    # saving stats also for later reference
    save_patient_stats(patient_stats, output_root)

    # now normalizing and saving all files
    for patient_id, file_items in patient_files.items():
        if patient_id not in patient_stats:
            continue

        channel_mean = patient_stats[patient_id]["mean"]
        channel_std = patient_stats[patient_id]["std"]

        for item in file_items:
            try:
                arr = np.load(item["file_path"])
                norm_arr = normalize_segment(arr, channel_mean, channel_std)
                save_normalized_segment(
                    norm_arr=norm_arr,
                    input_root=input_root,
                    output_root=output_root,
                    original_file_path=item["file_path"]
                )
                total_saved += 1
            except Exception as ex:
                print(f"Could not normalize file: {item['file_path']}")
                print(ex)

    print(f"Total normalized segments saved: {total_saved}")
    print(f"Normalized segments saved under: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Per-patient per-channel z-score normalization of EEG segment .npy files.")
    parser.add_argument(
        "input_root",
        type=str,
        help="Path to eeg_segments folder, e.g. /media/data/ukumar/iBehave/data_files/feb25/eeg_segments"
    )
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)

    # making normalized_eeg_segments in the parent folder parallel to eeg_segments
    parent_dir = os.path.dirname(input_root)
    output_root = os.path.join(parent_dir, "normalized_eeg_segments")

    normalize_all_segments(input_root=input_root, output_root=output_root)


if __name__ == "__main__":
    main()

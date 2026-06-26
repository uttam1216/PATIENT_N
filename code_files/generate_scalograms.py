import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image


# function to load eeg data from .npy and check if it has exactly 18 channels
def load_and_check_18ch_npy(npy_path, expected_channels=18):
    """
    Load EEG data from .npy and check if it has exactly 18 channels.

    Returns:
        (True, eeg_array) if valid
        (False, None) otherwise
    """
    try:
        eeg = np.load(npy_path)

        if not isinstance(eeg, np.ndarray):
            return False, None

        if eeg.ndim != 2:
            return False, None

        if eeg.shape[0] != expected_channels:
            return False, None

        return True, eeg

    except Exception:
        return False, None


# function to resize a grayscale image array to target size
def resize_grayscale_image(img_array, target_size=(484, 484)):
    """
    img_array is expected to be float array in [0, 1].
    target_size is (width, height).
    """
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    pil_img = pil_img.resize(target_size, resample=Image.BILINEAR)
    resized = np.array(pil_img).astype(np.float32) / 255.0
    return resized


# function to generate a single stacked Morlet scalogram and save it as PNG
def generate_and_save_stacked_scalogram(
    eeg,
    savepath,
    filename,
    fs=256,
    num_scales=64,
    wavelet_name="cmor1.5-1.0",
    target_size=(484, 484),
    cmap="jet"
):
    """
    Generates a single stacked CWT scalogram image from a normalized EEG segment
    of shape (18, timepoints), resizes it to target_size, and saves as PNG.
    """

    assert eeg.ndim == 2, "EEG must be 2D"
    assert eeg.shape[0] == 18, "EEG must have exactly 18 channels"

    os.makedirs(savepath, exist_ok=True)

    # using logarithmically spaced scales 
    scales = np.logspace(1, 3, num_scales)

    stacked_scalograms = []

    for ch in range(18):
        signal_1d = eeg[ch].astype(np.float32)

        coeffs, freqs = pywt.cwt(
            data=signal_1d,
            scales=scales,
            wavelet=wavelet_name,
            sampling_period=1.0 / fs
        )

        scalogram = np.abs(coeffs)

        # normalize per channel
        ch_min = scalogram.min()
        ch_max = scalogram.max()
        scalogram = (scalogram - ch_min) / (ch_max - ch_min + 1e-8)

        stacked_scalograms.append(scalogram)

    # stacking vertically -> shape becomes (18 * num_scales, timepoints)
    stacked_scalogram = np.vstack(stacked_scalograms)

    # normalize globally before resizing
    global_min = stacked_scalogram.min()
    global_max = stacked_scalogram.max()
    stacked_scalogram = (stacked_scalogram - global_min) / (global_max - global_min + 1e-8)

    # resize to fixed resolution
    resized_scalogram = resize_grayscale_image(
        img_array=stacked_scalogram,
        target_size=target_size
    )

    save_file = os.path.join(
        savepath,
        filename.replace(".npy", ".png")
    )

    # saving as color image using the chosen colormap
    plt.imsave(save_file, resized_scalogram, cmap=cmap, origin="lower")


# function to create output folders
def create_output_dirs(output_root):
    os.makedirs(os.path.join(output_root, "sz"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "ns"), exist_ok=True)


# function to collect all npy files from normalized_eeg_segments
# train/val/test are flattened, but sz/ns are preserved
def collect_input_files(input_root):
    all_files = []

    for split_name in ["train", "val", "test"]:
        for cls_name in ["sz", "ns"]:
            curr_dir = os.path.join(input_root, split_name, cls_name)

            if not os.path.exists(curr_dir):
                continue

            for fname in os.listdir(curr_dir):
                if fname.endswith(".npy"):
                    all_files.append({
                        "file_path": os.path.join(curr_dir, fname),
                        "file_name": fname,
                        "split_name": split_name,
                        "class_name": cls_name,
                    })

    return all_files


# function to generate scalograms from all normalized eeg segments
def generate_scalograms_from_npy_segments(
    input_root,
    output_root,
    fs=256,
    num_scales=64,
    wavelet_name="cmor1.5-1.0",
    target_size=(484, 484),
    cmap="jet"
):
    create_output_dirs(output_root)

    all_files = collect_input_files(input_root)

    if len(all_files) == 0:
        print("No .npy files found in input source folder.")
        return

    problem_files = []
    total_done = 0
    sz_done = 0
    ns_done = 0

    for item in all_files:
        npy_path = item["file_path"]
        npy_filename = item["file_name"]
        class_name = item["class_name"]   # sz or ns

        savepath = os.path.join(output_root, class_name)

        is_valid, eeg = load_and_check_18ch_npy(npy_path)

        if not is_valid:
            problem_files.append((npy_filename, "INVALID CHANNEL COUNT OR BAD FILE"))
            continue

        try:
            generate_and_save_stacked_scalogram(
                eeg=eeg,
                savepath=savepath,
                filename=npy_filename,
                fs=fs,
                num_scales=num_scales,
                wavelet_name=wavelet_name,
                target_size=target_size,
                cmap=cmap
            )
            total_done += 1

            if class_name == "sz":
                sz_done += 1
            elif class_name == "ns":
                ns_done += 1

        except Exception as ex:
            problem_files.append((npy_filename, str(ex)))

    print(f"Total scalograms saved: {total_done}")
    print(f"Saved in sz folder: {sz_done}")
    print(f"Saved in ns folder: {ns_done}")
    print(f"Problem files count: {len(problem_files)}")

    if len(problem_files) > 0:
        problem_df_path = os.path.join(output_root, "problem_files.csv")
        with open(problem_df_path, "w") as f:
            f.write("filename,issue\n")
            for fname, issue in problem_files:
                issue = str(issue).replace(",", ";")
                f.write(f"{fname},{issue}\n")
        print(f"Problem files log saved at: {problem_df_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stacked scalograms from normalized EEG segment .npy files."
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
        help="Sampling frequency of eeg segments"
    )
    parser.add_argument(
        "--num_scales",
        type=int,
        default=64,
        help="Number of wavelet scales"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=484,
        help="Output image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=484,
        help="Output image height"
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor1.5-1.0",
        help="PyWavelets wavelet name, e.g. cmor1.5-1.0"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Matplotlib colormap for saving images"
    )

    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    parent_dir = os.path.dirname(input_root)
    output_root = os.path.join(parent_dir, "scalograms")

    generate_scalograms_from_npy_segments(
        input_root=input_root,
        output_root=output_root,
        fs=args.fs,
        num_scales=args.num_scales,
        wavelet_name=args.wavelet,
        target_size=(args.width, args.height),
        cmap=args.cmap
    )


if __name__ == "__main__":
    main()
